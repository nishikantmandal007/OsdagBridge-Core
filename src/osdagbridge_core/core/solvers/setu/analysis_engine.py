import os
from time import perf_counter

from setu.postprocess.own_process import solve_in_own_process
from osdagbridge_core.core.bridge_types.plate_girder.designer import BridgeConfig
from osdagbridge_core.core.utils import common as C
from osdagbridge_core.core.utils.logger import bridge_logger
from .free_memory import available_memory_gb
from setu.irc6.temperature import effective_temperature_range
from setu.irc6.wind import design_wind_speed_mps, wind_pressure_pa
from .bridge_input import (bridge_input, construction_loads, construction_seismic_site, estimated_peak_gb, fill_unequal_depth_wind, normalise_gui_bracing,
                           reconcile_layout, seismic_site, temperature_site, wind_site, with_workers_gb, worker_gb, workers_that_fit)
from .girder_demands import case_demand, deflections_cache, girder_demands
from .live_load_report import live_load_report_values
from .run_progress import RunProgress
from .plot_dataset import CaseForces, governing_values_of_states, load_states, member_nodes, plot_dataset
from .unsupported_inputs import check_capability, simplifications


# One core is left for the analysis process, which solves every surface for the workers; measured (2026-11-07, setu 2693a09, 10-core
# Mac) it keeps at most about four busy: 2x11 m deck 99 / 103 / 108 s with 4 / 6 / 9 workers, 52 m deck 600 / 619 s with 4 / 6, each
# worker past four costing memory for nothing. Memory may allow fewer (workers_that_fit).
MOST_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))


# Say how big the run is, warn if it may not fit, and choose the workers that do.
def plan_run(d):
    nodes, alone = estimated_peak_gb(d)
    available = available_memory_gb()
    main, each = with_workers_gb(nodes), worker_gb(nodes)
    workers = workers_that_fit(MOST_WORKERS, main, each, available)
    gb = main + each * workers if workers > 1 else alone
    bridge_logger._emit("__progress__16", "progress")
    free = "free memory unknown, so no workers" if available is None else f"{available:.1f} GB free"
    bridge_logger.sub_step(f"setu: {nodes} deck nodes, estimated peak about {gb:.1f} GB with {workers} worker{'s' if workers > 1 else ''} ({free})")
    if available is not None and gb > available:
        bridge_logger.warning(f"This bridge may need about {gb:.1f} GB with only {available:.1f} GB free; close other programs or expect it to be slow")
    return workers


# The whole analysis in its own process: Stop ends it at once and its memory returns to the system when it finishes.
def solve(bridge_in, wind, seismic, temperature, braking_height_m, workers, **extras):
    tracker = RunProgress(extras.get("special_vehicle", False), extras.get("construction") is not None)
    values, forces = solve_in_own_process(
        bridge_in, wind, seismic, temperature, braking_height_m, workers=workers,
        phase=tracker.phase, progress=tracker.counts, check=bridge_logger.check_cancel, **extras)
    tracker.finish()
    return values, report_forces(forces)


# The design wind speed and pressure at the top of the superstructure (cl. 209.3, 209.3.5 as amended) for the report's wind table.
def wind_report(wind):
    if wind is None:
        return {}
    speed = design_wind_speed_mps(wind, wind.superstructure_top_m, wind.return_period_years)
    return {C.KEY_WL_HOURLY_MEAN_WIND: round(speed, 2), C.KEY_WL_HOURLY_WIND_PRESSURE: round(wind_pressure_pa(speed), 2)}


# The effective bridge temperature range and rise/fall (cl. 215.2) for the report's temperature table.
def temperature_report(temperature):
    if temperature is None:
        return {}
    t_min, t_max = effective_temperature_range(temperature.shade_max_c, temperature.shade_min_c, temperature.snowbound)
    half_range = (t_max - t_min) / 2
    return {C.KEY_TL_BRIDGE_TEMP_MIN: f"{t_min:.2f}", C.KEY_TL_BRIDGE_TEMP_MAX: f"{t_max:.2f}",
            C.KEY_TL_TEMP_RISE: f"{half_range:.2f}", C.KEY_TL_TEMP_FALL: f"{half_range:.2f}"}


# The wind and seismic forces the report's load tables print.
def report_forces(forces):
    report = {}
    if "wind_kn" in forces:
        wind_kn = forces["wind_kn"]
        report.update({C.KEY_WL_TRANSVERSE_WIND_FORCE: round(wind_kn["transverse"], 3),
                       C.KEY_WL_LONGITUDINAL_WIND_FORCE: round(wind_kn["longitudinal"], 3),
                       C.KEY_WL_VERTICAL_WIND_FORCE: round(wind_kn["vertical"], 3)})
    if "seismic_kn" in forces:
        report.update({C.KEY_SL_FORCE_LONGITUDINAL: round(forces["seismic_kn"]["longitudinal"], 3),
                       C.KEY_SL_FORCE_TRANSVERSE: round(forces["seismic_kn"]["transverse"], 3)})
    return report


class NotFromSetu(NotImplementedError, AttributeError):
    # An ospgrillage-only attribute: hasattr and getattr with a default see it as absent.
    pass


class DesignResultData(dict):
    # Grillage result records setu does not produce stop here.
    def __missing__(self, key):
        raise NotImplementedError(f"{key}: not provided by the setu engine")

    # Legacy consumers use get when looking for transverse members.
    def get(self, key, default=None):
        return self[key]


class NoGrillageElements:

    # setu's model has no grillage members, so every member query finds none.
    def get_element(self, member, options):
        return []


class SetuAnalysisEngine:

    # Run the public analysis once, pass its demands to the existing designer and its load states to the plots and tables.
    def __init__(self, bridge):
        normalise_gui_bracing(bridge.input_dict)
        reconcile_layout(bridge.input_dict)
        check_capability(bridge)
        bridge._build_dtos()
        bridge.material_props = bridge._build_material_props()
        self.bridge_input = bridge_input(bridge)
        self.wind = wind_site(bridge.input_dict)
        fill_unequal_depth_wind(self.wind, self.bridge_input)
        self.simplifications = simplifications(bridge)
        self.seismic = seismic_site(bridge.input_dict)
        self.temperature = temperature_site(bridge.input_dict)
        start = perf_counter()
        workers = plan_run(bridge.input_dict)
        d = bridge.input_dict
        values, self.report_forces = solve(self.bridge_input, self.wind, self.seismic, self.temperature, float(d[C.KEY_LL_ECCENTRICITY]), workers,
                                           special_vehicle=bool(d[C.KEY_LL_IRC_CLASS_SV]), construction=construction_loads(d),
                                           construction_seismic=construction_seismic_site(d))
        self.report_forces.update(wind_report(self.wind))
        self.report_forces.update(temperature_report(self.temperature))
        self.design_values, self.brace_values, self.diagrams = values.girders, values.braces, values.diagrams
        self.special_vehicle, self.construction = values.special_vehicle, values.construction
        self.analysis_seconds = perf_counter() - start
        self.configs = [BridgeConfig.from_plate_girder_bridge(bridge, g) for g in range(bridge._girder_count())]
        self.deflections_cache = deflections_cache(self.design_values, self.configs[0])
        self.per_girder_demands, self.per_girder_per_lc = girder_demands(
            self.design_values, self.configs, self.deflections_cache, self.seismic is not None, bridge.input_dict[C.KEY_LL_IRC_CLASS_FATIGUE],
            self.special_vehicle, self.construction)
        bridge._deflections_cache = self.deflections_cache
        self.model = NoGrillageElements()
        self._deduplicated_results = plot_dataset(values, bridge.get_edge_dist())
        self.states = load_states(values)
        self.state_values = governing_values_of_states(values.diagrams)

    # The output dock's design bars for one load case in its dropdown.
    def case_demand(self, girder_name, load_case):
        return case_demand(self.states, int(girder_name[1:]) - 1, load_case, self.per_girder_demands[girder_name],
                           self.design_values.dead, self.state_values)

    # Girder and deck design only need the canonical girder labels here.
    def get_result_data(self):
        return DesignResultData(girders={name: {} for name in self.per_girder_demands}, members=member_nodes(self._deduplicated_results),
                                forces=CaseForces(self._deduplicated_results),
                                loadcases=[str(name) for name in self._deduplicated_results.Loadcase.values])

    # ospgrillage's load-case objects and live model access are not part of the setu engine.
    def __getattr__(self, name):
        raise NotFromSetu(f"{name}: not provided by the setu engine")


# Replace the analysis stages while preserving girder and deck design.
def run_analysis(bridge):
    bridge.grillage_model = SetuAnalysisEngine(bridge)
    bridge_logger.sub_step(f"setu: analysis finished in {bridge.grillage_model.analysis_seconds:.0f} s")
    bridge.output_dict.update(live_load_report_values(bridge.grillage_model.bridge_input.span_m))
    bridge.output_dict.update(bridge.grillage_model.report_forces)
    bridge.output_dict[C.KEY_ANALYSIS_MESH] = bridge.input_dict[C.KEY_ANALYSIS_MESH]
    governed = [name for name, demand in bridge.grillage_model.per_girder_demands.items() if "special vehicle" in demand.governing_combination]
    if governed:
        bridge.grillage_model.simplifications.append(
            f"Class SV governs the ULS moment of {', '.join(governed)}: the plotted load states and case bars show the IRC traffic only")
    for note in bridge.grillage_model.simplifications:
        bridge_logger.warning(f"setu: {note}")
    bridge.output_dict["setu_simplifications"] = bridge.grillage_model.simplifications
