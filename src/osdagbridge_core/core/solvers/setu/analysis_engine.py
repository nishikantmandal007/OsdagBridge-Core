import os
import sys
from time import perf_counter

from setu.postprocess.own_process import solve_in_own_process
from osdagbridge_core.core.bridge_types.plate_girder.designer import BridgeConfig
from osdagbridge_core.core.utils import common as C
from osdagbridge_core.core.utils.logger import bridge_logger
from .bridge_input import bridge_input, estimated_peak_gb, fill_unequal_depth_wind, normalise_gui_bracing, reconcile_layout, seismic_site, temperature_site, wind_site, workers_that_fit
from .girder_demands import case_demand, deflections_cache, girder_demands
from .live_load_report import live_load_report_values
from .plot_dataset import CaseForces, load_states, member_nodes, plot_dataset
from .unsupported_inputs import check_capability, simplifications


# Workers forked after the factor exists read one copy of it, so they cost far less than separate models.
MOST_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))


# What the system can give out now, or None where it cannot be asked.
def available_memory_gb():
    try:
        if sys.platform == "darwin":
            from osdagbridge_core.core.utils import macos_memory
            return macos_memory.available_gb()
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as meminfo:
                return next(int(line.split()[1]) for line in meminfo if line.startswith("MemAvailable:")) / 1024 ** 2
    except (OSError, ValueError, AttributeError, StopIteration):
        pass
    return None


# Say how big the run is, warn if it may not fit, and choose the workers that do.
def plan_run(d):
    nodes, gb = estimated_peak_gb(d)
    available = available_memory_gb()
    workers = workers_that_fit(MOST_WORKERS, gb, available)
    bridge_logger.sub_step(f"setu: {nodes} deck nodes, estimated peak about {gb:.1f} GB, {workers} workers")
    if available is not None and gb > available:
        bridge_logger.warning(f"This bridge may need about {gb:.1f} GB with only {available:.1f} GB free; close other programs or expect it to be slow")
    return workers


# The whole analysis in its own process: Stop ends it at once and its memory returns to the system when it finishes.
def solve(bridge_in, wind, seismic, temperature, braking_height_m, workers):
    values, forces = solve_in_own_process(
        bridge_in, wind, seismic, temperature, braking_height_m, workers=workers,
        phase=lambda text: bridge_logger.sub_step(f"setu: {text}"), check=bridge_logger.check_cancel)
    return values, report_forces(forces)


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
        values, self.report_forces = solve(self.bridge_input, self.wind, self.seismic, self.temperature, float(bridge.input_dict[C.KEY_LL_ECCENTRICITY]), workers)
        self.design_values, self.brace_values, self.diagrams = values.girders, values.braces, values.diagrams
        self.analysis_seconds = perf_counter() - start
        self.configs = [BridgeConfig.from_plate_girder_bridge(bridge, g) for g in range(bridge._girder_count())]
        self.deflections_cache = deflections_cache(self.design_values, self.configs[0])
        self.per_girder_demands, self.per_girder_per_lc = girder_demands(
            self.design_values, self.configs, self.deflections_cache, self.seismic is not None, bridge.input_dict[C.KEY_LL_IRC_CLASS_FATIGUE])
        bridge._deflections_cache = self.deflections_cache
        self.model = NoGrillageElements()
        self._deduplicated_results = plot_dataset(values, bridge.get_edge_dist())
        self.states = load_states(values)

    # The output dock's design bars for one load case in its dropdown.
    def case_demand(self, girder_name, load_case):
        return case_demand(self.states, int(girder_name[1:]) - 1, load_case, self.per_girder_demands[girder_name])

    # Girder and deck design only need the canonical girder labels here.
    def get_result_data(self):
        return DesignResultData(girders={name: {} for name in self.per_girder_demands}, members=member_nodes(self._deduplicated_results),
                                forces=CaseForces(self._deduplicated_results))

    # ospgrillage's load-case objects and live model access are not part of the setu engine.
    def __getattr__(self, name):
        raise NotFromSetu(f"{name}: not provided by the setu engine")


# Replace the analysis stages while preserving girder and deck design.
def run_analysis(bridge):
    bridge.grillage_model = SetuAnalysisEngine(bridge)
    bridge_logger.sub_step(f"setu: analysis finished in {bridge.grillage_model.analysis_seconds:.0f} s")
    bridge.output_dict.update(live_load_report_values(bridge.grillage_model.bridge_input.span_m))
    bridge.output_dict.update(bridge.grillage_model.report_forces)
    for note in ("Class SV and Class 70R Bogie are not analysed: these results leave them out", *bridge.grillage_model.simplifications):
        bridge_logger.warning(f"setu: {note}")
    bridge.output_dict["setu_simplifications"] = bridge.grillage_model.simplifications
