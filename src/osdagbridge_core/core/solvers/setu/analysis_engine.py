from time import perf_counter

import setu
from osdagbridge_core.core.bridge_types.plate_girder.designer import BridgeConfig
from osdagbridge_core.core.utils import common as C
from .bridge_input import bridge_input, seismic_site, temperature_site, wind_site
from .girder_demands import deflections_cache, girder_demands
from .live_load_report import live_load_report_values
from .plot_dataset import plot_dataset
from .unsupported_inputs import check_capability


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
        check_capability(bridge)
        bridge._build_dtos()
        bridge.material_props = bridge._build_material_props()
        self.bridge_input = bridge_input(bridge)
        self.wind = wind_site(bridge.input_dict)
        self.seismic = seismic_site(bridge.input_dict)
        self.temperature = temperature_site(bridge.input_dict)
        start = perf_counter()
        values = setu.bridge_design_values(
            self.bridge_input, wind=self.wind, seismic=self.seismic, temperature=self.temperature,
            braking_height_above_deck_m=float(bridge.input_dict[C.KEY_LL_ECCENTRICITY]))
        self.design_values, self.brace_values, self.diagrams = values.girders, values.braces, values.diagrams
        self.analysis_seconds = perf_counter() - start
        self.configs = [BridgeConfig.from_plate_girder_bridge(bridge, g) for g in range(bridge._girder_count())]
        self.deflections_cache = deflections_cache(self.design_values, self.configs[0])
        self.per_girder_demands, self.per_girder_per_lc = girder_demands(
            self.design_values, self.configs, self.deflections_cache, self.seismic is not None, bridge.input_dict[C.KEY_LL_IRC_CLASS_FATIGUE])
        bridge._deflections_cache = self.deflections_cache
        self.model = NoGrillageElements()
        self._deduplicated_results = plot_dataset(values, bridge.get_edge_dist())

    # Girder and deck design only need the canonical girder labels here.
    def get_result_data(self):
        return DesignResultData(girders={name: {} for name in self.per_girder_demands})

    # ospgrillage's load-case objects and live model access are not part of the setu engine.
    def __getattr__(self, name):
        raise NotFromSetu(f"{name}: not provided by the setu engine")


# Replace the analysis stages while preserving girder and deck design.
def run_analysis(bridge):
    bridge.grillage_model = SetuAnalysisEngine(bridge)
    bridge.output_dict.update(live_load_report_values(bridge.grillage_model.bridge_input.span_m))
