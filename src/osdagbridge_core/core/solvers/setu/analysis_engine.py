from time import perf_counter

import setu
from osdagbridge_core.core.bridge_types.plate_girder.designer import BridgeConfig
from osdagbridge_core.core.utils import common as C
from .bridge_input import bridge_input, seismic_site, temperature_site, wind_site
from .girder_demands import deflections_cache, girder_demands
from .unsupported_inputs import check_capability


class DesignResultData(dict):
    # Model-backed consumers must stop at the phase boundary.
    def __missing__(self, key):
        raise NotImplementedError(f"{key}: setu engine phase 2/3 requires model-backed results")

    # Legacy consumers use get when looking for transverse members.
    def get(self, key, default=None):
        return self[key]


class SetuAnalysisEngine:
    _deduplicated_results = None

    # Run the public analysis once and pass its demands to the existing designer.
    def __init__(self, bridge):
        check_capability(bridge)
        bridge._build_dtos()
        bridge.material_props = bridge._build_material_props()
        self.bridge_input = bridge_input(bridge)
        self.wind = wind_site(bridge.input_dict)
        self.seismic = seismic_site(bridge.input_dict)
        self.temperature = temperature_site(bridge.input_dict)
        start = perf_counter()
        self.design_values = setu.girder_design_values(
            self.bridge_input, wind=self.wind, seismic=self.seismic, temperature=self.temperature,
            braking_height_above_deck_m=float(bridge.input_dict[C.KEY_LL_ECCENTRICITY]))
        self.analysis_seconds = perf_counter() - start
        self.configs = [BridgeConfig.from_plate_girder_bridge(bridge, g) for g in range(bridge._girder_count())]
        self.deflections_cache = deflections_cache(self.design_values, self.configs[0])
        self.per_girder_demands, self.per_girder_per_lc = girder_demands(
            self.design_values, self.configs, self.deflections_cache, self.seismic is not None, bridge.input_dict[C.KEY_LL_IRC_CLASS_FATIGUE])
        bridge._deflections_cache = self.deflections_cache

    # Girder and deck design only need the canonical girder labels here.
    def get_result_data(self):
        return DesignResultData(girders={name: {} for name in self.per_girder_demands})

    # Plots, reactions, load-effects and live model access need later phases.
    def __getattr__(self, name):
        raise NotImplementedError(f"{name}: setu engine phase 2/3 requires model-backed results")


# Replace the analysis stages while preserving girder and deck design.
def run_analysis(bridge):
    bridge.grillage_model = SetuAnalysisEngine(bridge)
