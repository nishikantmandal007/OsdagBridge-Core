import math

from setu import (AddedDeadLoads, Bracing, BridgeInput, Concrete, DeckCrossSection,
                  DeckSlab, Girders, MeshSettings, PlateGirderSection, SeismicSite,
                  Steel, TemperatureSite, WindSite)
from osdagbridge_core.core.bridge_types.plate_girder.analyser import BridgeGrillageModel
from osdagbridge_core.core.utils import common as C
from osdagbridge_core.core.utils.codes.irc6_2017 import IRC6_2017
from osdagbridge_core.core.bridge_types.plate_girder.bridge_geometry import CrossSectionLayout


# Lay out the strips from the left edge.
def cross_section(d):
    layout = CrossSectionLayout(
        carriageway_width=float(d[C.KEY_CARRIAGEWAY_WIDTH]), crash_barrier_width=float(d[C.KEY_CB_WIDTH]),
        railing_width=float(d[C.KEY_RL_WIDTH]), footpath_width=float(d[C.KEY_TS_FOOTPATH_WIDTH]),
        median_width=float(d[C.KEY_MD_WIDTH]), n_footpaths=int(d[C.KEY_TS_NO_OF_FOOTPATHS]))
    return DeckCrossSection.from_widths({strip.name: strip.width for strip in layout.components})


# Keep each girder's resolved plates in metres.
def girder_sections(bridge):
    keys = (C.KEY_MP_GIRDER_TOP_FLANGE_WIDTH, C.KEY_MP_GIRDER_TOP_FLANGE_THICKNESS,
            C.KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH, C.KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS,
            C.KEY_MP_GIRDER_WEB_DEPTH, C.KEY_MP_GIRDER_WEB_THICKNESS)
    return Girders(bridge._girder_count(), sections=[
        PlateGirderSection(*(float(bridge._girder_value(k, i)) for k in keys))
        for i in range(bridge._girder_count())])


# Read a member field without inventing a missing value.
def brace_value(d, key):
    return d[key + ".G1G2.B1M1"]


# Convert the angle area from square centimetres to square metres.
def bracing(d):
    arrangement = {"X-Bracing": "X", "K-Bracing": "K"}[brace_value(d, C.KEY_MP_CB_TYPE)]
    arrangement += "T" if brace_value(d, C.KEY_MP_CB_TOP_CHORD) == "Yes" else ""
    arrangement += "B" if brace_value(d, C.KEY_MP_CB_BOTTOM_CHORD) == "Yes" else ""
    area = C.get_angle_section_properties(brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION))["Area"] * 1e-4
    if brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE).startswith("Double Angle"):
        area *= 2
    return Bracing(arrangement, int(d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS]) + 2, area,
                   steel_grade=d[C.KEY_CROSS_BRACING], end_diaphragm_steel_grade=d[C.KEY_END_DIAPHRAGM])


# Use the resolved steel modulus in Pa and weight in N per cubic metre.
def steel_material(bridge):
    steel = bridge.material_props.steel_prop
    return Steel(steel.E / 1e6, steel.v, steel.rho / 1000, grade=steel.grade)


# Convert Ecm from GPa to MPa; the slab uses Poisson's ratio 0.2.
def concrete_material(d):
    return Concrete(float(d[C.KEY_MATERIAL_DECK_ECM]) * 1000, 0.2,
                    float(d[C.KEY_MATERIAL_DECK_DENSITY]), grade=d[C.KEY_DECK_CONCRETE_GRADE_BASIC])


# Only strips that exist can carry a superimposed load.
def added_dead_loads(d):
    return AddedDeadLoads(
        float(d[C.KEY_TS_FOOTPATH_THICKNESS]) / 1000 * float(d[C.KEY_MATERIAL_DECK_DENSITY]) if int(d[C.KEY_TS_NO_OF_FOOTPATHS]) else 0.0,
        0.0, float(d[C.KEY_MD_LOAD]) if float(d[C.KEY_MD_WIDTH]) else 0.0,
        float(d[C.KEY_CB_LOAD]) if float(d[C.KEY_CB_WIDTH]) else 0.0,
        float(d[C.KEY_RL_LOAD_VALUE]) if float(d[C.KEY_RL_WIDTH]) else 0.0)


# Assemble the bridge after OsdagBridge has converted the girder plates to metres.
def bridge_input(bridge):
    d = bridge.input_dict
    return BridgeInput(
        float(d[C.KEY_SPAN]), math.tan(math.radians(float(d[C.KEY_SKEW_ANGLE]))), cross_section(d),
        DeckSlab(float(d[C.KEY_TS_DECK_THICKNESS]) / 1000, float(d[C.KEY_TS_DECK_OVERHANG]), float(d[C.KEY_WC_THICKNESS]) / 1000),
        girder_sections(bridge), bracing(d), MeshSettings(16, 0.15),
        steel_material(bridge), concrete_material(d), float(d[C.KEY_WC_DENSITY]), added_dead_loads(d))


# Automatic is an explicit input mode, not a missing-value fallback.
def site_override(d, mode, value):
    if d[mode] in ("Automatic", "As per IRC 6"):
        return None
    if d[mode] == "Custom":
        return float(d[value])
    raise ValueError(f"{mode}={d[mode]!r}: choose the ospgrillage analysis engine")


# Pass the selected wind coefficients and site geometry through.
def wind_site(d):
    weather = d[C.KEY_PROJECT_LOCATION]["weather_data"]
    if float(weather["wind_speed"]) == 0:
        return None
    pairs = {
        "gust": (C.KEY_WL_GUST_FACTOR_MODE, C.KEY_WL_GUST_FACTOR_VALUE),
        "drag": (C.KEY_WL_DRAG_COEFF_MODE, C.KEY_WL_DRAG_COEFF_VALUE),
        "drag_on_live_load": (C.KEY_WL_DRAG_COEFF_LL_MODE, C.KEY_WL_DRAG_COEFF_LL_VALUE),
        "lift": (C.KEY_WL_LIFT_COEFF_MODE, C.KEY_WL_LIFT_COEFF_VALUE),
        "exposed_area_m2": (C.KEY_WL_SUPER_AREA_ELEV_MODE, C.KEY_WL_SUPER_AREA_ELEV_VALUE),
        "plan_area_m2": (C.KEY_WL_SUPER_AREA_PLAIN_MODE, C.KEY_WL_SUPER_AREA_PLAIN_VALUE),
        "live_load_exposed_area_m2": (C.KEY_WL_EXPOSED_FRONTAL_MODE, C.KEY_WL_EXPOSED_FRONTAL_VALUE),
        "force_height_above_deck_m": (C.KEY_WL_WIND_ECC_DECK_MODE, C.KEY_WL_WIND_ECC_DECK_VALUE),
    }
    return WindSite(float(weather["wind_speed"]), {"Plain Terrain": "plain", "Obstructed Terrain": "obstructed"}[d[C.KEY_WL_TERRAIN_TYPE]],
                    float(d[C.KEY_WL_AVG_EXPOSED_HEIGHT]), d[C.KEY_WL_SITE_TOPOGRAPHY] == "Funnelling",
                    float(d[C.KEY_CB_HEIGHT]) if float(d[C.KEY_CB_WIDTH]) else 0.0,
                    **{name: site_override(d, *keys) for name, keys in pairs.items()})


# Keep the specified seismic weights and coefficients when present.
def seismic_site(d):
    weather = d[C.KEY_PROJECT_LOCATION]["weather_data"]
    if float(weather["z_value"]) == 0:
        return None
    soil = str(d[C.KEY_SL_SOIL_TYPE]).split()[1]
    reduction = float(d[C.KEY_SL_RESPONSE_REDUCTION])
    horizontal = (float(d[C.KEY_SL_HORIZONTAL_COEFF]) if C.KEY_SL_HORIZONTAL_COEFF in d else
                  float(weather["z_value"]) / 2 * float(d[C.KEY_SL_IMPORTANCE_FACTOR]) *
                  BridgeGrillageModel._spectral_sa_g({"I": 1, "II": 2, "III": 3}[soil], float(d[C.KEY_SL_TIME_PERIOD])) *
                  IRC6_2017.table_18(float(d[C.KEY_SL_DAMPING])))
    vertical = float(d[C.KEY_SL_VERTICAL_COEFF]) if C.KEY_SL_VERTICAL_COEFF in d else 2 / 3 * horizontal
    return SeismicSite(weather["zone"], soil,
                       float(d[C.KEY_SL_IMPORTANCE_FACTOR]), float(d[C.KEY_SL_TIME_PERIOD]), float(d[C.KEY_SL_RESPONSE_REDUCTION]),
                       horizontal_coefficient=horizontal / reduction, vertical_coefficient=vertical / reduction, include_vertical=True,
                       dead_weight_kn=site_override(d, C.KEY_SL_DEAD_LOAD_MODE, C.KEY_SL_DEAD_LOAD_VALUE),
                       live_weight_kn=site_override(d, C.KEY_SL_LIVE_LOAD_MODE, C.KEY_SL_LIVE_LOAD_VALUE))


# Keep the shade temperatures separate from equivalent mechanical loads.
def temperature_site(d):
    weather = d[C.KEY_PROJECT_LOCATION]["weather_data"]
    return TemperatureSite(float(weather["max_temp"]), float(weather["min_temp"]))
