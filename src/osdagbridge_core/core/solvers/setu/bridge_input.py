import math
import re

from setu import (AddedDeadLoads, Bracing, BridgeInput, Concrete, DeckCrossSection,
                  DeckSlab, Girders, MeshSettings, PlateGirderSection, SeismicSite,
                  Steel, TemperatureSite, WindSite)
from setu.irc6.wind import drag_coefficient
from osdagbridge_core.core.bridge_types.plate_girder.analyser import BridgeGrillageModel
from osdagbridge_core.core.utils import common as C
from osdagbridge_core.core.utils.codes.irc6_2017 import IRC6_2017
from osdagbridge_core.core.bridge_types.plate_girder.bridge_geometry import CrossSectionLayout


PANELS_BETWEEN_BRACES = 16
TARGET_SIZE_ACROSS_M = 0.15
GB_PER_WORKER = 0.7  # measured: 8 workers used about 7.2 GB on a 20.85 m deck, 4 workers 2.74 GB


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


# The dialog saves the pair it shows as "∠ 100 ⅹ 100ⅹ 10" and True; every other pair keeps "IS 100 x 100 x 10" and "Yes".
def normalise_gui_bracing(d):
    for key, value in d.items():
        if not key.startswith(("member_properties.cross_bracing_details.", "member_properties.end_diaphragm_details.")):
            continue
        if isinstance(value, str) and "∠" in value:
            d[key] = "IS " + " x ".join(re.findall(r"\d+(?:\.\d+)?", value))
        elif isinstance(value, bool) and ("top_chord" in key or "bottom_chord" in key):
            d[key] = "Yes" if value else "No"


# Convert the angle area from square centimetres to square metres.
def bracing(d):
    arrangement = {"X-Bracing": "X", "K-Bracing": "K"}[brace_value(d, C.KEY_MP_CB_TYPE)]
    arrangement += "T" if brace_value(d, C.KEY_MP_CB_TOP_CHORD) == "Yes" else ""
    # setu's K-bracing has no bottom chord; simplifications() says so.
    arrangement += "B" if brace_value(d, C.KEY_MP_CB_BOTTOM_CHORD) == "Yes" and arrangement[0] == "X" else ""
    section = brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE)
    lookup = C.get_channel_section_properties if "Channel" in section else C.get_angle_section_properties
    area = lookup(brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION))["Area"] * 1e-4
    if section.startswith("Double"):
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
        # A railing strip exists only beside a footpath; OsdagBridge's layout counts it per footpath.
        float(d[C.KEY_RL_LOAD_VALUE]) if int(d[C.KEY_TS_NO_OF_FOOTPATHS]) and float(d[C.KEY_RL_WIDTH]) else 0.0)


# The dialog recomputes strip widths (railing 0.275 m) after the layout was solved with the default (0.375 m):
# keep the girders where they are and let the strips set the deck width and overhang.
def reconcile_layout(d):
    width = cross_section(d).total_width_m()
    overhang = (width - (int(d[C.KEY_TS_NO_OF_GIRDERS]) - 1) * float(d[C.KEY_TS_GIRDER_SPACING])) / 2
    if not math.isclose(width, float(d[C.KEY_TS_OVERALL_WIDTH]), abs_tol=1e-8) or not math.isclose(overhang, float(d[C.KEY_TS_DECK_OVERHANG]), abs_tol=1e-8):
        d[C.KEY_TS_OVERALL_WIDTH] = width
        d[C.KEY_TS_DECK_OVERHANG] = overhang


# Deck nodes and a rough peak memory in GB of the analysis process and its workers: a base plus one stiffness factor (banded estimate).
# ponytail: fitted to two measured runs (free RAM drop with 4 workers: 8.4 m 1.85 GB, 20.85 m 2.74 GB); refit when more widths are measured.
def estimated_peak_gb(d):
    girders, braces = int(d[C.KEY_TS_NO_OF_GIRDERS]), int(d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS])
    along = PANELS_BETWEEN_BRACES * (braces + 1) + 1
    across = round(cross_section(d).total_width_m() / TARGET_SIZE_ACROSS_M) + 1
    nodes = along * across
    return nodes, 1.7 + 0.85 * 6 * nodes * 8 * 6 * min(along, across) / 1024 ** 3


# Workers to start: up to the cap, each needing about this much on top of the analysis, and never fewer than one.
def workers_that_fit(cap, needed_gb, available_gb):
    if available_gb is None:
        return cap
    return max(1, min(cap, int((available_gb - needed_gb) / GB_PER_WORKER)))


# Assemble the bridge after OsdagBridge has converted the girder plates to metres.
def bridge_input(bridge):
    d = bridge.input_dict
    return BridgeInput(
        float(d[C.KEY_SPAN]), math.tan(math.radians(float(d[C.KEY_SKEW_ANGLE]))), cross_section(d),
        DeckSlab(float(d[C.KEY_TS_DECK_THICKNESS]) / 1000, float(d[C.KEY_TS_DECK_OVERHANG]), float(d[C.KEY_WC_THICKNESS]) / 1000),
        girder_sections(bridge), bracing(d), MeshSettings(PANELS_BETWEEN_BRACES, TARGET_SIZE_ACROSS_M),
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


# setu asks for explicit wind area, drag and height when girder depths differ: fill what the user left automatic from the deepest girder.
def fill_unequal_depth_wind(wind, bridge):
    depths = [section.depth_m for section in bridge.girders.section_of_each_girder]
    if wind is None or len(set(depths)) == 1:
        return
    depth = max(depths)
    deck = bridge.deck.thickness_m
    if wind.exposed_area_m2 is None:
        wind.exposed_area_m2 = bridge.span_m * (depth + deck + wind.solid_barrier_height_m)
    if wind.drag is None:
        spacing = (bridge.width_m() - 2 * bridge.deck.overhang_m) / (bridge.girders.count - 1)
        wind.drag = drag_coefficient(bridge.girders.count, spacing, depth)
    if wind.force_height_above_deck_m is None:
        wind.force_height_above_deck_m = (wind.solid_barrier_height_m - depth) / 2 - deck / 2


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
