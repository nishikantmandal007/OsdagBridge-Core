import math
import re

from setu import (AddedDeadLoads, Bracing, BridgeInput, Concrete, DeckCrossSection,
                  DeckSlab, Girders, MeshSettings, PlateGirderSection, SeismicSite,
                  Steel, TemperatureSite, WindSite)
from setu.builder.mesh import stations_along_span
from setu.irc6.wind import CYCLONE_COAST, INLAND, OFFSHORE, drag_depth_m, exposed_depth_m, service_drag_coefficient
from setu.postprocess.construction import ConstructionLoads
from osdagbridge_core.core.bridge_types.plate_girder.analyser import BridgeGrillageModel
from osdagbridge_core.core.utils import common as C
from osdagbridge_core.core.utils.codes.irc6_2017 import IRC6_2017
from osdagbridge_core.core.bridge_types.plate_girder.bridge_geometry import CrossSectionLayout


# Peak memory, measured as the system's physical footprint of every process (macOS, Standard mesh) on decks of 3021, 9911 and 18391
# nodes. Without workers (2026-11-07, setu 4531519): 0.33, 1.14 and 3.90 GB. With workers, which are spawned and search but never solve
# (setu 6e3ee1d), whole runs peaked at 0.53 / 0.81 GB (2 / 4 workers), 1.04 / 1.56 GB and 2.75 GB (4 workers): this process plus a
# share per worker, fitted below, reproduces all five to 0.01 GB. Refit when other decks or platforms are measured.
ANALYSIS_GB = (0.366, -0.0514, 0.0131)  # a + b k + c k**2, k thousands of deck nodes: the analysis alone, without workers
WITH_WORKERS_GB = (0.191, 0.0136, 0.00198)  # a + b k + c k**2: this process while workers search
WORKER_GB = (0.09, 0.0175)  # a + b k: each worker
RESERVE_GB = 0.5  # left free for the app, which receives the results


# Lay out the strips from the left edge.
def cross_section(d):
    layout = CrossSectionLayout(
        carriageway_width=float(d[C.KEY_CARRIAGEWAY_WIDTH]), crash_barrier_width=float(d[C.KEY_CB_WIDTH]),
        railing_width=float(d[C.KEY_RL_WIDTH]), footpath_width=float(d[C.KEY_TS_FOOTPATH_WIDTH]),
        median_width=float(d[C.KEY_MD_WIDTH]), n_footpaths=int(d[C.KEY_TS_NO_OF_FOOTPATHS]))
    return DeckCrossSection.from_widths({strip.name: strip.width for strip in layout.components},
                                        crowd_load=bool(d[C.KEY_LL_CROWD_LOAD]), congestion=bool(d[C.KEY_LL_CONGESTION]))


# The mesh the user chose (Fine, Standard, Coarse), as setu element lengths.
def mesh_settings(d):
    name = d[C.KEY_ANALYSIS_MESH]
    if name not in C.MESH_PRESETS:
        raise ValueError(f"mesh must be one of {', '.join(C.MESH_PRESETS)}, got {name!r}: choose it again in Additional Inputs")
    along_m, support_m, zone_m, across_m = C.MESH_PRESETS[name]
    return MeshSettings(None, across_m, element_length_m=along_m, support_element_length_m=support_m, support_zone_m=zone_m)


# Each girder's resolved plates in metres. The web is the overall depth less both flanges, as the designer takes it: the stored
# web depth is rounded up from an unrounded depth and can be 1 mm deeper than the plates add up to.
def girder_sections(bridge):
    def section(i):
        value = lambda key: float(bridge._girder_value(key, i))
        web_m = value(C.KEY_MP_GIRDER_DEPTH) - value(C.KEY_MP_GIRDER_TOP_FLANGE_THICKNESS) - value(C.KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS)
        return PlateGirderSection(value(C.KEY_MP_GIRDER_TOP_FLANGE_WIDTH), value(C.KEY_MP_GIRDER_TOP_FLANGE_THICKNESS),
                                  value(C.KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH), value(C.KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS),
                                  web_m, value(C.KEY_MP_GIRDER_WEB_THICKNESS))
    return Girders(bridge._girder_count(), sections=[section(i) for i in range(bridge._girder_count())])


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


# Deck nodes and the analysis process's peak memory in GB on its own (fitted above).
def estimated_peak_gb(d):
    braces, span, mesh = int(d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS]), float(d[C.KEY_SPAN]), mesh_settings(d)
    along = len(stations_along_span([span * i / (braces + 1) for i in range(braces + 2)], mesh))
    across = round(cross_section(d).total_width_m() / mesh.target_size_across_width_m) + 1
    nodes = along * across
    return nodes, analysis_gb(nodes)


def quadratic(coefficients, nodes):
    k = nodes / 1000
    return sum(c * k ** power for power, c in enumerate(coefficients))


# the analysis alone; this process while workers search; and each worker, in GB
def analysis_gb(nodes):
    return quadratic(ANALYSIS_GB, nodes)


def with_workers_gb(nodes):
    return quadratic(WITH_WORKERS_GB, nodes)


def worker_gb(nodes):
    return quadratic(WORKER_GB, nodes)


# Workers to start: as many as the cores allow and the free memory holds beside this process and a reserve, or one (the analysis
# alone, no workers) when two do not fit or the free memory is unknown.
def workers_that_fit(cap, main_gb, each_gb, available_gb):
    if available_gb is None:
        return 1
    fit = int((available_gb - RESERVE_GB - main_gb) / each_gb)
    return max(1, min(cap, fit)) if fit >= 2 else 1


# Assemble the bridge after OsdagBridge has converted the girder plates to metres.
def bridge_input(bridge):
    d = bridge.input_dict
    return BridgeInput(
        float(d[C.KEY_SPAN]), math.tan(math.radians(float(d[C.KEY_SKEW_ANGLE]))), cross_section(d),
        DeckSlab(float(d[C.KEY_TS_DECK_THICKNESS]) / 1000, float(d[C.KEY_TS_DECK_OVERHANG]), float(d[C.KEY_WC_THICKNESS]) / 1000),
        girder_sections(bridge), bracing(d), mesh_settings(d),
        steel_material(bridge), concrete_material(d), float(d[C.KEY_WC_DENSITY]), added_dead_loads(d))


# Automatic is an explicit input mode, not a missing-value fallback.
def site_override(d, mode, value):
    if d[mode] in ("Automatic", "As per IRC 6"):
        return None
    if d[mode] == "Custom":
        return float(d[value])
    raise ValueError(f"{mode}={d[mode]!r}: choose the ospgrillage analysis engine")


K4_REGIONS = {C.K4_OTHER: INLAND, C.K4_CYCLONE_COAST: CYCLONE_COAST, C.K4_OFFSHORE: OFFSHORE}


# IRC:6 cl. 209 as replaced by Amendment 10: the site inputs and any custom coefficient.
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
    return WindSite(float(weather["wind_speed"]), C.TERRAIN_CATEGORIES.index(d[C.KEY_WL_TERRAIN_CATEGORY]) + 1,
                    float(d[C.KEY_WL_AVG_EXPOSED_HEIGHT]), float(d[C.KEY_WL_ROAD_LEVEL]), float(d[C.KEY_WL_PIER_HEIGHT]),
                    int(d[C.KEY_WL_RETURN_PERIOD]), d[C.KEY_WL_K3_LOCATION] == C.K3_VALLEY_OR_LEEWARD, K4_REGIONS[d[C.KEY_WL_CYCLONIC_REGION]],
                    float(d[C.KEY_WL_CROSS_SLOPE]), float(d[C.KEY_CB_HEIGHT]) if float(d[C.KEY_CB_WIDTH]) else 0.0,
                    **{name: site_override(d, *keys) for name, keys in pairs.items()})


# setu asks for explicit wind area, drag and height when girder depths differ: fill what the user left automatic as setu would for
# equal girders of the deepest depth (Fig. 5 exposed depth, Fig. 7 / Table 8 drag, the area's centroid), without live load.
def fill_unequal_depth_wind(wind, bridge):
    depths = [section.depth_m for section in bridge.girders.section_of_each_girder]
    if wind is None or len(set(depths)) == 1:
        return
    depth, deck = max(depths), bridge.deck.thickness_m
    road_depth = depth + deck + bridge.wearing_course_thickness_m
    exposed = exposed_depth_m(road_depth, wind.solid_barrier_height_m, False)
    if wind.exposed_area_m2 is None:
        wind.exposed_area_m2 = bridge.span_m * exposed
    if wind.drag is None:
        wind.drag = service_drag_coefficient(bridge.width_m(), drag_depth_m(road_depth, wind.solid_barrier_height_m, False), wind.deck_cross_slope_deg)
    if wind.force_height_above_deck_m is None:
        wind.force_height_above_deck_m = exposed / 2 - depth - deck


# Keep the specified seismic weights and coefficients when present. Vertical seismic is included in every zone: IRC:6-2017
# cl. 218.3 requires it in zones IV and V and lets it be omitted in zones II and III except for bearings and linkages, stability
# checks, horizontal cantilevers, prestressed decks and near-field sites, so keeping it covers the bearings and stays conservative.
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
    return TemperatureSite(float(weather["max_temp"]), float(weather["min_temp"]), snowbound=bool(d[C.KEY_TL_SNOWBOUND]))


# The construction loads on the bare steel when the construction stage is included; IRC gives no values, so they must be entered.
def construction_loads(d):
    if d[C.KEY_DS_CONSTRUCTION_STAGE] != "Yes":
        return None
    values = {}
    for key, name in ((C.KEY_DS_SHUTTERING_LOAD, "shuttering"), (C.KEY_DS_CONSTRUCTION_LIVE_LOAD, "construction live load")):
        if d.get(key) in (None, ""):
            raise ValueError(f"{key}: enter the {name} on the bare steel in kN/m2 from the construction programme (0 if none); "
                             "IRC:6 cl. 218 gives no value. Or set Construction Stages to No")
        values[name] = float(d[key])
    return ConstructionLoads(values["shuttering"], values["construction live load"])


# Seismic during construction with importance factor 1 (IRC:SP:114 cl. 4.5 note), the coefficients worked out as for the bridge.
def construction_seismic_site(d):
    if seismic_site(d) is None:
        return None
    one = dict(d)
    one[C.KEY_SL_IMPORTANCE_FACTOR] = 1.0
    for key in (C.KEY_SL_HORIZONTAL_COEFF, C.KEY_SL_VERTICAL_COEFF):
        if key in one:
            one[key] = float(one[key]) / float(d[C.KEY_SL_IMPORTANCE_FACTOR])
    site = seismic_site(one)
    site.dead_weight_kn = site.live_weight_kn = None
    return site
