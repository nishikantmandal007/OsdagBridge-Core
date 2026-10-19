import math

from osdagbridge_core.core.utils import common as C
from .bridge_input import brace_value, cross_section


# Name the input the other engine can handle.
def refuse(key, value):
    raise ValueError(f"{key}={value!r} has no setu equivalent; choose the ospgrillage analysis engine")


# Name a vehicle that neither engine analyses: ospgrillage builds only Class A and 70R cases and leaves it out.
def refuse_vehicle(key, value):
    raise ValueError(f"{key}={value!r}: this vehicle is not analysed by either engine (ospgrillage leaves it out without saying so); untick it")


# A shared brace section must apply to every pair, station and end.
def check_bracing(d):
    for key in (C.KEY_MP_CB_SPACING, C.KEY_MP_CB_NO_OF_CROSS_BRACINGS):
        for full, value in d.items():
            if full.startswith(key + ".G") and value != d[key]:
                refuse(full, value)
    expected_spacing = float(d[C.KEY_SPAN]) / (int(d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS]) + 1)
    if not math.isclose(float(d[C.KEY_MP_CB_SPACING]), expected_spacing, abs_tol=0.0005):
        refuse(C.KEY_MP_CB_SPACING, d[C.KEY_MP_CB_SPACING])
    fields = (C.KEY_MP_CB_TYPE, C.KEY_MP_CB_BRACING_SECTION_TYPE, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION,
              C.KEY_MP_CB_TOP_CHORD, C.KEY_MP_CB_BOTTOM_CHORD)
    for key in fields:
        shared = brace_value(d, key)
        for full, value in d.items():
            if full.startswith(key + ".G") and value != shared:
                refuse(full, value)
    if brace_value(d, C.KEY_MP_CB_TYPE) not in ("X-Bracing", "K-Bracing"):
        refuse(C.KEY_MP_CB_TYPE, brace_value(d, C.KEY_MP_CB_TYPE))
    if brace_value(d, C.KEY_MP_CB_TYPE) == "K-Bracing" and brace_value(d, C.KEY_MP_CB_BOTTOM_CHORD) == "Yes":
        refuse(C.KEY_MP_CB_BOTTOM_CHORD, "K bottom chord")
    if not brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE).startswith(("Single Angle", "Double Angle")):
        refuse(C.KEY_MP_CB_BRACING_SECTION_TYPE, brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE))
    checks = {
        C.KEY_MP_ED_TYPE: "Cross Bracing", C.KEY_MP_ED_BRACING_TYPE: brace_value(d, C.KEY_MP_CB_TYPE),
        C.KEY_MP_ED_BRACING_SECTION: brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE),
        C.KEY_MP_ED_BRACING_SECTION_DESIGNATION: brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION),
        C.KEY_MP_ED_TOP_CHORD: brace_value(d, C.KEY_MP_CB_TOP_CHORD),
        C.KEY_MP_ED_BOTTOM_CHORD: brace_value(d, C.KEY_MP_CB_BOTTOM_CHORD),
    }
    for key, shared in checks.items():
        for full, value in d.items():
            if full.startswith(key + ".G") and value != shared:
                refuse(full, value)
    for chord, flag, section, desig in (
        ("top", C.KEY_MP_CB_TOP_CHORD, C.KEY_MP_CB_TOP_CHORD_SECTION_TYPE, C.KEY_MP_CB_TOP_CHORD_SECTION_DESIG),
        ("bottom", C.KEY_MP_CB_BOTTOM_CHORD, C.KEY_MP_CB_BOTTOM_CHORD_SECTION_TYPE, C.KEY_MP_CB_BOTTOM_CHORD_SECTION_DESIG),
    ):
        if brace_value(d, flag) == "Yes":
            for full, value in d.items():
                if full.startswith(section + ".G") and value != brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE):
                    refuse(full, value)
                if full.startswith(desig + ".G") and value != brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION):
                    refuse(full, value)
            ed_prefix = "member_properties.end_diaphragm_details."
            for field, expected in ((f"{chord}_chord_section_type", brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE)),
                                    (f"{chord}_chord_section_designation", brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION))):
                for full, value in d.items():
                    if full.startswith(ed_prefix + field + ".G") and value != expected:
                        refuse(full, value)


# Refuse choices the engine cannot preserve before building an OpenSees domain.
def check_capability(bridge):
    d = bridge.input_dict
    for key in (C.KEY_LL_IRC_CLASS_SV, C.KEY_BL_IRC_CLASS_SV, C.KEY_LL_IRC_70R_BOGIE, "loading.live_load.irc.class_b"):
        if key in d and d[key]:
            refuse_vehicle(key, d[key])
    if "temperature_load_kN_m2" in d and float(d["temperature_load_kN_m2"]):
        refuse("temperature_load_kN_m2", d["temperature_load_kN_m2"])
    if d[C.KEY_LL_CUSTOM_VEHICLES]:
        refuse(C.KEY_LL_CUSTOM_VEHICLES, d[C.KEY_LL_CUSTOM_VEHICLES])
    for key in (C.KEY_LL_IRC_CLASS_A, C.KEY_LL_IRC_70R_WHEELED, C.KEY_LL_IRC_70R_TRACKED, C.KEY_LL_IRC_AA_WHEELED, C.KEY_LL_IRC_AA_TRACKED):
        if not d[key]:
            refuse(key, "setu searches the full IRC vehicle family")
    if d[C.KEY_LL_FOOTPATH_PRESSURE_MODE] != "As per IRC 6":
        refuse(C.KEY_LL_FOOTPATH_PRESSURE_MODE, d[C.KEY_LL_FOOTPATH_PRESSURE_MODE])
    for key, expected in ((C.KEY_SC_LEFT_SUPPORT, "Pinned"), (C.KEY_SC_RIGHT_SUPPORT, "Roller"), (C.KEY_PL_SELF_WEIGHT_FACTOR, 1.0)):
        value = float(d[key]) if isinstance(expected, float) else d[key]
        if value != expected:
            refuse(key, value)
    for g in range(1, bridge._girder_count() + 1):
        key = f"{C.KEY_MP_GD_SEGMENT_TABLE}.G{g}"
        table = d[key]
        if len(table) != 1 or float(table[0]["start"]) != 0 or float(table[0]["end"]) != float(d[C.KEY_SPAN]):
            refuse(key, table)
    for key in (C.KEY_MATERIAL_GIRDER_G, C.KEY_MATERIAL_CROSS_BRACING_G, C.KEY_MATERIAL_END_DIAPHRAGM_G):
        if not math.isclose(float(d[key]), float(d[C.KEY_MATERIAL_GIRDER_E]) / (2 * (1 + float(d[C.KEY_MATERIAL_GIRDER_POISSON]))), rel_tol=1e-4):
            refuse(key, "independent shear modulus")
    if d[C.KEY_WL_SITE_TOPOGRAPHY] not in ("Flat", "Funnelling"):
        refuse(C.KEY_WL_SITE_TOPOGRAPHY, d[C.KEY_WL_SITE_TOPOGRAPHY])
    if d[C.KEY_WL_WIND_LL_ECC_MODE] != "As per IRC 6":
        refuse(C.KEY_WL_WIND_LL_ECC_MODE, d[C.KEY_WL_WIND_LL_ECC_MODE])
    depths = [bridge._girder_value(C.KEY_MP_GIRDER_DEPTH, i) for i in range(bridge._girder_count())]
    if len(set(depths)) > 1 and float(d[C.KEY_PROJECT_LOCATION]["weather_data"]["wind_speed"]):
        for key in (C.KEY_WL_SUPER_AREA_ELEV_MODE, C.KEY_WL_DRAG_COEFF_MODE, C.KEY_WL_WIND_ECC_DECK_MODE):
            if d[key] != "Custom":
                refuse(key, "unequal-depth girders require explicit wind area, drag and height")
    if not math.isclose(cross_section(d).total_width_m(), float(d[C.KEY_TS_OVERALL_WIDTH]), abs_tol=1e-8):
        refuse(C.KEY_TS_OVERALL_WIDTH, "inconsistent cross-section strips")
    spacing = (float(d[C.KEY_TS_OVERALL_WIDTH]) - 2 * float(d[C.KEY_TS_DECK_OVERHANG])) / (bridge._girder_count() - 1)
    if not math.isclose(spacing, float(d[C.KEY_TS_GIRDER_SPACING]), abs_tol=1e-8):
        refuse(C.KEY_TS_GIRDER_SPACING, "inconsistent overhang")
    for member in ("CROSS_BRACING", "END_DIAPHRAGM"):
        for suffix in ("E", "POISSON", "DENSITY"):
            key = getattr(C, f"KEY_MATERIAL_{member}_{suffix}")
            if d[key] != d[getattr(C, f"KEY_MATERIAL_GIRDER_{suffix}")]:
                refuse(key, d[key])
    check_bracing(d)
