import math

from osdagbridge_core.core.utils import common as C
from .bridge_input import brace_value


# Name the input the other engine can handle.
def refuse(key, value):
    raise ValueError(f"{key}={value!r} has no setu equivalent; choose the ospgrillage analysis engine")


# Name a vehicle that neither engine analyses: ospgrillage builds only Class A and 70R cases and leaves it out.
def refuse_vehicle(key, value):
    raise ValueError(f"{key}={value!r}: this vehicle is not analysed by either engine (ospgrillage leaves it out without saying so); untick it")


# IRC:6-2017 as amended no longer has these: Class AA (deleted by Amendment 10, cl. 201.1 and Annex A) and braking with Class SV
# (cl. 204.5.4: no braking on the SV); Class B is for timber bridges.
NOT_IN_IRC = {C.KEY_LL_IRC_AA_WHEELED: "Class AA was deleted by IRC:6 Amendment 10 (cl. 201.1, Annex A)",
              C.KEY_LL_IRC_AA_TRACKED: "Class AA was deleted by IRC:6 Amendment 10 (cl. 201.1, Annex A)",
              C.KEY_BL_IRC_CLASS_SV: "IRC:6 cl. 204.5.4 applies no braking with Class SV"}


# Refuse only what would make the results wrong or that setu cannot run at all.
def check_capability(bridge):
    d = bridge.input_dict
    for key, why in NOT_IN_IRC.items():
        if d.get(key):
            raise ValueError(f"{key}={d[key]!r}: {why}; untick it or choose the ospgrillage analysis engine")
    if d.get("loading.live_load.irc.class_b"):
        refuse_vehicle("loading.live_load.irc.class_b", d["loading.live_load.irc.class_b"])
    if d[C.KEY_LL_ROAD_CATEGORY] == C.ROAD_SV_REQUIRED and not d[C.KEY_LL_IRC_CLASS_SV]:
        raise ValueError(f"{C.KEY_LL_IRC_CLASS_SV} is unticked: IRC:6 cl. 201.1 as amended requires Class SV for all new bridges except over "
                         f"rural, district and village roads; tick it, or set {C.KEY_LL_ROAD_CATEGORY} to the rural, district or village road")
    if "temperature_load_kN_m2" in d and float(d["temperature_load_kN_m2"]):
        refuse("temperature_load_kN_m2", d["temperature_load_kN_m2"])
    for key in C.SETU_REQUIRED_VEHICLES:
        if not d[key]:
            raise ValueError(f"{key} is unticked: setu always searches this IRC vehicle, so tick it or choose the ospgrillage analysis engine")
    for key, expected in ((C.KEY_SC_LEFT_SUPPORT, "Pinned"), (C.KEY_SC_RIGHT_SUPPORT, "Roller")):
        if d[key] != expected:
            refuse(key, d[key])


# True when any per-member copy of this input differs from the value setu analyses.
def member_differs(d, key, expected):
    return any(full.startswith(key + ".G") and value != expected for full, value in d.items())


# Inputs the analysis does not follow, each stated instead of ignored. ospgrillage's analysis does not read these either;
# member design still checks every pair's own section against the forces.
def simplifications(bridge):
    d = bridge.input_dict
    notes = []
    if d[C.KEY_LL_CUSTOM_VEHICLES]:
        notes.append("Custom vehicles are not analysed")
    if d[C.KEY_LL_FOOTPATH_PRESSURE_MODE] != "As per IRC 6":
        notes.append("The custom footpath pressure is not used: footpath live load follows IRC:6")
    if float(d[C.KEY_PL_SELF_WEIGHT_FACTOR]) != 1.0:
        notes.append("The self-weight factor is not used: self weight is applied at 1.0")
    if d[C.KEY_WL_WIND_LL_ECC_MODE] != "As per IRC 6":
        notes.append("The custom eccentricity of wind on live load is not used: it follows IRC:6")
    span = float(d[C.KEY_SPAN])
    for g in range(1, bridge._girder_count() + 1):
        table = d[f"{C.KEY_MP_GD_SEGMENT_TABLE}.G{g}"]
        if len(table) != 1:
            notes.append("Girder segments are not analysed separately: each girder uses its first segment's section along the whole span")
            break
    if any(not math.isclose(float(d[key]), float(d[C.KEY_MATERIAL_GIRDER_E]) / (2 * (1 + float(d[C.KEY_MATERIAL_GIRDER_POISSON]))), abs_tol=0.05)
           for key in (C.KEY_MATERIAL_GIRDER_G, C.KEY_MATERIAL_CROSS_BRACING_G, C.KEY_MATERIAL_END_DIAPHRAGM_G)):
        notes.append("Shear modulus is taken as E / 2(1 + v), not the entered G")
    if any(d[getattr(C, f"KEY_MATERIAL_{member}_{suffix}")] != d[getattr(C, f"KEY_MATERIAL_GIRDER_{suffix}")]
           for member in ("CROSS_BRACING", "END_DIAPHRAGM") for suffix in ("E", "POISSON", "DENSITY")):
        notes.append("Bracing and end diaphragm stiffness and weight in the analysis use the girder steel's E, v and density")
    expected_spacing = span / (int(d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS]) + 1)
    if not math.isclose(float(d[C.KEY_MP_CB_SPACING]), expected_spacing, abs_tol=0.0005) or member_differs(d, C.KEY_MP_CB_SPACING, d[C.KEY_MP_CB_SPACING]) \
            or member_differs(d, C.KEY_MP_CB_NO_OF_CROSS_BRACINGS, d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS]):
        notes.append("Cross-bracing is analysed at equal spacing along the span, the same at every girder pair")
    if any(member_differs(d, key, brace_value(d, key)) for key in (
            C.KEY_MP_CB_TYPE, C.KEY_MP_CB_BRACING_SECTION_TYPE, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION, C.KEY_MP_CB_TOP_CHORD, C.KEY_MP_CB_BOTTOM_CHORD)):
        notes.append("All girder pairs are analysed with the G1-G2 bracing; member design still checks each pair's own section")
    if brace_value(d, C.KEY_MP_CB_TYPE) == "K-Bracing" and brace_value(d, C.KEY_MP_CB_BOTTOM_CHORD) == "Yes":
        notes.append("K-bracing is analysed without its bottom chord; the bottom chord is designed for the top chord's forces")
    if member_differs(d, C.KEY_MP_ED_TYPE, "Cross Bracing"):
        notes.append("Beam end diaphragms are analysed as cross-bracing at the support lines; the beam is sized by member design")
    elif any(member_differs(d, key, expected) for key, expected in (
            (C.KEY_MP_ED_BRACING_TYPE, brace_value(d, C.KEY_MP_CB_TYPE)), (C.KEY_MP_ED_BRACING_SECTION, brace_value(d, C.KEY_MP_CB_BRACING_SECTION_TYPE)),
            (C.KEY_MP_ED_BRACING_SECTION_DESIGNATION, brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION)))):
        notes.append("End diaphragms are analysed with the G1-G2 cross-bracing; member design still checks their own sections")
    changed = [key.rsplit(".", 1)[-1] for key, value in C.SETU_FIXED_DESIGN_FACTORS.items() if key in d and float(d[key]) != float(value)]
    if changed:
        notes.append(f"Partial factors {', '.join(changed)} are not read by the girder design: it uses "
                     + ", ".join(f"{key.rsplit('.', 1)[-1]} = {value}" for key, value in C.SETU_FIXED_DESIGN_FACTORS.items()))
    if any(key in d and not d[key] for key in C.SETU_ALWAYS_CHECKED):
        notes.append("Every limit-state check is run: unticking one in Design Options does not skip it")
    # setu holds the deck along the span at one bearing only (girder (count - 1) // 2 at the left end): holding every girder there
    # adds a restraint couple that lowers the girder moments (measured -2.5 % on an edge girder), so the pinned line is modelled so.
    held = (bridge._girder_count() - 1) // 2 + 1
    notes.append(f"The left support line is held along the span at G{held}'s bearing only: braking and longitudinal seismic go "
                 f"through that bearing, which gives G{held} an end moment (the force times the bearing's depth below the girder's "
                 f"centroid) and its bearing the whole longitudinal force; design that bearing and G{held}'s end for them")
    depths = {bridge._girder_value(C.KEY_MP_GIRDER_DEPTH, i) for i in range(bridge._girder_count())}
    if len(depths) > 1 and float(d[C.KEY_PROJECT_LOCATION]["weather_data"]["wind_speed"]):
        notes.append("Girders of different depths: automatic wind area, drag and height use the deepest girder")
    return notes
