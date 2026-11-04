from osdagbridge_core.core.bridge_types.plate_girder.designer import (
    DemandEnvelope, apply_camber_to_deflections_cache)
from osdagbridge_core.core.utils import common as C


# Keep the adverse magnitude used by the designer's force envelopes.
def governing(values):
    return max(values, key=lambda value: abs(value.value))


# Read basic ULS, enabled seismic ULS and rare SLS without recombining them.
def force_demands(values, seismic):
    states = ["ultimate, basic"] + (["ultimate, seismic"] if seismic else [])
    moment = governing(v for state in states for v in values["maximum composite moment"][state].values())
    shear = governing(v for state in states for v in values["support shear"][state].values())
    rare_m = governing(values["maximum composite moment"]["serviceability, rare"].values())
    rare_v = governing(values["support shear"]["serviceability, rare"].values())
    return dict(Mu_kNm=abs(moment.value), Vu_kN=abs(shear.value),
                M_sls_kNm=abs(rare_m.value), V_sls_kN=abs(rare_v.value),
                governing_combination=moment.combination, location=f"{moment.at_m} m")


# Read the bare-steel construction moments at the ULS design station.
def construction_demands(dead, girder, at_m):
    steel = dead.stages["steel self weight"][girder]
    slab = dead.stages["wet slab"][girder]
    station = abs(steel.stations_m - at_m).argmin()
    return dict(M_girder_sw_kNm=abs(float(steel.composite_moment_kn_m[station])),
                M_construction_kNm=abs(float(steel.composite_moment_kn_m[station] + slab.composite_moment_kn_m[station])))


# The bottom steel face cycles in bending; web area converts kN to MPa.
def fatigue_demands(values, girder, config, enabled):
    return dict(stress_range_MPa=values.fatigue_stresses_mpa[girder]["steel bottom"].range if enabled else 0.0,
                shear_range_MPa=values.fatigue[girder]["support shear"].range * 1000 / config.section.Aw if enabled else 0.0,
                Vr_kN=values.live_shear_range_kn[girder])


# Keep the cache's raw keys and actual stage names, converting metres to mm.
def raw_deflections(values):
    return {f"G{g + 1}": {
        C.KEY_SD_DEFL_LIVE_RAW: d["live_m"] * 1000,
        C.KEY_SD_DEFL_DL_RAW: sum(d["dead_m"].values()) * 1000,
        C.KEY_SD_DEFL_TOTAL_RAW: d["total_m"] * 1000,
        "per_lc": {stage: value * 1000 for stage, value in d["dead_m"].items()},
    } for g, d in values.deflections.items()}


# Let OsdagBridge apply its own camber rules and rounding.
def deflections_cache(values, config):
    return apply_camber_to_deflections_cache(
        raw_deflections(values), config.geometry.camber_mode, config.geometry.camber_value_mm)


# Carry the adjusted sags into the aggregate design checks.
def deflection_demands(cache):
    return dict(delta_live_mm=cache["live_mm"], delta_dl_mm=cache["dl_mm"],
                delta_total_mm=cache["total_mm"], camber_mm=cache["camber_mm"])


# The designer's check type for a load case in the plots' dropdown.
def case_kind(name):
    if name.startswith(("BASIC_", "SEISMIC_")) or name == "Envelope ULS":
        return "ULS"
    if name.startswith("SLS_FREQUENT_"):
        return "SLS_frequent"
    if name.startswith("SLS_") or name == "Envelope SLS":
        return "SLS"
    if name == "1.0 DL":
        return "DL"
    if name == "1.0 LL" or name.startswith("Case "):
        return "live_only"
    return "individual"


# One girder's demand under one load case of the plots' dropdown, worst over the states of an envelope.
def case_demand(states, girder, name, envelope):
    prefixes = {"Envelope ULS": ("BASIC_", "SEISMIC_"), "Envelope SLS": ("SLS_",)}.get(name)
    forces = [state[girder] for case, state in states.items() if (case.startswith(prefixes) if prefixes else case == name)]
    if not forces:
        return None
    moment = max(float(abs(f.composite_moment_kn_m).max()) for f in forces)
    shear = max(float(abs(f.shear_kn).max()) for f in forces)
    deflection = max(float(abs(f.deflection_m).max()) for f in forces) * 1000
    kind = case_kind(name)
    service = kind != "ULS"
    return {"Mu_kNm": moment, "Vu_kN": shear, "M_sls_kNm": moment if service else 0.0, "V_sls_kN": shear if service else 0.0,
            "M_construction_kNm": moment if kind == "DL" else 0.0, "M_girder_sw_kNm": moment if name == "SW" else 0.0,
            "delta_live_mm": deflection if kind == "live_only" else 0.0, "delta_dl_mm": deflection if kind == "DL" else 0.0,
            "stress_range_MPa": envelope.stress_range_MPa if kind == "SLS_frequent" else 0.0,
            "shear_range_MPa": envelope.shear_range_MPa if kind == "SLS_frequent" else 0.0, "lc_type": kind}


# Scope each native combination to the designer's check types.
def combination_demands(values, girder, config, fatigue):
    result = {}
    for name, directions in values.combinations[girder]["maximum composite moment"].items():
        moment = governing(directions.values())
        shear = governing(values.combinations[girder]["support shear"][name].values())
        kind = ("ULS" if name.startswith("ultimate,") else
                "SLS_frequent" if name.startswith("serviceability, frequent") else "SLS")
        result[name] = DemandEnvelope(
            Mu_kNm=abs(moment.value), Vu_kN=abs(shear.value),
            M_sls_kNm=abs(moment.value) if kind != "ULS" else 0.0,
            V_sls_kN=abs(shear.value) if kind != "ULS" else 0.0,
            stress_range_MPa=fatigue["stress_range_MPa"] if kind == "SLS_frequent" else 0.0,
            shear_range_MPa=fatigue["shear_range_MPa"] if kind == "SLS_frequent" else 0.0,
            Nsc=config.fatigue.Nsc, governing_combination=moment.combination,
            location=f"{moment.at_m} m", member=f"G{girder + 1}", source="setu", lc_type=kind)
    return result


# Build the designer's existing demand seam directly from the public results.
def girder_demands(values, configs, cache, seismic, fatigue_enabled):
    demands, per_lc = {}, {}
    for g, config in enumerate(configs):
        name = f"G{g + 1}"
        forces = force_demands(values.girders[g], seismic)
        states = ["ultimate, basic"] + (["ultimate, seismic"] if seismic else [])
        moment = governing(v for state in states for v in values.girders[g]["maximum composite moment"][state].values())
        fatigue = fatigue_demands(values, g, config, fatigue_enabled)
        # The steel axial force belongs to the internal composite couple.
        demands[name] = DemandEnvelope(
            **forces, **construction_demands(values.dead, g, moment.at_m),
            **fatigue, **deflection_demands(cache[name]), Nu_kN=0.0,
            Nsc=config.fatigue.Nsc, member=name, source="setu")
        per_lc[name] = combination_demands(values, g, config, fatigue)
    return demands, per_lc
