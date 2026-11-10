from setu.irc6.irc_constants import CONSTRUCTION_DEAD_FACTORS, VARIABLE_LOAD_FACTORS
from setu.postprocess.girder_response import STEEL_SELF_WEIGHT, SUPERIMPOSED, SURFACING_STAGE, WET_SLAB
from setu.utils.constants import DEAD, FREQUENT, LIVE, SURFACING
from osdagbridge_core.core.bridge_types.plate_girder.designer import (
    DemandEnvelope, apply_camber_to_deflections_cache)

# Annex B cl. 7.2 as amended by Amendment 10: deflection limits in all structures use Table B.3 column 3 (frequent), live load leading
FREQUENT_LIVE_LOAD_FACTOR = VARIABLE_LOAD_FACTORS[FREQUENT][LIVE][0]
# Table B.2: construction-stage dead load, adverse
CONSTRUCTION_DEAD_FACTOR = CONSTRUCTION_DEAD_FACTORS[0]
CONSTRUCTION_STATES = ("ultimate, construction", "ultimate, seismic during construction")


# Every governing value of one girder response in these limit states, with Class SV's where it was analysed.
def candidates(values, special_vehicle, girder, response, states):
    found = [v for state in states for v in values.girders[girder][response].get(state, {}).values()]
    if special_vehicle is not None:
        found += [v for state in states for v in special_vehicle.girders[girder][response].get(state, {}).values()]
    return found
from osdagbridge_core.core.utils import common as C


# Keep the adverse magnitude used by the designer's force envelopes.
def governing(values):
    return max(values, key=lambda value: abs(value.value))


# Read basic ULS, enabled seismic ULS and rare SLS without recombining them; Class SV competes where it was analysed.
def force_demands(values, girder, seismic, special_vehicle=None):
    states = ["ultimate, basic"] + (["ultimate, seismic"] if seismic else [])
    moment = governing(candidates(values, special_vehicle, girder, "maximum composite moment", states))
    shear = governing(candidates(values, special_vehicle, girder, "support shear", states))
    rare_m = governing(candidates(values, special_vehicle, girder, "maximum composite moment", ["serviceability, rare"]))
    rare_v = governing(candidates(values, special_vehicle, girder, "support shear", ["serviceability, rare"]))
    return dict(Mu_kNm=abs(moment.value), Vu_kN=abs(shear.value),
                M_sls_kNm=abs(rare_m.value), V_sls_kN=abs(rare_v.value),
                M_sls_stages_kNm=sls_stages(values.dead, rare_m, girder),
                governing_combination=moment.combination, location=f"{moment.at_m} m")


# An SLS moment split by the section that carries each part on an unpropped deck (IRC:22 cl. 604.3): the steel's own weight and the
# wet slab on the steel alone, superimposed dead load and surfacing on the long-term composite section, the rest (live load, wind,
# temperature) on the short-term composite section. setu's dead share is its factor times the dead stages at the governing station,
# so each stage takes that factor; signed so the parts add up to the moment's magnitude.
def sls_stages(dead, value, girder):
    station = station_of(dead, girder, value.at_m)
    dead_factor, surfacing_factor = state_factors(dead, value, girder)
    return stages_at(dead, girder, station, value.value, dead_factor, surfacing_factor)


# The station index at a position along the girder.
def station_of(dead, girder, at_m):
    return list(dead.stages[STEEL_SELF_WEIGHT][girder].stations_m).index(at_m)


# A governing value's dead and surfacing factors: its shares over the stages they factor, at its own station.
def state_factors(dead, value, girder):
    station = station_of(dead, girder, value.at_m)
    at = lambda stage: float(dead.stages[stage][girder].composite_moment_kn_m[station])
    dead_total = at(STEEL_SELF_WEIGHT) + at(WET_SLAB) + at(SUPERIMPOSED)
    surfacing = at(SURFACING_STAGE)
    return (value.shares.get(DEAD, 0.0) / dead_total if dead_total else 0.0,
            value.shares.get(SURFACING, 0.0) / surfacing if surfacing else 0.0)


# One girder's moment at a station split into the steel, long-term and short-term parts, signed so they add to its magnitude.
def stages_at(dead, girder, station, moment_kn_m, dead_factor, surfacing_factor):
    at = lambda stage: float(dead.stages[stage][girder].composite_moment_kn_m[station])
    sign = 1.0 if moment_kn_m >= 0 else -1.0
    steel = sign * dead_factor * (at(STEEL_SELF_WEIGHT) + at(WET_SLAB))
    long_term = sign * (dead_factor * at(SUPERIMPOSED) + surfacing_factor * at(SURFACING_STAGE))
    return {"steel": steel, "long_term": long_term, "short_term": abs(moment_kn_m) - steel - long_term}


# ULS construction demands on the bare steel for the designer's LTB checks: stage 1 the girder's own weight times the Table B.2
# construction dead-load factor, stage 2 setu's governing construction combination (steel, wet slab, shuttering, construction live
# load, wind and seismic during construction). Both 0 when the construction stage is left out, so the designer skips them.
def construction_demands(dead, construction, girder):
    if construction is None:
        return dict(M_girder_sw_kNm=0.0, M_construction_kNm=0.0)
    steel = dead.stages["steel self weight"][girder]
    stage_2 = governing(v for state in CONSTRUCTION_STATES for v in construction.girders[girder]["maximum composite moment"].get(state, {}).values())
    return dict(M_girder_sw_kNm=CONSTRUCTION_DEAD_FACTOR * float(abs(steel.composite_moment_kn_m).max()),
                M_construction_kNm=abs(stage_2.value))


# The bottom steel face cycles in bending; web area converts kN to MPa.
def fatigue_demands(values, girder, config, enabled):
    return dict(stress_range_MPa=values.fatigue_stresses_mpa[girder]["steel bottom"].range if enabled else 0.0,
                shear_range_MPa=values.fatigue[girder]["support shear"].range * 1000 / config.section.Aw if enabled else 0.0,
                Vr_kN=values.live_shear_range_kn[girder])


# Keep the cache's raw keys and actual stage names, converting metres to mm; live load deflection in the frequent combination.
def raw_deflections(values):
    return {f"G{g + 1}": {
        C.KEY_SD_DEFL_LIVE_RAW: FREQUENT_LIVE_LOAD_FACTOR * d["live_m"] * 1000,
        C.KEY_SD_DEFL_DL_RAW: sum(d["dead_m"].values()) * 1000,
        # dead load deflection stays unfactored: it sets the camber
        C.KEY_SD_DEFL_TOTAL_RAW: (sum(d["dead_m"].values()) + FREQUENT_LIVE_LOAD_FACTOR * d["live_m"]) * 1000,
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


# One girder's demand under one load case of the plots' dropdown, worst over the states of an envelope. An SLS combination's moment
# is split by stage (sls_stages) with the dead and surfacing factors of the value it governs; other cases leave it whole.
def case_demand(states, girder, name, envelope, dead=None, governing_values=None):
    prefixes = {"Envelope ULS": ("BASIC_", "SEISMIC_"), "Envelope SLS": ("SLS_",)}.get(name)
    named = [(case, state[girder]) for case, state in states.items() if (case.startswith(prefixes) if prefixes else case == name)]
    if not named:
        return None
    forces = [f for _, f in named]
    worst_case, worst = max(named, key=lambda item: float(abs(item[1].composite_moment_kn_m).max()))
    moment = float(abs(worst.composite_moment_kn_m).max())
    shear = max(float(abs(f.shear_kn).max()) for f in forces)
    deflection = max(float(abs(f.deflection_m).max()) for f in forces) * 1000
    kind = case_kind(name)
    service = kind != "ULS"
    stages = None
    diagram = (governing_values or {}).get(worst_case)
    if service and dead is not None and diagram is not None and worst_case.startswith("SLS_"):
        station = int(abs(worst.composite_moment_kn_m).argmax())
        stages = stages_at(dead, girder, station, float(worst.composite_moment_kn_m[station]),
                           *state_factors(dead, diagram.governing, diagram.girder))
    return {"Mu_kNm": moment, "Vu_kN": shear, "M_sls_kNm": moment if service else 0.0, "V_sls_kN": shear if service else 0.0,
            "M_sls_stages_kNm": stages,
            "M_construction_kNm": moment if kind == "DL" else 0.0, "M_girder_sw_kNm": moment if name == "SW" else 0.0,
            "delta_live_mm": deflection if kind == "live_only" else 0.0, "delta_dl_mm": deflection if kind == "DL" else 0.0,
            "stress_range_MPa": envelope.stress_range_MPa if kind == "SLS_frequent" else 0.0,
            "shear_range_MPa": envelope.shear_range_MPa if kind == "SLS_frequent" else 0.0, "lc_type": kind}


# Each Class SV combination's governing moment and shear, by combination name, as the native combinations are held.
def special_vehicle_combinations(special_vehicle, girder):
    by_name = {}
    for response in ("maximum composite moment", "support shear"):
        for by_direction in special_vehicle.girders[girder][response].values():
            for value in by_direction.values():
                by_name.setdefault(value.combination, {}).setdefault(response, {})[value.adverse] = value
    return {name: responses for name, responses in by_name.items() if len(responses) == 2}


# Scope each native combination to the designer's check types.
def combination_demands(values, girder, config, fatigue, special_vehicle=None):
    result = {}
    combinations = {name: {"maximum composite moment": directions, "support shear": values.combinations[girder]["support shear"][name]}
                    for name, directions in values.combinations[girder]["maximum composite moment"].items()}
    if special_vehicle is not None:
        combinations.update(special_vehicle_combinations(special_vehicle, girder))
    for name, responses in combinations.items():
        moment = governing(responses["maximum composite moment"].values())
        shear = governing(responses["support shear"].values())
        kind = ("ULS" if name.startswith("ultimate,") else
                "SLS_frequent" if name.startswith("serviceability, frequent") else "SLS")
        result[name] = DemandEnvelope(
            Mu_kNm=abs(moment.value), Vu_kN=abs(shear.value),
            M_sls_kNm=abs(moment.value) if kind != "ULS" else 0.0,
            V_sls_kN=abs(shear.value) if kind != "ULS" else 0.0,
            M_sls_stages_kNm=sls_stages(values.dead, moment, girder) if kind != "ULS" else None,
            stress_range_MPa=fatigue["stress_range_MPa"] if kind == "SLS_frequent" else 0.0,
            shear_range_MPa=fatigue["shear_range_MPa"] if kind == "SLS_frequent" else 0.0,
            Nsc=config.fatigue.Nsc, governing_combination=moment.combination,
            location=f"{moment.at_m} m", member=f"G{girder + 1}", source="setu", lc_type=kind)
    return result


# Build the designer's existing demand seam directly from the public results.
def girder_demands(values, configs, cache, seismic, fatigue_enabled, special_vehicle=None, construction=None):
    demands, per_lc = {}, {}
    for g, config in enumerate(configs):
        name = f"G{g + 1}"
        forces = force_demands(values, g, seismic, special_vehicle)
        fatigue = fatigue_demands(values, g, config, fatigue_enabled)
        # The steel axial force belongs to the internal composite couple.
        demands[name] = DemandEnvelope(
            **forces, **construction_demands(values.dead, construction, g),
            **fatigue, **deflection_demands(cache[name]), Nu_kN=0.0,
            Nsc=config.fatigue.Nsc, member=name, source="setu")
        per_lc[name] = combination_demands(values, g, config, fatigue, special_vehicle)
    return demands, per_lc
