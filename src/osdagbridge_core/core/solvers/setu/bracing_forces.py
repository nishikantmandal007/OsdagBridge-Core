from osdagbridge_core.core.bridge_types.plate_girder.crossbracingforces import CrossBracingForces, EndDiaphragmForces

DIAGONALS = ("diagonal_down", "diagonal_up", "k_top_left", "k_top_right", "k_bottom_left", "k_bottom_right")
CHORDS = ("top_chord", "bottom_chord")
SMALLEST_REPORTED_KN = 0.005  # OsdagBridge reports nothing at or below 5 N
REPORTED_DECIMALS = 3


# The ultimate limit states braces are designed for, the same ones the girder demands read.
def ultimate_limit_states(engine):
    return ["ultimate, basic"] + (["ultimate, seismic"] if engine.seismic is not None else [])


# Brace stations between the two support lines.
def intermediate_stations(engine):
    return range(1, engine.bridge_input.bracing.station_count - 1)


# The two support lines, where the end diaphragms sit.
def end_stations(engine):
    return (0, engine.bridge_input.bracing.station_count - 1)


# OsdagBridge's name for the girder pair setu numbers from 0 at G1-G2.
def pair_name(pair_index):
    return f"G{pair_index + 1}-G{pair_index + 2}"


# Every governing ULS axial force (kN, tension positive) of these roles in one pair at these stations.
def governing_forces(engine, pair_index, stations, roles):
    states = ultimate_limit_states(engine)
    return [value for (pair, station, role), by_state in engine.brace_values.members.items()
            if pair == pair_index and station in stations and role in roles
            for state in states for value in by_state[state].values()]


# Tension and compression of one member kind in OsdagBridge's keys: compression as a magnitude, None when negligible.
def tension_and_compression(prefix, forces):
    summary = dict.fromkeys((f"{prefix}_tension_kN", f"{prefix}_tension_gov_lc", f"{prefix}_compression_kN", f"{prefix}_compression_gov_lc"))
    if not forces:
        return summary
    tension = max(forces, key=lambda force: force.value)
    compression = min(forces, key=lambda force: force.value)
    if tension.value > SMALLEST_REPORTED_KN:
        summary[f"{prefix}_tension_kN"] = round(tension.value, REPORTED_DECIMALS)
        summary[f"{prefix}_tension_gov_lc"] = tension.combination
    if compression.value < -SMALLEST_REPORTED_KN:
        summary[f"{prefix}_compression_kN"] = round(abs(compression.value), REPORTED_DECIMALS)
        summary[f"{prefix}_compression_gov_lc"] = compression.combination
    return summary


# One girder pair's design forces: setu's member axial forces, already along each member, never resolved again.
def pair_forces(engine, pair_index, stations):
    return {**tension_and_compression("diag", governing_forces(engine, pair_index, stations, DIAGONALS)),
            **tension_and_compression("chord", governing_forces(engine, pair_index, stations, CHORDS))}


# Cross-bracing designed from setu's intermediate brace members instead of grillage shears.
class SetuCrossBracingForces(CrossBracingForces):

    # Intermediate panels in the setu model, one per pair at each station between the supports.
    def get_crossbracing_count(self):
        engine = self.bridge.grillage_model
        return len(intermediate_stations(engine)) * (engine.bridge_input.girders.count - 1)

    # The parent's per-pair configuration and geometry, with setu's governing forces.
    def get_design_forces_dict(self):
        engine = self.bridge.grillage_model
        pairs = [pair_name(k) for k in range(engine.bridge_input.girders.count - 1)]
        configs = {pair: self._get_pair_config(pair) for pair in pairs}
        return {
            "brace_type": {pair: configs[pair]["brace_type"] for pair in pairs},
            "top_chord": {pair: configs[pair]["top_chord"] for pair in pairs},
            "bottom_chord": {pair: configs[pair]["bottom_chord"] for pair in pairs},
            "geometry": {pair: self.get_brace_geometry_info(pair) for pair in pairs},
            "pairs": {pair_name(k): pair_forces(engine, k, intermediate_stations(engine)) for k in range(len(pairs))},
        }


# End diaphragms designed from setu's brace members on both support lines instead of grillage edge elements.
class SetuEndDiaphragmForces(EndDiaphragmForces):

    # setu's governing forces over both ends; cos α is unused because setu forces are already along each member.
    def get_pair_forces(self, girder_pair, cos_alpha):
        engine = self.bridge.grillage_model
        pair_index = int(girder_pair.split("-")[0].removeprefix("G")) - 1
        return pair_forces(engine, pair_index, end_stations(engine))
