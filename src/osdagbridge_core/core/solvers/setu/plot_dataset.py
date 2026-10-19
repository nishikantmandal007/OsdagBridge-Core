from importlib import import_module

import numpy as np
import xarray as xr

from setu.postprocess.girder_response import STEEL_SELF_WEIGHT, SUPERIMPOSED, SURFACING_STAGE, WET_SLAB
from setu.postprocess.result_dataset import DISPLACEMENT_COMPONENTS, FORCE_COMPONENTS
from setu.utils.constants import (
    BASIC, BIGGER_IS_WORSE, DEAD, FREQUENT, MAX_MOMENT, QUASI_PERMANENT, RARE, SEISMIC_COMBINATION, SURFACING)

ops = import_module("openseespy.opensees")  # a compiled module: imported by name so type checkers do not judge its commands against an empty stub
N_PER_KN = 1000.0
DEAD_STAGE_CASES = {STEEL_SELF_WEIGHT: "SW", WET_SLAB: "DD", SUPERIMPOSED: "SIDL", SURFACING_STAGE: "1.0 DW"}
DEAD_LOAD_CASE = "1.0 DL"
LIVE_LOAD_CASE = "1.0 LL"
BRAKING_CASE = "Braking Load"
COMBINATION_PREFIX = {BASIC: "BASIC_", SEISMIC_COMBINATION: "SEISMIC_", RARE: "SLS_RARE_", FREQUENT: "SLS_FREQUENT_",
                      QUASI_PERMANENT: "SLS_QP_"}
ENVELOPES = {"Envelope ULS": ("BASIC_", "SEISMIC_"), "Envelope SLS": ("SLS_",)}
SKELETON_MATERIAL = 1
SKELETON_AREA = 1.0  # the skeleton is geometry for OsdagBridge's readers and is never analysed


# Every setu load state under OsdagBridge's load-case names, each girder's forces along it.
def load_states(values):
    girders, diagrams = values.girders, values.diagrams
    states = {DEAD_STAGE_CASES[stage]: forces for stage, forces in girders.dead.stages.items()}
    states[DEAD_LOAD_CASE] = girders.dead.factored({DEAD: 1.0, SURFACING: 0.0})
    governing_girder, _ = girders.governing(MAX_MOMENT, BASIC, BIGGER_IS_WORSE)
    states[LIVE_LOAD_CASE] = diagrams.live[governing_girder, MAX_MOMENT, BIGGER_IS_WORSE]
    states[BRAKING_CASE] = diagrams.braking[governing_girder, MAX_MOMENT, BIGGER_IS_WORSE]
    for (girder, response, adverse), forces in diagrams.live.items():
        states[f"Case G{girder + 1} {response}, {adverse}"] = forces
    for name, forces in diagrams.wind.items():
        states[f"WL {name}"] = forces
    for prefix, (targets, forces) in distinct_governing_states(diagrams.governing).items():
        states[f"{prefix}: {'; '.join(targets)}"] = forces
    return states


# Identical solved states named once per limit state, listing every girder value they govern.
def distinct_governing_states(governing):
    by_forces = {}
    for (girder, response, limit_state, adverse), diagram in governing.items():
        key = (limit_state, b"".join(forces_bytes(diagram.forces[g]) for g in sorted(diagram.forces)))
        targets, _ = by_forces.setdefault(key, ([], diagram.forces))
        targets.append(f"G{girder + 1} {response}, {adverse} ({diagram.governing.combination})")
    counts = dict.fromkeys(COMBINATION_PREFIX.values(), 0)
    named = {}
    for (limit_state, _), state in by_forces.items():
        prefix = COMBINATION_PREFIX[limit_state]
        counts[prefix] += 1
        named[f"{prefix}{counts[prefix]}"] = state
    return named


# Every number a girder's forces hold, for telling identical states apart.
def forces_bytes(forces):
    return forces.composite_moment_kn_m.tobytes() + forces.shear_kn.tobytes() + forces.deflection_m.tobytes()


# Plan positions of every node line left to right: the deck edges join the girders when OsdagBridge names edge beams.
def node_lines(diagrams, edge_dist):
    girders = [diagrams.girder_positions_m[girder] for girder in sorted(diagrams.girder_positions_m)]
    if edge_dist > 0:
        return [diagrams.edge_positions_m["left"], *girders, diagrams.edge_positions_m["right"]]
    return girders


# Node tag of a station on a line, numbered station by station across the deck.
def node_tag(station, line, line_count):
    return 1 + station * line_count + line


# Element tag of a panel on a line, numbered line by line.
def element_tag(panel, line, panel_count):
    return 1 + line * panel_count + panel


# The node lines and their elements in the live OpenSees domain, which OsdagBridge's plot readers walk.
def build_skeleton(lines):
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.uniaxialMaterial("Elastic", SKELETON_MATERIAL, 1.0)
    panels = len(lines[0]) - 1
    for line, positions in enumerate(lines):
        for station, (x_m, z_m) in enumerate(positions):
            ops.node(node_tag(station, line, len(lines)), x_m, 0.0, z_m)
    for line in range(len(lines)):
        for panel in range(panels):
            ops.element("truss", element_tag(panel, line, panels), node_tag(panel, line, len(lines)),
                        node_tag(panel + 1, line, len(lines)), SKELETON_AREA, SKELETON_MATERIAL)


# setu's internal forces as ospgrillage's element end forces in N and N·m: +M and +V at the i end, -M and -V at the j end.
def girder_end_forces(forces):
    moment = N_PER_KN * forces.composite_moment_kn_m
    shear = N_PER_KN * forces.shear_kn
    return {"Mz_i": moment[:-1], "Mz_j": -moment[1:], "Vy_i": shear[:-1], "Vy_j": -shear[1:]}


# Forces and displacements of every load state on the skeleton, in ospgrillage's dataset layout; edge lines carry none.
def states_dataset(states, lines, first_girder_line):
    panels = len(lines[0]) - 1
    stations = panels + 1
    components = FORCE_COMPONENTS + DISPLACEMENT_COMPONENTS
    elements = [element_tag(panel, line, panels) for line in range(len(lines)) for panel in range(panels)]
    nodes = [node_tag(station, line, len(lines)) for station in range(stations) for line in range(len(lines))]
    forces = np.full((len(states), len(elements), len(components)), np.nan)
    displacements = np.full((len(states), len(nodes), len(components)), np.nan)
    vertical = components.index("y")
    for case, by_girder in enumerate(states.values()):
        for girder, girder_forces in by_girder.items():
            line = first_girder_line + girder
            for component, values in girder_end_forces(girder_forces).items():
                forces[case, line * panels:(line + 1) * panels, components.index(component)] = values
            displacements[case, line::len(lines), vertical] = girder_forces.deflection_m
    coords = {"Loadcase": list(states), "Element": elements, "Node": nodes, "Component": components}
    return xr.Dataset({"forces": (("Loadcase", "Element", "Component"), forces),
                       "displacements": (("Loadcase", "Node", "Component"), displacements)}, coords=coords)


# The ULS and SLS envelopes added as load cases: per element, node and component, the worst signed magnitude.
def with_envelopes(dataset):
    names = [str(name) for name in dataset.Loadcase.values]
    rows = [dataset]
    for label, prefixes in ENVELOPES.items():
        members = dataset.sel(Loadcase=[name for name in names if name.startswith(prefixes)])
        high, low = members.max("Loadcase"), members.min("Loadcase")
        rows.append(xr.where(abs(high) >= abs(low), high, low).expand_dims(Loadcase=[label]))
    return xr.concat(rows, dim="Loadcase")


# setu's solved load states as the dataset and live model geometry OsdagBridge plots and tabulates.
def plot_dataset(values, edge_dist):
    lines = node_lines(values.diagrams, edge_dist)
    build_skeleton(lines)
    return with_envelopes(states_dataset(load_states(values), lines, 1 if edge_dist > 0 else 0))
