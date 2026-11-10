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
    # no braking state where a congested or crowd case governs (IRC:6 cl. 204.4 leaves braking out)
    if (governing_girder, MAX_MOMENT, BIGGER_IS_WORSE) in diagrams.braking:
        states[BRAKING_CASE] = diagrams.braking[governing_girder, MAX_MOMENT, BIGGER_IS_WORSE]
    for (girder, response, adverse), forces in diagrams.live.items():
        states[f"Case G{girder + 1} {response}, {adverse}"] = forces
    for name, forces in diagrams.wind.items():
        states[f"WL {name}"] = forces
    for name, (forces, _) in governing_states(diagrams).items():
        states[name] = forces
    return states


# Each distinct governing state by its load case name: its girder forces and its brace members' axial forces.
def governing_states(diagrams):
    return {f"{prefix}: {'; '.join(targets)}": (forces, braces) for prefix, (targets, forces, braces, _) in distinct_governing_states(diagrams.governing).items()}


# The first governing value behind each named governing state: its combination's shares give the state's dead and surfacing factors.
def governing_values_of_states(diagrams):
    return {f"{prefix}: {'; '.join(targets)}": first for prefix, (targets, _, _, first) in distinct_governing_states(diagrams.governing).items()}


# Identical solved states named once per limit state, listing every girder value they govern.
def distinct_governing_states(governing):
    by_forces = {}
    for (girder, response, limit_state, adverse), diagram in governing.items():
        braces = diagram.brace_forces_kn
        key = (limit_state, b"".join(forces_bytes(diagram.forces[g]) for g in sorted(diagram.forces)), None if braces is None else braces.tobytes())
        targets, _, _, _ = by_forces.setdefault(key, ([], diagram.forces, braces, diagram))
        targets.append(f"G{girder + 1} {response}, {adverse} ({diagram.governing.combination})")
    counts = dict.fromkeys(COMBINATION_PREFIX.values(), 0)
    named = {}
    for (limit_state, _, _), state in by_forces.items():
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
    torsion = N_PER_KN * forces.torsion_kn_m
    axial = N_PER_KN * forces.axial_kn
    end_forces = {"Mz_i": moment[:-1], "Mz_j": -moment[1:], "Vy_i": shear[:-1], "Vy_j": -shear[1:], "Mx_i": torsion[:-1], "Mx_j": -torsion[1:],
                  "Vx_i": axial[:-1], "Vx_j": -axial[1:]}
    if forces.weak_axis_shear_kn is not None:
        weak_shear, weak_moment = N_PER_KN * forces.weak_axis_shear_kn, N_PER_KN * forces.weak_axis_moment_kn_m
        end_forces.update({"Vz_i": weak_shear[:-1], "Vz_j": -weak_shear[1:], "My_i": weak_moment[:-1], "My_j": -weak_moment[1:]})
    return end_forces


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
            if girder_forces.displacements_m is not None:
                for column, component in enumerate(DISPLACEMENT_COMPONENTS):
                    displacements[case, line::len(lines), components.index(component)] = girder_forces.displacements_m[:, column]
            displacements[case, line::len(lines), vertical] = girder_forces.deflection_m
    coords = {"Loadcase": list(states), "Element": elements, "Node": nodes, "Component": components}
    # the engine tag lets the plots label setu's results (the deflection summary's columns)
    return xr.Dataset({"forces": (("Loadcase", "Element", "Component"), forces),
                       "displacements": (("Loadcase", "Node", "Component"), displacements)}, coords=coords, attrs={"engine": "setu"})


# Each skeleton element's end nodes, in the live domain the plot readers walk.
def member_nodes(dataset):
    return {str(element): [int(node) for node in ops.eleNodes(int(element))] for element in dataset["Element"].values.tolist()}


# A load case's element-end forces as the load arrows read them, built when a case is asked for.
class CaseForces:
    def __init__(self, dataset):
        self.dataset = dataset

    def get(self, load_case, default=None):
        if load_case not in self.dataset["Loadcase"].values:
            return default
        forces = self.dataset["forces"].sel(Loadcase=load_case).fillna(0.0)
        names = [str(component) for component in forces["Component"].values]
        elements = forces["Element"].values.tolist()
        return {str(element): dict(zip(names, row)) for element, row in zip(elements, forces.transpose("Element", "Component").values.tolist())}


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
