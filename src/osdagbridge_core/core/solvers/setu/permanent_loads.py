from types import SimpleNamespace

import numpy as np

from setu.builder.mesh import build_mesh
from setu.irc6.irc_constants import WET_CONCRETE_EXTRA_KN_M3
from setu.loads.dead_loads import girder_tributary_width_m, line_loads_across
from setu.models.sections import girder_properties


def permanent_loads_per_girder(bridge):
    lines_m = build_mesh(bridge).girder_lines_m
    model = SimpleNamespace(bridge=bridge, mesh=SimpleNamespace(girder_lines_m=lines_m))
    edges_m = [0.0] + [(a + b) / 2 for a, b in zip(lines_m, lines_m[1:])] + [bridge.width_m()]
    strips = bridge.cross_section.strips

    def on_each_girder(kpa, carries):
        return [sum(kpa * max(0.0, min(hi, s.z_to_m) - max(lo, s.z_from_m)) for s in strips if carries(s))
                for lo, hi in zip(edges_m, edges_m[1:])]

    wet_slab_kpa = (bridge.concrete.unit_weight_kn_m3 + WET_CONCRETE_EXTRA_KN_M3) * bridge.deck.thickness_m
    dw = on_each_girder(bridge.wearing_course.pressure_kpa, lambda s: s.carries_traffic())
    sidl = on_each_girder(bridge.added_dead_loads.footpath_kpa, lambda s: s.carries_pedestrians())
    for z_m, kn_per_m in line_loads_across(bridge):
        z_m = min(max(z_m, lines_m[0]), lines_m[-1])
        i = min(int(np.searchsorted(lines_m, z_m, side="right")) - 1, len(lines_m) - 1)
        t = (z_m - lines_m[i]) / (lines_m[i + 1] - lines_m[i]) if i + 1 < len(lines_m) else 0.0
        sidl[i] += kn_per_m * (1 - t)
        if t:
            sidl[i + 1] += kn_per_m * t
    return [{"SW":   bridge.steel.unit_weight_kn_m3 * girder_properties(bridge.girders.section_at(k)).area_m2,
             "DD":   wet_slab_kpa * girder_tributary_width_m(model, k),
             "DW":   dw[k],
             "SIDL": sidl[k]}
            for k in range(bridge.girders.count)]
