import math
from dataclasses import asdict, fields
from unittest.mock import patch

import numpy as np
import pytest

from setu_helper import default_input, supported_input
from osdagbridge_core.core.utils import common as C
from osdagbridge_core.core.utils.codes.irc6_2017 import IRC6_2017
from osdagbridge_core.core.bridge_types.plate_girder.plategirderbridge import PlateGirderBridge
from osdagbridge_core.core.bridge_types.plate_girder.designer import (
    BridgeConfig, DemandEnvelope, _extract_demands_from_result_data, apply_camber_to_deflections_cache)
from osdagbridge_core.core.bridge_types.plate_girder.results_data import build_deflections_cache
from osdagbridge_core.core.solvers.setu.analysis_engine import SetuAnalysisEngine
from osdagbridge_core.core.solvers.setu.unsupported_inputs import check_capability
from osdagbridge_core.core.solvers.setu.girder_demands import girder_demands, raw_deflections
from osdagbridge_core.core.solvers.setu.bridge_input import (
    added_dead_loads, brace_value, bridge_input, cross_section, seismic_site, temperature_site, wind_site)


# Prepare the GUI inputs without building a structural model.
def prepared(d):
    b = PlateGirderBridge()
    b.set_input(d)
    b._resolve_optimized_bounds_to_mm()
    b._convert_girder_dims_mm_to_m()
    b._build_dtos()
    b.material_props = b._build_material_props()
    return b


# Record completed design stages around the real stage runner.
def record_stages(b):
    completed = []
    run = b._run_stage

    # Keep failed stages out of the completion list.
    def stage(number, function, *args, **kwargs):
        value = run(number, function, *args, **kwargs)
        completed.append(number)
        return value

    b._run_stage = stage
    return completed


@pytest.fixture(scope="module")
# Run both design stages through the real public API once.
def setu_bridge():
    b = PlateGirderBridge()
    b.set_input(supported_input())
    completed = record_stages(b)
    with patch("setu.girder_design_values", wraps=__import__("setu").girder_design_values) as solve:
        with pytest.raises(NotImplementedError, match="setu engine phase 2/3"):
            b.design()
    assert solve.call_count == 1
    assert completed == ["1", "2", "5", "6"]
    assert b.design_results["per_girder"] and b.deck_design_results
    print(f"SETU_ANALYSIS_SECONDS={b.grillage_model.analysis_seconds:.6f}")
    assert b.grillage_model.analysis_seconds <= 300, "Time the 0.25 m mesh and stop for the reviewer"
    return b


# Check the ospgrillage stages and its existing stage-seven dependency boundary.
def designed_osp(d):
    d[C.KEY_ANALYSIS_ENGINE] = C.ENGINE_OSPGRILLAGE
    b = PlateGirderBridge()
    b.set_input(d)
    completed = record_stages(b)
    try:
        b.design()
    except ModuleNotFoundError as error:
        assert error.name == "osdag_core"
    assert completed[:11] == ["1", "2", "3", "4A", "4B", "4C", "4D", "4E", "4F", "4G", "5"]
    assert "6" in completed
    assert b.result_data["crossbracings"]
    from osdagbridge_core.core.bridge_types.plate_girder.crossbracingforces import CrossBracingForces
    assert CrossBracingForces(b).get_design_forces_dict()["pairs"]
    return b


@pytest.fixture(scope="module")
# Both engines see the same supported default bridge.
def osp_bridge():
    return designed_osp(supported_input())


# Check geometry, all material units, strip loads and each independent girder.
def test_input_mapping():
    b = prepared(supported_input())
    d = b.input_dict
    d[C.KEY_SKEW_ANGLE] = 15.0
    d[f"{C.KEY_MP_GIRDER_TOP_FLANGE_WIDTH}.G2.M1"] = 0.7
    mapped = bridge_input(b)
    assert mapped.span_m == float(d[C.KEY_SPAN])
    assert mapped.skew == math.tan(math.radians(15))
    assert mapped.width_m() == float(d[C.KEY_TS_OVERALL_WIDTH])
    assert mapped.deck.thickness_m == float(d[C.KEY_TS_DECK_THICKNESS]) / 1000
    assert mapped.deck.overhang_m == float(d[C.KEY_TS_DECK_OVERHANG])
    assert mapped.deck.wearing_course_thickness_m == float(d[C.KEY_WC_THICKNESS]) / 1000
    assert mapped.wearing_course_unit_weight_kn_m3 == float(d[C.KEY_WC_DENSITY])
    assert mapped.steel.elastic_modulus_mpa == b.material_props.steel_prop.E / 1e6
    assert mapped.steel.elastic_modulus_mpa == float(d[C.KEY_MATERIAL_GIRDER_E]) * 1000
    assert mapped.steel.poissons_ratio == float(d[C.KEY_MATERIAL_GIRDER_POISSON])
    assert mapped.steel.unit_weight_kn_m3 == b.material_props.steel_prop.rho / 1000 == float(d[C.KEY_MATERIAL_GIRDER_DENSITY])
    assert mapped.concrete.elastic_modulus_mpa == float(d[C.KEY_MATERIAL_DECK_ECM]) * 1000
    assert mapped.concrete.poissons_ratio == 0.2
    assert mapped.concrete.unit_weight_kn_m3 == float(d[C.KEY_MATERIAL_DECK_DENSITY])
    assert mapped.girders.count == b._girder_count()
    plate_keys = {
        "top_flange_width_m": C.KEY_MP_GIRDER_TOP_FLANGE_WIDTH,
        "top_flange_thickness_m": C.KEY_MP_GIRDER_TOP_FLANGE_THICKNESS,
        "bottom_flange_width_m": C.KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH,
        "bottom_flange_thickness_m": C.KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS,
        "web_height_m": C.KEY_MP_GIRDER_WEB_DEPTH, "web_thickness_m": C.KEY_MP_GIRDER_WEB_THICKNESS,
    }
    for g in range(mapped.girders.count):
        for attr, key in plate_keys.items():
            assert getattr(mapped.girders.section_at(g), attr) == b._girder_value(key, g)
    assert mapped.bracing.arrangement == "XTB"
    assert mapped.bracing.station_count == int(d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS]) + 2
    assert mapped.bracing.area_m2 == C.get_angle_section_properties(brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION))["Area"] * 1e-4 * 2
    assert mapped.bracing.steel_grade == d[C.KEY_CROSS_BRACING]
    assert mapped.bracing.end_diaphragm_steel_grade == d[C.KEY_END_DIAPHRAGM]
    assert mapped.mesh.panels_between_braces == 16
    assert mapped.mesh.target_size_across_width_m == 0.15
    assert vars(mapped.added_dead_loads) == {"footpath_kpa": 0.0, "kerb_kn_per_m": 0.0, "median_kn_per_m": 0.0,
                                             "crash_barrier_kn_per_m": float(d[C.KEY_CB_LOAD]), "railing_kn_per_m": 0.0}
    assert [(s.name, s.width_m) for s in mapped.cross_section.strips] == [
        ("crash_barrier_left", 0.45), ("carriageway", 7.5), ("crash_barrier_right", 0.45)]


@pytest.mark.parametrize("count", [1, 2])
# Verify asymmetric footpaths and each-way carriageways with an explicit median.
def test_cross_section_strips(count):
    d = supported_input()
    d.update({C.KEY_TS_NO_OF_FOOTPATHS: count, C.KEY_TS_FOOTPATH_WIDTH: 1.5,
              C.KEY_RL_WIDTH: 0.075, C.KEY_MD_WIDTH: 0.6, C.KEY_MD_LOAD: 3.0})
    section = cross_section(d)
    assert section.total_width_m() == 2 * 7.5 + 0.9 + 0.6 + count * 1.575
    assert len(section.footways()) == count
    assert [s.width_m() for s in section.carriageways()] == [7.5, 7.5]
    loads = added_dead_loads(d)
    assert loads.footpath_kpa == 0.25 * 25
    assert loads.median_kn_per_m == 3.0
    assert loads.railing_kn_per_m == float(d[C.KEY_RL_LOAD_VALUE])


# Map every site mode, including effective seismic damping and reduction.
def test_site_mapping():
    d = supported_input()
    wind = wind_site(d)
    assert (wind.basic_wind_speed_mps, wind.terrain, wind.height_m, wind.funnelling, wind.solid_barrier_height_m) == (44, "plain", 10, False, 1.1)
    fields = {"gust": (C.KEY_WL_GUST_FACTOR_MODE, C.KEY_WL_GUST_FACTOR_VALUE),
              "drag": (C.KEY_WL_DRAG_COEFF_MODE, C.KEY_WL_DRAG_COEFF_VALUE),
              "drag_on_live_load": (C.KEY_WL_DRAG_COEFF_LL_MODE, C.KEY_WL_DRAG_COEFF_LL_VALUE),
              "lift": (C.KEY_WL_LIFT_COEFF_MODE, C.KEY_WL_LIFT_COEFF_VALUE),
              "exposed_area_m2": (C.KEY_WL_SUPER_AREA_ELEV_MODE, C.KEY_WL_SUPER_AREA_ELEV_VALUE),
              "plan_area_m2": (C.KEY_WL_SUPER_AREA_PLAIN_MODE, C.KEY_WL_SUPER_AREA_PLAIN_VALUE),
              "live_load_exposed_area_m2": (C.KEY_WL_EXPOSED_FRONTAL_MODE, C.KEY_WL_EXPOSED_FRONTAL_VALUE),
              "force_height_above_deck_m": (C.KEY_WL_WIND_ECC_DECK_MODE, C.KEY_WL_WIND_ECC_DECK_VALUE)}
    for i, (attr, (mode, value)) in enumerate(fields.items()):
        assert getattr(wind, attr) is None
        d[mode], d[value] = "Custom", 1.1 + i
    for i, attr in enumerate(fields):
        assert getattr(wind_site(d), attr) == 1.1 + i
    d[C.KEY_SL_RESPONSE_REDUCTION] = 2
    seismic = seismic_site(d)
    assert (seismic.zone, seismic.soil, seismic.importance_factor, seismic.period_s, seismic.response_reduction) == ("III", "I", 1, 0.5, 2)
    assert seismic.horizontal_coefficient == 0.16 / 2 * 2 * IRC6_2017.table_18(2) / 2
    assert seismic.vertical_coefficient == 2 / 3 * (0.16 / 2 * 2 * IRC6_2017.table_18(2)) / 2
    assert seismic.include_vertical
    assert seismic.dead_weight_kn is None and seismic.live_weight_kn is None
    d.update({C.KEY_SL_HORIZONTAL_COEFF: 0.3, C.KEY_SL_VERTICAL_COEFF: 0.2,
              C.KEY_SL_DEAD_LOAD_MODE: "Custom", C.KEY_SL_DEAD_LOAD_VALUE: 1000,
              C.KEY_SL_LIVE_LOAD_MODE: "Custom", C.KEY_SL_LIVE_LOAD_VALUE: 200})
    custom = seismic_site(d)
    assert (custom.horizontal_coefficient, custom.vertical_coefficient, custom.dead_weight_kn, custom.live_weight_kn) == (0.15, 0.1, 1000, 200)
    assert vars(temperature_site(d)) == {"shade_max_c": 40.0, "shade_min_c": 15.0}


@pytest.mark.parametrize("key,value", [
    (C.KEY_LL_IRC_CLASS_SV, True), (C.KEY_BL_IRC_CLASS_SV, True), (C.KEY_LL_IRC_70R_BOGIE, True), ("loading.live_load.irc.class_b", True),
    (C.KEY_LL_IRC_CLASS_A, False), (C.KEY_LL_CUSTOM_VEHICLES, [{"name": "custom"}]),
    (C.KEY_MP_CB_TYPE + ".G2G3.B2M1", "K-Bracing"),
    (C.KEY_MP_ED_BRACING_TYPE + ".G1G2.E1M1", "K-Bracing"),
    (C.KEY_MP_ED_TYPE + ".G1G2.E1M1", "Rolled Beam"),
    (C.KEY_MP_CB_SPACING + ".G2G3.B2M1", 5.0),
    ("temperature_load_kN_m2", 1.5),
    (C.KEY_SC_RIGHT_SUPPORT, "Fixed"), (C.KEY_PL_SELF_WEIGHT_FACTOR, 1.2),
    (C.KEY_MP_GD_SEGMENT_TABLE + ".G1", [{"start": 0, "end": 10}, {"start": 10, "end": 35}]),
])
# Unsupported selections must fail instead of switching engines or dropping a load.
def test_capability(key, value):
    b = prepared(supported_input())
    b.input_dict[key] = value
    with pytest.raises(ValueError, match="choose the ospgrillage analysis engine") as error:
        check_capability(b)
    assert key in str(error.value)


# The literal GUI defaults conflict with setu's unsupported-vehicle boundary.
def test_literal_default_rejected():
    assert default_input()[C.KEY_ANALYSIS_ENGINE] == C.ENGINE_SETU
    with pytest.raises(ValueError, match="ospgrillage analysis engine"):
        check_capability(prepared(default_input()))


# Required material inputs cannot silently become a default value.
def test_missing_input_raises():
    b = prepared(supported_input())
    del b.input_dict[C.KEY_MATERIAL_DECK_ECM]
    with pytest.raises(KeyError, match=C.KEY_MATERIAL_DECK_ECM):
        bridge_input(b)


# Independently state every mapped field so no default or unit error can hide.
def expected_demand(engine, g, name=None):
    values, config = engine.design_values, engine.configs[g]
    if name is None:
        states = ["ultimate, basic"] + (["ultimate, seismic"] if engine.seismic is not None else [])
        moments = [v for state in states for v in values.girders[g]["maximum composite moment"][state].values()]
        shears = [v for state in states for v in values.girders[g]["support shear"][state].values()]
        rare_m = max(values.girders[g]["maximum composite moment"]["serviceability, rare"].values(), key=lambda v: abs(v.value))
        rare_v = max(values.girders[g]["support shear"]["serviceability, rare"].values(), key=lambda v: abs(v.value))
        kind = ""
    else:
        moments = list(values.combinations[g]["maximum composite moment"][name].values())
        shears = list(values.combinations[g]["support shear"][name].values())
        kind = "ULS" if name.startswith("ultimate,") else "SLS_frequent" if name.startswith("serviceability, frequent") else "SLS"
        rare_m, rare_v = max(moments, key=lambda v: abs(v.value)), max(shears, key=lambda v: abs(v.value))
    m, v = max(moments, key=lambda v: abs(v.value)), max(shears, key=lambda v: abs(v.value))
    expected = dict.fromkeys([f.name for f in fields(DemandEnvelope)], 0.0)
    expected.update(Mu_kNm=abs(m.value), Vu_kN=abs(v.value), Nsc=config.fatigue.Nsc,
                    governing_combination=m.combination, location=f"{m.at_m} m", member=f"G{g + 1}", source="setu", lc_type=kind)
    if kind != "ULS":
        expected.update(M_sls_kNm=abs(rare_m.value), V_sls_kN=abs(rare_v.value))
    if kind in ("", "SLS_frequent"):
        expected.update(stress_range_MPa=values.fatigue_stresses_mpa[g]["steel bottom"].range,
                        shear_range_MPa=values.fatigue[g]["support shear"].range * 1000 / config.section.Aw)
    if name is None:
        sw, wet = (values.dead.stages[s][g] for s in ("steel self weight", "wet slab"))
        i = np.argmin(np.abs(sw.stations_m - m.at_m))
        cache = engine.deflections_cache[f"G{g + 1}"]
        expected.update(M_girder_sw_kNm=abs(float(sw.composite_moment_kn_m[i])),
                        M_construction_kNm=abs(float(sw.composite_moment_kn_m[i] + wet.composite_moment_kn_m[i])),
                        Vr_kN=values.live_shear_range_kn[g], delta_live_mm=cache["live_mm"],
                        delta_dl_mm=cache["dl_mm"], delta_total_mm=cache["total_mm"], camber_mm=cache["camber_mm"])
    return expected


# Exact comparisons cover every field, girder, native combination and check scope.
def test_demand_mapping(setu_bridge):
    engine = setu_bridge.grillage_model
    for g in engine.design_values.girders:
        label = f"G{g + 1}"
        assert asdict(engine.per_girder_demands[label]) == expected_demand(engine, g)
        assert set(engine.per_girder_per_lc[label]) == set(engine.design_values.combinations[g]["maximum composite moment"])
        for name, demand in engine.per_girder_per_lc[label].items():
            assert asdict(demand) == expected_demand(engine, g, name)
        assert setu_bridge.design_results["per_girder"][label]["demand"]["Mu_kNm"] == engine.per_girder_demands[label].Mu_kNm


# Direct metres-to-mm conversion precedes the existing camber function.
def test_deflections_cache(setu_bridge):
    engine = setu_bridge.grillage_model
    raw = raw_deflections(engine.design_values)
    for g, d in engine.design_values.deflections.items():
        assert raw[f"G{g + 1}"] == {
            C.KEY_SD_DEFL_LIVE_RAW: d["live_m"] * 1000,
            C.KEY_SD_DEFL_DL_RAW: sum(d["dead_m"].values()) * 1000,
            C.KEY_SD_DEFL_TOTAL_RAW: d["total_m"] * 1000,
            "per_lc": {stage: value * 1000 for stage, value in d["dead_m"].items()}}
    config = engine.configs[0]
    assert engine.deflections_cache == apply_camber_to_deflections_cache(raw, config.geometry.camber_mode, config.geometry.camber_value_mm)
    assert setu_bridge._deflections_cache is engine.deflections_cache
    with patch("osdagbridge_core.core.solvers.setu.girder_demands.apply_camber_to_deflections_cache", wraps=apply_camber_to_deflections_cache) as camber:
        from osdagbridge_core.core.solvers.setu.girder_demands import deflections_cache
        assert deflections_cache(engine.design_values, config) == engine.deflections_cache
        camber.assert_called_once_with(raw, config.geometry.camber_mode, config.geometry.camber_value_mm)
    custom = apply_camber_to_deflections_cache(raw, "Custom", 10.0)
    demands, _ = girder_demands(engine.design_values, engine.configs, custom, True, True)
    assert demands["G1"].delta_total_mm == custom["G1"]["total_mm"]


# Large negative responses and enabled seismic must both enter the envelope.
def test_adverse_and_seismic(setu_bridge):
    from copy import deepcopy
    engine = setu_bridge.grillage_model
    values = deepcopy(engine.design_values)
    for g in values.girders:
        for response in ("maximum composite moment", "support shear"):
            values.girders[g][response]["ultimate, basic"]["maximum"].value = 10
            values.girders[g][response]["ultimate, basic"]["minimum"].value = -20
            values.girders[g][response]["ultimate, seismic"]["maximum"].value = 30
            values.girders[g][response]["ultimate, seismic"]["minimum"].value = -300
    for seismic, expected in ((False, 20), (True, 300)):
        demands, _ = girder_demands(values, engine.configs, engine.deflections_cache, seismic, True)
        assert all(d.Mu_kNm == expected and d.Vu_kN == expected for d in demands.values())


# Work out steel weight from plate areas and the actual brace member lengths.
def steel_weight_per_m(bridge):
    areas = [s.top_flange_width_m * s.top_flange_thickness_m + s.bottom_flange_width_m * s.bottom_flange_thickness_m + s.web_height_m * s.web_thickness_m
             for s in bridge.girders.sections]
    depths = [s.depth_m for s in bridge.girders.sections]
    assert bridge.bracing.arrangement == "XTB" and len(set(depths)) == 1
    spacing = (bridge.width_m() - 2 * bridge.deck.overhang_m) / (bridge.girders.count - 1)
    braces = bridge.bracing.station_count * (bridge.girders.count - 1) * (2 * math.hypot(spacing, depths[0]) + 2 * spacing)
    return (sum(areas) + braces * bridge.bracing.area_m2 / bridge.span_m) * bridge.steel.unit_weight_kn_m3


# Balance girder UDL and brace point weights without smearing the support braces.
def test_independent_dead_statics(setu_bridge):
    engine = setu_bridge.grillage_model
    bridge = engine.bridge_input
    length = bridge.span_m
    steel = steel_weight_per_m(bridge)
    girder = sum(s.top_flange_width_m * s.top_flange_thickness_m + s.bottom_flange_width_m * s.bottom_flange_thickness_m + s.web_height_m * s.web_thickness_m
                 for s in bridge.girders.sections) * bridge.steel.unit_weight_kn_m3
    station_weight = (steel - girder) * length / bridge.bracing.station_count
    brace_x = np.linspace(0, length, bridge.bracing.station_count)
    steel_moment = girder * length ** 2 / 8 + station_weight * sum(np.minimum(brace_x, length - brace_x) / 2)
    wet_moment = bridge.width_m() * bridge.deck.thickness_m * bridge.concrete.unit_weight_kn_m3 * length ** 2 / 8
    for stage, expected, tolerance in (("steel self weight", steel_moment, 0.01), ("wet slab", wet_moment, 0.02)):
        forces = engine.design_values.dead.stages[stage].values()
        actual = sum(float(f.composite_moment_kn_m[np.argmin(abs(f.stations_m - length / 2))]) for f in forces)
        print("Independent statics", stage, actual, expected)
        assert actual == pytest.approx(expected, rel=tolerance)
    print("All-steel UDL comparison (brace weights smeared)", steel_moment, steel * length ** 2 / 8)


# Estimate edge ULS from tributary dead weight and two Class A influence lines.
def test_independent_edge_uls(setu_bridge):
    engine = setu_bridge.grillage_model
    b = engine.bridge_input
    length = b.span_m
    spacing = (b.width_m() - 2 * b.deck.overhang_m) / (b.girders.count - 1)
    tributary = b.deck.overhang_m + spacing / 2
    steel = steel_weight_per_m(b) / b.girders.count
    slab = tributary * b.deck.thickness_m * b.concrete.unit_weight_kn_m3
    barriers = b.added_dead_loads.crash_barrier_kn_per_m
    surfacing = (tributary - 0.45) * b.deck.wearing_course_thickness_m * b.wearing_course_unit_weight_kn_m3
    # IRC Class A axle weights in tonnes and successive spacings in metres.
    weights = np.array([2.7, 2.7, 11.4, 11.4, 6.8, 6.8, 6.8, 6.8]) * 9.81
    offsets = np.cumsum([0, 1.1, 3.2, 1.2, 4.3, 3, 3, 3])
    axle_x = np.arange(-offsets[-1], length, 0.01)[:, None] + offsets
    line = np.maximum(np.minimum(axle_x, length - axle_x), 0) / 2
    lane_moment = float(np.max(line @ weights)) * (1 + 9 / (13.5 + length))
    # Courbon sharing for two legal lane centres on this symmetric deck.
    girder_z = np.linspace(-1.5 * spacing, 1.5 * spacing, b.girders.count)
    lane_centres = [-b.width_m() / 2 + 0.45 + 0.15 + 2.3 / 2,
                    -b.width_m() / 2 + 0.45 + 0.15 + 2.3 / 2 + 3.5]
    share = sum(1 / b.girders.count + lane * girder_z[0] / sum(girder_z ** 2) for lane in lane_centres)
    expected = (1.35 * (steel + slab + barriers) + 1.75 * surfacing) * length ** 2 / 8 + 1.5 * lane_moment * share
    actual = engine.per_girder_demands["G1"].Mu_kNm
    print("Hand edge ULS", actual, expected, "lane moment", lane_moment, "share", share)
    assert actual == pytest.approx(expected, rel=0.15)


# Minimal result data must refuse model-backed features explicitly.
def test_phase_boundaries(setu_bridge):
    engine = setu_bridge.grillage_model
    assert engine._deduplicated_results is None
    data = engine.get_result_data()
    assert dict(data) == {"girders": {f"G{g + 1}": {} for g in range(4)}}
    for key in ("crossbracings", "forces", "nodes", "displacements", "loadcases", "members"):
        with pytest.raises(NotImplementedError, match="setu engine phase 2/3"):
            data[key]
        with pytest.raises(NotImplementedError, match="setu engine phase 2/3"):
            data.get(key)
    for name in ("model", "dataset", "reactions", "get_element", "live_load_report_values"):
        with pytest.raises(NotImplementedError, match="setu engine phase 2/3"):
            getattr(engine, name)
    with pytest.raises(NotImplementedError, match="setu engine phase 2/3"):
        setu_bridge.get_results_dataset()
    with pytest.raises(NotImplementedError, match="setu engine phase 2/3"):
        setu_bridge.compute_load_effects_cache()


# Both real engines run girder and deck design on the same supported input.
def test_end_to_end_comparison(setu_bridge, osp_bridge):
    print("\nEngine | Girder | Mu kNm | Vu kN | Nu kN | M_sls kNm | live mm | governing DCR check | DCR")
    for name, b in (("setu", setu_bridge), ("ospgrillage", osp_bridge)):
        if name == "setu":
            demands = b.grillage_model.per_girder_demands
        else:
            config = BridgeConfig.from_plate_girder_bridge(b)
            cache = apply_camber_to_deflections_cache(build_deflections_cache(config, b.result_data), config.geometry.camber_mode, config.geometry.camber_value_mm)
            demands, _ = _extract_demands_from_result_data(config, cache, b.result_data)
        for girder, result in b.design_results["per_girder"].items():
            d = demands[girder]
            check = max((c for c in result["checks"] if c["dcr"] is not None), key=lambda c: c["dcr"])
            print(f"{name} | {girder} | {d.Mu_kNm:.6f} | {d.Vu_kN:.6f} | {d.Nu_kN:.6f} | {d.M_sls_kNm:.6f} | {d.delta_live_mm:.6f} | {check['name']} | {result['max_dcr']:.6f}")


# Deselected fatigue still leaves the native live shear range available for studs.
def test_disabled_fatigue(setu_bridge):
    e = setu_bridge.grillage_model
    demands, per_lc = girder_demands(e.design_values, e.configs, e.deflections_cache, True, False)
    for g, demand in demands.items():
        assert demand.stress_range_MPa == demand.shear_range_MPa == 0
        assert demand.Vr_kN == e.per_girder_demands[g].Vr_kN
        assert all(d.stress_range_MPa == d.shear_range_MPa == 0 for d in per_lc[g].values())


@pytest.mark.parametrize("engine", [C.ENGINE_SETU, C.ENGINE_OSPGRILLAGE])
# The approved hook supplies demands only on the setu branch.
def test_dcr_hook(engine):
    from types import SimpleNamespace
    b = PlateGirderBridge()
    b.input_dict = {C.KEY_ANALYSIS_ENGINE: engine}
    b.grillage_model = SimpleNamespace(per_girder_demands={"G1": "demand"}, per_girder_per_lc={"G1": {}}, deflections_cache={"G1": "cache"})
    module = "osdagbridge_core.core.bridge_types.plate_girder.plategirderbridge"
    with patch(module + ".run_design_check", return_value=(None, "dcr", {"result": 1})) as check, patch.object(b, "store_design_results"), patch(module + ".bridge_logger.girder_verdict"):
        b._run_dcr_checks()
    if engine == C.ENGINE_SETU:
        check.assert_called_once_with(b, b.grillage_model.per_girder_demands, b.grillage_model.per_girder_per_lc, b.grillage_model.deflections_cache)
    else:
        check.assert_called_once_with(plate_girder_bridge=b, print_report=True)
    assert b._dcr_engine == "dcr" and b.design_results == {"result": 1}
