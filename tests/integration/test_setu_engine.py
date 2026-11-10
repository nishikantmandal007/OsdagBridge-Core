import gc
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
from osdagbridge_core.core.solvers.setu.analysis_engine import SetuAnalysisEngine, temperature_report
from osdagbridge_core.core.solvers.setu.unsupported_inputs import check_capability, simplifications
from osdagbridge_core.core.solvers.setu.girder_demands import girder_demands, raw_deflections, sls_stages
from osdagbridge_core.core.solvers.setu.bridge_input import (
    added_dead_loads, brace_value, bracing, bridge_input, cross_section, analysis_gb, estimated_peak_gb, normalise_gui_bracing, with_workers_gb, worker_gb, workers_that_fit, reconcile_layout, seismic_site, temperature_site, wind_site)


# Prepare the GUI inputs without building a structural model.
def prepared(d):
    b = PlateGirderBridge()
    b.set_input(d)
    b._resolve_optimized_bounds_to_mm()
    b._convert_girder_dims_mm_to_m()
    b._build_dtos()
    b.material_props = b._build_material_props()
    return b


# The mapped setu bridge of these inputs.
def prepared_bridge_input(d):
    return bridge_input(prepared(d))


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
# Run the whole design, Osdag member design and CAD included, through one setu analysis.
def setu_bridge():
    b = PlateGirderBridge()
    b.set_input(supported_input())
    completed = record_stages(b)
    from osdagbridge_core.core.solvers.setu import analysis_engine
    with patch.object(analysis_engine, "solve_in_own_process", wraps=analysis_engine.solve_in_own_process) as solve:
        b.design()
    assert solve.call_count == 1
    assert completed == ["1", "2", "4G", "5", "6", "7", "8"]
    assert set(b.crossbracing_design_results) == set(b.end_diaphragm_design_results) == {"G1-G2", "G2-G3", "G3-G4"}
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
        "web_thickness_m": C.KEY_MP_GIRDER_WEB_THICKNESS,
    }
    for g in range(mapped.girders.count):
        for attr, key in plate_keys.items():
            assert getattr(mapped.girders.section_at(g), attr) == b._girder_value(key, g)
        # the web the designer checks: overall depth less both flanges
        web_m = (b._girder_value(C.KEY_MP_GIRDER_DEPTH, g) - b._girder_value(C.KEY_MP_GIRDER_TOP_FLANGE_THICKNESS, g)
                 - b._girder_value(C.KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS, g))
        assert mapped.girders.section_at(g).web_height_m == pytest.approx(web_m)
        assert mapped.girders.section_at(g).web_height_m * 1000 == pytest.approx(BridgeConfig.from_plate_girder_bridge(b, g).section.dw)
    assert mapped.bracing.arrangement == "XTB"
    assert mapped.bracing.station_count == int(d[C.KEY_MP_CB_NO_OF_CROSS_BRACINGS]) + 2
    assert mapped.bracing.area_m2 == C.get_angle_section_properties(brace_value(d, C.KEY_MP_CB_BRACING_SECTION_DESIGNATION))["Area"] * 1e-4 * 2
    assert mapped.bracing.steel_grade == d[C.KEY_CROSS_BRACING]
    assert mapped.bracing.end_diaphragm_steel_grade == d[C.KEY_END_DIAPHRAGM]
    assert d[C.KEY_ANALYSIS_MESH] == "Standard"
    assert (mapped.mesh.element_length_m, mapped.mesh.support_element_length_m, mapped.mesh.support_zone_m,
            mapped.mesh.target_size_across_width_m) == (0.5, 0.1, 1.5, 0.15)
    assert not mapped.cross_section.crowd_load and not mapped.cross_section.congestion
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
    assert (wind.basic_wind_speed_mps, wind.terrain_category, wind.superstructure_top_m, wind.road_level_m, wind.pier_height_m,
            wind.return_period_years, wind.valley_or_leeward_hill, wind.cyclonic_region, wind.deck_cross_slope_deg,
            wind.solid_barrier_height_m) == (44, 2, 10, 10, 10, 100, False, "inland", 0, 1.1)
    d.update({C.KEY_WL_TERRAIN_CATEGORY: "TC 3", C.KEY_WL_RETURN_PERIOD: "50", C.KEY_WL_ROAD_LEVEL: "8.5", C.KEY_WL_PIER_HEIGHT: "6",
              C.KEY_WL_K3_LOCATION: C.K3_VALLEY_OR_LEEWARD, C.KEY_WL_CYCLONIC_REGION: C.K4_CYCLONE_COAST, C.KEY_WL_CROSS_SLOPE: "1.5"})
    site = wind_site(d)
    assert (site.terrain_category, site.return_period_years, site.road_level_m, site.pier_height_m, site.valley_or_leeward_hill,
            site.cyclonic_region, site.deck_cross_slope_deg) == (3, 50, 8.5, 6, True, "cyclone coast", 1.5)
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
    assert vars(temperature_site(d)) == {"shade_max_c": 40.0, "shade_min_c": 15.0, "snowbound": False}
    assert temperature_report(temperature_site(d)) == {
        C.KEY_TL_BRIDGE_TEMP_MIN: "5.00", C.KEY_TL_BRIDGE_TEMP_MAX: "55.00",
        C.KEY_TL_TEMP_RISE: "25.00", C.KEY_TL_TEMP_FALL: "25.00"}
    d[C.KEY_TL_SNOWBOUND] = True
    assert temperature_site(d).snowbound
    assert temperature_report(temperature_site(d)) == {
        C.KEY_TL_BRIDGE_TEMP_MIN: "-35.00", C.KEY_TL_BRIDGE_TEMP_MAX: "50.00",
        C.KEY_TL_TEMP_RISE: "42.50", C.KEY_TL_TEMP_FALL: "42.50"}


@pytest.mark.parametrize("key,value", [("loading.live_load.irc.class_b", True)])
# A vehicle neither engine analyses stops the run and says so, instead of pointing at ospgrillage.
def test_vehicle_neither_engine_analyses(key, value):
    b = prepared(supported_input())
    b.input_dict[key] = value
    with pytest.raises(ValueError, match="not analysed by either engine") as error:
        check_capability(b)
    assert key in str(error.value)


# Every setu run states its bearing layout: the deck is held along the span at one bearing (four girders: G2's).
BEARING_NOTE = ("The left support line is held along the span at G2's bearing only: braking and longitudinal seismic go through that "
                "bearing, which gives G2 an end moment (the force times the bearing's depth below the girder's centroid) and its bearing "
                "the whole longitudinal force; design that bearing and G2's end for them")


# Inputs ospgrillage's analysis also ignores: the run goes ahead, and setu says what it did not follow.
@pytest.mark.parametrize("key,value,says", [
    (C.KEY_LL_CUSTOM_VEHICLES, [{"name": "custom"}], "Custom vehicles"),
    (C.KEY_LL_FOOTPATH_PRESSURE_MODE, "Custom", "footpath pressure"),
    (C.KEY_PL_SELF_WEIGHT_FACTOR, 1.2, "self-weight factor"),
    (C.KEY_WL_WIND_LL_ECC_MODE, "Custom", "eccentricity"),
    (C.KEY_MP_CB_TYPE + ".G2G3.B2M1", "K-Bracing", "G1-G2 bracing"),
    (C.KEY_MP_CB_BRACING_SECTION_DESIGNATION + ".G2G3.B2M1", "IS 110 x 110 x 10", "G1-G2 bracing"),
    (C.KEY_MP_ED_BRACING_TYPE + ".G1G2.E1M1", "K-Bracing", "End diaphragms"),
    (C.KEY_MP_ED_TYPE + ".G1G2.E1M1", "Rolled Beam", "Beam end diaphragms"),
    (C.KEY_MP_CB_SPACING + ".G2G3.B2M1", 5.0, "equal spacing"),
    (C.KEY_MATERIAL_CROSS_BRACING_E, 210, "girder steel"),
    (C.KEY_MP_GD_SEGMENT_TABLE + ".G1", [{"start": 0, "end": 10}, {"start": 10, "end": 35}], "segments"),
    (C.KEY_DO_GAMMA_C_BASIC, "1.40", "gamma_c_basic"),
    (C.KEY_DO_ULS_SHEAR, False, "Every limit-state check is run"),
])
def test_simplification_is_stated(key, value, says):
    b = prepared(supported_input())
    assert simplifications(b) == [BEARING_NOTE]
    b.input_dict[key] = value
    check_capability(b)
    assert any(says in note for note in simplifications(b))


# What setu cannot run at all still stops, naming the input.
@pytest.mark.parametrize("key,value", [(C.KEY_LL_IRC_CLASS_A, False), (C.KEY_SC_RIGHT_SUPPORT, "Fixed"), ("temperature_load_kN_m2", 1.5)])
def test_capability(key, value):
    b = prepared(supported_input())
    b.input_dict[key] = value
    with pytest.raises(ValueError, match="choose the ospgrillage analysis engine") as error:
        check_capability(b)
    assert key in str(error.value)


# The GUI defaults run on setu as they are: it is the default engine and they leave out what it refuses.
def test_literal_default_accepted():
    assert default_input()[C.KEY_ANALYSIS_ENGINE] == C.ENGINE_SETU
    check_capability(prepared(default_input()))


# ospgrillage keeps its defaults ticked.
def test_defaults_follow_engine():
    from osdagbridge_core.core.bridge_types.plate_girder.defaults import _update_loading_tab_defaults
    d = {C.KEY_ANALYSIS_ENGINE: C.ENGINE_OSPGRILLAGE}
    _update_loading_tab_defaults(d)
    assert all(d[key] for key in C.SETU_UNSUPPORTED_VEHICLES)


# The database's rounded shear modulus is not an independent one; a really different value still is.
def test_shear_modulus_rounding():
    b = prepared(supported_input())
    b.input_dict[C.KEY_MATERIAL_GIRDER_G] = 76.9
    assert simplifications(b) == [BEARING_NOTE]
    b.input_dict[C.KEY_MATERIAL_GIRDER_G] = 80.0
    assert any("Shear modulus" in note for note in simplifications(b))


# Without a footpath there is no railing strip, so its load must not reach setu.
def test_no_footpath_no_railing_load():
    d = supported_input()
    d[C.KEY_TS_NO_OF_FOOTPATHS] = 0
    assert added_dead_loads(d).railing_kn_per_m == 0


# A railing width changed after the layout solve moves the deck edge, not the girders.
def test_layout_follows_strips():
    d = supported_input()
    d[C.KEY_TS_NO_OF_FOOTPATHS], d[C.KEY_TS_FOOTPATH_WIDTH], d[C.KEY_RL_WIDTH] = 2, 1.5, 0.275
    spacing = d[C.KEY_TS_GIRDER_SPACING]
    reconcile_layout(d)
    assert d[C.KEY_TS_OVERALL_WIDTH] == cross_section(d).total_width_m()
    assert d[C.KEY_TS_GIRDER_SPACING] == spacing
    assert d[C.KEY_TS_DECK_OVERHANG] * 2 + spacing * (d[C.KEY_TS_NO_OF_GIRDERS] - 1) == pytest.approx(d[C.KEY_TS_OVERALL_WIDTH])


# The influence surfaces do not outlive the run, and the results that came from them do.
def test_surfaces_dropped(setu_bridge):
    engine = setu_bridge.grillage_model
    assert engine.design_values.surfaces == engine.design_values.place_surfaces == engine.brace_values.surfaces == {}
    assert engine.brace_values.members
    from setu.analysis.influence_surface import InfluenceSurface
    gc.collect()
    assert not [o for o in gc.get_objects() if isinstance(o, InfluenceSurface)]


# The report's wind and seismic forces, and the design bars for a dropdown load case, match ospgrillage's feature set.
def test_report_forces_and_case_bars(setu_bridge):
    for key in (C.KEY_WL_TRANSVERSE_WIND_FORCE, C.KEY_WL_LONGITUDINAL_WIND_FORCE, C.KEY_WL_VERTICAL_WIND_FORCE, C.KEY_SL_FORCE_LONGITUDINAL, C.KEY_SL_FORCE_TRANSVERSE):
        assert setu_bridge.output_dict[key] > 0
    assert setu_bridge.output_dict[C.KEY_SL_FORCE_TRANSVERSE] > setu_bridge.output_dict[C.KEY_SL_FORCE_LONGITUDINAL]
    for load_case in setu_bridge.get_available_loadcases():
        assert setu_bridge.get_dcr_for_selection("G1", load_case), load_case
    engine = setu_bridge.grillage_model
    uls = engine.case_demand("G1", "Envelope ULS")
    # The plotted load states are the IRC traffic's; Class SV governs this bridge's design moment and the run says so.
    traffic = max(abs(v.value) for state in ("ultimate, basic", "ultimate, seismic") for v in engine.design_values.girders[0]["maximum composite moment"][state].values())
    assert uls["lc_type"] == "ULS" and uls["Mu_kNm"] == pytest.approx(traffic, rel=1e-3)
    assert engine.per_girder_demands["G1"].Mu_kNm > uls["Mu_kNm"]
    assert any("Class SV governs" in note and "G1" in note for note in setu_bridge.output_dict["setu_simplifications"])


# Peak memory grows as the deck is widened.
def test_peak_estimate_grows_with_width():
    narrow, wide = supported_input(), supported_input()
    wide[C.KEY_CARRIAGEWAY_WIDTH], wide[C.KEY_TS_NO_OF_GIRDERS] = 46.0, 22
    assert estimated_peak_gb(wide)[1] > estimated_peak_gb(narrow)[1]


# Workers fit the free memory beside the analysis and a reserve, up to the cap; one (no workers) when two do not fit or memory is unknown.
@pytest.mark.parametrize("available,workers", [(None, 1), (20.0, 4), (5.0, 2), (4.0, 1), (0.5, 1)])
def test_workers_that_fit(available, workers):
    assert workers_that_fit(4, 2.0, 1.0, available) == workers


# The estimates give the measured peaks: the analysis alone, and whole runs with 2 or 4 workers.
def test_memory_estimate_matches_the_measured_decks():
    for nodes, measured_gb in ((3021, 0.33), (9911, 1.14), (18391, 3.84)):
        assert analysis_gb(nodes) == pytest.approx(measured_gb, abs=0.02)
    for nodes, workers, measured_gb in ((3021, 2, 0.53), (3021, 4, 0.81), (9911, 2, 1.04), (9911, 4, 1.56), (18391, 4, 2.75)):
        assert with_workers_gb(nodes) + workers * worker_gb(nodes) == pytest.approx(measured_gb, abs=0.02)


# Channel and single angle sections give the database area, doubled for double sections.
@pytest.mark.parametrize("section,designation,area_cm2", [
    ("Channel", "MC 100", 12.1), ("Double Channel", "MC 100", 24.2), ("Angle", "100 x 100 x 10", 19.1), ("Double Angle (Short Leg)", "100 x 100 x 10", 38.2)])
def test_bracing_section_area(section, designation, area_cm2):
    d = supported_input()
    d[C.KEY_MP_CB_BRACING_SECTION_TYPE + ".G1G2.B1M1"] = section
    d[C.KEY_MP_CB_BRACING_SECTION_DESIGNATION + ".G1G2.B1M1"] = designation
    assert bracing(d).area_m2 == pytest.approx(area_cm2 * 1e-4)


# The peak location is the node of the largest magnitude, first one on a tie.
def test_peak_location():
    import xarray as xr
    from osdagbridge_core.core.bridge_types.plate_girder.results_data import _peak_location
    values = np.array([[[1000.0, -5000.0], [5000.0, 2000.0]]])
    ds = xr.Dataset({"forces": (("Loadcase", "Element", "Component"), values)},
                    coords={"Loadcase": ["c"], "Element": [1, 2], "Component": ["Mz_i", "Mz_j"]})
    coords = {1: (0.0,), 2: (1.5,), 3: (3.0,)}
    assert _peak_location(ds, "c", [(1, 1, 2), (2, 2, 3)], "Mz", coords) == 1.5
    assert _peak_location(ds, "c", [(9, 1, 2)], "Mz", coords) is None


# What IRC:6 as amended no longer has stops the run with its clause: Class AA (Amendment 10) and braking with Class SV (204.5.4).
@pytest.mark.parametrize("key,says", [(C.KEY_LL_IRC_AA_WHEELED, "Amendment 10"), (C.KEY_LL_IRC_AA_TRACKED, "Amendment 10"),
                                      (C.KEY_BL_IRC_CLASS_SV, "204.5.4")])
def test_not_in_irc(key, says):
    b = prepared(supported_input())
    b.input_dict[key] = True
    with pytest.raises(ValueError, match=says):
        check_capability(b)


# Class SV is required on new bridges except over rural, district and village roads (cl. 201.1 as amended).
def test_class_sv_follows_the_road_category():
    b = prepared(supported_input())
    b.input_dict[C.KEY_LL_IRC_CLASS_SV] = False
    with pytest.raises(ValueError, match="201.1"):
        check_capability(b)
    b.input_dict[C.KEY_LL_ROAD_CATEGORY] = C.ROAD_SV_OPTIONAL
    check_capability(b)


# The construction loads have no IRC value, so leaving them blank stops the run; Construction Stages No leaves the stage out.
def test_construction_loads_are_entered():
    from osdagbridge_core.core.solvers.setu.bridge_input import construction_loads
    d = supported_input()
    assert vars(construction_loads(d)) == {"shuttering_kpa": 0.5, "construction_live_load_kpa": 1.5}
    d[C.KEY_DS_SHUTTERING_LOAD] = ""
    with pytest.raises(ValueError, match="cl. 218"):
        construction_loads(d)
    d[C.KEY_DS_CONSTRUCTION_STAGE] = "No"
    assert construction_loads(d) is None


# Seismic during construction: importance factor 1 (IRC:SP:114 cl. 4.5 note), the bridge's own coefficients scaled back from its I.
def test_construction_seismic_site():
    from osdagbridge_core.core.solvers.setu.bridge_input import construction_seismic_site
    d = supported_input()
    d[C.KEY_SL_IMPORTANCE_FACTOR] = 1.2
    site, bridge = construction_seismic_site(d), seismic_site(d)
    assert site.importance_factor == 1.0
    assert site.horizontal_coefficient == pytest.approx(bridge.horizontal_coefficient / 1.2)


# The mesh choice sets the element lengths and the memory estimate; an unknown name is refused.
def test_mesh_preset():
    coarse, fine = supported_input(), supported_input()
    coarse[C.KEY_ANALYSIS_MESH], fine[C.KEY_ANALYSIS_MESH] = "Coarse", "Fine"
    assert estimated_peak_gb(coarse)[0] < estimated_peak_gb(supported_input())[0] < estimated_peak_gb(fine)[0]
    assert vars(prepared_bridge_input(fine).mesh) == {"panels_between_braces": None, "target_size_across_width_m": 0.10,
        "element_length_m": 0.25, "support_element_length_m": 0.05, "support_zone_m": 1.5}
    assert prepared_bridge_input(coarse).mesh.element_length_m == 1.0
    old = supported_input()
    old[C.KEY_ANALYSIS_MESH] = "Fast"
    with pytest.raises(ValueError, match="mesh must be one of Fine, Standard, Coarse"):
        estimated_peak_gb(old)


# A required vehicle left unticked says to tick it.
def test_required_vehicle_message():
    b = prepared(supported_input())
    b.input_dict[C.KEY_LL_IRC_70R_TRACKED] = False
    with pytest.raises(ValueError, match="70r_tracked is unticked: setu always searches this IRC vehicle, so tick it"):
        check_capability(b)


# The Additional Inputs dropdown offers both engines under the key the backend branches on.
def test_engine_field_in_additional_inputs():
    from osdagbridge_core.core.bridge_types.plate_girder.ui_fields_additional_input import DESIGN_OPTIONS_SCHEMA
    fields = [f for section in DESIGN_OPTIONS_SCHEMA["sections"] for row in section.get("rows", []) for f in row["fields"]]
    field = next(f for f in fields if f["id"] == C.KEY_ANALYSIS_ENGINE)
    assert field["choices"] == [C.ENGINE_SETU, C.ENGINE_OSPGRILLAGE]


# The dialog's saved form of a bracing pair matches the untouched pairs, so it is not refused.
def test_gui_bracing_values_accepted():
    d = supported_input()
    suffix = ".G1G2.B1M1"
    d[C.KEY_MP_CB_BRACING_SECTION_DESIGNATION + suffix] = "∠ 100 ⅹ 100ⅹ 10"
    d[C.KEY_MP_CB_TOP_CHORD + suffix] = True
    d[C.KEY_MP_CB_BOTTOM_CHORD + suffix] = False
    normalise_gui_bracing(d)
    assert d[C.KEY_MP_CB_BRACING_SECTION_DESIGNATION + suffix] == "IS 100 x 100 x 10"
    assert d[C.KEY_MP_CB_TOP_CHORD + suffix] == "Yes"
    assert d[C.KEY_MP_CB_BOTTOM_CHORD + suffix] == "No"


# Required material inputs cannot silently become a default value.
def test_missing_input_raises():
    b = prepared(supported_input())
    del b.input_dict[C.KEY_MATERIAL_DECK_ECM]
    with pytest.raises(KeyError, match=C.KEY_MATERIAL_DECK_ECM):
        bridge_input(b)


# Class SV values of one girder response in these limit states (cl. 204.5: 1.15 ULS basic, 1.0 rare, 0.75 frequent).
def sv_values(engine, g, response, states):
    sv = engine.special_vehicle
    return [] if sv is None else [v for state in states for v in sv.girders[g][response].get(state, {}).values()]


# Class SV values of one girder response that came from this combination.
def sv_combination(engine, g, response, name):
    return [v for by in engine.special_vehicle.girders[g][response].values() for v in by.values() if v.combination == name]


# Independently state every mapped field so no default or unit error can hide.
def expected_demand(engine, g, name=None):
    values, config = engine.design_values, engine.configs[g]
    if name is None:
        states = ["ultimate, basic"] + (["ultimate, seismic"] if engine.seismic is not None else [])
        moments = [v for state in states for v in values.girders[g]["maximum composite moment"][state].values()] + sv_values(engine, g, "maximum composite moment", states)
        shears = [v for state in states for v in values.girders[g]["support shear"][state].values()] + sv_values(engine, g, "support shear", states)
        rare = ["serviceability, rare"]
        rare_m = max([*values.girders[g]["maximum composite moment"]["serviceability, rare"].values(), *sv_values(engine, g, "maximum composite moment", rare)], key=lambda v: abs(v.value))
        rare_v = max([*values.girders[g]["support shear"]["serviceability, rare"].values(), *sv_values(engine, g, "support shear", rare)], key=lambda v: abs(v.value))
        kind = ""
    else:
        native = name in values.combinations[g]["maximum composite moment"]
        moments = list(values.combinations[g]["maximum composite moment"][name].values()) if native else sv_combination(engine, g, "maximum composite moment", name)
        shears = list(values.combinations[g]["support shear"][name].values()) if native else sv_combination(engine, g, "support shear", name)
        kind = "ULS" if name.startswith("ultimate,") else "SLS_frequent" if name.startswith("serviceability, frequent") else "SLS"
        rare_m, rare_v = max(moments, key=lambda v: abs(v.value)), max(shears, key=lambda v: abs(v.value))
    m, v = max(moments, key=lambda v: abs(v.value)), max(shears, key=lambda v: abs(v.value))
    expected = dict.fromkeys([f.name for f in fields(DemandEnvelope)], 0.0)
    expected.update(Mu_kNm=abs(m.value), Vu_kN=abs(v.value), Nsc=config.fatigue.Nsc,
                    governing_combination=m.combination, location=f"{m.at_m} m", member=f"G{g + 1}", source="setu", lc_type=kind)
    expected["M_sls_stages_kNm"] = None
    if kind != "ULS":
        expected.update(M_sls_kNm=abs(rare_m.value), V_sls_kN=abs(rare_v.value), M_sls_stages_kNm=sls_stages(values.dead, rare_m, g))
    if kind in ("", "SLS_frequent"):
        expected.update(stress_range_MPa=values.fatigue_stresses_mpa[g]["steel bottom"].range,
                        shear_range_MPa=values.fatigue[g]["support shear"].range * 1000 / config.section.Aw)
    if name is None:
        sw = values.dead.stages["steel self weight"][g]
        construction = engine.construction.girders[g]["maximum composite moment"]
        cache = engine.deflections_cache[f"G{g + 1}"]
        # Table B.2: girder self weight at 1.35 (stage 1); the governing ULS construction combination (stage 2)
        expected.update(M_girder_sw_kNm=1.35 * float(np.abs(sw.composite_moment_kn_m).max()),
                        M_construction_kNm=max(abs(v.value) for state in ("ultimate, construction", "ultimate, seismic during construction")
                                               for v in construction.get(state, {}).values()),
                        Vr_kN=values.live_shear_range_kn[g], delta_live_mm=cache["live_mm"],
                        delta_dl_mm=cache["dl_mm"], delta_total_mm=cache["total_mm"], camber_mm=cache["camber_mm"])
    return expected


# Exact comparisons cover every field, girder, native combination and check scope.
def test_demand_mapping(setu_bridge):
    engine = setu_bridge.grillage_model
    for g in engine.design_values.girders:
        label = f"G{g + 1}"
        assert asdict(engine.per_girder_demands[label]) == expected_demand(engine, g)
        sv_names = {v.combination for by in engine.special_vehicle.girders[g]["maximum composite moment"].values() for v in by.values()}
        assert set(engine.per_girder_per_lc[label]) == set(engine.design_values.combinations[g]["maximum composite moment"]) | sv_names
        assert any("special vehicle" in name for name in engine.per_girder_per_lc[label])
        for name, demand in engine.per_girder_per_lc[label].items():
            assert asdict(demand) == expected_demand(engine, g, name)
        assert setu_bridge.design_results["per_girder"][label]["demand"]["Mu_kNm"] == engine.per_girder_demands[label].Mu_kNm


# Direct metres-to-mm conversion precedes the existing camber function.
def test_deflections_cache(setu_bridge):
    engine = setu_bridge.grillage_model
    raw = raw_deflections(engine.design_values)
    for g, d in engine.design_values.deflections.items():
        assert raw[f"G{g + 1}"] == {
            # Annex B cl. 7.2 as amended: deflection limits use Table B.3 column 3, live load leading at 0.75
            C.KEY_SD_DEFL_LIVE_RAW: 0.75 * d["live_m"] * 1000,
            C.KEY_SD_DEFL_DL_RAW: sum(d["dead_m"].values()) * 1000,
            C.KEY_SD_DEFL_TOTAL_RAW: (sum(d["dead_m"].values()) + 0.75 * d["live_m"]) * 1000,
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
    # IRC:6 Table 1 note as amended: wet concrete at its unit weight + 1 kN/m3
    wet_moment = bridge.width_m() * bridge.deck.thickness_m * (bridge.concrete.unit_weight_kn_m3 + 1.0) * length ** 2 / 8
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


# The Transverse Member Design dialog's per-load-case table: each governing state's brace forces, which are the replay's own solves.
# Per pair and load case, the table's largest and smallest diagonal and chord forces are those of the members at the intermediate stations.
def test_per_load_case_brace_forces(setu_bridge):
    from osdagbridge_core.core.solvers.setu.bracing_forces import CHORDS, DIAGONALS, intermediate_stations, pair_name
    from osdagbridge_core.core.solvers.setu.plot_dataset import governing_states
    engine = setu_bridge.grillage_model
    states = governing_states(engine.diagrams)
    table = setu_bridge.cross_bracing_forces().compute_panel_forces()
    assert set(table["LoadCase"]) == set(states)
    assert set(states) <= set(engine.get_result_data()["loadcases"])
    members = engine.diagrams.brace_members
    stations = intermediate_stations(engine)
    for name, (_, braces) in states.items():
        assert len(braces) == len(members)
        for pair in range(engine.bridge_input.girders.count - 1):
            rows = table[(table["LoadCase"] == name) & (table["Girder Pair"] == pair_name(pair))]
            for roles, column in ((DIAGONALS, "F_diag (kN)"), (CHORDS, "F_chord (kN)")):
                forces = [braces[k] for k, (p, station, role) in enumerate(members) if p == pair and station in stations and role in roles]
                assert rows[column].max() == max(forces) and rows[column].min() == min(forces)


# Grillage-only records and attributes are refused by name; the results dataset and the tables built from it are setu's.
def test_engine_boundaries(setu_bridge):
    engine = setu_bridge.grillage_model
    data = engine.get_result_data()
    assert data["girders"] == {f"G{g + 1}": {} for g in range(4)}
    assert data["members"] and data["forces"].get("1.0 DL") and data["forces"].get("no such case", {}) == {}
    assert data["loadcases"] == [str(name) for name in engine._deduplicated_results.Loadcase.values]
    for key in ("crossbracings", "nodes", "displacements"):
        with pytest.raises(NotImplementedError, match="not provided by the setu engine"):
            data[key]
        with pytest.raises(NotImplementedError, match="not provided by the setu engine"):
            data.get(key)
    for name in ("dataset", "reactions", "live_load_report_values", "vehicle_load_cases_list", "self_weight_load_case"):
        with pytest.raises(NotImplementedError, match="not provided by the setu engine"):
            getattr(engine, name)
        assert not hasattr(engine, name)
    assert engine.model.get_element(member="start_edge", options="elements") == []
    assert setu_bridge.get_results_dataset() is engine._deduplicated_results
    assert set(setu_bridge._load_effects_cache) == {"G1", "G2", "G3", "G4"}
    assert setu_bridge._lc_summary and setu_bridge._reaction_summary


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


# Every governing ULS axial force of these roles, written out independently of the adapter: IRC traffic, Class SV, construction stage.
def expected_forces(engine, pair, stations, roles):
    states = ["ultimate, basic"] + (["ultimate, seismic"] if engine.seismic is not None else [])
    sources = [(engine.brace_values.members, states), (engine.special_vehicle.braces, ["ultimate, basic"]),
               (engine.construction.braces, ["ultimate, construction", "ultimate, seismic during construction"])]
    return [value for members, wanted in sources for (k, station, role), by_state in members.items()
            if k == pair and station in stations and role in roles for state in wanted for value in by_state.get(state, {}).values()]


# The pair summary OsdagBridge expects, built from the expected forces with its own 5 N cut-off and rounding.
def expected_summary(engine, pair, stations):
    from osdagbridge_core.core.solvers.setu.bracing_forces import CHORDS, DIAGONALS
    summary = {}
    for prefix, roles in (("diag", DIAGONALS), ("chord", CHORDS)):
        forces = expected_forces(engine, pair, stations, roles)
        assert forces, f"pair {pair} has no {prefix} members at stations {list(stations)}"
        tension = max(forces, key=lambda force: force.value)
        compression = min(forces, key=lambda force: force.value)
        summary[f"{prefix}_tension_kN"] = round(tension.value, 3) if tension.value > 0.005 else None
        summary[f"{prefix}_tension_gov_lc"] = tension.combination if tension.value > 0.005 else None
        summary[f"{prefix}_compression_kN"] = round(-compression.value, 3) if compression.value < -0.005 else None
        summary[f"{prefix}_compression_gov_lc"] = compression.combination if compression.value < -0.005 else None
    return summary


# Intermediate cross-bracing reads only the stations between the supports, with setu's member forces as they are.
def test_cross_bracing_forces(setu_bridge):
    from osdagbridge_core.core.solvers.setu.bracing_forces import SetuCrossBracingForces
    engine = setu_bridge.grillage_model
    last = engine.bridge_input.bracing.station_count - 1
    forces = SetuCrossBracingForces(setu_bridge).get_design_forces_dict()
    assert list(forces["pairs"]) == ["G1-G2", "G2-G3", "G3-G4"]
    assert set(forces) == {"brace_type", "top_chord", "bottom_chord", "geometry", "pairs"}
    for pair, summary in enumerate(forces["pairs"].values()):
        assert summary == expected_summary(engine, pair, range(1, last))
        member_forces = {round(abs(force.value), 3) for force in expected_forces(engine, pair, range(1, last), ("diagonal_down", "diagonal_up"))}
        assert summary["diag_tension_kN"] in member_forces and summary["diag_compression_kN"] in member_forces
    assert SetuCrossBracingForces(setu_bridge).get_crossbracing_count() == (last - 1) * 3


# End diaphragms read only the two support lines, use the engine's one analysis, and leave the engine as they found it.
def test_end_diaphragm_forces(setu_bridge):
    from osdagbridge_core.core.solvers.setu.bracing_forces import SetuEndDiaphragmForces
    engine = setu_bridge.grillage_model
    last = engine.bridge_input.bracing.station_count - 1
    end = SetuEndDiaphragmForces(setu_bridge, pair_to_elements={})
    run = engine.brace_values
    for pair in range(3):
        assert end.get_pair_forces(f"G{pair + 1}-G{pair + 2}", cos_alpha=0.5) == expected_summary(engine, pair, (0, last))
    assert any(expected_summary(engine, pair, (0, last)) != expected_summary(engine, pair, range(1, last)) for pair in range(3)), \
        "end and intermediate stations carry identical forces, so this test could not catch them being mixed"
    assert engine.brace_values is run


# The live-load report table is filled from setu's IRC vehicles and impact factors.
def test_live_load_report_values(setu_bridge):
    b = setu_bridge
    span_m = float(b.input_dict[C.KEY_SPAN])
    assert b.output_dict[C.KEY_LL_IMPACT_FACTOR_CLASS_A] == round(1 + 9 / (13.5 + span_m), 3)
    assert b.output_dict[C.KEY_LL_IMPACT_FACTOR_CLASS_AA_70R] == b.output_dict[C.KEY_LL_IMPACT_FACTOR_70R_TRACKED] == 1.1
    # cl. 208.3 with Amendment 10: 70R wheeled 25 % up to 23 m, then the Class A curve
    assert b.output_dict[C.KEY_LL_IMPACT_FACTOR_70R_WHEELED] == round(1 + 9 / (13.5 + span_m), 3)
    assert b.output_dict[C.KEY_LL_VEHICLE_TOTAL_LOAD_CLASS_SV] == 3850.0
    assert b.output_dict[C.KEY_ANALYSIS_MESH] == "Standard"
    # IRC:6-2017 Amendment 10 (and Amendment 8 for the fatigue truck) print these in kN
    assert b.output_dict[C.KEY_LL_VEHICLE_TOTAL_LOAD_CLASS_A] == 554.0
    assert b.output_dict[C.KEY_LL_VEHICLE_TOTAL_LOAD_70R_WHEELED] == 1000.0
    assert b.output_dict[C.KEY_LL_VEHICLE_TOTAL_LOAD_70R_TRACKED] == 700.0
    assert b.output_dict[C.KEY_LL_VEHICLE_TOTAL_LOAD_FATIGUE] == 400.0


@pytest.fixture
# The setu engine's skeleton back in the live OpenSees domain, which any other bridge built since has replaced.
def setu_geometry(setu_bridge):
    from osdagbridge_core.core.solvers.setu.plot_dataset import build_skeleton, node_lines
    build_skeleton(node_lines(setu_bridge.grillage_model.diagrams, setu_bridge.get_edge_dist()))
    return setu_bridge


# OsdagBridge's girder name for setu girder k: the deck edges are lines EB1 and EB2 when the deck overhangs.
def line_name(bridge, girder):
    return f"G{girder + 2}" if bridge.get_edge_dist() > 0 else f"G{girder + 1}"


# The plot dataset has ospgrillage's layout and load-case groups, and OsdagBridge finds setu's girders in it by itself.
def test_plot_dataset_contract(setu_geometry, osp_bridge):
    from osdagbridge_core.core.bridge_types.plate_girder.analysis_results import PlateGirderAnalysisResults
    b = setu_geometry
    ds, osp = b.get_results_dataset(), osp_bridge.get_results_dataset()
    assert ds["forces"].dims == osp["forces"].dims and ds["displacements"].dims == osp["displacements"].dims
    assert set(map(str, ds.Component.values)) <= set(map(str, osp.Component.values))
    groups = b.get_result_handler().classify_loadcases()
    assert groups["sw"] == ["SW"] and groups["dl_only"] == ["1.0 DL"]
    assert groups["envelope_uls"] == ["Envelope ULS"] and groups["envelope_sls"] == ["Envelope SLS"]
    assert "1.0 LL" in groups["vehicle_static"] and groups["uls_basic"] and groups["uls_seismic"] and groups["sls_rare"]
    girders, _ = PlateGirderAnalysisResults(ds, b.grillage_model, edge_dist=b.get_edge_dist()).build_girders(verbose=False)
    span_m = float(b.input_dict[C.KEY_SPAN])
    assert list(girders) == ["EB1", "G2", "G3", "G4", "G5", "EB2"]
    assert all(g["length"] == pytest.approx(span_m) for g in girders.values())


# OsdagBridge's own readers give back setu's numbers: dead stage moments, live deflection, and each design moment where it governs.
def test_plotted_values_are_setus(setu_geometry):
    from setu.utils.constants import BASIC, BIGGER_IS_WORSE, MAX_MOMENT
    b = setu_geometry
    engine = b.grillage_model
    handler = b.get_result_handler()
    governing_girder, _ = engine.design_values.governing(MAX_MOMENT, BASIC, BIGGER_IS_WORSE)
    for girder in range(4):
        name = line_name(b, girder)
        steel = engine.design_values.dead.stages["steel self weight"][girder]
        plotted = handler._get_forces_df("SW", name, "Mz_i").iloc[:, 1].to_numpy()
        assert plotted == pytest.approx(steel.composite_moment_kn_m[:-1], abs=1e-3)
        live_mm = handler._get_displacements_df("1.0 LL", name, "dy")["dy"].to_numpy()
        assert live_mm == pytest.approx(1000 * engine.diagrams.live[governing_girder, MAX_MOMENT, BIGGER_IS_WORSE][girder].deflection_m, abs=1e-6)
        design = engine.design_values.girders[girder][MAX_MOMENT][BASIC][BIGGER_IS_WORSE]
        envelope = handler._get_forces_df("Envelope ULS", name, "Mz_i").iloc[:, 1].to_numpy()
        station = list(steel.stations_m).index(design.at_m)
        # A state governing another section can beat this section's search slightly: the search samples positions.
        assert design.value * (1 - 1e-6) <= envelope[station] <= design.value * (1 + 5e-4)
    left, right = b._reaction_summary["SW"]["left_kN"], b._reaction_summary["SW"]["right_kN"]
    assert left < 0 < right, "reactions keep ospgrillage's signs: left as Vy_i, right as -Vy_j"


# Every OsdagBridge figure, the graph engine and the report's figures draw from the setu dataset.
def test_figures_render(setu_geometry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from osdagbridge_core.core.bridge_types.plate_girder.plot_generator import capture_report_figures
    b = setu_geometry
    ds = b.get_results_dataset()
    envelope = ds.sel(Loadcase="Envelope ULS")
    for build, key in ((b.build_figure_bmd, "Mz"), (b.build_figure_sfd, "Fy"), (b.build_figure_deflection, "Dy")):
        build(envelope, key)
    b.build_figure_grillage()
    nodes, members = b.get_nodes_members()
    assert set(capture_report_figures(ds, nodes, members, edge_dist=b.get_edge_dist())) == {"bm_envelope", "sf_envelope", "defl_ll"}
    figure = plt.figure()
    engine = b.build_graph_engine(figure, *[figure.add_subplot(4, 1, k + 1) for k in range(4)])
    for key in engine.get_girder_keys():
        assert engine.extract_member_results(key, "1.0 LL") is not None
    plt.close("all")


# A congested or crowd placement takes no braking (IRC:6 cl. 204.4), so setu gives no braking diagram where one governs:
# the load states and the plot dataset are built without a braking state.
def test_load_states_without_braking(setu_bridge):
    import copy
    from types import SimpleNamespace
    from osdagbridge_core.core.solvers.setu.plot_dataset import BRAKING_CASE, LIVE_LOAD_CASE, load_states, plot_dataset
    engine = setu_bridge.grillage_model
    diagrams = copy.copy(engine.diagrams)
    diagrams.braking = {}
    values = SimpleNamespace(girders=engine.design_values, diagrams=diagrams)
    states = load_states(values)
    assert BRAKING_CASE not in states and LIVE_LOAD_CASE in states
    assert BRAKING_CASE not in set(map(str, plot_dataset(values, setu_bridge.get_edge_dist()).Loadcase.values))


# setu runs apply IS 800:2007 as written: the high-shear interaction takes the governing shear strength of cl. 8.4 (here the web's
# buckling strength), LTB takes βb of cl. 8.2.2, and the SLS moment acts stage by stage (steel, long-term, short-term composite).
def test_setu_design_corrections(setu_bridge):
    from osdagbridge_core.core.bridge_types.plate_girder.designer import IRC22CapacityCalculator
    engine = setu_bridge.grillage_model
    for g, name in enumerate(engine.per_girder_demands):
        demand, config = engine.per_girder_demands[name], engine.configs[g]
        assert config.setu
        stages = demand.M_sls_stages_kNm
        assert stages["steel"] > 0 and stages["long_term"] > 0 and stages["short_term"] > 0
        assert sum(stages.values()) == pytest.approx(demand.M_sls_kNm)
        calc = IRC22CapacityCalculator(config)
        cap = calc.compute_all(Vu_kN=demand.Vu_kN, M_sls_kNm=demand.M_sls_kNm, V_sls_kN=demand.V_sls_kN, M_sls_stages_kNm=stages)
        assert cap.details["combined_bending_shear"]["Vd_used_kN"] == pytest.approx(min(cap.Vd_kN, cap.Vcr_kN), abs=1e-3)
        ltb, sec, mat = cap.details["buckling_resistance"], config.section, config.material
        assert ltb["Mb_kNm"] == pytest.approx(ltb["chi_LT"] * ltb["beta_b"] * sec.Zp_steel * mat.fy / mat.gamma_m0 / 1e6, rel=1e-3)
        comp = calc.compute_composite_section_props(cap.beff_mm)
        bottom = stages["steel"] * 1e6 * sec.y_cg_from_bot / sec.Iz_steel + sum(
            stages[part] * 1e6 * comp[part][C.KEY_COMP_Y_FROM_BOT] / comp[part][C.KEY_COMP_I] for part in ("long_term", "short_term"))
        assert cap.details["sls_actual_stresses"]["fbt_MPa"] == pytest.approx(bottom, abs=1e-2)
        # the bare steel carries its share on a far smaller section than the composite one
        whole = calc.compute_all(Vu_kN=demand.Vu_kN, M_sls_kNm=demand.M_sls_kNm, V_sls_kN=demand.V_sls_kN)
        assert cap.details["sls_actual_stresses"]["fbt_MPa"] > whole.details["sls_actual_stresses"]["fbt_MPa"]
    sls_case = next(case for case in engine.states if case.startswith("SLS_RARE_"))
    case = engine.case_demand("G1", sls_case)
    assert sum(case["M_sls_stages_kNm"].values()) == pytest.approx(case["M_sls_kNm"])


# The moment diagram's summary keeps the sign of an end moment at the far support (setu stores -M at an element's j end), and
# the design report counts the main girders, not the deck-edge lines.
def test_plot_summary_and_girder_count(setu_bridge):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from osdagbridge_core.core.bridge_types.plate_girder.plot_generator import build_figure_bmd
    b = setu_bridge
    states, ds = b.grillage_model.states, b.get_results_dataset()
    nodes, members = b.get_nodes_members()
    fig, summary = build_figure_bmd(ds.sel(Loadcase="Envelope ULS"), "Mz", nodes, members, edge_dist=b.get_edge_dist())
    plt.close(fig)
    for g in range(b._girder_count()):
        members_ = [states[name][g].composite_moment_kn_m for name in states if name.startswith(("BASIC_", "SEISMIC_"))]
        high, low = np.max(members_, axis=0), np.min(members_, axis=0)
        envelope = np.where(abs(high) >= abs(low), high, low)
        assert summary[f"G{g + 1}"]["max"] == pytest.approx(envelope.max(), abs=0.01)
        assert summary[f"G{g + 1}"]["min"] == pytest.approx(envelope.min(), abs=0.01)
    assert b.design_results["n_girders"] == b._girder_count()
