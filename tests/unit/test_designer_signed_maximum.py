from types import SimpleNamespace

from osdagbridge_core.core.bridge_types.plate_girder import designer


# A large negative force must beat a smaller negative force in both designer loops.
def test_designer_negative_maximum(monkeypatch):
    monkeypatch.setattr(designer, "composite_stiffness_props", lambda config: ({}, 1.0))
    config = SimpleNamespace(section=SimpleNamespace(Ze_steel=1e6, Aw=1000), fatigue=SimpleNamespace(Nsc=2000000))
    names = ["Envelope ULS", "Envelope SLS", "SW", "1.0 DL", "1.0 DL + 1.0 LL", "Case1"]
    row = {f"{c}_{end}": -300000.0 for c in ("Vx", "Vy", "Vz", "Mx", "My", "Mz") for end in ("i", "j")}
    smaller = {key: -20000.0 for key in row}
    data = {"girders": {"G1": {"members": ["1", "2"], "nodes": [1, 2, 3]}}, "loadcases": names,
            "forces": {name: {"1": row, "2": smaller} for name in names},
            "displacements": {name: {str(n): {"x": 0.0, "y": 0.0, "z": 0.0} for n in (1, 2, 3)} for name in names}}
    cache = {"G1": {"live_mm": 0, "dl_mm": 0, "total_mm": 0, "camber_mm": 0, "per_lc": {name: 0 for name in names}}}
    aggregate, per_case = designer._extract_demands_from_result_data(config, cache, data)
    assert (aggregate["G1"].Mu_kNm, aggregate["G1"].Vu_kN, aggregate["G1"].Nu_kN, aggregate["G1"].M_sls_kNm) == (300, 300, 300, 300)
    assert per_case["G1"]["Case1"].Mu_kNm == 300
    assert per_case["G1"]["Case1"].Vu_kN == 300
