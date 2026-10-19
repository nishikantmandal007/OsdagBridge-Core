import copy
from osdagbridge_core.core.bridge_types.plate_girder import defaults
from osdagbridge_core.core.utils import common as C


# the default GUI inputs plus the fields the GUI dialogs fill in
def default_input():
    d = copy.deepcopy(defaults.BASIC_INPUT_DICT)
    d[C.KEY_SPAN], d[C.KEY_CARRIAGEWAY_WIDTH], d[C.KEY_SKEW_ANGLE] = 35.0, 7.5, 0.0
    d[C.KEY_PROJECT_LOCATION] = {"weather_data": {"latitude": 19.07, "longitude": 72.88, "zone": "III", "z_value": 0.16,
                                                  "wind_speed": 44.0, "max_temp": 40.0, "min_temp": 15.0}}
    for member in ("GIRDER", "CROSS_BRACING", "END_DIAPHRAGM"):
        for key, value in (("FY", 350), ("FU", 490), ("E", 200), ("G", round(200 / 2.6, 3)), ("POISSON", 0.3), ("THERMAL", 1.2e-5), ("DENSITY", 78.5)):
            d.setdefault(getattr(C, f"KEY_MATERIAL_{member}_{key}"), value)
    for key, value in (("FCK", 40.0), ("FCTM", 3.0), ("ECM", 33.0), ("THERMAL", 1.2e-5), ("DENSITY", 25.0)):
        d.setdefault(getattr(C, f"KEY_MATERIAL_DECK_{key}"), value)
    defaults.solve_extend_basic_input_dict(d)
    return d


# Explicitly deselect the two vehicle families setu cannot analyse.
def supported_input():
    d = default_input()
    d[C.KEY_LL_IRC_CLASS_SV] = False
    d[C.KEY_BL_IRC_CLASS_SV] = False
    d[C.KEY_LL_IRC_70R_BOGIE] = False
    return d
