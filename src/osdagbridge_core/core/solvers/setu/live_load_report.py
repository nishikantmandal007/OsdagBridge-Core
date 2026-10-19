from setu.irc6.impact import impact_factor
from setu.irc6.irc_constants import FATIGUE_TRUCK_AXLE_LOADS_T
from setu.irc6.vehicles import CLASS_70R_TRACKED, CLASS_70R_WHEELED, CLASS_A
from setu.utils.constants import GRAVITY_KN_PER_TONNE
from osdagbridge_core.core.utils import common as C

IMPACT_DECIMALS = 3
LOAD_DECIMALS = 2


# A vehicle's weight in kN, rounded as the live-load report prints it.
def total_load_kn(load_t):
    return round(load_t * GRAVITY_KN_PER_TONNE, LOAD_DECIMALS)


# The report's live-load table from setu's IRC vehicles and impact factors; its one 70R impact key holds the tracked value, as OsdagBridge's does.
def live_load_report_values(span_m):
    return {
        C.KEY_LL_IMPACT_FACTOR_CLASS_A: round(impact_factor(CLASS_A.name, span_m), IMPACT_DECIMALS),
        C.KEY_LL_IMPACT_FACTOR_CLASS_AA_70R: round(impact_factor(CLASS_70R_TRACKED.name, span_m), IMPACT_DECIMALS),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_CLASS_A: total_load_kn(CLASS_A.total_load_t()),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_70R_WHEELED: total_load_kn(CLASS_70R_WHEELED.total_load_t()),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_70R_TRACKED: total_load_kn(CLASS_70R_TRACKED.total_load_t()),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_FATIGUE: total_load_kn(sum(FATIGUE_TRUCK_AXLE_LOADS_T)),
    }
