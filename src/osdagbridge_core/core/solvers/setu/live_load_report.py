from setu.irc6.impact import impact_factor
from setu.irc6.irc_constants import FATIGUE_TRUCK_AXLE_LOADS_KN, SPECIAL_VEHICLE_AXLE_LOADS_KN
from setu.irc6.vehicles import CLASS_70R_TRACKED, CLASS_70R_WHEELED, CLASS_A
from osdagbridge_core.core.utils import common as C

IMPACT_DECIMALS = 3
LOAD_DECIMALS = 2


# A vehicle's weight in kN (setu holds IRC loads in kN as Amendment 10 prints them), rounded as the live-load report prints it.
def total_load_kn(load_kn):
    return round(load_kn, LOAD_DECIMALS)


# The report's live-load table from setu's IRC vehicles and impact factors (cl. 208): Class A, 70R wheeled and 70R tracked each get
# their own impact factor; the shared 70R key keeps OsdagBridge's tracked value. Class SV takes no impact (cl. 204.5.4).
def live_load_report_values(span_m):
    return {
        C.KEY_LL_IMPACT_FACTOR_CLASS_A: round(impact_factor(CLASS_A.name, span_m), IMPACT_DECIMALS),
        C.KEY_LL_IMPACT_FACTOR_CLASS_AA_70R: round(impact_factor(CLASS_70R_TRACKED.name, span_m), IMPACT_DECIMALS),
        C.KEY_LL_IMPACT_FACTOR_70R_WHEELED: round(impact_factor(CLASS_70R_WHEELED.name, span_m), IMPACT_DECIMALS),
        C.KEY_LL_IMPACT_FACTOR_70R_TRACKED: round(impact_factor(CLASS_70R_TRACKED.name, span_m), IMPACT_DECIMALS),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_CLASS_A: total_load_kn(CLASS_A.total_load_kn()),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_70R_WHEELED: total_load_kn(CLASS_70R_WHEELED.total_load_kn()),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_70R_TRACKED: total_load_kn(CLASS_70R_TRACKED.total_load_kn()),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_FATIGUE: total_load_kn(sum(FATIGUE_TRUCK_AXLE_LOADS_KN)),
        C.KEY_LL_VEHICLE_TOTAL_LOAD_CLASS_SV: total_load_kn(sum(SPECIAL_VEHICLE_AXLE_LOADS_KN)),
    }
