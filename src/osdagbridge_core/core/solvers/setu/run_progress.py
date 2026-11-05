import json
from time import perf_counter

from osdagbridge_core.core.utils.logger import bridge_logger

# The setu analysis fills the progress bar from FIRST_PCT to LAST_PCT (Stage 4G).
FIRST_PCT, LAST_PCT = 16, 66

# Each setu stage: key, title shown to the user, what its counts are, share of a run's time, and the phase texts setu announces it with.
# The shares are measured (2 x 11 m deck with 2 workers, 45 m x 52.85 m deck with 4 workers); they only spread the time before a
# stage has run, and every stage's own measured time replaces its share as soon as there is one.
STAGES = (
    ("build", "Model and dead load stages", None, 1.4, ("building the model and solving the dead load stages",)),
    ("factor", "Stiffness matrix factorised", None, 1.4, ("factorising the stiffness matrix",)),
    ("girders", "Girders", "places", 8.4, ("solving and searching the girders", "searching the girders")),
    ("braces", "Cross bracing members", "members", 77.8, ("solving and searching the braces", "searching the braces")),
    ("special_vehicle", "Class SV", "places", 4.1, ("solving and searching the Class SV",)),
    ("replay", "Governing load states", "load cases", 6.7, ("replaying the governing load states",)),
    ("construction", "Construction stage", None, 0.7, ("the construction stage on the bare steel",)),
)
COUNTED_STAGES = ("girders", "braces")  # a time remaining is only given once one of these has a measured pace
LEAST_FRACTION_FOR_PACE = 0.02
LEAST_ITEMS_FOR_PACE = 3  # one early arrival is no pace
# How far off the time left can be, as a fraction of it: measured on the runs above, about 16 % while only the girders give the pace,
# 10 % early in the braces, 5 % once most of the work is behind. It is shown as a range until the range is narrower than a minute.
UNCERTAINTY_GIRDERS_ONLY, UNCERTAINTY_EARLY, UNCERTAINTY_LATE = 0.20, 0.10, 0.05


class Stage:
    def __init__(self, key, title, unit, share):
        self.key, self.title, self.unit, self.share = key, title, unit, share
        self.started = self.finished = None
        self.done = self.total = 0

    def seconds(self, now):
        if self.started is None:
            return 0.0
        return (self.finished or now) - self.started

    # this stage's whole time projected from its own pace, or None before enough of it has been counted
    def projected_s(self, now):
        if self.finished is not None:
            return self.seconds(now)
        if self.total and self.done >= max(LEAST_ITEMS_FOR_PACE, LEAST_FRACTION_FOR_PACE * self.total):
            return self.seconds(now) * self.total / self.done
        return None


# Follows setu's phases and counts, and tells the user where the run is, how long each stage took and about how long is left.
class RunProgress:
    def __init__(self, special_vehicle, construction):
        skipped = {"special_vehicle"} if not special_vehicle else set()
        if not construction:
            skipped.add("construction")
        self.stages = [Stage(key, title, unit, share) for key, title, unit, share, _ in STAGES if key not in skipped]
        self.by_phase = {text: key for key, _, _, _, texts in STAGES for text in texts}
        self.current = None
        self.started = perf_counter()
        self.pct = FIRST_PCT

    def stage(self, key):
        return next(stage for stage in self.stages if stage.key == key)

    def phase(self, text):
        bridge_logger.sub_step(f"setu: {text}")
        key = self.by_phase.get(text)
        if key is None:
            return
        now = perf_counter()
        if self.current is not None:
            self.current.finished = now
        self.current = self.stage(key)
        self.current.started = now
        self.send(now)

    def counts(self, done, total):
        if self.current is not None:
            self.current.done, self.current.total = done, total
            self.send(perf_counter())

    def finish(self):
        now = perf_counter()
        if self.current is not None:
            self.current.finished = now
        self.current = None
        self.send(now)

    # seconds left, scaled from what has been measured so far; None until a counted stage has a pace
    def remaining_s(self, now):
        measured_s = measured_share = 0.0
        paced = False
        for stage in self.stages:
            projected = stage.projected_s(now)
            if projected is None:
                continue
            measured_s += projected
            measured_share += stage.share
            paced = paced or stage.key in COUNTED_STAGES
        if not paced:
            return None
        waiting_share = sum(stage.share for stage in self.stages if stage.started is None)
        total_s = measured_s + measured_s * waiting_share / measured_share
        if self.current is not None and self.current.projected_s(now) is None:
            total_s += measured_s * self.current.share / measured_share
        elapsed_s = sum(stage.seconds(now) for stage in self.stages)
        return max(0.0, total_s - elapsed_s)

    # fraction either way the time left may be out, from how much of the work has been measured
    def uncertainty(self, now):
        braces = next((stage for stage in self.stages if stage.key == "braces"), None)
        if braces is None or braces.projected_s(now) is None:
            return UNCERTAINTY_GIRDERS_ONLY
        if braces.finished is None and braces.done < braces.total / 2:
            return UNCERTAINTY_EARLY
        return UNCERTAINTY_LATE

    def fraction(self, now, remaining_s):
        if self.current is None and any(stage.finished is not None for stage in self.stages):
            return 1.0  # finished
        if remaining_s is not None:
            elapsed_s = sum(stage.seconds(now) for stage in self.stages)
            return elapsed_s / (elapsed_s + remaining_s) if elapsed_s + remaining_s > 0 else 0.0
        total_share = sum(stage.share for stage in self.stages)
        done_share = sum(stage.share for stage in self.stages if stage.finished is not None)
        if self.current is not None and self.current.total:
            done_share += self.current.share * self.current.done / self.current.total
        return done_share / total_share

    def send(self, now):
        remaining = self.remaining_s(now)
        self.pct = max(self.pct, FIRST_PCT + int((LAST_PCT - FIRST_PCT) * self.fraction(now, remaining)))  # the bar never goes back
        bridge_logger._emit(f"__progress__{self.pct}", "progress")
        status = {
            "stages": [{"title": stage.title, "state": "done" if stage.finished is not None else "running" if stage.started is not None else "waiting",
                        "seconds": round(stage.seconds(now), 1)} for stage in self.stages],
            "current": None if self.current is None else {"title": self.current.title, "done": self.current.done, "total": self.current.total, "unit": self.current.unit},
            "elapsed_s": round(now - self.started, 1),
            "remaining_s": None if remaining is None else round(remaining),
            "remaining_range_s": None if remaining is None else [round(remaining * (1 - self.uncertainty(now))), round(remaining * (1 + self.uncertainty(now)))],
        }
        bridge_logger.status(json.dumps(status), "setu")
