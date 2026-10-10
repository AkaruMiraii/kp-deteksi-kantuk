from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MouthOpeningResult:
    state: str
    duration_seconds: float


class MouthOpeningTracker:
    def __init__(self, threshold, duration_seconds, max_gap_seconds):
        if any(
            not isfinite(value) or value <= 0
            for value in (threshold, duration_seconds, max_gap_seconds)
        ):
            raise ValueError("Threshold, duration and observation gap must be finite and positive.")
        self.threshold = threshold
        self.required_duration = duration_seconds
        self.max_gap = max_gap_seconds
        self.started_at = None
        self.previous_timestamp = None

    def update(self, timestamp_seconds, mar):
        if not isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("Timestamp must be finite and nonnegative.")
        if self.previous_timestamp is not None and timestamp_seconds <= self.previous_timestamp:
            raise ValueError("Timestamp must strictly increase.")
        if self.previous_timestamp is not None:
            if timestamp_seconds - self.previous_timestamp > self.max_gap:
                self.started_at = None
        self.previous_timestamp = timestamp_seconds
        if mar is None or not isfinite(mar) or mar < 0:
            self.started_at = None
            return MouthOpeningResult("not_assessable", 0.0)
        if mar <= self.threshold:
            self.started_at = None
            return MouthOpeningResult("no_large_opening", 0.0)
        if self.started_at is None:
            self.started_at = timestamp_seconds
        elapsed = timestamp_seconds - self.started_at
        state = "prolonged_opening" if elapsed >= self.required_duration else "opening_candidate"
        return MouthOpeningResult(state, elapsed)
