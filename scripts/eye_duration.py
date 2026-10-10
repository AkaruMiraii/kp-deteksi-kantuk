import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class EyeClosureResult:
    state: str
    duration_seconds: float


class EyeClosureTracker:
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

    def update(self, timestamp_seconds, ear_r, ear_l):
        if not isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("Timestamp must be finite and nonnegative.")
        if self.previous_timestamp is not None and timestamp_seconds <= self.previous_timestamp:
            raise ValueError("Timestamp must strictly increase.")
        if self.previous_timestamp is not None:
            if timestamp_seconds - self.previous_timestamp > self.max_gap:
                self.started_at = None
        self.previous_timestamp = timestamp_seconds
        if any(value is None or not isfinite(value) or value < 0 for value in (ear_r, ear_l)):
            self.started_at = None
            return EyeClosureResult("not_assessable", 0.0)
        if ear_r >= self.threshold or ear_l >= self.threshold:
            self.started_at = None
            return EyeClosureResult("no_bilateral_closure", 0.0)
        if self.started_at is None:
            self.started_at = timestamp_seconds
        elapsed = timestamp_seconds - self.started_at
        state = "prolonged_closure" if elapsed >= self.required_duration else "closure_candidate"
        return EyeClosureResult(state, elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Replay eye duration on a diagnostic log, not final evaluation."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--max-gap", type=float, required=True)
    args = parser.parse_args()
    tracker = EyeClosureTracker(args.threshold, args.duration, args.max_gap)
    states = Counter()
    transitions = []
    previous_state = None
    frames = 0
    with args.log.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            invalid = bool(row["reason"])
            result = tracker.update(
                row["timestamp_ms"] / 1000,
                None if invalid else row["ear_r"],
                None if invalid else row["ear_l"],
            )
            states[result.state] += 1
            frames += 1
            if result.state != previous_state:
                transitions.append(
                    dict(
                        timestamp_seconds=row["timestamp_ms"] / 1000,
                        activity=row["activity"],
                        **asdict(result),
                    )
                )
                previous_state = result.state
    print(
        json.dumps(
            dict(
                source=str(args.log),
                frames=frames,
                states=states,
                parameters=dict(
                    threshold=args.threshold,
                    duration_seconds=args.duration,
                    max_gap_seconds=args.max_gap,
                    bilateral_rule="both EAR below threshold",
                ),
                purpose="development replay, not calibrated classification accuracy",
                transitions=transitions,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
