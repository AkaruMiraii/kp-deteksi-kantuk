"""Experimental per-track event counts, never clinical sleep labels."""

from collections import deque

WINDOW_SECONDS = 300.0
MIN_COVERAGE = 0.8
MIN_OBSERVATION_SECONDS = 30.0
OPENING_COUNT_THRESHOLD = 2
CLOSURE_COUNT_THRESHOLD = 1


class EventWindow:
    def __init__(self, max_gap_seconds, window_seconds=WINDOW_SECONDS):
        self.max_gap = max_gap_seconds
        self.window = window_seconds
        self.events = deque()
        self.intervals = deque()
        self.previous = None
        self.started = None
        self.valid = (False, False)
        self.closed = False
        self.opened = False
        self.total_eye = self.total_mouth = 0

    def update(self, time, eye, mouth):
        if self.started is None:
            self.started = time
        if self.previous is not None:
            gap = time - self.previous
            if 0 < gap <= self.max_gap:
                flags = (
                    self.valid[0] and eye != "not_assessable",
                    self.valid[1] and mouth != "not_assessable",
                )
                self.intervals.append((self.previous, time, flags))
        self.previous = time
        self.valid = (eye != "not_assessable", mouth != "not_assessable")
        for state, normal, prolonged, latch, kind in (
            (eye, "no_bilateral_closure", "prolonged_closure", "closed", "eye"),
            (mouth, "no_large_opening", "prolonged_opening", "opened", "mouth"),
        ):
            if state == normal:
                setattr(self, latch, False)
            elif state == prolonged and not getattr(self, latch):
                self.events.append((time, kind))
                setattr(self, latch, True)
                if kind == "eye":
                    self.total_eye += 1
                else:
                    self.total_mouth += 1
        cutoff = time - self.window
        while self.events and self.events[0][0] <= cutoff:
            self.events.popleft()
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        span = min(self.window, time - self.started)
        valid_seconds = [0.0, 0.0]
        for start, end, flags in self.intervals:
            duration = end - max(start, cutoff)
            for i in range(2):
                if flags[i]:
                    valid_seconds[i] += duration
        coverage = [seconds / span if span else 0.0 for seconds in valid_seconds]
        eyes = sum(kind == "eye" for _, kind in self.events)
        mouths = sum(kind == "mouth" for _, kind in self.events)
        enough = span >= MIN_OBSERVATION_SECONDS and min(coverage) >= MIN_COVERAGE
        pattern = "insufficient_observation"
        if enough:
            pattern = (
                "increased_visual_drowsiness"
                if eyes >= CLOSURE_COUNT_THRESHOLD and mouths >= OPENING_COUNT_THRESHOLD
                else "prolonged_eye_event"
                if eyes
                else "repeated_opening_candidates"
                if mouths >= OPENING_COUNT_THRESHOLD
                else "no_combined_pattern"
            )
        return dict(
            eye_events_window=eyes,
            opening_events_window=mouths,
            eye_events_session=self.total_eye,
            opening_events_session=self.total_mouth,
            eye_coverage=coverage[0],
            mouth_coverage=coverage[1],
            observation_seconds=span,
            event_window_seconds=self.window,
            visual_pattern=pattern,
            event_policy="experimental_opening_not_confirmed_yawn_v1",
        )
