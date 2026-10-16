"""Causal temporal support for passive scores, not calibrated confidence."""

from collections import deque
from math import isfinite
from statistics import median


class AuthenticityWindow:
    def __init__(self, max_gap_ms=500):
        self.max_gap_ms = max_gap_ms
        self.samples = deque(maxlen=60)
        self.previous = None

    def update(self, timestamp_ms, score):
        if (
            not isfinite(timestamp_ms)
            or timestamp_ms < 0
            or (self.previous is not None and timestamp_ms <= self.previous)
        ):
            raise ValueError("Timestamp must strictly increase")
        if self.previous is not None and timestamp_ms - self.previous > self.max_gap_ms:
            self.samples.clear()
        self.previous = timestamp_ms
        if score is None or not isfinite(score) or not 0 <= score <= 1:
            self.samples.clear()
            return None
        self.samples.append((timestamp_ms, score))
        while self.samples and timestamp_ms - self.samples[0][0] > 1000:
            self.samples.popleft()
        if len(self.samples) < 5 or timestamp_ms - self.samples[0][0] < 400:
            return None
        return median(value for _, value in self.samples)
