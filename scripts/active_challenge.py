"""Experimental random pose challenge. A passed challenge is not attack-proof."""

from math import isfinite
from secrets import choice


class ActiveChallenge:
    def __init__(self, sequence=None, hold_seconds=0.4):
        self.sequence = sequence or [choice(("left", "right")), choice(("up", "down"))]
        self.hold_seconds = hold_seconds
        self.step = 0
        self.started = None
        self.neutral_required = False
        self.previous = None
        self.passed_at = None

    def update(self, timestamp_ms, direction, score):
        if (
            not isfinite(timestamp_ms)
            or timestamp_ms < 0
            or (self.previous is not None and timestamp_ms <= self.previous)
        ):
            raise ValueError("Invalid challenge timestamp")
        if self.previous is not None and timestamp_ms - self.previous > 500:
            self.step = 0
            self.started = None
            self.neutral_required = False
        self.previous = timestamp_ms
        if self.passed_at is not None and timestamp_ms - self.passed_at >= 15000:
            self.step = 0
            self.started = None
            self.neutral_required = False
            self.passed_at = None
            self.sequence = [choice(("left", "right")), choice(("up", "down"))]
        state = "uncertain"
        if (
            direction not in ("left", "right", "up", "down", "center")
            or score is None
            or not isfinite(score)
            or score < 0.8
        ):
            self.step = 0
            self.started = None
            self.neutral_required = False
            if score is not None and isfinite(score) and score < 0.2:
                state = "spoof"
        elif self.step >= len(self.sequence):
            state = "live"
        elif self.neutral_required:
            if direction == "center":
                self.neutral_required = False
        elif direction == self.sequence[self.step]:
            if self.started is None:
                self.started = timestamp_ms
            if timestamp_ms - self.started >= self.hold_seconds * 1000:
                self.step += 1
                self.started = None
                self.neutral_required = True
                if self.step == len(self.sequence):
                    self.passed_at = timestamp_ms
                    state = "live"
        else:
            self.started = None
        return dict(
            authenticity=state,
            challenge_step=self.step,
            challenge_request="complete"
            if self.step == len(self.sequence)
            else ("center" if self.neutral_required else self.sequence[self.step]),
            authenticity_reason="experimental_rgb_and_challenge",
            validated_attack_coverage=False,
        )
