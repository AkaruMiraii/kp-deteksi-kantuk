"""Opt-in task-response proxy, never a confirmed mental-state classifier."""
from math import isfinite


class TaskResponseProbe:
    def __init__(self):
        self.started = None
        self.deadline = None
        self.expected = None
        self.status = 'idle'
        self.latency_ms = None

    def start(self, timestamp_ms, expected, timeout_seconds=5):
        if expected not in ('left', 'right') or not isfinite(timestamp_ms) or not 0 < timeout_seconds <= 30:
            raise ValueError('Valid timestamp, direction and timeout required')
        self.started = timestamp_ms
        self.deadline = timestamp_ms + timeout_seconds * 1000
        self.expected = expected
        self.status = 'pending'
        self.latency_ms = None

    def respond(self, timestamp_ms, response):
        if self.status != 'pending' or not self.started <= timestamp_ms < self.deadline:
            return False
        if response != self.expected:
            return False
        self.status = 'responded'
        self.latency_ms = timestamp_ms - self.started
        return True

    def update(self, timestamp_ms, assessable, eyes_open):
        if not assessable or not eyes_open:
            if self.status in ('pending', 'missed'):
                self.status = 'cancelled'
        elif self.status == 'pending' and timestamp_ms >= self.deadline:
            self.status = 'missed'
        return dict(
            task_response_status=self.status,
            task_response_request=self.expected if self.status == 'pending' else 'none',
            task_response_latency_ms=self.latency_ms,
            zone_out_indicator='task_nonresponse_candidate' if self.status == 'missed' else 'unknown',
            zone_out_indicator_source='opt_in_task_response_not_mental_state',
        )
