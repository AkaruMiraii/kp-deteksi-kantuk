"""Track-local experimental observations. No inferred internal mental states."""

from collections import deque
from math import isfinite
from statistics import median

from .event_window import EventWindow
from .eye_duration import EyeClosureTracker
from .mouth_duration import MouthOpeningTracker
from .task_response import TaskResponseProbe


class PersonState:
    def __init__(self, track_id, config):
        self.track_id = track_id
        self.config = config
        self.reset()

    def reset(self):
        c = self.config
        self.eyes = EyeClosureTracker(c.ear_threshold, c.closure_seconds, c.max_gap_seconds)
        self.mouth = MouthOpeningTracker(c.mar_threshold, c.mouth_seconds, c.max_gap_seconds)
        self.raw_eyes = EyeClosureTracker(c.ear_threshold, c.closure_seconds, c.max_gap_seconds)
        self.raw_mouth = MouthOpeningTracker(c.mar_threshold, c.mouth_seconds, c.max_gap_seconds)
        self.event_window = EventWindow(c.max_gap_seconds)
        self.history = deque(maxlen=300)
        self.focus_start = None
        self.previous = None
        self.baseline = None
        self.baseline_source = None
        self.zone_label = None
        self.zone_source = "no_independent_label"
        self.task_probe = TaskResponseProbe()

    def calibrate_open_eyes(self, right, left, source):
        if source not in ("user_confirmed_open_eyes", "automatic_frontal_open_eye_proxy") or any(
            not isfinite(v) or v < 0.08 or v > 1 for v in (right, left)
        ):
            raise ValueError("Explicit valid open-eye calibration required")
        events = self.event_window
        self.reset()
        self.event_window = events
        self.baseline = (right, left)
        self.baseline_source = source
        c = self.config
        self.eyes = EyeClosureTracker(min(right, left) * 0.65, c.closure_seconds, c.max_gap_seconds)
        self.raw_eyes = EyeClosureTracker(
            min(right, left) * 0.65, c.closure_seconds, c.max_gap_seconds
        )

    def calibrate_from_history(self, source="user_confirmed_open_eyes"):
        rows = list(self.history)
        if not rows:
            raise ValueError("No calibration observations")
        end = rows[-1]["timestamp_ms"]
        rows = [r for r in rows if end - r["timestamp_ms"] <= 3000]
        if len(rows) < 5 or end - rows[0]["timestamp_ms"] < 2000:
            raise ValueError("Keep eyes open for at least two seconds before C")
        if any(
            r["ear_r"] is None
            or r["ear_l"] is None
            or min(r["ear_r"], r["ear_l"]) < 0.08
            or max(r["ear_r"], r["ear_l"]) > 0.5
            for r in rows
        ):
            raise ValueError("Calibration needs valid open-eye geometry")
        if any(
            b["timestamp_ms"] - a["timestamp_ms"] > self.config.max_gap_seconds * 1000
            for a, b in zip(rows, rows[1:])
        ):
            raise ValueError("Calibration observation gap")
        self.calibrate_open_eyes(
            median(r["ear_r"] for r in rows),
            median(r["ear_l"] for r in rows),
            source,
        )

    def set_zone_out_label(self, value, source):
        if source not in ("self_report", "task_response") or type(value) is not bool:
            raise ValueError("Independent label source required")
        self.zone_label = value
        self.zone_source = source

    def update(
        self, timestamp_ms, right, left, mar, toward_task, authenticity, threshold_only=False
    ):
        t = timestamp_ms / 1000
        raw_eye = self.raw_eyes.update(t, right, left)
        raw_mouth = self.raw_mouth.update(t, mar)
        trusted = authenticity == "live" or threshold_only
        eye = self.eyes.update(t, right if trusted else None, left if trusted else None)
        mouth = self.mouth.update(t, mar if trusted else None)
        gap = self.previous is not None and t - self.previous > self.config.max_gap_seconds
        self.previous = t
        if gap or not trusted or toward_task is not True or eye.state != "no_bilateral_closure":
            self.focus_start = None
        elif self.focus_start is None:
            self.focus_start = t
        focus_elapsed = t - self.focus_start if self.focus_start is not None else 0
        if not trusted or eye.state == "not_assessable":
            primary_status = "tidak_dapat_dinilai"
            primary_reason = "authenticity_unverified" if not trusted else "eye_unavailable"
        elif eye.state == "prolonged_closure":
            primary_status = "indikasi_kantuk"
            primary_reason = "prolonged_bilateral_closure"
        else:
            primary_status = "tidak_terdeteksi_indikasi_kantuk"
            primary_reason = "no_prolonged_bilateral_closure"
        row = dict(
            primary_status=primary_status,
            primary_reason=primary_reason,
            primary_trigger="eye" if primary_status == "indikasi_kantuk" else "none",
            decision_policy="threshold_only_unverified_v1"
            if threshold_only
            else "kp_eye_evidence_mouth_observation_v1",
            authenticity_required=not threshold_only,
            track_id=self.track_id,
            timestamp_ms=timestamp_ms,
            authenticity=authenticity,
            ear_r=right,
            ear_l=left,
            mar=mar,
            eye_state=eye.state,
            mouth_state=mouth.state,
            closed_seconds=eye.duration_seconds,
            raw_closed_seconds=raw_eye.duration_seconds,
            raw_eye_state=raw_eye.state,
            raw_mouth_state=raw_mouth.state,
            raw_mouth_seconds=raw_mouth.duration_seconds,
            raw_orientation_indicator="unknown"
            if right is None or left is None
            else "toward_camera"
            if toward_task is True
            else "away_or_unknown",
            raw_observation_policy="geometry_only_not_authenticity_or_mental_state",
            mouth_seconds=mouth.duration_seconds,
            focus_seconds=focus_elapsed,
            focus_indicator="task_orientation_candidate"
            if self.focus_start is not None and focus_elapsed >= self.config.focus_seconds
            else "unknown",
            microsleep_indicator="closure_candidate"
            if eye.state == "prolonged_closure"
            else "unknown",
            zone_out="unknown"
            if self.zone_label is None
            else ("self_reported" if self.zone_label else "reported_absent"),
            zone_out_label_source=self.zone_source,
            baseline=self.baseline,
            baseline_source=self.baseline_source,
            experimental=True,
            threshold_provenance=self.config.provenance,
        )
        row.update(
            self.task_probe.update(
                timestamp_ms,
                trusted and right is not None and left is not None,
                eye.state == "no_bilateral_closure",
            )
        )
        row.update(self.event_window.update(t, raw_eye.state, raw_mouth.state))
        self.history.append(row)
        return row
