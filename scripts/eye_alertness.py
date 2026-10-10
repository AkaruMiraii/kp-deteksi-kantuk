from collections import deque
from math import isfinite
from statistics import median

import cv2

from .eye_features import calculate_ear

EYE_INDICES = ([33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380])


def assess_eyes(result, width, height):
    if not result.face_landmarks:
        return None, None, "no_face_detected"
    landmarks = result.face_landmarks[0]
    if len(landmarks) <= max(i for eye in EYE_INDICES for i in eye):
        return None, None, "insufficient_eye_landmarks"
    values = []
    for eye in EYE_INDICES:
        points = [(landmarks[i].x * width, landmarks[i].y * height) for i in eye]
        if any(
            not isfinite(x) or not isfinite(y) or not 0 <= x < width or not 0 <= y < height
            for x, y in points
        ):
            return None, None, "invalid_eye_geometry"
        try:
            values.append(calculate_ear(points))
        except ValueError:
            return None, None, "zero_eye_width"
    return values[0], values[1], ""


class EyeAlertness:
    def __init__(self, calibration_seconds=5):
        if not isfinite(calibration_seconds) or calibration_seconds <= 0:
            raise ValueError("Calibration duration must be positive.")
        self.calibration_seconds = calibration_seconds
        self.reset()

    def reset(self):
        self.samples = []
        self.baselines = None
        self.calibration_start = None
        self.previous = None
        self.started = None
        self.window = deque()
        self.closed = False

    def update(self, timestamp_ms, right, left):
        if not isfinite(timestamp_ms) or timestamp_ms < 0:
            raise ValueError("Invalid timestamp.")
        if self.previous is not None and timestamp_ms <= self.previous:
            raise ValueError("Timestamps must increase.")
        gap = self.previous is not None and timestamp_ms - self.previous > 500
        if gap:
            self.started = None
            self.closed = False
            self.window.clear()
            if self.baselines is None:
                self.samples.clear()
                self.calibration_start = None
        self.previous = timestamp_ms
        output = dict(
            eye_alertness_state="not_assessable",
            eye_closed_duration_seconds=0.0,
            eye_threshold_right=None,
            eye_threshold_left=None,
            eye_closed_fraction_30s=None,
            eye_observed_seconds_30s=0.0,
        )
        if any(v is None or not isfinite(v) or v < 0 for v in (right, left)):
            self.started = None
            self.closed = False
            self.window.clear()
            if self.baselines is None:
                self.samples.clear()
                self.calibration_start = None
            return output
        if self.baselines is None:
            self.calibration_start = (
                timestamp_ms if self.calibration_start is None else self.calibration_start
            )
            self.samples.append((right, left))
            output["eye_alertness_state"] = "calibrating_open_eyes"
            if (
                timestamp_ms - self.calibration_start < self.calibration_seconds * 1000
                or len(self.samples) < 5
            ):
                return output
            self.baselines = (
                median(v[0] for v in self.samples),
                median(v[1] for v in self.samples),
            )
            if min(self.baselines) < 0.08:
                self.samples.clear()
                self.baselines = None
                self.calibration_start = None
                output["eye_alertness_state"] = "calibration_failed_open_eyes_required"
                return output
        thresholds = (self.baselines[0] * 0.65, self.baselines[1] * 0.65)
        # Hysteresis prevents rapid reopening decisions near the closed boundary.
        limit = 0.75 if self.closed else 0.65
        self.closed = right < self.baselines[0] * limit and left < self.baselines[1] * limit
        if self.closed:
            self.started = timestamp_ms if self.started is None else self.started
        else:
            self.started = None
        elapsed = (timestamp_ms - self.started) / 1000 if self.started is not None else 0
        self.window.append((timestamp_ms, self.closed))
        while self.window and self.window[0][0] < timestamp_ms - 30000:
            self.window.popleft()
        observed = sum(b[0] - a[0] for a, b in zip(self.window, list(self.window)[1:])) / 1000
        closed_time = (
            sum(b[0] - a[0] for a, b in zip(self.window, list(self.window)[1:]) if a[1]) / 1000
        )
        fraction = closed_time / observed if observed else None
        state = "no_eye_indicator"
        if elapsed >= 8:
            state = "sleep_like_visual_indicator"
        elif elapsed >= 3:
            state = "near_sleep_visual_indicator"
        elif elapsed >= 1.5:
            state = "drowsiness_visual_indicator"
        elif elapsed > 0:
            state = "closure_pending"
        output.update(
            eye_alertness_state=state,
            eye_closed_duration_seconds=elapsed,
            eye_threshold_right=thresholds[0],
            eye_threshold_left=thresholds[1],
            eye_closed_fraction_30s=fraction,
            eye_observed_seconds_30s=observed,
        )
        return output


class EyeAwareDetector:
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker
        self.result = None

    def detect_for_video(self, image, timestamp):
        self.result = self.detector.detect_for_video(image, timestamp)
        return self.result

    def add_eye_status(self, frame, record, overlay, timestamp):
        right, left, reason = assess_eyes(self.result, frame.shape[1], frame.shape[0])
        record.update(self.tracker.update(timestamp, right, left))
        record.update(
            independent_ear_r=right, independent_ear_l=left, independent_eye_reason=reason
        )
        # Replace only the conflicting legacy final status display; original columns stay auditable.
        cv2.rectangle(overlay, (0, 140), (frame.shape[1], 205), (18, 18, 18), -1)
        names = {
            "sleep_like_visual_indicator": "INDIKASI TIDUR (VISUAL)",
            "near_sleep_visual_indicator": "INDIKASI HAMPIR TIDUR (VISUAL)",
            "drowsiness_visual_indicator": "INDIKASI KANTUK (MATA)",
            "no_eye_indicator": "BELUM ADA INDIKATOR MATA",
            "closure_pending": "MATA MENUTUP: MENUNGGU DURASI",
            "not_assessable": "MATA TIDAK DAPAT DINILAI",
            "calibrating_open_eyes": "KALIBRASI: BUKA MATA, TATAP KAMERA",
            "calibration_failed_open_eyes_required": "KALIBRASI GAGAL: BUKA MATA",
        }
        cv2.putText(
            overlay,
            names[record["eye_alertness_state"]],
            (10, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 210, 255),
            1,
        )
        cv2.putText(
            overlay,
            f"Closed {record['eye_closed_duration_seconds']:.2f}s | R reset | EXPERIMENTAL",
            (10, 191),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (230, 230, 230),
            1,
        )
        return record, overlay
