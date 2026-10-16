"""Multi-face routing using existing geometry and track-local timers."""

from math import isfinite
from types import SimpleNamespace

from .active_challenge import ActiveChallenge
from .authenticity_window import AuthenticityWindow
from .eye_alertness import assess_eyes
from .face_tracking import FaceTracker
from .head_pose import head_direction
from .landmark_diagnostics import inspect_frame
from .person_state import PersonState


def observation_quality(right, left, mar):
    if right is None or left is None:
        return "eye_unavailable"
    if any(not isfinite(v) or v < 0 or v > 0.5 for v in (right, left)):
        return "implausible_eye_geometry"
    return ""


class MultiPersonPipeline:
    def __init__(
        self, config, authenticity_engine=None, threshold_only=False, identity_engine=None
    ):
        self.config = config
        self.identity_engine = identity_engine
        self.threshold_only = threshold_only
        self.tracker = FaceTracker(config.max_gap_seconds * 1000)
        self.states = {}
        self.challenges = {}
        self.auth_windows = {}
        self.authenticity_engine = authenticity_engine

    @staticmethod
    def merge_pending(target, pending, now):
        old, new = target.event_window, pending.event_window
        if new.started is None or now - new.started > 60:
            return
        if old.started is None:
            old.started = new.started
        existing = set(old.events)
        for event in new.events:
            if now - 60 <= event[0] <= now and event not in existing:
                old.events.append(event)
                if event[1] == "eye":
                    old.total_eye += 1
                else:
                    old.total_mouth += 1
        old.events = type(old.events)(sorted(set(old.events)))
        known = set(old.intervals)
        for interval in new.intervals:
            if interval[1] >= now - 60 and interval not in known:
                old.intervals.append(interval)
        old.intervals = type(old.intervals)(sorted(set(old.intervals)))
        old.closed = new.closed
        old.opened = new.opened
        old.previous = now
        old.valid = (False, False)

    def reset_track(self, identity):
        self.states[identity].reset()
        self.challenges.pop(identity, None)
        self.auth_windows.pop(identity, None)

    def update(self, timestamp_ms, result, width, height, frame=None):
        import numpy as np

        faces, boxes = [], []
        for landmarks in result.face_landmarks[: self.config.max_people]:
            if not landmarks or any(not isfinite(p.x) or not isfinite(p.y) for p in landmarks):
                continue
            xs, ys = [p.x for p in landmarks], [p.y for p in landmarks]
            box = min(xs), min(ys), max(xs), max(ys)
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            faces.append(landmarks)
            boxes.append(box)
        ids = self.tracker.update(timestamp_ms, boxes)
        if self.identity_engine is not None:
            if frame is None:
                raise ValueError("Session identity requires source frame")
            features = [self.identity_engine.feature(frame, points) for points in faces]
            if hasattr(self.identity_engine, "resolve_at"):
                ids = self.identity_engine.resolve_at(timestamp_ms, ids, features)
                for pending_id in getattr(self.identity_engine, "reset_pending_ids", ()):
                    self.states.pop(pending_id, None)
                    self.challenges.pop(pending_id, None)
                    self.auth_windows.pop(pending_id, None)
                for restored, pending_id in self.identity_engine.restored_from.items():
                    if pending_id in self.states:
                        pending_state = self.states.pop(pending_id)
                        if restored not in self.states:
                            self.states[restored] = PersonState(restored, self.config)
                            self.merge_pending(
                                self.states[restored], pending_state, timestamp_ms / 1000
                            )
                        else:
                            self.merge_pending(
                                self.states[restored], pending_state, timestamp_ms / 1000
                            )
            else:
                ids = self.identity_engine.resolve(ids, features)
        retained = (
            (set(self.identity_engine.gallery) | set(ids))
            if self.identity_engine is not None
            else set(self.tracker.boxes)
        )
        for identity in retained - set(ids):
            if identity in self.states:
                self.states[identity].update(timestamp_ms, None, None, None, False, "uncertain")
                self.challenges.pop(identity, None)
                self.auth_windows.pop(identity, None)
        self.states = {k: v for k, v in self.states.items() if k in retained}
        self.challenges = {k: v for k, v in self.challenges.items() if k in ids}
        self.auth_windows = {k: v for k, v in self.auth_windows.items() if k in ids}
        rows = []
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for identity, landmarks, box in zip(ids, faces, boxes):
            state = self.states.setdefault(identity, PersonState(identity, self.config))
            single = SimpleNamespace(face_landmarks=[landmarks])
            right, left, eye_reason = assess_eyes(single, width, height)
            metrics, _ = inspect_frame(blank, single, "unmarked")
            mar = None if metrics["reason"] else metrics["mar"]
            direction = head_direction(landmarks)
            auth = dict(authenticity="uncertain", passive_live_score=None)
            if frame is not None and self.authenticity_engine is not None:
                try:
                    auth = self.authenticity_engine.predict(frame, box)
                except (RuntimeError, ValueError, OSError) as error:
                    auth = dict(
                        authenticity="uncertain",
                        passive_live_score=None,
                        authenticity_reason="model_inference_error",
                        model_error=str(error),
                    )
            challenge = self.challenges.setdefault(identity, ActiveChallenge())
            window = self.auth_windows.setdefault(
                identity, AuthenticityWindow(self.config.max_gap_seconds * 1000)
            )
            temporal_score = window.update(timestamp_ms, auth["passive_live_score"])
            verified = challenge.update(timestamp_ms, direction, temporal_score)
            if self.threshold_only:
                verified = dict(authenticity="not_checked", challenge_request="none")
            quality = observation_quality(right, left, mar)
            row = state.update(
                timestamp_ms,
                None if quality else right,
                None if quality else left,
                mar,
                direction == "center",
                verified["authenticity"],
                threshold_only=self.threshold_only,
            )
            row.update(
                ear_r=right,
                ear_l=left,
                observation_quality="invalid" if quality else "valid",
                observation_reason=quality,
            )
            row.update(auth)
            row["temporal_live_score"] = temporal_score
            row.update(verified)
            if auth.get("passive_live_score") is None:
                row["authenticity_reason"] = auth.get("authenticity_reason", "model_unavailable")
            row.update(
                box=box,
                eye_reason=eye_reason,
                mouth_reason=metrics["reason"],
                head_direction=direction,
                toward_task=direction == "center",
                points_px=metrics["points_px"],
            )
            if hasattr(self.identity_engine, "resolve_at"):
                row["identity_pending"] = identity < 0
                row["enrollment_remaining"] = self.identity_engine.remaining
                if identity < 0:
                    row["primary_status"] = "tidak_dapat_dinilai"
                    row["primary_reason"] = "identity_confirmation_pending"
                    row["identity_candidates"] = [
                        k for _, k in self.identity_engine.candidates(features[ids.index(identity)])
                    ][:3]
                row["identity_confirm_options"] = list(self.identity_engine.gallery)
            rows.append(row)
        return rows
