import hashlib
import json
from collections import deque
from math import exp, isfinite
from pathlib import Path
from statistics import mean, pstdev


class TemporalFeatures:
    names = [
        "mar",
        "ear_mean",
        "ear_difference",
        "mar_range",
        "mar_mean",
        "mar_std",
        "ear_window_mean",
        "mar_slope",
        "window_seconds",
    ]

    def __init__(self):
        self.history = deque()
        self.previous = None
        self.gap_reset = False

    def update(self, timestamp_ms, ear_r, ear_l, mar):
        if not isfinite(timestamp_ms) or timestamp_ms < 0:
            raise ValueError("Invalid timestamp.")
        if self.previous is not None and timestamp_ms <= self.previous:
            raise ValueError("Timestamps must increase.")
        self.gap_reset = self.previous is not None and timestamp_ms - self.previous > 500
        if self.gap_reset:
            self.history.clear()
        self.previous = timestamp_ms
        if any(v is None or not isfinite(v) or v < 0 for v in (ear_r, ear_l, mar)):
            self.history.clear()
            return None
        self.history.append((timestamp_ms, mar, (ear_r + ear_l) / 2))
        while self.history and self.history[0][0] < timestamp_ms - 1500:
            self.history.popleft()
        mouths = [v[1] for v in self.history]
        eyes = [v[2] for v in self.history]
        elapsed = (timestamp_ms - self.history[0][0]) / 1000
        slope = (mar - mouths[0]) / elapsed if elapsed else 0
        return [
            mar,
            (ear_r + ear_l) / 2,
            abs(ear_r - ear_l),
            max(mouths) - min(mouths),
            mean(mouths),
            pstdev(mouths),
            mean(eyes),
            slope,
            elapsed,
        ]


class SmartYawn:
    def __init__(self, path):
        path = Path(path)
        self.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model = json.loads(path.read_text(encoding="utf-8"))
        model = self.model
        if model["schema_version"] != 1 or model["feature_names"] != TemporalFeatures.names:
            raise ValueError("Unsupported smart yawn model schema.")
        n = len(TemporalFeatures.names)
        for key in ("mean", "scale", "coefficients"):
            if len(model[key]) != n or not all(isfinite(v) for v in model[key]):
                raise ValueError(f"Invalid model vector: {key}")
        if any(v <= 0 for v in model["scale"]):
            raise ValueError("Model scale must be positive.")
        if not isfinite(model["intercept"]) or not 0 < model["score_threshold"] < 1:
            raise ValueError("Invalid model decision parameters.")
        if not isfinite(model["persistence_seconds"]) or model["persistence_seconds"] < 0:
            raise ValueError("Invalid persistence.")
        self.features = TemporalFeatures()
        self.started = None

    def update(self, timestamp_ms, ear_r, ear_l, mar):
        values = self.features.update(timestamp_ms, ear_r, ear_l, mar)
        if values is None:
            self.started = None
            return dict(
                smart_yawn_state="not_assessable",
                smart_yawn_score=None,
                smart_yawn_duration_seconds=0.0,
            )
        if self.features.gap_reset:
            self.started = None
        m = self.model
        logit = m["intercept"] + sum(
            c * (v - u) / s for c, v, u, s in zip(m["coefficients"], values, m["mean"], m["scale"])
        )
        score = 1 / (1 + exp(-max(-60, min(60, logit))))
        if score < m["score_threshold"]:
            self.started = None
            state = "no_candidate"
            elapsed = 0.0
        else:
            if self.started is None:
                self.started = timestamp_ms
            elapsed = (timestamp_ms - self.started) / 1000
            state = "yawn_candidate" if elapsed >= m["persistence_seconds"] else "candidate_pending"
        return dict(
            smart_yawn_state=state, smart_yawn_score=score, smart_yawn_duration_seconds=elapsed
        )
