"""Opt-in session-only face matching. Embeddings never leave memory."""

import cv2
import numpy as np

MATCH_THRESHOLD = 0.5
CONTINUITY_THRESHOLD = 0.25
MISMATCH_FRAMES = 8
MATCH_MARGIN = 0.1
MAX_SESSION_IDENTITIES = 64
MAX_TEMPLATES = 8
MIN_FACE_PIXELS = 100


class SessionIdentity:
    def __init__(self, model):
        self.recognizer = cv2.FaceRecognizerSF_create(str(model), "")
        self.gallery = {}
        self.bindings = {}
        self.next_id = 1

    def feature(self, frame, landmarks):
        h, w = frame.shape[:2]
        # Validate the actual SFace alignment inputs, not a behavioral pose proxy.
        indices = (33, 133, 362, 263, 1, 61, 291)
        if len(landmarks) <= max(indices) or any(
            not np.isfinite(v) or not 0 <= v <= 1
            for i in indices
            for v in (landmarks[i].x, landmarks[i].y)
        ):
            return None
        xs, ys = [p.x * w for p in landmarks], [p.y * h for p in landmarks]
        if min(max(xs) - min(xs), max(ys) - min(ys)) < MIN_FACE_PIXELS:
            return None
        pairs = ((33, 133), (362, 263))
        eyes = sorted(
            [
                (
                    sum(landmarks[i].x * w for i in pair) / 2,
                    sum(landmarks[i].y * h for i in pair) / 2,
                )
                for pair in pairs
            ]
        )
        mouth = sorted([(landmarks[i].x * w, landmarks[i].y * h) for i in (61, 291)])
        points = eyes + [(landmarks[1].x * w, landmarks[1].y * h)] + mouth
        face = np.array(
            [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            + [v for point in points for v in point]
            + [1],
            dtype=np.float32,
        )
        aligned = self.recognizer.alignCrop(frame, face)
        value = self.recognizer.feature(aligned).reshape(-1)
        norm = np.linalg.norm(value)
        return value / norm if norm > 0 else None

    def resolve(self, spatial_ids, features):
        if not hasattr(self, "mismatches"):
            self.mismatches = {}
        if not hasattr(self, "templates"):
            self.templates = {k: [v.copy()] for k, v in self.gallery.items()}
        active = set(spatial_ids)
        self.mismatches = {k: v for k, v in self.mismatches.items() if k in active}
        self.bindings = {k: v for k, v in self.bindings.items() if k in active}
        used = set(self.bindings.values())
        output = []
        for spatial, feature in zip(spatial_ids, features):
            if feature is not None:
                norm = np.linalg.norm(feature)
                feature = feature / norm if np.isfinite(feature).all() and norm > 0 else None
            if spatial in self.bindings:
                bound = self.bindings[spatial]
                template = self.gallery.get(bound)
                if (
                    feature is None
                    or template is None
                    or max(float(feature @ old) for old in self.templates.get(bound, [template]))
                    >= CONTINUITY_THRESHOLD
                ):
                    if feature is not None and template is not None:
                        score = max(
                            float(feature @ old) for old in self.templates.get(bound, [template])
                        )
                        if score >= MATCH_THRESHOLD:
                            samples = self.templates.setdefault(bound, [template])
                            if len(samples) < MAX_TEMPLATES:
                                samples.append(feature.copy())
                    self.mismatches[spatial] = 0
                    output.append(bound)
                    continue
                self.mismatches[spatial] = self.mismatches.get(spatial, 0) + 1
                if self.mismatches[spatial] < MISMATCH_FRAMES:
                    output.append(bound)
                    continue
                del self.bindings[spatial]
                self.mismatches[spatial] = 0
            matches = (
                sorted(
                    (
                        (
                            max(
                                float(feature @ sample)
                                for sample in self.templates.get(identity, [old])
                            ),
                            identity,
                        )
                        for identity, old in self.gallery.items()
                        if identity not in used
                    ),
                    reverse=True,
                )
                if feature is not None
                else []
            )
            if (
                matches
                and matches[0][0] >= MATCH_THRESHOLD
                and (len(matches) == 1 or matches[0][0] - matches[1][0] >= MATCH_MARGIN)
            ):
                identity = matches[0][1]
            else:
                identity = self.next_id
                self.next_id += 1
                if feature is not None and len(self.gallery) < MAX_SESSION_IDENTITIES:
                    self.gallery[identity] = feature.copy()
                    self.templates[identity] = [feature.copy()]
            self.bindings[spatial] = identity
            used.add(identity)
            output.append(identity)
        return output
