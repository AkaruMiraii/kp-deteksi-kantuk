"""Conservative short-lived spatial tracks, not biometric identities."""

from math import isfinite


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0


class FaceTracker:
    def __init__(self, max_gap_ms=500):
        self.max_gap_ms = max_gap_ms
        self.previous = None
        self.boxes = {}
        self.next_id = 1
        self.last_seen = {}
        self.active_ids = set()

    def update(self, timestamp_ms, boxes):
        if (
            not isfinite(timestamp_ms)
            or timestamp_ms < 0
            or (self.previous is not None and timestamp_ms <= self.previous)
        ):
            raise ValueError("Timestamp must be finite and strictly increasing")
        for b in boxes:
            if len(b) != 4 or any(not isfinite(x) for x in b) or b[0] >= b[2] or b[1] >= b[3]:
                raise ValueError("Invalid face box")
        old = {
            k: b
            for k, b in self.boxes.items()
            if timestamp_ms - self.last_seen[k] <= self.max_gap_ms
        }
        candidates = [
            [k for k, b in old.items() if iou(box, b) >= (0.25 if k in self.active_ids else 0.7)]
            for box in boxes
        ]
        result = []
        for i, box in enumerate(boxes):
            ambiguous = any(j != i and iou(box, other) > 0.1 for j, other in enumerate(boxes))
            unique = len(candidates[i]) == 1 and sum(candidates[i][0] in c for c in candidates) == 1
            if unique and not ambiguous:
                identity = candidates[i][0]
            else:
                identity = self.next_id
                self.next_id += 1
            result.append(identity)
        self.boxes = {
            k: b
            for k, b in old.items()
            if k in result or not any(iou(b, current) > 0.1 for current in boxes)
        }
        self.boxes.update(zip(result, boxes))
        self.last_seen = {k: t for k, t in self.last_seen.items() if k in old}
        self.last_seen.update((k, timestamp_ms) for k in result)
        self.active_ids = set(result)
        self.previous = timestamp_ms
        return result
