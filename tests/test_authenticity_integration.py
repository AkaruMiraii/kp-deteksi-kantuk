import unittest
from types import SimpleNamespace

import numpy as np

from scripts.detection_config import DetectionConfig
from scripts.multi_person import MultiPersonPipeline


class TestAuthenticityIntegration(unittest.TestCase):
    def test_temporal_support_is_per_track_and_lost_tracks_expire(self):
        class Engine:
            def predict(self, frame, box):
                return dict(authenticity="uncertain", passive_live_score=0.9)

        points = [
            SimpleNamespace(x=0.1 + (i % 10) * 0.01, y=0.2 + (i % 7) * 0.01) for i in range(478)
        ]
        other = [SimpleNamespace(x=p.x + 0.5, y=p.y) for p in points]
        pipeline = MultiPersonPipeline(DetectionConfig(), Engine())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for t in (0, 100, 200, 300):
            rows = pipeline.update(
                t, SimpleNamespace(face_landmarks=[points, other]), 640, 480, frame
            )
            self.assertTrue(all(r["temporal_live_score"] is None for r in rows))
        rows = pipeline.update(
            400, SimpleNamespace(face_landmarks=[other, points]), 640, 480, frame
        )
        self.assertTrue(all(r["temporal_live_score"] == 0.9 for r in rows))
        self.assertIsNot(pipeline.auth_windows[1], pipeline.auth_windows[2])
        pipeline.update(500, SimpleNamespace(face_landmarks=[]), 640, 480, frame)
        self.assertFalse(pipeline.auth_windows)
