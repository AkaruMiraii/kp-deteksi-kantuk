"""Identity alignment validity is independent of the behavioral pose proxy."""

import unittest
from types import SimpleNamespace as P

import numpy as np

from scripts.detection_config import DetectionConfig
from scripts.multi_person import MultiPersonPipeline
from scripts.session_identity import SessionIdentity


class TestIdentityPoseContract(unittest.TestCase):
    def points(self):
        points = [P(x=0.5, y=0.5) for _ in range(478)]
        for i, xy in {
            33: (0.3, 0.4),
            133: (0.4, 0.4),
            362: (0.6, 0.4),
            263: (0.7, 0.4),
            1: (0.5, 0.42),
            152: (0.5, 0.8),
            61: (0.4, 0.6),
            291: (0.6, 0.6),
            10: (0.5, 0.2),
        }.items():
            points[i] = P(x=xy[0], y=xy[1])
        return points

    def test_up_pose_does_not_suppress_safe_identity_alignment(self):
        class Identity:
            gallery = {}

            def feature(self, frame, points):
                self.called = True
                return np.array([1.0, 0.0])

            def resolve(self, ids, features):
                self.features = features
                return ids

        engine = Identity()
        engine.called = False
        pipeline = MultiPersonPipeline(
            DetectionConfig(), threshold_only=True, identity_engine=engine
        )
        pipeline.update(
            0,
            P(face_landmarks=[self.points()]),
            640,
            480,
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        self.assertTrue(engine.called, "behavioral up proxy incorrectly suppresses SFace")
        self.assertIsNotNone(engine.features[0])

    def test_out_of_frame_alignment_point_is_rejected_before_crop(self):
        engine = SessionIdentity.__new__(SessionIdentity)

        class Recognizer:
            def alignCrop(self, *args):
                raise AssertionError("unsafe alignment reached recognizer")

        engine.recognizer = Recognizer()
        points = self.points()
        points[61].y = 1.1
        self.assertIsNone(engine.feature(np.zeros((480, 640, 3), dtype=np.uint8), points))
