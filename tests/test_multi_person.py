import unittest
from types import SimpleNamespace

from scripts.detection_config import DetectionConfig


class TestMultiPerson(unittest.TestCase):
    def test_one_detector_result_routes_each_face_to_its_own_state(self):
        from scripts.multi_person import MultiPersonPipeline

        points = [
            SimpleNamespace(x=0.15 + (i % 10) * 0.005, y=0.2 + (i % 7) * 0.005) for i in range(478)
        ]
        other = [SimpleNamespace(x=p.x + 0.5, y=p.y) for p in points]
        pipeline = MultiPersonPipeline(DetectionConfig())
        rows = pipeline.update(0, SimpleNamespace(face_landmarks=[points, other]), 640, 480)
        self.assertEqual([r["track_id"] for r in rows], [1, 2])
        self.assertTrue(all(r["authenticity"] == "uncertain" for r in rows))
        rows = pipeline.update(100, SimpleNamespace(face_landmarks=[other, points]), 640, 480)
        self.assertEqual([r["track_id"] for r in rows], [2, 1])
        self.assertIsNot(pipeline.states[1].history, pipeline.states[2].history)
        pipeline.update(200, SimpleNamespace(face_landmarks=[]), 640, 480)
        self.assertEqual(set(pipeline.states), {1, 2})
        self.assertEqual(pipeline.states[1].history[-1]["raw_closed_seconds"], 0)
        pipeline.update(700, SimpleNamespace(face_landmarks=[]), 640, 480)
        self.assertFalse(pipeline.states)

    def test_passive_engine_receives_frame_and_challenge_remains_uncertain(self):
        import numpy as np

        from scripts.multi_person import MultiPersonPipeline

        class Engine:
            def predict(self, frame, box):
                return dict(passive_live_score=0.9, authenticity="uncertain")

        points = [
            SimpleNamespace(x=0.2 + (i % 10) * 0.015, y=0.2 + (i % 7) * 0.02) for i in range(478)
        ]
        pipeline = MultiPersonPipeline(DetectionConfig(), authenticity_engine=Engine())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rows = pipeline.update(0, SimpleNamespace(face_landmarks=[points]), 640, 480, frame=frame)
        self.assertEqual(rows[0]["passive_live_score"], 0.9)
        self.assertEqual(rows[0]["authenticity"], "uncertain")
        self.assertIn(rows[0]["challenge_request"], ["left", "right"])

    def test_implausible_ear_does_not_reach_behavior_timer(self):
        from scripts.multi_person import observation_quality

        self.assertEqual(observation_quality(0.2, 0.7, 0.1), "implausible_eye_geometry")
        self.assertEqual(observation_quality(0.2, 0.21, 0.1), "")
        self.assertEqual(observation_quality(None, 0.2, 0.1), "eye_unavailable")

    def test_model_error_is_uncertain_and_reason_is_not_overwritten(self):
        import numpy as np

        from scripts.multi_person import MultiPersonPipeline

        class Engine:
            def predict(self, frame, box):
                raise RuntimeError("inference unavailable")

        points = [
            SimpleNamespace(x=0.2 + (i % 10) * 0.015, y=0.2 + (i % 7) * 0.02) for i in range(478)
        ]
        rows = MultiPersonPipeline(DetectionConfig(), Engine()).update(
            0,
            SimpleNamespace(face_landmarks=[points]),
            640,
            480,
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        self.assertEqual(rows[0]["authenticity"], "uncertain")
        self.assertEqual(rows[0]["authenticity_reason"], "model_inference_error")
