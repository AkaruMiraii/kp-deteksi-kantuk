import unittest

from scripts.detection_config import DetectionConfig
from scripts.person_state import PersonState


class TestAttentionBaseline(unittest.TestCase):
    def test_explicit_open_eye_baseline_and_independent_label(self):
        p = PersonState(1, DetectionConfig())
        p.calibrate_open_eyes(0.3, 0.32, source="user_confirmed_open_eyes")
        self.assertEqual(p.baseline, (0.3, 0.32))
        self.assertAlmostEqual(p.eyes.threshold, 0.195)
        p.set_zone_out_label(True, source="self_report")
        result = p.update(0, 0.3, 0.32, 0.1, True, "live")
        self.assertEqual(result["zone_out"], "self_reported")
        self.assertEqual(result["zone_out_label_source"], "self_report")
        p.reset()
        self.assertIsNone(p.baseline)
        with self.assertRaises(ValueError):
            p.set_zone_out_label(True, source="gaze_away")

    def test_calibration_uses_causal_open_eye_window_not_one_frame(self):
        p = PersonState(1, DetectionConfig())
        p.update(0, 0.3, 0.32, 0.1, True, "live")
        with self.assertRaises(ValueError):
            p.calibrate_from_history()
        for t in range(400, 3200, 400):
            p.update(t, 0.3, 0.32, 0.1, True, "live")
        p.calibrate_from_history()
        self.assertEqual(p.baseline, (0.3, 0.32))
        self.assertEqual(len(p.history), 0)
        with self.assertRaises(ValueError):
            p.calibrate_open_eyes(0.05, 0.3, "user_confirmed_open_eyes")
