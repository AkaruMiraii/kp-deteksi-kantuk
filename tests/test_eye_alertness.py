import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.eye_alertness import EyeAlertness, EyeAwareDetector, assess_eyes


class TestEyeAlertness(unittest.TestCase):
    def test_closed_mouth_is_irrelevant_and_tiers_reset_on_reopening(self):
        tracker = EyeAlertness(calibration_seconds=1)
        for ts in (0, 250, 500, 750, 1000):
            state = tracker.update(ts, 0.3, 0.28)
        self.assertEqual(state["eye_alertness_state"], "no_eye_indicator")
        self.assertLess(state["eye_threshold_right"], 0.3)
        for ts in range(1250, 9751, 250):
            state = tracker.update(ts, 0.04, 0.04)
            if ts == 2750:
                self.assertEqual(state["eye_alertness_state"], "drowsiness_visual_indicator")
            if ts == 4250:
                self.assertEqual(state["eye_alertness_state"], "near_sleep_visual_indicator")
        self.assertEqual(state["eye_alertness_state"], "sleep_like_visual_indicator")
        state = tracker.update(10000, 0.3, 0.28)
        self.assertEqual(state["eye_alertness_state"], "no_eye_indicator")
        self.assertEqual(state["eye_closed_duration_seconds"], 0)

    def test_invalid_eyes_or_gap_do_not_accumulate_closure(self):
        tracker = EyeAlertness(calibration_seconds=1)
        for ts in (0, 250, 500, 750, 1000):
            tracker.update(ts, 0.3, 0.3)
        tracker.update(1250, 0.04, 0.04)
        state = tracker.update(1500, None, None)
        self.assertEqual(state["eye_alertness_state"], "not_assessable")
        self.assertEqual(tracker.update(1750, 0.04, 0.04)["eye_closed_duration_seconds"], 0)
        self.assertEqual(tracker.update(3000, 0.04, 0.04)["eye_closed_duration_seconds"], 0)

    def test_invalid_mouth_landmarks_do_not_invalidate_eye_geometry(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        for indices in ([33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]):
            for index, (x, y) in zip(
                indices,
                [(0.3, 0.4), (0.35, 0.38), (0.45, 0.38), (0.5, 0.4), (0.45, 0.42), (0.35, 0.42)],
            ):
                points[index] = SimpleNamespace(x=x, y=y)
        points[13] = SimpleNamespace(x=2, y=2)
        right_ear, left_ear, reason = assess_eyes(
            SimpleNamespace(face_landmarks=[points]), 640, 480
        )
        self.assertEqual(reason, "")
        self.assertGreater(right_ear, 0)
        self.assertGreater(left_ear, 0)

    def test_invalid_eye_geometry_is_not_assessable(self):
        self.assertEqual(
            assess_eyes(SimpleNamespace(face_landmarks=[]), 640, 480)[2], "no_face_detected"
        )
        with self.assertRaises(ValueError):
            EyeAlertness(calibration_seconds=0)

    def test_eye_mode_cli_is_opt_in(self):
        import scripts.smart_live as live

        with patch("sys.argv", ["smart", "--eye-alertness"]), patch.object(live, "run") as run:
            live.main()
        self.assertTrue(run.call_args.args[0].eye_alertness)

    def test_proxy_uses_one_inference_and_eye_status_not_mouth(self):
        calls = []
        result = SimpleNamespace(face_landmarks=[])
        detector = SimpleNamespace(detect_for_video=lambda image, ts: calls.append(ts) or result)
        proxy = EyeAwareDetector(detector, EyeAlertness(calibration_seconds=1))
        self.assertIs(proxy.detect_for_video(None, 0), result)
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("scripts.eye_alertness.assess_eyes", return_value=(0.3, 0.3, "")):
            record, overlay = proxy.add_eye_status(raw, {"reason": "invalid_mouth"}, raw.copy(), 0)
        self.assertEqual(calls, [0])
        self.assertEqual(record["independent_eye_reason"], "")
        self.assertEqual(record["eye_alertness_state"], "calibrating_open_eyes")
        self.assertGreater(np.count_nonzero(overlay), 0)
        np.testing.assert_array_equal(raw, np.zeros_like(raw))


if __name__ == "__main__":
    unittest.main()
