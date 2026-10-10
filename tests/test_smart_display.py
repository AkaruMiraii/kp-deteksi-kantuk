import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.smart_display import render_dashboard


class TestSmartDisplay(unittest.TestCase):
    def test_camera_pixels_remain_unmodified_and_metrics_are_outside(self):
        raw = np.full((480, 640, 3), 91, dtype=np.uint8)
        before = raw.copy()
        record = dict(
            eye_alertness_state="no_eye_indicator",
            eye_closed_duration_seconds=0,
            independent_ear_r=0.2,
            independent_ear_l=0.22,
            mar=0.01,
            smart_yawn_state="no_candidate",
            smart_yawn_score=0.1,
            smart_yawn_duration_seconds=0,
            processing_ms=9,
        )
        with patch("scripts.smart_display.cv2.putText", wraps=cv2.putText) as text:
            panel = render_dashboard(raw, record)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_array_equal(panel[16:496, 16:656], before)
        self.assertGreater(panel.shape[1], 640)
        for call in text.call_args_list:
            x, y = call.args[2]
            self.assertTrue(x >= 672 or y >= 512)

    def test_calibration_and_uncalibrated_score_are_explicit(self):
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        row = dict(
            eye_alertness_state="calibrating_open_eyes",
            smart_yawn_state="candidate_pending",
            smart_yawn_score=0.8,
        )
        with patch("scripts.smart_display.cv2.putText", wraps=cv2.putText) as text:
            render_dashboard(raw, row)
        strings = [call.args[1] for call in text.call_args_list]
        self.assertIn("KALIBRASI MATA", strings)
        self.assertIn("Buka mata alami selama 5 detik", strings)
        self.assertIn("Skor model, bukan probabilitas", strings)

    def test_small_camera_and_missing_values_still_render(self):
        raw = np.zeros((120, 160, 3), dtype=np.uint8)
        row = dict(
            final_status="tidak_dapat_dinilai",
            smart_yawn_state="not_assessable",
            smart_yawn_score=None,
        )
        with patch("scripts.smart_display.cv2.putText", wraps=cv2.putText) as text:
            panel = render_dashboard(raw, row)
        self.assertGreaterEqual(panel.shape[0], 600)
        self.assertIn("TIDAK DAPAT DINILAI", [call.args[1] for call in text.call_args_list])

    def test_landmark_contours_return_without_changing_source(self):
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        points = {
            str(index): [100 + i * 6, 120 + (i % 2) * 5]
            for i, index in enumerate(
                [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 78, 308, 13, 14]
            )
        }
        row = dict(
            final_status="tidak_terdeteksi_indikasi_kantuk",
            smart_yawn_state="no_candidate",
            ear_r=0.2,
            ear_l=0.2,
            mar=0.1,
            points_px=points,
        )
        panel = render_dashboard(raw, row)
        self.assertGreater(np.count_nonzero(panel[16:496, 16:656]), 0)
        np.testing.assert_array_equal(raw, np.zeros_like(raw))
        row.update(ear_r=None, ear_l=None, mar=None)
        invalid = render_dashboard(raw, row)
        np.testing.assert_array_equal(invalid[16:496, 16:656], raw)


if __name__ == "__main__":
    unittest.main()
