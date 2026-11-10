import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.smart_live import process_smart_frame


class TestSmartLive(unittest.TestCase):
    def test_invalid_geometry_is_not_given_to_predictor(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        received = []
        predictor = SimpleNamespace(
            update=lambda *args: (
                received.append(args)
                or dict(
                    smart_yawn_state="not_assessable",
                    smart_yawn_score=None,
                    smart_yawn_duration_seconds=0.0,
                )
            )
        )
        base = dict(
            reason="selected_points_outside_image",
            ear_r=0.1,
            ear_l=0.1,
            mar=0.8,
            final_status="tidak_dapat_dinilai",
        )
        with patch("scripts.smart_live.process_frame", return_value=(base, frame.copy())):
            row, overlay = process_smart_frame(frame, None, 10, predictor, None, None)
        self.assertEqual(received, [(10, None, None, None)])
        self.assertEqual(row["smart_yawn_state"], "not_assessable")
        self.assertGreater(np.count_nonzero(overlay), 0)
        self.assertGreater(overlay.shape[1], frame.shape[1])
        np.testing.assert_array_equal(overlay[16:496, 16:656], frame)

    def test_candidate_does_not_rewrite_eye_decision(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        predictor = SimpleNamespace(
            update=lambda *args: dict(
                smart_yawn_state="yawn_candidate",
                smart_yawn_score=0.8,
                smart_yawn_duration_seconds=0.5,
            )
        )
        base = dict(
            reason="",
            ear_r=0.2,
            ear_l=0.2,
            mar=0.7,
            final_status="tidak_terdeteksi_indikasi_kantuk",
        )
        with patch("scripts.smart_live.process_frame", return_value=(base, frame.copy())):
            row, _ = process_smart_frame(frame, None, 10, predictor, None, None)
        self.assertEqual(row["final_status"], "tidak_terdeteksi_indikasi_kantuk")
        self.assertEqual(row["smart_yawn_state"], "yawn_candidate")


if __name__ == "__main__":
    unittest.main()
