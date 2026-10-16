import unittest

import numpy as np


class TestMultiDisplay(unittest.TestCase):
    def test_side_panel_preserves_input_and_has_room_for_two_people(self):
        from scripts.multi_display import render_multi

        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        original = frame.copy()
        rows = [
            dict(
                track_id=i,
                box=(0.1, 0.1, 0.3, 0.4),
                points_px={},
                authenticity="uncertain",
                ear_r=0.3,
                ear_l=0.3,
                mar=0.1,
                closed_seconds=0,
                focus_indicator="unknown",
                microsleep_indicator="unknown",
                zone_out="unknown",
                challenge_request="left",
            )
            for i in (1, 2)
        ]
        output = render_multi(frame, rows)
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(output.shape, (762, 1320, 3))
        from scripts.detection_config import DetectionConfig

        selected = render_multi(frame, rows, selected_id=2, config=DetectionConfig())
        self.assertFalse(np.array_equal(output, selected))
        invalid = [
            dict(
                rows[0],
                observation_quality="invalid",
                observation_reason="implausible_eye_geometry",
            )
        ]
        good = [dict(rows[0], observation_quality="valid", observation_reason="")]
        from scripts.multi_display import MonitorUI

        invalid_ui, good_ui = MonitorUI(), MonitorUI()
        invalid_ui.debug = good_ui.debug = True
        invalid_ui.update(invalid)
        good_ui.update(good)
        self.assertFalse(
            np.array_equal(
                invalid_ui.render(frame, invalid, selected_id=1),
                good_ui.render(frame, good, selected_id=1),
            )
        )
