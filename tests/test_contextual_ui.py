import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scripts.multi_display import STATUS_FRAMES, MonitorUI


def row():
    return dict(
        track_id=1,
        box=(0.2, 0.2, 0.5, 0.6),
        points_px={},
        primary_status="tidak_terdeteksi_indikasi_kantuk",
        observation_quality="valid",
        authenticity="live",
        head_direction="center",
        raw_eye_state="no_bilateral_closure",
        closed_seconds=0,
        baseline=None,
    )


class TestContextualUI(unittest.TestCase):
    def test_fade_and_hysteresis_do_not_change_input(self):
        now = [0.0]
        ui = MonitorUI(clock=lambda: now[0])
        normal = row()
        ui.update([normal])
        closed = dict(normal, primary_status="indikasi_kantuk")
        for _ in range(STATUS_FRAMES - 1):
            self.assertEqual(ui.update([closed])[0][2], "normal")
        self.assertEqual(ui.update([closed])[0][2], "closed")
        now[0] = 0.5
        self.assertAlmostEqual(ui.update([])[0][1], 0.5)
        now[0] = 1.1
        self.assertEqual(ui.update([]), [])
        self.assertEqual(normal["primary_status"], "tidak_terdeteksi_indikasi_kantuk")

    def test_empty_view_has_no_panel_and_help_toggle(self):
        ui = MonitorUI(clock=lambda: 20)
        output = ui.render(np.zeros((480, 640, 3), dtype=np.uint8), [])
        self.assertEqual(output.shape, (762, 960, 3))
        ui.handle_key(ord("h"))
        self.assertTrue(ui.debug)
        ui.handle_key(ord("?"))
        self.assertTrue(ui.help)

    def test_auto_calibration_requires_trusted_stable_evidence(self):
        now = [0.0]
        ui = MonitorUI(clock=lambda: now[0])
        state = SimpleNamespace(baseline=None, calibrate_from_history=Mock())
        pipeline = SimpleNamespace(states={1: state})
        sample = row()
        ui.update([sample])
        self.assertEqual(ui.auto_calibrate([sample], pipeline), [])
        now[0] = 3.1
        ui.update([sample])
        self.assertEqual(ui.auto_calibrate([sample], pipeline), [])
        now[0] = 6.2
        ui.update([sample])
        ui.auto_calibrate([sample], pipeline)
        # Observation gaps must restart calibration, not bridge missing frames.
        state.calibrate_from_history.assert_not_called()
