import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.multi_display import MonitorUI


class TestUIRegressions(unittest.TestCase):
    def test_unassessable_is_never_normal(self):
        self.assertEqual(
            MonitorUI.status(
                dict(
                    primary_status="tidak_dapat_dinilai",
                    authenticity="live",
                    observation_quality="valid",
                )
            ),
            "attention",
        )

    def test_reappearing_track_does_not_reuse_pending_frames(self):
        now = [0.0]
        ui = MonitorUI(clock=lambda: now[0])
        normal = dict(
            track_id=1,
            primary_status="tidak_terdeteksi_indikasi_kantuk",
            authenticity="live",
            observation_quality="valid",
        )
        closed = dict(normal, primary_status="indikasi_kantuk")
        ui.update([normal])
        for _ in range(5):
            ui.update([closed])
        now[0] = 0.2
        ui.update([])
        now[0] = 0.3
        self.assertEqual(ui.update([closed])[0][2], "normal")

    def test_auto_baseline_logs_automatic_provenance(self):
        now = [0.0]
        ui = MonitorUI(clock=lambda: now[0])
        state = SimpleNamespace(baseline=None, calibrate_from_history=Mock())
        sample = dict(
            track_id=1,
            observation_quality="valid",
            authenticity="live",
            head_direction="center",
            raw_eye_state="no_bilateral_closure",
        )
        pipeline = SimpleNamespace(states={1: state})
        for i in range(33):
            now[0] = i / 10
            ui.update([sample])
            ui.auto_calibrate([sample], pipeline)
        state.calibrate_from_history.assert_called_with(source="automatic_frontal_open_eye_proxy")
