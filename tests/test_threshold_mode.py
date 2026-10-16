import unittest

from scripts.detection_config import DetectionConfig
from scripts.multi_live import parse_args
from scripts.person_state import PersonState


class TestThresholdMode(unittest.TestCase):
    def test_explicit_mode_preserves_unverified_authenticity(self):
        self.assertFalse(parse_args([]).threshold_only)
        state = PersonState(1, DetectionConfig())
        for i in range(21):
            row = state.update(i * 100, 0.05, 0.05, 0.1, True, "not_checked", threshold_only=True)
        self.assertEqual(row["primary_status"], "indikasi_kantuk")
        self.assertEqual(row["authenticity"], "not_checked")
        self.assertFalse(row["authenticity_required"])
        row = state.update(2200, None, None, None, False, "not_checked", threshold_only=True)
        self.assertEqual(row["primary_status"], "tidak_dapat_dinilai")
        self.assertEqual(row["closed_seconds"], 0)

    def test_default_keeps_authenticity_gate(self):
        state = PersonState(1, DetectionConfig())
        for i in range(21):
            row = state.update(i * 100, 0.05, 0.05, 0.1, True, "uncertain")
        self.assertEqual(row["primary_status"], "tidak_dapat_dinilai")
        self.assertTrue(row["authenticity_required"])
