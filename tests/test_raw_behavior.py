import unittest

from scripts.detection_config import DetectionConfig
from scripts.person_state import PersonState


class TestRawBehavior(unittest.TestCase):
    def test_raw_mouth_survives_unverified_authenticity_without_bypass(self):
        state = PersonState(1, DetectionConfig())
        for i in range(26):
            row = state.update(i * 100, 0.2, 0.2, 0.7, True, "uncertain")
        self.assertEqual(row["raw_mouth_state"], "prolonged_opening")
        self.assertGreaterEqual(row["raw_mouth_seconds"], 2)
        self.assertEqual(row["primary_status"], "tidak_dapat_dinilai")
        self.assertEqual(row["authenticity"], "uncertain")
        self.assertEqual(row["mouth_state"], "not_assessable")
        self.assertEqual(row["raw_orientation_indicator"], "toward_camera")

    def test_invalid_observation_resets_raw_timers(self):
        state = PersonState(1, DetectionConfig())
        state.update(0, 0.05, 0.05, 0.7, True, "uncertain")
        state.update(400, 0.05, 0.05, 0.7, True, "uncertain")
        row = state.update(500, None, None, None, False, "uncertain")
        self.assertEqual(row["raw_mouth_seconds"], 0)
        self.assertEqual(row["raw_closed_seconds"], 0)
