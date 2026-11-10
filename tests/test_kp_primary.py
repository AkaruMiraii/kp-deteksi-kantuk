import unittest
from scripts.detection_config import DetectionConfig
from scripts.person_state import PersonState


class PrimaryDecisionTests(unittest.TestCase):
    def test_primary_eye_decision_does_not_depend_on_mouth(self):
        state = PersonState(1, DetectionConfig())
        for timestamp in (0, 400, 800, 1200, 1600):
            row = state.update(timestamp, 0.05, 0.05, None, True, 'live')
        self.assertEqual(row['primary_status'], 'indikasi_kantuk')
        self.assertEqual(row['primary_trigger'], 'eye')

    def test_untrusted_input_is_not_negative_drowsiness(self):
        state = PersonState(1, DetectionConfig())
        row = state.update(0, 0.3, 0.3, 0.2, True, 'uncertain')
        self.assertEqual(row['primary_status'], 'tidak_dapat_dinilai')

    def test_mouth_alone_is_not_primary_positive(self):
        state = PersonState(1, DetectionConfig())
        for timestamp in range(0, 2401, 400):
            row = state.update(timestamp, 0.3, 0.3, 0.9, True, 'live')
        self.assertEqual(row['primary_status'], 'tidak_terdeteksi_indikasi_kantuk')
