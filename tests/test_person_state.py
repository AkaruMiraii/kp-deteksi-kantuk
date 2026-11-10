import unittest

from scripts.detection_config import DetectionConfig


class TestPersonState(unittest.TestCase):
    def test_individual_timers_and_unknown_authenticity_gate(self):
        from scripts.person_state import PersonState

        a = PersonState(1, DetectionConfig())
        b = PersonState(2, DetectionConfig())
        a.update(0, 0.05, 0.05, 0.1, True, "live")
        b.update(0, 0.3, 0.3, 0.1, True, "live")
        for t in (400, 800, 1200, 1600):
            x = a.update(t, 0.05, 0.05, 0.1, True, "live")
            y = b.update(t, 0.3, 0.3, 0.1, True, "live")
        self.assertEqual(x["microsleep_indicator"], "closure_candidate")
        self.assertEqual(y["closed_seconds"], 0)
        self.assertEqual(y["zone_out"], "unknown")
        z = a.update(2000, 0.05, 0.05, 0.1, True, "uncertain")
        self.assertEqual(z["microsleep_indicator"], "unknown")
        self.assertEqual(z["closed_seconds"], 0)
        a.reset()
        self.assertEqual(a.update(2400, 0.3, 0.3, 0.1, True, "live")["closed_seconds"], 0)

    def test_untrusted_face_keeps_raw_observation_timer_separate(self):
        from scripts.person_state import PersonState

        a = PersonState(1, DetectionConfig())
        for t in (0, 400, 800, 1200, 1600):
            row = a.update(t, 0.05, 0.05, 0.1, True, "uncertain")
        self.assertAlmostEqual(row["raw_closed_seconds"], 1.6)
        self.assertEqual(row["microsleep_indicator"], "unknown")
        self.assertEqual(row["closed_seconds"], 0)
