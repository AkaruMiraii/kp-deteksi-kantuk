import unittest

from scripts.mouth_duration import MouthOpeningTracker


class TestMouthDuration(unittest.TestCase):
    def test_short_opening_stays_candidate_and_sustained_opening_triggers(self):
        tracker = MouthOpeningTracker(0.5, 2.0, 0.5)
        self.assertEqual(tracker.update(0.0, 0.6).state, "opening_candidate")
        for timestamp in (0.5, 1.0, 1.5):
            self.assertEqual(tracker.update(timestamp, 0.6).state, "opening_candidate")
        result = tracker.update(2.0, 0.6)
        self.assertEqual(result.state, "prolonged_opening")
        self.assertEqual(result.duration_seconds, 2.0)
        self.assertEqual(tracker.update(2.1, 0.1).state, "no_large_opening")
        self.assertEqual(tracker.update(2.2, 0.6).duration_seconds, 0.0)

    def test_invalid_observation_resets_timer(self):
        tracker = MouthOpeningTracker(0.5, 2.0, 0.5)
        tracker.update(0.0, 0.6)
        tracker.update(0.5, 0.6)
        for time, value in [(0.6, None), (0.7, float("nan")), (0.8, -0.1)]:
            self.assertEqual(tracker.update(time, value).state, "not_assessable")
        self.assertEqual(tracker.update(0.9, 0.6).duration_seconds, 0.0)

    def test_observation_gap_resets_timer(self):
        tracker = MouthOpeningTracker(0.5, 2.0, 0.5)
        tracker.update(0.0, 0.6)
        self.assertEqual(tracker.update(2.0, 0.6).duration_seconds, 0.0)

    def test_threshold_equality_does_not_start_timer(self):
        self.assertEqual(
            MouthOpeningTracker(0.5, 2.0, 0.5).update(0.0, 0.5).state, "no_large_opening"
        )

    def test_invalid_parameters_and_timestamps_are_rejected(self):
        for params in [(0, 2, 0.5), (0.5, 0, 0.5), (0.5, 2, 0), (float("inf"), 2, 0.5)]:
            with self.subTest(parameters=params), self.assertRaises(ValueError):
                MouthOpeningTracker(*params)
        tracker = MouthOpeningTracker(0.5, 2.0, 0.5)
        tracker.update(1.0, 0.6)
        for timestamp in (1.0, 0.5, -1.0, float("nan")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                tracker.update(timestamp, 0.6)
        self.assertAlmostEqual(tracker.update(1.2, 0.6).duration_seconds, 0.2)


if __name__ == "__main__":
    unittest.main()
