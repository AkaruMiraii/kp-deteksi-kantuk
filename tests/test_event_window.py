import unittest

from scripts.event_window import EventWindow


class TestEventWindow(unittest.TestCase):
    def test_sustained_event_counts_once_and_requires_reopening(self):
        tracker = EventWindow(0.5)
        for i in range(20):
            row = tracker.update(i * 0.1, "prolonged_closure", "prolonged_opening")
        self.assertEqual(row["eye_events_window"], 1)
        self.assertEqual(row["opening_events_window"], 1)
        tracker.update(2, "not_assessable", "not_assessable")
        row = tracker.update(2.1, "prolonged_closure", "prolonged_opening")
        self.assertEqual(row["eye_events_window"], 1)
        tracker.update(2.2, "no_bilateral_closure", "no_large_opening")
        row = tracker.update(2.3, "prolonged_closure", "prolonged_opening")
        self.assertEqual(row["eye_events_window"], 2)

    def test_combination_and_window_expiry(self):
        tracker = EventWindow(0.5)
        for i in range(401):
            t = i * 0.1
            eye = "prolonged_closure" if 10 <= t < 12 else "no_bilateral_closure"
            mouth = "prolonged_opening" if 15 <= t < 18 or 25 <= t < 28 else "no_large_opening"
            row = tracker.update(t, eye, mouth)
        self.assertEqual(row["visual_pattern"], "increased_visual_drowsiness")
        self.assertEqual(row["opening_events_window"], 2)
        row = tracker.update(340, "no_bilateral_closure", "no_large_opening")
        self.assertEqual(row["eye_events_window"], 0)
        self.assertEqual(row["opening_events_window"], 0)
        self.assertEqual(row["visual_pattern"], "insufficient_observation")

    def test_individual_state_and_invalid_coverage(self):
        a, b = EventWindow(0.5), EventWindow(0.5)
        a.update(0, "prolonged_closure", "prolonged_opening")
        row = b.update(0, "not_assessable", "not_assessable")
        self.assertEqual(row["eye_events_window"], 0)
        self.assertEqual(row["eye_coverage"], 0)
