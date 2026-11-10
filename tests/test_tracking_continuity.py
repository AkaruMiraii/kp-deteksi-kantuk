import unittest

from scripts.face_tracking import FaceTracker


class TestTrackingContinuity(unittest.TestCase):
    def test_brief_dropout_retains_unique_stationary_track(self):
        tracker = FaceTracker()
        box = (0.1, 0.1, 0.3, 0.4)
        self.assertEqual(tracker.update(0, [box]), [1])
        tracker.update(100, [])
        self.assertEqual(tracker.update(200, [box]), [1])

    def test_expired_track_is_not_reidentified(self):
        tracker = FaceTracker()
        box = (0.1, 0.1, 0.3, 0.4)
        tracker.update(0, [box])
        tracker.update(100, [])
        self.assertNotEqual(tracker.update(600, [box]), [1])

    def test_dropout_matching_requires_high_overlap(self):
        tracker = FaceTracker()
        tracker.update(0, [(0.1, 0.1, 0.3, 0.4)])
        tracker.update(100, [])
        self.assertNotEqual(tracker.update(200, [(0.2, 0.1, 0.4, 0.4)]), [1])
