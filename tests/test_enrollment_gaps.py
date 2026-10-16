import numpy as np

from tests.test_enrolled_identity import TestEnrolledIdentity


class TestEnrollmentGaps(TestEnrolledIdentity):
    def test_short_invalid_gap_excludes_missing_evidence(self):
        obj = self.engine()
        a = np.array([1.0, 0.0])
        for t in range(0, 3000, 100):
            obj.resolve_at(t, [1], [a])
        for t in (3000, 3100):
            self.assertEqual(obj.resolve_at(t, [1], [None]), [-1])
        for t in range(3200, 5300, 100):
            result = obj.resolve_at(t, [1], [a])
        self.assertEqual(result, [-1])  # only 4.9s supported intervals
        self.assertEqual(obj.resolve_at(5400, [1], [a]), [1])

    def test_long_invalid_gap_resets(self):
        obj = self.engine()
        a = np.array([1.0, 0.0])
        for t in range(0, 4900, 100):
            obj.resolve_at(t, [1], [a])
        for t in range(4900, 5600, 100):
            obj.resolve_at(t, [1], [None])
        self.assertEqual(obj.resolve_at(5600, [1], [a]), [-1])
        self.assertLessEqual(len(obj.pending[1]["samples"]), 1)

    def test_mismatch_resets_samples(self):
        obj = self.engine()
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        for t in range(0, 4900, 100):
            obj.resolve_at(t, [1], [a])
        obj.resolve_at(4900, [1], [None])
        self.assertEqual(obj.resolve_at(5000, [1], [b]), [-1])
        self.assertEqual(len(obj.pending[1]["samples"]), 1)

    def test_track_change_discards_evidence(self):
        obj = self.engine()
        a = np.array([1.0, 0.0])
        for t in range(0, 4900, 100):
            obj.resolve_at(t, [1], [a])
        self.assertEqual(obj.resolve_at(4900, [2], [a]), [-2])
        self.assertNotIn(1, obj.pending)
        self.assertEqual(len(obj.pending[2]["samples"]), 1)
