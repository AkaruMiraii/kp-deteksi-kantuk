import unittest

import numpy as np

from scripts.enrolled_identity import EnrolledIdentity


class TestEnrolledIdentity(unittest.TestCase):
    def engine(self):
        obj = EnrolledIdentity.__new__(EnrolledIdentity)
        obj.gallery, obj.bindings, obj.templates, obj.pending = {}, {}, {}, {}
        obj.started, obj.next_id, obj.enrollment_seconds = None, 1, 60
        obj.restored_from = {}
        return obj

    def test_enroll_then_return_requires_temporal_evidence(self):
        obj = self.engine()
        a = np.array([1.0, 0.0])
        for i in range(61):
            result = obj.resolve_at(i * 100, [1], [a])
        self.assertEqual(result, [1])
        obj.resolve_at(7000, [], [])
        self.assertEqual(obj.resolve_at(8000, [2], [a]), [-2])
        for i in range(81, 92):
            result = obj.resolve_at(i * 100, [2], [a])
        self.assertEqual(result, [1])

    def test_locked_unknown_does_not_enroll(self):
        obj = self.engine()
        obj.resolve_at(0, [], [])
        for i in range(610, 680):
            result = obj.resolve_at(i * 100, [9], [np.array([0.0, 1.0])])
        self.assertEqual(result, [-9])
        self.assertEqual(obj.gallery, {})
        self.assertEqual(obj.confirm(9, 0, 68000), 1)

    def test_no_feature_never_enrolls(self):
        obj = self.engine()
        for i in range(100):
            result = obj.resolve_at(i * 100, [1], [None])
        self.assertEqual(result, [-1])
        self.assertEqual(obj.gallery, {})
