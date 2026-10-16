import unittest


class TestFaceTracking(unittest.TestCase):
    def test_order_changes_and_ambiguity_never_reuse_uncertain_identity(self):
        from scripts.face_tracking import FaceTracker

        t = FaceTracker()
        a, b = (0.1, 0.1, 0.2, 0.3), (0.7, 0.1, 0.8, 0.3)
        self.assertEqual(t.update(0, [a, b]), [1, 2])
        self.assertEqual(t.update(100, [b, a]), [2, 1])
        self.assertEqual(t.update(200, []), [])
        ids = t.update(300, [a, b])
        self.assertEqual(ids, [1, 2])
        overlap = (0.4, 0.1, 0.6, 0.3)
        ids = t.update(400, [overlap, overlap])
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(set(ids).isdisjoint(t.update(500, [overlap, overlap])))
        with self.assertRaises(ValueError):
            t.update(500, [])
