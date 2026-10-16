import unittest

import numpy as np

from scripts.session_identity import SessionIdentity


class TestIdentityStability(unittest.TestCase):
    def engine(self):
        engine = SessionIdentity.__new__(SessionIdentity)
        engine.gallery, engine.bindings, engine.next_id = {}, {}, 1
        return engine

    def test_single_bad_frame_does_not_split_continuous_identity(self):
        engine = self.engine()
        a, b = np.array([1., 0.]), np.array([0., 1.])
        self.assertEqual(engine.resolve([1], [a]), [1])
        self.assertEqual(engine.resolve([1], [b]), [1])
        self.assertEqual(engine.resolve([1], [a]), [1])

    def test_persistent_different_face_breaks_binding(self):
        engine = self.engine()
        engine.resolve([1], [np.array([1., 0.])])
        for _ in range(8):
            result = engine.resolve([1], [np.array([0., 1.])])
        self.assertNotEqual(result, [1])

    def test_corrupt_feature_cannot_create_template(self):
        engine = self.engine()
        engine.resolve([1], [np.array([float('nan'), 0.])])
        self.assertEqual(engine.gallery, {})
