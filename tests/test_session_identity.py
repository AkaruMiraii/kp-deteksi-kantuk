import unittest

import numpy as np

from scripts.session_identity import SessionIdentity


class TestSessionIdentity(unittest.TestCase):
    def gallery(self):
        engine = SessionIdentity.__new__(SessionIdentity)
        engine.gallery, engine.bindings, engine.next_id = {}, {}, 1
        return engine

    def test_return_reuses_id_but_other_face_does_not(self):
        engine = self.gallery()
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        self.assertEqual(engine.resolve([1], [a]), [1])
        engine.resolve([], [])
        self.assertEqual(engine.resolve([9], [a]), [1])
        engine.resolve([], [])
        self.assertEqual(engine.resolve([10], [b]), [2])

    def test_same_template_cannot_authorize_two_simultaneous_tracks(self):
        engine = self.gallery()
        a = np.array([1.0, 0.0])
        ids = engine.resolve([1, 2], [a, a])
        self.assertEqual(len(set(ids)), 2)

    def test_missing_features_do_not_match_old_id(self):
        engine = self.gallery()
        engine.resolve([1], [np.array([1.0, 0.0])])
        engine.resolve([], [])
        self.assertEqual(engine.resolve([2], [None]), [2])
