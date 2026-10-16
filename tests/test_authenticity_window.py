import unittest


class TestAuthenticityWindow(unittest.TestCase):
    def test_temporal_score_requires_support_and_resets_on_gap_or_unknown(self):
        from scripts.authenticity_window import AuthenticityWindow

        window = AuthenticityWindow()
        self.assertIsNone(window.update(0, 0.9))
        self.assertIsNone(window.update(100, 0.9))
        self.assertIsNone(window.update(200, 0.9))
        self.assertIsNone(window.update(300, 0.9))
        self.assertEqual(window.update(400, 0.9), 0.9)
        self.assertIsNone(window.update(1000, 0.9))
        self.assertIsNone(window.update(1100, None))
        self.assertEqual(len(window.samples), 0)
        with self.assertRaises(ValueError):
            window.update(1100, 0.9)
