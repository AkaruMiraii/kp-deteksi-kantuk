import unittest


class TestActiveChallenge(unittest.TestCase):
    def test_random_sequence_requires_neutral_return_and_high_passive_score(self):
        from scripts.active_challenge import ActiveChallenge

        c = ActiveChallenge(sequence=["left", "right"], hold_seconds=0.2)
        self.assertEqual(c.update(0, "left", 0.9)["authenticity"], "uncertain")
        self.assertEqual(c.update(300, "left", 0.9)["challenge_step"], 1)
        c.update(400, "right", 0.9)
        self.assertEqual(c.update(700, "right", 0.9)["challenge_step"], 1)
        c.update(800, "center", 0.9)
        c.update(900, "right", 0.9)
        self.assertEqual(c.update(1200, "right", 0.9)["authenticity"], "live")
        self.assertEqual(c.update(1300, "center", 0.1)["authenticity"], "spoof")
        self.assertEqual(c.update(1400, "center", None)["authenticity"], "uncertain")

    def test_success_expires_and_invalid_direction_does_not_authorize(self):
        from scripts.active_challenge import ActiveChallenge

        c = ActiveChallenge(sequence=["left"], hold_seconds=0.2)
        c.update(0, "left", 0.9)
        self.assertEqual(c.update(300, "left", 0.9)["authenticity"], "live")
        self.assertEqual(c.update(400, None, 0.9)["authenticity"], "uncertain")
        c = ActiveChallenge(sequence=["left"], hold_seconds=0.2)
        c.update(0, "left", 0.9)
        c.update(300, "left", 0.9)
        for t in range(700, 15700, 400):
            result = c.update(t, "center", 0.9)
        self.assertEqual(result["authenticity"], "uncertain")
