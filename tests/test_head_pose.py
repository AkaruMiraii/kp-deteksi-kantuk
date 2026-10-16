import unittest
from types import SimpleNamespace


class TestHeadPose(unittest.TestCase):
    def test_missing_and_frontal_landmarks(self):
        from scripts.head_pose import head_direction

        self.assertIsNone(head_direction([]))
        p = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        p[33] = SimpleNamespace(x=0.3, y=0.4)
        p[263] = SimpleNamespace(x=0.7, y=0.4)
        p[1] = SimpleNamespace(x=0.5, y=0.5)
        p[152] = SimpleNamespace(x=0.5, y=0.8)
        self.assertEqual(head_direction(p), "center")
        p[1].x = 0.65
        self.assertEqual(head_direction(p), "right")
