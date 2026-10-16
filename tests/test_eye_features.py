import unittest

from scripts.eye_features import calculate_ear


class TestEyeAspectRatio(unittest.TestCase):
    def test_known_geometry_returns_half(self):
        points = [
            (0.0, 0.0),  # P1
            (1.0, 1.0),  # P2
            (3.0, 1.0),  # P3
            (4.0, 0.0),  # P4
            (3.0, -1.0),  # P5
            (1.0, -1.0),  # P6
        ]

        result = calculate_ear(points)

        self.assertEqual(result, 0.5)

    def test_smaller_eye_opening_returns_lower_ear(self):
        points = [
            (0.0, 0.0),  # P1
            (1.0, 0.2),  # P2
            (3.0, 0.2),  # P3
            (4.0, 0.0),  # P4
            (3.0, -0.2),  # P5
            (1.0, -0.2),  # P6
        ]

        result = calculate_ear(points)

        self.assertAlmostEqual(result, 0.1)
        self.assertLess(result, 0.5)

    def test_fully_closed_eye_returns_zero(self):
        points = [
            (0.0, 0.0),  # P1
            (1.0, 0.0),  # P2
            (3.0, 0.0),  # P3
            (4.0, 0.0),  # P4
            (3.0, 0.0),  # P5
            (1.0, 0.0),  # P6
        ]

        result = calculate_ear(points)

        self.assertAlmostEqual(result, 0.0)

    def test_zero_eye_width_is_rejected(self):
        points = [
            (0.0, 0.0),  # P1
            (1.0, 1.0),  # P2
            (3.0, 1.0),  # P3
            (0.0, 0.0),  # P4 berimpit dengan P1
            (3.0, -1.0),  # P5
            (1.0, -1.0),  # P6
        ]

        with self.assertRaisesRegex(ValueError, "Lebar mata"):
            calculate_ear(points)


if __name__ == "__main__":
    unittest.main()
