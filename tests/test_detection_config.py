import unittest


class TestDetectionConfig(unittest.TestCase):
    def test_validated_configuration_rejects_nonfinite_and_invalid_ranges(self):
        from scripts.detection_config import DetectionConfig

        config = DetectionConfig()
        self.assertEqual(config.max_people, 2)
        self.assertEqual(config.provenance, "experimental_user_configuration")
        for kwargs in (
            {"ear_threshold": float("nan")},
            {"mar_threshold": -1},
            {"closure_seconds": 0},
            {"max_gap_seconds": float("inf")},
            {"max_people": 0},
            {"max_people": 2.5},
            {"max_people": True},
            {"focus_seconds": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DetectionConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
