import unittest

import numpy as np


class TestSpoofCrop(unittest.TestCase):
    def test_detector_crop_is_matched_to_person_not_largest_face(self):
        from scripts.face_authenticity import match_detection

        detections = np.array(
            [[0, 1, 0.95, 0.1, 0.1, 0.3, 0.4], [0, 1, 0.99, 0.65, 0.1, 0.85, 0.4]]
        )
        self.assertEqual(match_detection(detections, (0.1, 0.1, 0.3, 0.4)), (0.1, 0.1, 0.3, 0.4))
        self.assertIsNone(match_detection(detections, (0.4, 0.6, 0.5, 0.8)))
