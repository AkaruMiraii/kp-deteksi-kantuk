import unittest

import numpy as np


class TestAuthenticity(unittest.TestCase):
    def test_model_failure_and_small_crop_remain_uncertain(self):
        from scripts.face_authenticity import FaceAuthenticity

        engine = FaceAuthenticity()
        output = engine.predict(np.zeros((480, 640, 3), dtype=np.uint8), (0.1, 0.1, 0.11, 0.11))
        self.assertEqual(output["authenticity"], "uncertain")
        output = engine.predict(np.zeros((480, 640, 3), dtype=np.uint8), (0.1, 0.1, 0.4, 0.5))
        self.assertEqual(output["authenticity"], "uncertain")
        self.assertEqual(output["authenticity_reason"], "model_unavailable")

    def test_landmark_fallback_crop_cannot_authorize_authenticity(self):
        from unittest.mock import Mock

        from scripts.face_authenticity import FaceAuthenticity

        engine = FaceAuthenticity()
        engine.model = Mock()
        result = engine.predict(np.zeros((480, 640, 3), dtype=np.uint8), (0.1, 0.1, 0.4, 0.5))
        self.assertEqual(result["authenticity"], "uncertain")
        self.assertEqual(result["authenticity_reason"], "reference_crop_detector_unavailable")
        engine.model.assert_not_called()
