import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.smart_yawn import SmartYawn, TemporalFeatures


class TestSmartYawn(unittest.TestCase):
    def test_history_is_causal_and_invalid_resets(self):
        features = TemporalFeatures()
        a = features.update(0, 0.2, 0.2, 0.1)
        self.assertIsNotNone(a)
        b = features.update(300, 0.1, 0.1, 0.6)
        self.assertGreater(b[0], a[0])
        self.assertGreater(b[3], a[3])
        self.assertIsNone(features.update(400, None, None, None))
        c = features.update(450, 0.2, 0.2, 0.1)
        self.assertEqual(c[3], 0)
        with self.assertRaises(ValueError):
            features.update(450, 0.2, 0.2, 0.1)

    def test_json_model_candidate_requires_persistence(self):
        model = dict(
            schema_version=1,
            feature_names=TemporalFeatures.names,
            mean=[0] * len(TemporalFeatures.names),
            scale=[1] * len(TemporalFeatures.names),
            coefficients=[0] * len(TemporalFeatures.names),
            intercept=10,
            score_threshold=0.5,
            persistence_seconds=0.4,
            target="draft_visual_yawn_candidate",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(model))
            predictor = SmartYawn(path)
            self.assertEqual(
                predictor.update(0, 0.2, 0.2, 0.7)["smart_yawn_state"], "candidate_pending"
            )
            self.assertEqual(
                predictor.update(400, 0.2, 0.2, 0.7)["smart_yawn_state"], "yawn_candidate"
            )
            self.assertEqual(
                predictor.update(500, None, None, None)["smart_yawn_state"], "not_assessable"
            )
            self.assertEqual(
                predictor.update(600, 0.2, 0.2, 0.7)["smart_yawn_state"], "candidate_pending"
            )


if __name__ == "__main__":
    unittest.main()
