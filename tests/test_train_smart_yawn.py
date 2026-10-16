import unittest

from scripts.train_smart_yawn import label_at


class TestTrainingLabels(unittest.TestCase):
    def test_unknown_and_occluded_are_not_negative(self):
        intervals = [
            dict(start_frame=10, end_frame=20, label="yawn_like_candidate_wide"),
            dict(start_frame=30, end_frame=40, label="mouth_occluded"),
            dict(start_frame=50, end_frame=60, label="small_or_closed"),
        ]
        self.assertEqual(label_at(15, intervals), 1)
        self.assertIsNone(label_at(25, intervals))
        self.assertIsNone(label_at(35, intervals))
        self.assertEqual(label_at(55, intervals), 0)

    def test_interval_edges_are_excluded_from_fit(self):
        intervals = [dict(start_frame=10, end_frame=100, label="yawn_like_candidate_wide")]
        self.assertIsNone(label_at(10, intervals, margin=10))
        self.assertEqual(label_at(50, intervals, margin=10), 1)
        self.assertIsNone(label_at(95, intervals, margin=10))


if __name__ == "__main__":
    unittest.main()
