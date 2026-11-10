import unittest

import numpy as np

from scripts.detection_config import DetectionConfig
from scripts.face_tracking import FaceTracker
from scripts.multi_display import render_multi
from scripts.multi_live import parse_args
from scripts.person_selection import PersonSelection


class TestSixFaces(unittest.TestCase):
    def test_capacity_and_selection(self):
        self.assertEqual(parse_args(["--max-people", "6"]).config.max_people, 6)
        with self.assertRaises(ValueError):
            DetectionConfig(max_people=7)
        rows = [dict(track_id=i) for i in range(1, 7)]
        selection = PersonSelection()
        self.assertEqual(selection.update(rows, ord("6")), 6)
        self.assertEqual(selection.update(list(reversed(rows)), -1), 6)

    def test_six_tracks_keep_ids_when_order_changes(self):
        boxes = [(i / 6, 0.1, i / 6 + 0.1, 0.3) for i in range(6)]
        tracker = FaceTracker()
        ids = tracker.update(0, boxes)
        self.assertEqual(len(set(ids)), 6)
        self.assertEqual(tracker.update(30, list(reversed(boxes))), list(reversed(ids)))

    def test_six_cards_are_bounded_and_source_is_unchanged(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rows = [
            dict(
                track_id=i,
                box=(0.1, 0.1, 0.2, 0.2),
                points_px={},
                authenticity="uncertain",
                ear_r=0.2,
                ear_l=0.2,
                mar=0.1,
                closed_seconds=0,
                focus_indicator="unknown",
                microsleep_indicator="unknown",
                challenge_request="center",
            )
            for i in range(1, 7)
        ]
        output = render_multi(frame, rows, selected_id=6, config=DetectionConfig(max_people=6))
        self.assertEqual(output.shape, (762, 1320, 3))
        self.assertFalse(frame.any())
        self.assertTrue(output[:400, 960:].any())
