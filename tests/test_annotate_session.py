import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.annotate_session import AnnotationEditor, get_frame, handle_key, run_gui


class TestAnnotationEditor(unittest.TestCase):
    def test_inclusive_interval_saves_separate_labels_and_source_hash(self):
        editor = AnnotationEditor([100, 140, 180, 220], "pilot", "video-hash", "mapping-hash")
        editor.seek(1)
        editor.mark_start()
        editor.seek(2)
        editor.mark_end()
        editor.commit("eyes_closed_both")
        with TemporaryDirectory() as directory:
            csv_path = editor.save(Path(directory), "reviewer")
            with csv_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["start_frame"], "1")
            self.assertEqual(rows[0]["end_frame"], "2")
            self.assertEqual(rows[0]["start_timestamp_ms"], "140")
            self.assertEqual(rows[0]["end_timestamp_ms"], "180")
            metadata = json.loads(csv_path.with_suffix(".json").read_text())
            self.assertEqual(metadata["video_sha256"], "video-hash")
            self.assertFalse(metadata["annotation_complete"])
            self.assertEqual(metadata["unreviewed_default"], "unknown")
            other = editor.save(Path(directory), "reviewer")
            self.assertNotEqual(other, csv_path)
            self.assertTrue(csv_path.exists())

    def test_overlap_rejected_in_same_channel_but_other_channel_allowed(self):
        editor = AnnotationEditor([100, 140, 180, 220], "pilot", "video-hash", "mapping-hash")
        editor.seek(0)
        editor.mark_start()
        editor.seek(2)
        editor.mark_end()
        editor.commit("eyes_closed_both")
        editor.seek(1)
        editor.mark_start()
        editor.seek(3)
        editor.mark_end()
        with self.assertRaises(ValueError):
            editor.commit("eyes_open")
        editor.commit("mouth_relaxed")
        self.assertEqual(len(editor.intervals), 2)

    def test_navigation_bounds_and_invalid_selection_are_explicit(self):
        editor = AnnotationEditor([100, 140], "pilot", "video-hash", "mapping-hash")
        editor.seek(-20)
        self.assertEqual(editor.index, 0)
        editor.seek(20)
        self.assertEqual(editor.index, 1)
        with self.assertRaises(ValueError):
            editor.commit("eyes_closed_both")
        editor.mark_start()
        editor.seek(0)
        editor.mark_end()
        with self.assertRaises(ValueError):
            editor.commit("eyes_closed_both")

    def test_panel_never_mutates_raw_pixels(self):
        editor = AnnotationEditor([100], "pilot", "video-hash", "mapping-hash")
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        display = editor.render(raw)
        np.testing.assert_array_equal(raw, np.zeros_like(raw))
        np.testing.assert_array_equal(display[:480, :640], raw)
        self.assertGreater(display.shape[0], 480)

    def test_keys_navigate_mark_commit_and_save_without_predictions(self):
        editor = AnnotationEditor([100, 140, 180], "pilot", "video-hash", "mapping-hash")
        with TemporaryDirectory() as directory:
            playing, deadline = False, 0
            saved = None
            for key in "dbde1s":
                playing, deadline, path = handle_key(
                    ord(key), editor, playing, deadline, Path(directory), "reviewer"
                )
                if path is not None:
                    saved = path
            self.assertEqual(len(editor.intervals), 1)
            self.assertEqual(editor.intervals[0]["start_frame"], 1)
            self.assertEqual(editor.intervals[0]["end_frame"], 2)
            self.assertTrue(saved.is_file())
            handle_key(ord("u"), editor, playing, deadline, Path(directory), "reviewer")
            self.assertEqual(editor.intervals, [])
            self.assertTrue(saved.is_file())

    def test_seek_returns_verified_frame_and_rejects_wrong_pixels(self):
        raw = np.zeros((64, 96, 3), dtype=np.uint8)
        capture = SimpleNamespace(set=lambda key, value: None, read=lambda: (True, raw))
        mapping = [{"pixel_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}]
        np.testing.assert_array_equal(get_frame(capture, 0, mapping), raw)
        mapping[0]["pixel_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            get_frame(capture, 0, mapping)

    def test_gui_events_save_interval_and_release_video(self):
        editor = AnnotationEditor([100, 140, 180], "pilot", "video-hash", "mapping-hash")
        raw = np.zeros((64, 96, 3), dtype=np.uint8)
        mapping = [{"pixel_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}] * 3
        released = []
        capture = SimpleNamespace(
            isOpened=lambda: True,
            set=lambda key, value: None,
            read=lambda: (True, raw),
            release=lambda: released.append(True),
        )
        slider = {"value": 0}
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(session=Path(directory), annotator="reviewer")
            with (
                patch("scripts.annotate_session.cv2.namedWindow"),
                patch("scripts.annotate_session.cv2.createTrackbar"),
                patch(
                    "scripts.annotate_session.cv2.getTrackbarPos",
                    side_effect=lambda *args: slider["value"],
                ),
                patch(
                    "scripts.annotate_session.cv2.setTrackbarPos",
                    side_effect=lambda name, window, value: slider.update(value=value),
                ),
                patch("scripts.annotate_session.cv2.imshow"),
                patch("scripts.annotate_session.cv2.getWindowProperty", return_value=1),
                patch(
                    "scripts.annotate_session.cv2.waitKey", side_effect=[ord(k) for k in "bdde1q"]
                ),
                patch("scripts.annotate_session.cv2.destroyAllWindows"),
            ):
                run_gui(args, editor, capture, mapping)
            paths = list((Path(directory) / "annotations").glob("*.csv"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(editor.intervals[0]["start_frame"], 0)
            self.assertEqual(editor.intervals[0]["end_frame"], 2)
            self.assertEqual(released, [True])

    def test_feedback_tracks_selection_commit_save_and_cancel(self):
        editor = AnnotationEditor([100, 140, 180], "pilot", "video-hash", "mapping-hash")
        self.assertEqual(editor.feedback()[0], "NOT SELECTING")
        editor.mark_start()
        self.assertEqual(editor.feedback()[0], "SELECTING INTERVAL")
        editor.seek(2)
        editor.mark_end()
        self.assertEqual(editor.feedback()[0], "END MARKED")
        editor.commit("eyes_closed_both")
        self.assertEqual(editor.feedback()[0], "LABEL ADDED - UNSAVED")
        with TemporaryDirectory() as directory:
            editor.save(Path(directory), "reviewer")
            self.assertEqual(editor.feedback()[0], "SAVED TO DISK")
            editor.mark_start()
            handle_key(ord("x"), editor, False, 0, Path(directory), "reviewer")
            self.assertEqual(editor.feedback()[0], "SAVED TO DISK")

    def test_selection_feedback_is_visible_outside_raw_video(self):
        editor = AnnotationEditor([100, 140], "pilot", "video-hash", "mapping-hash")
        editor.mark_start()
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("scripts.annotate_session.cv2.putText", wraps=__import__("cv2").putText) as draw:
            display = editor.render(raw, playing=True)
        texts = [call.args[1] for call in draw.call_args_list]
        self.assertIn("SELECTING INTERVAL", texts)
        self.assertIn("PLAYING", texts)
        self.assertIn("Press E to mark the end", texts)
        np.testing.assert_array_equal(display[:480, :640], raw)

    def test_reversed_range_shows_invalid_selection_not_ready(self):
        editor = AnnotationEditor([100, 140], "pilot", "video-hash", "mapping-hash")
        editor.seek(1)
        editor.mark_start()
        editor.seek(0)
        editor.mark_end()
        self.assertEqual(editor.feedback()[0], "INVALID RANGE")


if __name__ == "__main__":
    unittest.main()
