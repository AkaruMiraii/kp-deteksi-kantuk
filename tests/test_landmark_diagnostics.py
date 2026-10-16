import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.landmark_diagnostics import DiagnosticSession, calculate_mar, inspect_frame


class TestMouthRatio(unittest.TestCase):
    def test_known_geometry(self):
        self.assertAlmostEqual(calculate_mar([(0, 0), (4, 0), (2, 1), (2, -1)]), 0.5)

    def test_closed_mouth(self):
        self.assertEqual(calculate_mar([(0, 0), (4, 0), (2, 0), (2, 0)]), 0.0)

    def test_zero_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Lebar mulut"):
            calculate_mar([(0, 0), (0, 0), (2, 1), (2, -1)])


class TestFrameInspection(unittest.TestCase):
    def test_missing_face_has_no_metrics_and_raw_is_unchanged(self):
        raw = np.zeros((120, 640, 3), dtype=np.uint8)
        record, overlay = inspect_frame(raw, SimpleNamespace(face_landmarks=[]), "neutral")
        self.assertFalse(record["face_detected"])
        self.assertIsNone(record["ear_r"])
        self.assertIsNone(record["mar"])
        self.assertEqual(record["reason"], "no_face_detected")
        self.assertEqual(np.count_nonzero(raw), 0)
        self.assertGreater(np.count_nonzero(overlay), 0)


class TestDiagnosticSession(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.frame = np.zeros((24, 32, 3), dtype=np.uint8)
        self.record = {"face_detected": True, "ear_r": 0.0, "mar": 0.5}

    def test_periodic_snapshot_limit_and_missing_face_are_logged(self):
        session = DiagnosticSession(Path(self.temp.name), {}, interval_ms=2000, max_snapshots=2)
        self.addCleanup(session.close, "test")
        for timestamp in (0, 1000, 2000, 4000):
            session.write(timestamp, self.record, self.frame, self.frame)
        session.write(
            5000, {"face_detected": False, "ear_r": None, "mar": None}, self.frame, self.frame
        )
        session.close("q")
        rows = [
            json.loads(line)
            for line in (session.directory / "frames.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["ear_r"], 0.0)
        self.assertIsNone(rows[-1]["ear_r"])
        self.assertFalse(rows[-1]["face_detected"])
        self.assertEqual(sum(bool(row["snapshot_raw"]) for row in rows), 2)
        self.assertEqual(len(list((session.directory / "snapshots").glob("*.png"))), 4)
        summary = json.loads((session.directory / "session.json").read_text())
        self.assertEqual(summary["frames"], 5)
        self.assertEqual(summary["no_face_frames"], 1)
        self.assertEqual(summary["snapshots"], 2)
        self.assertEqual(summary["exit_reason"], "q")

    def test_manual_snapshot_can_capture_between_intervals(self):
        session = DiagnosticSession(Path(self.temp.name), {}, interval_ms=2000, max_snapshots=2)
        self.addCleanup(session.close, "test")
        session.write(0, self.record, self.frame, self.frame)
        session.write(10, self.record, self.frame, self.frame, force_snapshot=True)
        session.close("q")
        self.assertEqual(len(list((session.directory / "snapshots").glob("*.png"))), 4)

    def test_failed_image_write_is_not_reported_as_success(self):
        session = DiagnosticSession(Path(self.temp.name), {}, interval_ms=2000, max_snapshots=2)
        self.addCleanup(session.close, "test")
        with patch("scripts.landmark_diagnostics.cv2.imwrite", return_value=False):
            with self.assertRaises(OSError):
                session.write(0, self.record, self.frame, self.frame)
        self.assertEqual(session.snapshots, 0)


if __name__ == "__main__":
    unittest.main()
