import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.landmark_diagnostics import DiagnosticSession, parse_arguments
from scripts.raw_video import read_raw_session


class TestRawIntegration(unittest.TestCase):
    def test_session_records_raw_pixels_not_overlay(self):
        with TemporaryDirectory() as directory:
            session = DiagnosticSession(Path(directory), {}, max_snapshots=0, record_video=True)
            raw = np.zeros((64, 96, 3), dtype=np.uint8)
            overlay = np.full_like(raw, 255)
            for timestamp in (0, 36, 101):
                session.write(timestamp, {"face_detected": False}, raw, overlay)
            session.close("q")
            decoded = list(read_raw_session(session.directory))
            self.assertEqual(len(decoded), 3)
            for _, _, frame in decoded:
                np.testing.assert_array_equal(frame, raw)
            metadata = json.loads((session.directory / "session.json").read_text())
            self.assertTrue(metadata["raw_video"]["complete"])
            self.assertFalse(metadata["raw_video"]["audio"])
            self.assertFalse(metadata["raw_video"]["overlay"])

    def test_plain_log_mode_does_not_create_video(self):
        with TemporaryDirectory() as directory:
            session = DiagnosticSession(Path(directory), {}, max_snapshots=0)
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            session.write(10, {"face_detected": False}, frame, frame)
            session.close("q")
            self.assertFalse((session.directory / "raw.avi").exists())

    def test_cli_video_opt_in_enables_logs(self):
        with patch("sys.argv", ["diagnostics", "--record-video"]):
            args = parse_arguments()
        self.assertTrue(args.record_video)
        self.assertTrue(args.record)
        with patch("sys.argv", ["diagnostics"]):
            args = parse_arguments()
        self.assertFalse(args.record_video)


if __name__ == "__main__":
    unittest.main()
