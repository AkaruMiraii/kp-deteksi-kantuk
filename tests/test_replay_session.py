import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts import replay_session
from scripts.landmark_diagnostics import DiagnosticSession


class TestReplaySession(unittest.TestCase):
    def test_verify_cli_checks_complete_source_without_webcam(self):
        with TemporaryDirectory() as directory:
            session = DiagnosticSession(Path(directory), {}, max_snapshots=0, record_video=True)
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            for timestamp in (10, 44, 102):
                session.write(timestamp, {"face_detected": False}, frame, frame)
            session.close("q")
            output = io.StringIO()
            with (
                patch("sys.argv", ["replay", str(session.directory), "--verify-only"]),
                patch("sys.stdout", output),
            ):
                replay_session.main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["verified_frames"], 3)
            self.assertEqual(report["first_timestamp_ms"], 10)
            self.assertEqual(report["last_timestamp_ms"], 102)
            self.assertEqual(report["raw_pixels_verified"], True)

    def test_verify_detects_prediction_timestamp_mismatch(self):
        with TemporaryDirectory() as directory:
            session = DiagnosticSession(Path(directory), {}, max_snapshots=0, record_video=True)
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            session.write(10, {"face_detected": False}, frame, frame)
            session.close("q")
            (session.directory / "frames.jsonl").write_text(
                json.dumps({"frame_id": 0, "timestamp_ms": 20}) + "\n"
            )
            with patch("sys.argv", ["replay", str(session.directory), "--verify-only"]):
                with self.assertRaisesRegex(ValueError, "not synchronized"):
                    replay_session.main()


if __name__ == "__main__":
    unittest.main()
