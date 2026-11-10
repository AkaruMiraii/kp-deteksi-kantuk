import csv
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import landmark_diagnostics as diagnostics
from scripts.eye_duration import EyeClosureTracker
from scripts.mouth_duration import MouthOpeningTracker


class TestMouthLive(unittest.TestCase):
    def test_channels_run_independently_and_invalid_geometry_resets_both(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = SimpleNamespace(detect_for_video=lambda image, timestamp: None)
        eyes = EyeClosureTracker(0.12, 1.5, 0.5)
        mouth = MouthOpeningTracker(0.5, 2.0, 0.5)
        records = []
        for timestamp in (0, 500, 1000, 1500, 2000, 2100, 2200):
            base = dict(
                activity="speaking", ear_r=0.21, ear_l=0.24, mar=0.6, face_detected=True, reason=""
            )
            if timestamp == 2100:
                base["reason"] = "selected_points_outside_image"
            with patch.object(diagnostics, "inspect_frame", return_value=(base, image.copy())):
                record, overlay = diagnostics.process_frame(
                    image, detector, timestamp, "speaking", eyes, mouth
                )
            records.append(record)
            self.assertGreater(np.count_nonzero(overlay), 0)
        self.assertEqual(records[4]["mouth_state"], "prolonged_opening")
        self.assertEqual(records[4]["eye_state"], "no_bilateral_closure")
        self.assertEqual(records[4]["mouth_duration_seconds"], 2.0)
        self.assertEqual(records[5]["mouth_state"], "not_assessable")
        self.assertEqual(records[5]["eye_state"], "not_assessable")
        self.assertEqual(records[6]["mouth_duration_seconds"], 0.0)
        self.assertEqual(records[6]["activity"], "speaking")
        with TemporaryDirectory() as directory:
            session = diagnostics.DiagnosticSession(Path(directory), {}, max_snapshots=0)
            self.addCleanup(session.close, "test")
            for index, record in enumerate(records):
                session.write(index, record, image, image)
            session.close("q")
            with (session.directory / "metrics.csv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[4]["mouth_state"], "prolonged_opening")
            self.assertEqual(float(rows[4]["mouth_duration_seconds"]), 2.0)

    def test_mouth_mode_off_by_default_and_parameters_are_explicit(self):
        with patch("sys.argv", ["diagnostics"]):
            args = diagnostics.parse_arguments()
        self.assertIsNone(args.mouth_threshold)
        with patch(
            "sys.argv",
            [
                "diagnostics",
                "--mouth-threshold",
                "0.5",
                "--mouth-duration",
                "2",
                "--mouth-max-gap",
                "0.5",
            ],
        ):
            args = diagnostics.parse_arguments()
        with TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"
            model.write_bytes(b"metadata fixture")
            metadata = diagnostics.session_metadata(args, model)
        self.assertTrue(metadata["mouth_duration_rule"]["enabled"])
        self.assertEqual(metadata["mouth_duration_rule"]["duration_seconds"], 2.0)
        self.assertFalse(metadata["mouth_duration_rule"]["calibrated"])
        self.assertFalse(metadata["eye_duration_rule"]["enabled"])

    def test_partial_and_invalid_mouth_parameters_are_rejected(self):
        for options in [
            ["--mouth-threshold", "0.5"],
            ["--mouth-threshold", "nan", "--mouth-duration", "2", "--mouth-max-gap", "0.5"],
        ]:
            with (
                self.subTest(options=options),
                patch("sys.argv", ["diagnostics"] + options),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    diagnostics.parse_arguments()


if __name__ == "__main__":
    unittest.main()
