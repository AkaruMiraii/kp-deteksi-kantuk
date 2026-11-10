import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import landmark_diagnostics as diagnostics
from scripts.eye_duration import EyeClosureTracker


class TestLiveEyeDuration(unittest.TestCase):
    def test_processing_keeps_tracker_and_logs_reset_on_invalid_frame(self):
        tracker = EyeClosureTracker(0.12, 1.5, 0.5)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = SimpleNamespace(detect_for_video=lambda image, timestamp: None)
        statuses = []
        records = []
        for timestamp in (0, 300, 600, 900, 1200, 1500, 1600, 1700):
            base = dict(activity="neutral", face_detected=True, ear_r=0.06, ear_l=0.07, reason="")
            if timestamp == 1600:
                base.update(reason="selected_points_outside_image")
            with patch.object(diagnostics, "inspect_frame", return_value=(base, image.copy())):
                record, overlay = diagnostics.process_frame(
                    image, detector, timestamp, "neutral", tracker
                )
            records.append(record)
            statuses.append(record["eye_state"])
            self.assertEqual(record["activity"], "neutral")
            self.assertGreater(np.count_nonzero(overlay), 0)
        self.assertEqual(statuses[0], "closure_candidate")
        self.assertEqual(statuses[5], "prolonged_closure")
        self.assertEqual(statuses[6], "not_assessable")
        self.assertEqual(statuses[7], "closure_candidate")
        self.assertEqual(records[7]["eye_duration_seconds"], 0.0)
        with TemporaryDirectory() as directory:
            session = diagnostics.DiagnosticSession(Path(directory), {}, max_snapshots=0)
            self.addCleanup(session.close, "test")
            for index, record in enumerate(records):
                session.write(index, record, image, image)
            session.close("q")
            with (session.directory / "metrics.csv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[5]["eye_state"], "prolonged_closure")
            self.assertEqual(float(rows[5]["eye_duration_seconds"]), 1.5)
            log = [
                json.loads(line)
                for line in (session.directory / "frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual(log[6]["eye_state"], "not_assessable")

    def test_duration_is_disabled_without_parameters(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = SimpleNamespace(detect_for_video=lambda image, timestamp: None)
        with patch.object(
            diagnostics,
            "inspect_frame",
            return_value=({"face_detected": True, "reason": ""}, image.copy()),
        ):
            record, overlay = diagnostics.process_frame(image, detector, 0, "neutral", None)
        self.assertIsNone(record["eye_state"])
        self.assertIsNone(record["eye_duration_seconds"])
        self.assertGreater(np.count_nonzero(overlay), 0)

    def test_cli_requires_all_three_positive_finite_parameters(self):
        bad_options = [
            ["--eye-threshold", "0.12"],
            ["--eye-threshold", "nan", "--eye-duration", "1.5", "--eye-max-gap", "0.5"],
            ["--eye-threshold", "0.12", "--eye-duration", "0", "--eye-max-gap", "0.5"],
        ]
        for options in bad_options:
            with (
                self.subTest(options=options),
                patch("sys.argv", ["diagnostics"] + options),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    diagnostics.parse_arguments()

    def test_cli_disabled_by_default_and_metadata_records_explicit_parameters(self):
        with patch("sys.argv", ["diagnostics"]):
            args = diagnostics.parse_arguments()
        self.assertIsNone(args.eye_threshold)
        with patch(
            "sys.argv",
            [
                "diagnostics",
                "--eye-threshold",
                "0.12",
                "--eye-duration",
                "1.5",
                "--eye-max-gap",
                "0.5",
            ],
        ):
            args = diagnostics.parse_arguments()
        with TemporaryDirectory() as directory:
            model = Path(directory) / "model.task"
            model.write_bytes(b"test metadata fixture")
            metadata = diagnostics.session_metadata(args, model)
        self.assertEqual(metadata["eye_duration_rule"]["threshold"], 0.12)
        self.assertEqual(metadata["eye_duration_rule"]["duration_seconds"], 1.5)
        self.assertFalse(metadata["eye_duration_rule"]["calibrated"])


if __name__ == "__main__":
    unittest.main()
