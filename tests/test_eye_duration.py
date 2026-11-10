import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import eye_duration
from scripts.eye_duration import EyeClosureTracker


class TestEyeReplay(unittest.TestCase):
    def test_cli_replays_log_without_modifying_source(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "frames.jsonl"
            content = (
                "\n".join(
                    json.dumps(
                        dict(timestamp_ms=t, ear_r=0.06, ear_l=0.07, reason="", activity="unmarked")
                    )
                    for t in (0, 500, 1000, 1500)
                )
                + "\n"
            )
            source.write_text(content, encoding="utf-8")
            output = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "eye_duration",
                        str(source),
                        "--threshold",
                        "0.12",
                        "--duration",
                        "1.5",
                        "--max-gap",
                        "0.5",
                    ],
                ),
                patch("sys.stdout", output),
            ):
                eye_duration.main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["frames"], 4)
            self.assertEqual(report["states"]["prolonged_closure"], 1)
            self.assertEqual(source.read_text(encoding="utf-8"), content)


class TestEyeDuration(unittest.TestCase):
    def test_short_closure_does_not_trigger_but_sustained_closure_does(self):
        tracker = EyeClosureTracker(threshold=0.12, duration_seconds=1.5, max_gap_seconds=0.5)
        first = tracker.update(0.0, 0.06, 0.07)
        self.assertEqual(first.state, "closure_candidate")
        self.assertEqual(first.duration_seconds, 0.0)
        for timestamp in (0.3, 0.6, 0.9, 1.2):
            self.assertEqual(tracker.update(timestamp, 0.06, 0.07).state, "closure_candidate")
        final = tracker.update(1.5, 0.06, 0.07)
        self.assertEqual(final.state, "prolonged_closure")
        self.assertAlmostEqual(final.duration_seconds, 1.5)
        self.assertEqual(tracker.update(1.6, 0.21, 0.24).state, "no_bilateral_closure")

    def test_missing_metrics_reset_continuity(self):
        tracker = EyeClosureTracker(0.12, 1.5, 0.5)
        tracker.update(0.0, 0.06, 0.07)
        tracker.update(0.4, 0.06, 0.07)
        self.assertEqual(tracker.update(0.5, None, None).state, "not_assessable")
        self.assertEqual(tracker.update(0.6, 0.06, 0.07).duration_seconds, 0.0)

    def test_long_observation_gap_resets_continuity(self):
        tracker = EyeClosureTracker(0.12, 1.5, 0.5)
        tracker.update(0.0, 0.06, 0.07)
        self.assertEqual(tracker.update(2.0, 0.06, 0.07).duration_seconds, 0.0)

    def test_one_closed_eye_does_not_start_bilateral_closure(self):
        tracker = EyeClosureTracker(0.12, 1.5, 0.5)
        self.assertEqual(tracker.update(0.0, 0.06, 0.24).state, "no_bilateral_closure")

    def test_invalid_parameters_are_rejected(self):
        for threshold, duration, gap in [
            (0, 1, 0.5),
            (0.12, 0, 0.5),
            (0.12, 1, 0),
            (float("nan"), 1, 0.5),
        ]:
            with self.subTest(parameters=(threshold, duration, gap)):
                with self.assertRaises(ValueError):
                    EyeClosureTracker(threshold, duration, gap)

    def test_invalid_time_is_rejected_without_changing_state(self):
        tracker = EyeClosureTracker(0.12, 1.5, 0.5)
        tracker.update(1.0, 0.06, 0.07)
        for timestamp in [1.0, 0.9, float("nan"), float("inf")]:
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    tracker.update(timestamp, 0.06, 0.07)
        self.assertAlmostEqual(tracker.update(1.2, 0.06, 0.07).duration_seconds, 0.2)

    def test_nonfinite_and_negative_ear_are_not_assessable(self):
        for value in [float("nan"), float("inf"), -0.1]:
            tracker = EyeClosureTracker(0.12, 1.5, 0.5)
            with self.subTest(value=value):
                self.assertEqual(tracker.update(0.0, value, 0.07).state, "not_assessable")


if __name__ == "__main__":
    unittest.main()
