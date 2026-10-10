import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import landmark_diagnostics as diagnostics
from scripts.drowsiness_decision import (
    STATUS_INDIKASI_KANTUK,
    STATUS_TIDAK_DAPAT_DINILAI,
    STATUS_TIDAK_TERDETEKSI,
)
from scripts.eye_duration import EyeClosureTracker
from scripts.mouth_duration import MouthOpeningTracker


class TestDecisionLiveIntegration(unittest.TestCase):
    def test_process_frame_evaluates_and_records_3_states(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detector = SimpleNamespace(detect_for_video=lambda img, ts: None)
        eyes = EyeClosureTracker(0.12, 1.5, 0.5)
        mouth = MouthOpeningTracker(0.5, 2.0, 0.5)

        test_steps = []

        # 1. Normal frames at t = 0 to 600 ms (gap = 200ms)
        for ts in (0, 200, 400, 600):
            test_steps.append(
                dict(
                    ts=ts,
                    ear_r=0.22,
                    ear_l=0.24,
                    mar=0.02,
                    reason="",
                    expected_status=STATUS_TIDAK_TERDETEKSI,
                    expected_trigger="none",
                )
            )

        # 2. Eye closure starts at 800 ms and continues until 2400 ms (elapsed = 1.6s >= 1.5s)
        # Prior to 2300 ms, duration < 1.5s -> candidate only
        for ts in (800, 1000, 1200, 1400, 1600, 1800, 2000, 2200):
            test_steps.append(
                dict(
                    ts=ts,
                    ear_r=0.05,
                    ear_l=0.05,
                    mar=0.02,
                    reason="",
                    expected_status=STATUS_TIDAK_TERDETEKSI,
                    expected_trigger="none",
                )
            )
        # At 2400 ms, duration = 1.6s >= 1.5s -> prolonged_closure
        test_steps.append(
            dict(
                ts=2400,
                ear_r=0.05,
                ear_l=0.05,
                mar=0.02,
                reason="",
                expected_status=STATUS_INDIKASI_KANTUK,
                expected_trigger="eye",
            )
        )

        # 3. Invalid frame at 2600 ms (face missing) -> immediately tidak_dapat_dinilai
        test_steps.append(
            dict(
                ts=2600,
                ear_r=None,
                ear_l=None,
                mar=None,
                reason="no_face_detected",
                expected_status=STATUS_TIDAK_DAPAT_DINILAI,
                expected_trigger="none",
            )
        )

        # 4. Re-entry normal at 2800 ms -> timer was reset, returns to normal
        test_steps.append(
            dict(
                ts=2800,
                ear_r=0.22,
                ear_l=0.24,
                mar=0.02,
                reason="",
                expected_status=STATUS_TIDAK_TERDETEKSI,
                expected_trigger="none",
            )
        )

        # 5. Mouth opening starts at 3000 ms and continues until 5200 ms (elapsed = 2.2s >= 2.0s)
        for ts in (3000, 3300, 3600, 3900, 4200, 4500, 4800):
            test_steps.append(
                dict(
                    ts=ts,
                    ear_r=0.22,
                    ear_l=0.24,
                    mar=0.70,
                    reason="",
                    expected_status=STATUS_TIDAK_TERDETEKSI,
                    expected_trigger="none",
                )
            )
        # At 5100 ms, elapsed = 2.1s >= 2.0s -> prolonged_opening
        test_steps.append(
            dict(
                ts=5100,
                ear_r=0.22,
                ear_l=0.24,
                mar=0.70,
                reason="",
                expected_status=STATUS_TIDAK_TERDETEKSI,
                expected_trigger="none",
            )
        )

        # 6. Mouth closed again at 5300 ms -> returns to normal
        test_steps.append(
            dict(
                ts=5300,
                ear_r=0.22,
                ear_l=0.24,
                mar=0.02,
                reason="",
                expected_status=STATUS_TIDAK_TERDETEKSI,
                expected_trigger="none",
            )
        )

        records = []
        for step in test_steps:
            base = dict(
                activity="test",
                face_detected=(step["reason"] == ""),
                ear_r=step["ear_r"],
                ear_l=step["ear_l"],
                mar=step["mar"],
                reason=step["reason"],
            )
            with patch.object(diagnostics, "inspect_frame", return_value=(base, image.copy())):
                record, overlay = diagnostics.process_frame(
                    image, detector, step["ts"], "test", eyes, mouth
                )
            records.append(record)
            self.assertEqual(
                record["final_status"], step["expected_status"], f"Failed at ts={step['ts']}"
            )
            self.assertEqual(
                record["drowsiness_trigger"],
                step["expected_trigger"],
                f"Failed trigger at ts={step['ts']}",
            )
            self.assertGreater(np.count_nonzero(overlay), 0)

        # Verify CSV & JSON logging of 3 states
        with TemporaryDirectory() as directory:
            session = diagnostics.DiagnosticSession(Path(directory), {}, max_snapshots=0)
            self.addCleanup(session.close, "test")
            for idx, r in enumerate(records):
                session.write(idx, r, image, image)
            session.close("q")

            with (session.directory / "metrics.csv").open(encoding="utf-8", newline="") as src:
                csv_rows = list(csv.DictReader(src))
            self.assertEqual(csv_rows[0]["final_status"], STATUS_TIDAK_TERDETEKSI)
            self.assertEqual(csv_rows[12]["final_status"], STATUS_INDIKASI_KANTUK)
            self.assertEqual(csv_rows[12]["drowsiness_trigger"], "eye")
            self.assertEqual(csv_rows[13]["final_status"], STATUS_TIDAK_DAPAT_DINILAI)
            self.assertEqual(csv_rows[22]["final_status"], STATUS_TIDAK_TERDETEKSI)
            self.assertEqual(csv_rows[22]["drowsiness_trigger"], "none")
            self.assertEqual(csv_rows[22]["decision_reason"], "mouth_opening_not_validated")
            self.assertEqual(csv_rows[22]["decision_policy"], "v2_eye_evidence_mouth_observation")
            self.assertEqual(csv_rows[23]["final_status"], STATUS_TIDAK_TERDETEKSI)


if __name__ == "__main__":
    unittest.main()
