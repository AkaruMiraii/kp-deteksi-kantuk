import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import landmark_diagnostics as diagnostics


class TestDiagnosticTiming(unittest.TestCase):
    def test_processing_uses_high_resolution_clock(self):
        args = SimpleNamespace(
            record=False,
            record_video=False,
            input_image=Path("fixture.png"),
            camera=0,
            snapshot_interval=2.0,
            max_snapshots=120,
            eye_threshold=None,
            eye_duration=None,
            eye_max_gap=None,
            mouth_threshold=None,
            mouth_duration=None,
            mouth_max_gap=None,
        )
        output = io.StringIO()
        with (
            patch.object(diagnostics, "parse_arguments", return_value=args),
            patch.object(diagnostics, "session_metadata", return_value={}),
            patch.object(diagnostics, "monotonic_ns", return_value=100_000_000),
            patch.object(
                diagnostics, "perf_counter_ns", create=True, side_effect=[1_000_000, 2_250_000]
            ) as clock,
            patch.object(
                diagnostics.cv2, "imread", return_value=np.zeros((120, 640, 3), dtype=np.uint8)
            ),
            patch.object(diagnostics.vision.FaceLandmarker, "create_from_options") as factory,
            patch("sys.stdout", output),
        ):
            factory.return_value.__enter__.return_value.detect_for_video.return_value = (
                SimpleNamespace(face_landmarks=[])
            )
            diagnostics.main()
        record = next(
            json.loads(line) for line in output.getvalue().splitlines() if line.startswith("{")
        )
        self.assertEqual(record["processing_ms"], 1.25)
        self.assertEqual(clock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
