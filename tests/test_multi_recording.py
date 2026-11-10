import json
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import multi_live
from scripts.raw_video import read_raw_session


class TestMultiRecording(unittest.TestCase):
    def test_offline_recording_has_replay_contract_and_no_camera(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            stack.enter_context(
                patch.object(multi_live, "__file__", str(root / "scripts/multi_live.py"))
            )
            stack.enter_context(
                patch.object(multi_live, "uuid4", return_value=SimpleNamespace(hex="fixture"))
            )
            stack.enter_context(patch.object(multi_live, "FaceAuthenticity"))
            stack.enter_context(patch.object(multi_live, "file_sha256", return_value="fixture"))
            stack.enter_context(
                patch.object(
                    multi_live.vision.FaceLandmarker,
                    "create_from_options",
                    return_value=nullcontext(
                        SimpleNamespace(
                            detect_for_video=lambda *a: SimpleNamespace(face_landmarks=[])
                        )
                    ),
                )
            )

            def source():
                yield 0, 0, frame
                yield 1, 100, frame

            stack.enter_context(patch.object(multi_live, "read_raw_session", return_value=source()))
            camera = stack.enter_context(
                patch.object(
                    multi_live.cv2,
                    "VideoCapture",
                    side_effect=lambda source: (
                        cv2_capture(source)
                        if isinstance(source, str)
                        else (_ for _ in ()).throw(AssertionError("camera forbidden"))
                    ),
                )
            )
            stack.enter_context(patch.object(multi_live.cv2, "destroyAllWindows"))
            image_writer = stack.enter_context(
                patch.object(multi_live.cv2, "imwrite", return_value=True)
            )
            stack.enter_context(patch("builtins.print"))
            multi_live.run(multi_live.parse_args(["--session", "fixture", "--record-video"]))
            directory = root / "outputs/multiperson/fixture"
            self.assertTrue(all(isinstance(call.args[0], str) for call in camera.call_args_list))
            summary = json.loads((directory / "session.json").read_text())
            self.assertTrue(summary["raw_video"]["complete"])
            with patch.object(multi_live.cv2, "VideoCapture", new=cv2_capture):
                replay = list(read_raw_session(directory))
            self.assertEqual([t for _, t, _ in replay], [0, 100])
            for _, _, image in replay:
                np.testing.assert_array_equal(image, frame)
            image_writer.assert_not_called()

    def test_source_failure_preserves_error_and_marks_recording_incomplete(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            stack.enter_context(
                patch.object(multi_live, "__file__", str(root / "scripts/multi_live.py"))
            )
            stack.enter_context(
                patch.object(multi_live, "uuid4", return_value=SimpleNamespace(hex="fixture"))
            )
            stack.enter_context(patch.object(multi_live, "FaceAuthenticity"))
            stack.enter_context(patch.object(multi_live, "file_sha256", return_value="fixture"))
            stack.enter_context(
                patch.object(
                    multi_live.vision.FaceLandmarker,
                    "create_from_options",
                    return_value=nullcontext(
                        SimpleNamespace(
                            detect_for_video=lambda *a: SimpleNamespace(face_landmarks=[])
                        )
                    ),
                )
            )

            def source():
                yield 0, 0, frame
                raise OSError("source failed")

            stack.enter_context(patch.object(multi_live, "read_raw_session", return_value=source()))
            stack.enter_context(patch.object(multi_live.cv2, "destroyAllWindows"))
            stack.enter_context(patch("builtins.print"))
            with self.assertRaisesRegex(OSError, "source failed"):
                multi_live.run(multi_live.parse_args(["--session", "fixture", "--record-video"]))
            directory = root / "outputs/multiperson/fixture"
            summary = json.loads((directory / "session.json").read_text())
            self.assertFalse(summary["raw_video"]["complete"])
            self.assertEqual(summary["exit_reason"], "error")


cv2_capture = multi_live.cv2.VideoCapture
