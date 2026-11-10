import csv
import json
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import smart_live
from scripts.raw_video import decoded_frames, read_raw_session

REAL_VIDEO_CAPTURE = smart_live.cv2.VideoCapture


class TestSmartRecording(unittest.TestCase):
    def test_cli_record_video_is_opt_in(self):
        with patch("scripts.smart_live.run") as run:
            with patch("sys.argv", ["smart-live", "--record-video"]):
                smart_live.main()
            self.assertTrue(run.call_args.args[0].record_video)
            with patch("sys.argv", ["smart-live"]):
                smart_live.main()
            self.assertFalse(run.call_args.args[0].record_video)

    def exercise(self, root, recording=True, source=None, args_extra=None):
        raw = np.random.default_rng(11).integers(0, 256, (64, 96, 3), dtype=np.uint8)
        dashboard = np.random.default_rng(12).integers(0, 256, (65, 97, 3), dtype=np.uint8)
        row = {key: 0 for key in smart_live.FIELDS}
        row.update(reason="", final_status="unchanged", smart_yawn_state="not_yawning")
        args = SimpleNamespace(
            model=root / "model.json",
            eye_alertness=False,
            session=root / "input",
            input_image=None,
            camera=0,
        )
        if recording is not None:
            args.record_video = recording
        for key, value in (args_extra or {}).items():
            setattr(args, key, value)
        predictor = SimpleNamespace(
            model={"landmark_model_sha256": "hash", "target": "yawn"}, model_hash="hash"
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.object(smart_live, "__file__", str(root / "scripts" / "smart_live.py"))
        )
        stack.enter_context(
            patch.object(smart_live, "uuid4", return_value=SimpleNamespace(hex="testsession"))
        )
        stack.enter_context(patch.object(smart_live, "SmartYawn", return_value=predictor))
        stack.enter_context(patch.object(smart_live, "file_sha256", return_value="hash"))
        stack.enter_context(
            patch.object(
                smart_live.vision.FaceLandmarker,
                "create_from_options",
                return_value=nullcontext(None),
            )
        )
        stack.enter_context(
            patch.object(
                smart_live,
                "read_raw_session",
                return_value=source if source is not None else [(4, 120, raw), (9, 243, raw)],
            )
        )
        stack.enter_context(
            patch.object(smart_live, "process_smart_frame", return_value=(row, dashboard))
        )
        stack.enter_context(patch.object(smart_live.cv2, "destroyAllWindows"))
        stack.enter_context(patch("builtins.print"))
        return args, raw, dashboard, root / "outputs" / "smart_yawn" / "testsession"

    def test_real_recording_replays_original_bgr_and_padded_dashboard(self):
        with TemporaryDirectory() as temporary:
            args, raw, dashboard, directory = self.exercise(Path(temporary))
            smart_live.run(args)
            summary = json.loads((directory / "session.json").read_text())
            self.assertTrue(summary["raw_video"]["complete"])
            self.assertTrue(summary["dashboard_video"]["complete"])
            self.assertEqual(summary["exit_reason"], "source_complete")
            decoded = list(read_raw_session(directory))
            self.assertEqual([(i, t) for i, t, _ in decoded], [(0, 120), (1, 243)])
            for _, _, frame in decoded:
                np.testing.assert_array_equal(frame, raw)
            for _, _, frame in decoded_frames(directory / "dashboard"):
                self.assertEqual(frame.shape, (66, 98, 3))
                np.testing.assert_array_equal(frame[:65, :97], dashboard)
                self.assertFalse(frame[65].any())
                self.assertFalse(frame[:, 97].any())
            with (directory / "predictions.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            rows = [
                json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["source_frame_id"] for r in rows], [4, 9])
            for logged, replay in zip(csv_rows, rows):
                for key in smart_live.FIELDS:
                    self.assertEqual(logged[key], str(replay[key]))
            self.assertEqual(summary["dashboard_video"]["padding_bottom"], 1)
            self.assertEqual(summary["dashboard_video"]["padding_right"], 1)

    def test_source_error_and_interrupt_mark_both_recordings_incomplete(self):
        for error in (OSError("source failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw = np.zeros((64, 96, 3), dtype=np.uint8)

                def source():
                    yield 0, 10, raw
                    raise error

                args, _, _, directory = self.exercise(root, source=source())
                with self.assertRaises(type(error)):
                    smart_live.run(args)
                summary = json.loads((directory / "session.json").read_text())
                self.assertEqual(
                    summary["exit_reason"],
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                )
                for key in ("raw_video", "dashboard_video"):
                    self.assertFalse(summary[key]["complete"])
                    self.assertEqual(summary[key]["verified_frames"], 1)
                with self.assertRaisesRegex(ValueError, "not finalized"):
                    list(read_raw_session(directory))

    def test_failing_raw_finalization_still_closes_dashboard_and_camera(self):
        from scripts.raw_video import RawVideoRecorder

        with TemporaryDirectory() as temporary:
            args, _, _, directory = self.exercise(Path(temporary), args_extra={"session": None})
            original_close = RawVideoRecorder.close
            closed = []

            def close(recorder):
                result = original_close(recorder)
                closed.append(recorder.directory)
                if recorder.directory == directory:
                    raise OSError("raw verification failed")
                return result

            with (
                patch.object(smart_live.cv2, "VideoCapture") as capture,
                patch.object(smart_live, "webcam_frames", return_value=[]),
                patch.object(RawVideoRecorder, "close", close),
            ):
                with self.assertRaisesRegex(OSError, "raw verification failed"):
                    smart_live.run(args)
                capture.return_value.release.assert_called_once()
            self.assertEqual(closed, [directory, directory / "dashboard"])
            summary = json.loads((directory / "session.json").read_text())
            self.assertEqual(summary["exit_reason"], "error")
            self.assertFalse(summary["raw_video"]["complete"])
            self.assertFalse(summary["dashboard_video"]["complete"])
            self.assertIn("raw verification failed", summary["recording_errors"][0])

    def test_q_and_window_close_finalize_and_close_source(self):
        for key, visible, reason in ((ord("Q"), 1, "q"), (-1, 0, "window_closed")):
            with self.subTest(reason=reason), TemporaryDirectory() as temporary:
                args, raw, _, directory = self.exercise(
                    Path(temporary), args_extra={"session": None}
                )
                source_closed = []

                def source():
                    try:
                        yield 0, 10, raw
                        yield 1, 20, raw
                    finally:
                        source_closed.append(True)

                with (
                    patch.object(smart_live.cv2, "VideoCapture") as capture,
                    patch.object(smart_live, "webcam_frames", return_value=source()),
                    patch.object(smart_live.cv2, "imshow"),
                    patch.object(smart_live.cv2, "waitKey", return_value=key),
                    patch.object(smart_live.cv2, "getWindowProperty", return_value=visible),
                ):
                    camera = capture.return_value
                    capture.side_effect = lambda source: (
                        camera if isinstance(source, int) else REAL_VIDEO_CAPTURE(source)
                    )
                    smart_live.run(args)
                    camera.release.assert_called_once()
                self.assertEqual(source_closed, [True])
                summary = json.loads((directory / "session.json").read_text())
                self.assertEqual(summary["exit_reason"], reason)
                self.assertEqual(summary["frames"], 1)
                self.assertTrue(summary["raw_video"]["complete"])
                self.assertTrue(summary["dashboard_video"]["complete"])

    def test_recording_off_preserves_csv_preview_and_source_indices(self):
        for recording in (False, None):
            with self.subTest(recording=recording), TemporaryDirectory() as temporary:
                args, _, _, directory = self.exercise(Path(temporary), recording=recording)
                smart_live.run(args)
                self.assertTrue((directory / "preview.png").exists())
                with (directory / "predictions.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([r["frame_id"] for r in rows], ["4", "9"])
                self.assertFalse((directory / "raw.avi").exists())
                self.assertFalse((directory / "dashboard").exists())
                self.assertFalse((directory / "frames.jsonl").exists())
                self.assertNotIn("raw_video", json.loads((directory / "session.json").read_text()))

    def test_dashboard_write_failure_keeps_raw_session_unreplayable(self):
        from scripts.raw_video import RawVideoRecorder

        with TemporaryDirectory() as temporary:
            args, _, _, directory = self.exercise(Path(temporary))
            original_write = RawVideoRecorder.write

            def write(recorder, timestamp, frame):
                if recorder.directory.name == "dashboard":
                    raise OSError("dashboard write failed")
                return original_write(recorder, timestamp, frame)

            with patch.object(RawVideoRecorder, "write", write):
                with self.assertRaisesRegex(OSError, "dashboard write failed"):
                    smart_live.run(args)
            summary = json.loads((directory / "session.json").read_text())
            self.assertEqual(summary["frames"], 0)
            self.assertEqual(summary["raw_video"]["frames"], 1)
            self.assertFalse(summary["raw_video"]["complete"])
            self.assertFalse(summary["dashboard_video"]["complete"])
            with self.assertRaisesRegex(ValueError, "not finalized"):
                list(read_raw_session(directory))

    def test_codec_unavailable_propagates_and_finalizes_metadata(self):
        with TemporaryDirectory() as temporary:
            args, _, _, directory = self.exercise(Path(temporary))
            with patch.object(smart_live.cv2, "VideoWriter") as writer:
                writer.return_value.isOpened.return_value = False
                with self.assertRaisesRegex(OSError, "FFV1"):
                    smart_live.run(args)
                writer.return_value.release.assert_called_once()
            summary = json.loads((directory / "session.json").read_text())
            self.assertEqual(summary["exit_reason"], "error")
            self.assertFalse(summary["raw_video"]["complete"])
            self.assertFalse(summary["dashboard_video"]["complete"])

    def test_preview_failure_finalizes_both_without_claiming_complete(self):
        with TemporaryDirectory() as temporary:
            args, _, _, directory = self.exercise(Path(temporary))
            with patch.object(smart_live.cv2, "imwrite", return_value=False):
                with self.assertRaisesRegex(OSError, "Preview"):
                    smart_live.run(args)
            summary = json.loads((directory / "session.json").read_text())
            for key in ("raw_video", "dashboard_video"):
                self.assertFalse(summary[key]["complete"])
                self.assertEqual(summary[key]["verified_frames"], 1)

    def test_original_exception_survives_dashboard_finalization_failure(self):
        from scripts.raw_video import RawVideoRecorder

        with TemporaryDirectory() as temporary:
            args, _, _, directory = self.exercise(Path(temporary))
            original_close = RawVideoRecorder.close

            def close(recorder):
                result = original_close(recorder)
                if recorder.directory.name == "dashboard":
                    raise OSError("dashboard verification failed")
                return result

            with (
                patch.object(
                    smart_live, "process_smart_frame", side_effect=ValueError("inference failed")
                ),
                patch.object(RawVideoRecorder, "close", close),
            ):
                with self.assertRaisesRegex(ValueError, "inference failed") as caught:
                    smart_live.run(args)
            self.assertIn("dashboard verification failed", caught.exception.__notes__[0])
            summary = json.loads((directory / "session.json").read_text())
            self.assertIn("inference failed", summary["error"])
            self.assertIn("dashboard verification failed", summary["recording_errors"][0])
            self.assertFalse(summary["raw_video"]["complete"])
            self.assertFalse(summary["dashboard_video"]["complete"])

    def test_image_source_records_one_frame_at_zero(self):
        with TemporaryDirectory() as temporary:
            args, raw, _, directory = self.exercise(
                Path(temporary), args_extra={"session": None, "input_image": Path("image.png")}
            )
            with patch.object(smart_live.cv2, "imread", return_value=raw):
                smart_live.run(args)
            self.assertEqual([(i, t) for i, t, _ in read_raw_session(directory)], [(0, 0)])

    def test_empty_source_does_not_claim_recording_complete(self):
        with TemporaryDirectory() as temporary:
            args, _, _, directory = self.exercise(Path(temporary), source=[])
            smart_live.run(args)
            summary = json.loads((directory / "session.json").read_text())
            self.assertEqual(summary["frames"], 0)
            self.assertFalse(summary["raw_video"]["complete"])
            self.assertFalse(summary["dashboard_video"]["complete"])

    def test_close_failure_before_metadata_reports_actual_frame_count(self):
        from scripts.raw_video import RawVideoRecorder

        with TemporaryDirectory() as temporary:
            args, _, _, directory = self.exercise(Path(temporary))
            original_close = RawVideoRecorder.close

            def close(recorder):
                if recorder.directory == directory:
                    raise OSError("release failed before metadata")
                return original_close(recorder)

            with patch.object(RawVideoRecorder, "close", close):
                with self.assertRaisesRegex(OSError, "release failed"):
                    smart_live.run(args)
            summary = json.loads((directory / "session.json").read_text())
            self.assertEqual(summary["raw_video"]["frames"], 2)
            self.assertEqual(summary["dashboard_video"]["verified_frames"], 2)
            self.assertFalse(summary["raw_video"]["complete"])
            self.assertFalse(summary["dashboard_video"]["complete"])


if __name__ == "__main__":
    unittest.main()
