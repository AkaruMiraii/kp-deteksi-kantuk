import argparse
import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import monotonic_ns, perf_counter_ns
from uuid import uuid4

import cv2
from mediapipe.tasks.python import BaseOptions, vision

from .eye_alertness import EyeAlertness, EyeAwareDetector
from .eye_duration import EyeClosureTracker
from .landmark_diagnostics import process_frame
from .mouth_duration import MouthOpeningTracker
from .raw_video import RawVideoRecorder, file_sha256, read_raw_session
from .smart_display import render_dashboard
from .smart_yawn import SmartYawn

FIELDS = [
    "frame_id",
    "timestamp_ms",
    "ear_r",
    "ear_l",
    "mar",
    "reason",
    "final_status",
    "smart_yawn_state",
    "smart_yawn_score",
    "smart_yawn_duration_seconds",
    "processing_ms",
]
WINDOW = "KP - Smart yawn experimental"


def process_smart_frame(frame, detector, timestamp_ms, predictor, eyes, mouth):
    started = perf_counter_ns()
    record, overlay = process_frame(frame, detector, timestamp_ms, "unmarked", eyes, mouth)
    values = [record[k] for k in ("ear_r", "ear_l", "mar")]
    if record["reason"]:
        values = [None, None, None]
    record.update(predictor.update(timestamp_ms, *values))
    if isinstance(detector, EyeAwareDetector):
        record, overlay = detector.add_eye_status(frame, record, overlay, timestamp_ms)
    record["processing_ms"] = (perf_counter_ns() - started) / 1_000_000
    display = render_dashboard(frame, record)
    record["processing_ms"] = (perf_counter_ns() - started) / 1_000_000
    return record, display


def run(args):
    root = Path(__file__).resolve().parents[1]
    predictor = SmartYawn(args.model)
    face_model = root / "models" / "face_landmarker.task"
    if file_sha256(face_model) != predictor.model["landmark_model_sha256"]:
        raise ValueError("Face landmark model differs from training source.")
    eyes = EyeClosureTracker(0.12, 1.5, 0.5)
    mouth = MouthOpeningTracker(0.5, 2, 0.5)
    eye_alertness = EyeAlertness() if args.eye_alertness else None
    eye_fields = [
        "eye_alertness_state",
        "eye_closed_duration_seconds",
        "eye_threshold_right",
        "eye_threshold_left",
        "eye_closed_fraction_30s",
        "eye_observed_seconds_30s",
        "independent_ear_r",
        "independent_ear_l",
        "independent_eye_reason",
    ]
    directory = root / "outputs" / "smart_yawn" / uuid4().hex[:12]
    directory.mkdir(parents=True)
    capture = None
    counts = Counter()
    frames = 0
    reason = "error"
    recording = getattr(args, "record_video", False)
    raw_recorder = dashboard_recorder = None
    dashboard_padding = {}
    failure = None
    try:
        if recording:
            raw_recorder = RawVideoRecorder(directory)
            dashboard_directory = directory / "dashboard"
            dashboard_directory.mkdir()
            dashboard_recorder = RawVideoRecorder(dashboard_directory)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        with ExitStack() as stack:
            stream = stack.enter_context(
                (directory / "predictions.csv").open("w", encoding="utf-8", newline="")
            )
            writer = csv.DictWriter(
                stream, fieldnames=FIELDS + eye_fields if eye_alertness else FIELDS
            )
            writer.writeheader()
            jsonl = (
                stack.enter_context((directory / "frames.jsonl").open("w", encoding="utf-8"))
                if recording
                else None
            )
            detector = stack.enter_context(vision.FaceLandmarker.create_from_options(options))
            if eye_alertness is not None:
                detector = EyeAwareDetector(detector, eye_alertness)
                print(
                    "Keep eyes naturally OPEN and look at camera for 5 seconds. R recalibrates after a subject change."
                )
            if args.session:
                source = read_raw_session(args.session)
            elif args.input_image:
                image = cv2.imread(str(args.input_image))
                if image is None:
                    raise OSError("Input image cannot be read.")
                source = [(0, 0, image)]
            else:
                capture = cv2.VideoCapture(args.camera)
                if not capture.isOpened():
                    raise OSError("Camera cannot be opened.")
                source = webcam_frames(capture)
            source_close = getattr(source, "close", None)
            if callable(source_close):
                stack.callback(source_close)
            for index, timestamp, frame in source:
                record, overlay = process_smart_frame(
                    frame, detector, timestamp, predictor, eyes, mouth
                )
                row = dict(record, frame_id=frames if recording else index, timestamp_ms=timestamp)
                if recording:
                    assert raw_recorder is not None and dashboard_recorder is not None
                    assert jsonl is not None
                    raw_recorder.write(timestamp, frame)
                    height, width = overlay.shape[:2]
                    bottom, right = height % 2, width % 2
                    dashboard_padding = dict(
                        original_width=width,
                        original_height=height,
                        padding_bottom=bottom,
                        padding_right=right,
                    )
                    padded = cv2.copyMakeBorder(
                        overlay, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    dashboard_recorder.write(timestamp, padded)
                    row["source_frame_id"] = index
                    jsonl.write(json.dumps(row) + "\n")
                    jsonl.flush()
                writer.writerow({k: row[k] for k in writer.fieldnames})
                stream.flush()
                counts[record["smart_yawn_state"]] += 1
                frames += 1
                if frames == 1 and not cv2.imwrite(str(directory / "preview.png"), overlay):
                    raise OSError("Preview could not be saved.")
                if not args.session and not args.input_image:
                    cv2.imshow(WINDOW, overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("r"), ord("R")) and eye_alertness is not None:
                        eye_alertness.reset()
                    if key in (ord("q"), ord("Q")):
                        reason = "q"
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        reason = "window_closed"
                        break
            else:
                reason = "source_complete"
    except BaseException as error:
        failure = error
        reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        raise
    finally:
        cleanup_errors = []
        for close in ([capture.release] if capture is not None else []) + [cv2.destroyAllWindows]:
            try:
                close()
            except BaseException as error:
                cleanup_errors.append(error)
        summary = dict(
            frames=frames,
            exit_reason=reason,
            states=dict(counts),
            model_sha256=predictor.model_hash,
            target=predictor.model["target"],
            source=str(args.session or args.input_image or f"webcam:{args.camera}"),
            experimental=True,
            calibrated_probability=False,
            final_eye_decision_unchanged=True,
            adaptive_eye_display_enabled=args.eye_alertness,
            eye_tier_policy="experimental closure duration 1.5/3/8 seconds; not clinical sleep stages",
            closed_fraction_note="30s EAR-based closed fraction, not validated 80-percent pupil PERCLOS",
            processing_ms_scope="existing detection plus causal ML, excludes capture/disk/GUI",
        )
        if recording:
            for key, recorder in (
                ("raw_video", raw_recorder),
                ("dashboard_video", dashboard_recorder),
            ):
                metadata = dict(enabled=True, complete=False, frames=0, verified_frames=0)
                if recorder is not None:
                    try:
                        metadata = dict(recorder.close())
                    except BaseException as error:
                        cleanup_errors.append(error)
                        metadata.update(
                            recorder.metadata or {}, frames=recorder.frames, complete=False
                        )
                        # A release failure must not leave the other resource open.
                        for resource in (recorder.writer, recorder.mapping):
                            if resource is not None:
                                try:
                                    if resource is recorder.writer:
                                        resource.release()
                                    else:
                                        resource.close()
                                except BaseException as release_error:
                                    cleanup_errors.append(release_error)
                summary[key] = metadata
            summary["dashboard_video"].update(
                overlay=True, directory="dashboard", **dashboard_padding
            )
            if failure is not None or cleanup_errors:
                for key in ("raw_video", "dashboard_video"):
                    summary[key]["complete"] = False
        if cleanup_errors:
            summary["exit_reason"] = "error"
            summary["recording_errors"] = [repr(error) for error in cleanup_errors]
        if failure is not None:
            summary["error"] = repr(failure)
        try:
            (directory / "session.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if failure is not None:
                for error in cleanup_errors:
                    failure.add_note(f"Cleanup failure: {error!r}")
            else:
                raise cleanup_errors[0]
    print(json.dumps(dict(output=str(directory), **summary), indent=2))


def webcam_frames(capture):
    start = monotonic_ns()
    index = 0
    previous = -1
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise OSError("Camera frame unavailable.")
        timestamp = max((monotonic_ns() - start) // 1_000_000, previous + 1)
        previous = timestamp
        yield index, timestamp, frame
        index += 1


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Experimental real-time EAR/MAR yawn candidate model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "smart_yawn.json",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--session", type=Path, help="Headless raw replay with original timestamps."
    )
    source.add_argument("--input-image", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--eye-alertness", action="store_true", help="Experimental adaptive eye-only visual tiers."
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Record lossless raw and dashboard videos."
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
