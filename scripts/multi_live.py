"""Experimental integrated local runner. Offline session never opens camera."""

import argparse
import json
import secrets
import shutil
import sys
from pathlib import Path
from time import monotonic_ns, perf_counter
from uuid import uuid4

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .detection_config import DetectionConfig
from .enrolled_identity import EnrolledIdentity
from .face_authenticity import FaceAuthenticity
from .multi_display import MonitorUI
from .multi_person import MultiPersonPipeline
from .person_selection import PersonSelection
from .raw_video import RawVideoRecorder, file_sha256, read_raw_session


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--sound-alert", action="store_true")
    parser.add_argument("--threshold-only", action="store_true")
    parser.add_argument("--session-identity", action="store_true")
    parser.add_argument("--enrollment-seconds", type=float, default=60)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--max-storage-mb", type=int, default=512)
    parser.add_argument("--max-people", type=int, default=6, choices=tuple(range(1, 7)))
    for name, default in [
        ("ear-threshold", 0.12),
        ("mar-threshold", 0.5),
        ("closure-seconds", 1.5),
        ("mouth-seconds", 2),
        ("focus-seconds", 2),
        ("max-gap-seconds", 0.5),
    ]:
        parser.add_argument("--" + name, type=float, default=default)
    args = parser.parse_args(argv)
    try:
        args.config = DetectionConfig(
            **{
                name: getattr(args, name)
                for name in (
                    "ear_threshold",
                    "mar_threshold",
                    "closure_seconds",
                    "mouth_seconds",
                    "focus_seconds",
                    "max_gap_seconds",
                    "max_people",
                )
            }
        )
        if not 5 <= args.enrollment_seconds <= 300:
            raise ValueError("Enrollment duration must be 5..300 seconds")
        if not 0 < args.max_seconds <= 1800 or not 16 <= args.max_storage_mb <= 2048:
            raise ValueError("Duration must be 0..1800 seconds, storage 16..2048 MB")
    except ValueError as error:
        parser.error(str(error))
    return args


def handle_participant_key(selection, rows, key, pipeline, timestamp, events):
    selected = selection.update(rows, key)
    if selected is not None:
        row = next(r for r in rows if r["track_id"] == selected)
        identity = selected
        state = pipeline.states[identity]
        event = None
        if key in (ord("["), ord("]")) and identity < 0:
            options = [0] + row.get("identity_confirm_options", [])
            index = getattr(pipeline, "confirmation_index", 0)
            pipeline.confirmation_index = (index + (1 if key == ord("]") else -1)) % len(options)
        if key == ord("v") and identity < 0:
            options = [0] + row.get("identity_confirm_options", [])
            choice = options[getattr(pipeline, "confirmation_index", 0) % len(options)]
            try:
                restored = pipeline.identity_engine.confirm(-identity, choice, timestamp)
                pending = pipeline.states.pop(identity)
                if restored in pipeline.states:
                    pipeline.merge_pending(pipeline.states[restored], pending, timestamp / 1000)
                else:
                    from .person_state import PersonState

                    pipeline.states[restored] = PersonState(restored, pipeline.config)
                    pipeline.merge_pending(pipeline.states[restored], pending, timestamp / 1000)
                selection.track_id = restored
                event = "operator_identity_confirmed"
            except ValueError:
                event = "operator_identity_rejected"
        if key == ord("c") and row["ear_r"] is not None and row["ear_l"] is not None:
            try:
                state.calibrate_from_history()
                event = "open_eye_baseline"
            except ValueError:
                event = "calibration_rejected"
        if key == ord("r"):
            pipeline.reset_track(identity)
            event = "reset"
        if key == ord("t"):
            if row["authenticity"] == "live" and row["eye_state"] == "no_bilateral_closure":
                state.task_probe.start(timestamp, secrets.choice(("left", "right")))
                event = "task_probe_started"
            else:
                event = "task_probe_rejected"
        if key in (ord("a"), ord("d")):
            response = "left" if key == ord("a") else "right"
            event = (
                "task_response_accepted"
                if state.task_probe.respond(timestamp, response)
                else "task_response_rejected"
            )
        if key in (ord("z"), ord("n")):
            state.set_zone_out_label(key == ord("z"), "self_report")
            event = "self_report"
        if event:
            events.write(
                json.dumps(
                    dict(
                        timestamp_ms=timestamp,
                        track_id=identity,
                        event=event,
                        key=chr(key),
                    )
                )
                + "\n"
            )
            events.flush()


def finalize_session(failure, resources, result, args, models, start):
    source, cap, recorder = resources
    output, count, reason = result
    model, anti_model = models
    cleanup_errors = []
    actions = (
        ([source.close] if source is not None else [])
        + ([cap.release] if cap is not None else [])
        + [cv2.destroyAllWindows]
    )
    for action in actions:
        try:
            action()
        except BaseException as error:
            cleanup_errors.append(error)
    raw = None
    if recorder is not None:
        try:
            raw = recorder.close()
        except BaseException as error:
            cleanup_errors.append(error)
            raw = dict(recorder.metadata or {}, complete=False, frames=recorder.frames)
    if failure is not None or cleanup_errors:
        reason = "error"
        if raw is not None:
            raw["complete"] = False
    summary = dict(
        frames=count,
        exit_reason=reason,
        config=args.config.to_dict(),
        source=str(args.session or f"webcam:{args.camera}"),
        raw_video=raw,
        landmark_sha256=file_sha256(model),
        anti_spoof_sha256=file_sha256(anti_model),
        experimental=True,
        threshold_only=getattr(args, "threshold_only", False),
        session_identity=getattr(args, "session_identity", False),
        identity_storage="session_ram_only",
        validated_attack_types=[],
        elapsed_seconds=perf_counter() - start,
        cleanup_errors=[str(e) for e in cleanup_errors],
    )
    for filename in ("summary.json", "session.json"):
        try:
            (output / filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except BaseException as error:
            cleanup_errors.append(error)
    if failure is not None:
        for error in cleanup_errors:
            failure.add_note(f"Cleanup error: {error}")
    elif cleanup_errors:
        raise cleanup_errors[0]
    return summary


def open_source(args):
    cap = None
    if args.session:
        source = read_raw_session(args.session)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise OSError("Camera cannot be opened")

        def camera_frames():
            origin = monotonic_ns()
            previous = -1
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise OSError("Camera frame unavailable")
                timestamp = max(previous + 1, (monotonic_ns() - origin) // 1000000)
                previous = timestamp
                yield index, timestamp, frame
                index += 1

        source = camera_frames()
    return source, cap


def run(args):
    root = Path(__file__).resolve().parents[1]
    model = root / "models/face_landmarker.task"
    anti_model = root / "models/anti_spoof/2.7_80x80_MiniFASNetV2.pth"
    threshold_only = getattr(args, "threshold_only", False)
    engine = None if threshold_only else FaceAuthenticity(anti_model)
    identity = (
        EnrolledIdentity(root / "models/sface.onnx", args.enrollment_seconds)
        if getattr(args, "session_identity", False)
        else None
    )
    pipeline = MultiPersonPipeline(
        args.config, engine, threshold_only=threshold_only, identity_engine=identity
    )
    output = root / "outputs/multiperson" / uuid4().hex[:12]
    output.mkdir(parents=True)
    cap = recorder = source = None
    selection = PersonSelection()
    ui_time = [0.0]
    monitor = MonitorUI(clock=lambda: ui_time[0], sound=getattr(args, "sound_alert", False))
    count = 0
    reason = "error"
    start = perf_counter()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=args.config.max_people,
    )
    try:
        if args.record_video:
            recorder = RawVideoRecorder(output)
        source, cap = open_source(args)
        with (
            vision.FaceLandmarker.create_from_options(options) as detector,
            (output / "persons.jsonl").open("x", encoding="utf-8") as log,
            (output / "events.jsonl").open("x", encoding="utf-8") as events,
            (output / "frames.jsonl").open("x", encoding="utf-8") as mapping,
        ):
            first_timestamp = None
            for index, timestamp, frame in source:
                ui_time[0] = timestamp / 1000
                if first_timestamp is None:
                    first_timestamp = timestamp
                if timestamp - first_timestamp >= args.max_seconds * 1000:
                    reason = "duration_limit"
                    break
                used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if (
                    used >= args.max_storage_mb * 1024 * 1024
                    or shutil.disk_usage(output).free < 100 * 1024 * 1024
                ):
                    reason = "storage_limit"
                    break
                rgb = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                result = detector.detect_for_video(rgb, timestamp)
                rows = pipeline.update(
                    timestamp, result, frame.shape[1], frame.shape[0], frame=frame
                )
                if len(rows) == 1 and selection.track_id is None:
                    selection.track_id = rows[0]["track_id"]
                if getattr(args, "session_identity", False):
                    for row in rows:
                        if row.get("identity_pending"):
                            options = [0] + row.get("identity_confirm_options", [])
                            row["confirmation_choice"] = options[
                                getattr(pipeline, "confirmation_index", 0) % len(options)
                            ]
                monitor.update(rows)
                for calibration_event in monitor.auto_calibrate(rows, pipeline):
                    events.write(json.dumps(dict(calibration_event, timestamp_ms=timestamp)) + "\n")
                    events.flush()
                if recorder:
                    recorder.write(timestamp, frame)
                    mapping.write(
                        json.dumps(
                            dict(frame_id=count, timestamp_ms=timestamp, source_frame_id=index)
                        )
                        + "\n"
                    )
                    mapping.flush()
                for row in rows:
                    log.write(
                        json.dumps(
                            dict(row, frame_id=index, config=args.config.to_dict()), allow_nan=False
                        )
                        + "\n"
                    )
                log.flush()
                display = monitor.render(
                    frame, rows, selected_id=selection.track_id, config=args.config
                )
                count += 1
                if (
                    args.save_preview
                    and count == 1
                    and not cv2.imwrite(str(output / "preview.png"), display)
                ):
                    raise OSError("Preview write failed")
                if args.session:
                    continue
                cv2.imshow("KP / Multi-person experimental", display)
                key = cv2.waitKey(1) & 255
                monitor.handle_key(key)
                if (
                    key == ord("q")
                    or cv2.getWindowProperty("KP / Multi-person experimental", cv2.WND_PROP_VISIBLE)
                    < 1
                ):
                    reason = "user_exit"
                    break
                handle_participant_key(selection, rows, key, pipeline, timestamp, events)
            else:
                reason = "source_complete"
    finally:
        summary = finalize_session(
            sys.exc_info()[1],
            (source, cap, recorder),
            (output, count, reason),
            args,
            (model, anti_model),
            start,
        )
    print(json.dumps(dict(output=str(output), **summary)))


if __name__ == "__main__":
    run(parse_args())
