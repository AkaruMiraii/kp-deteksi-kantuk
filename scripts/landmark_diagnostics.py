import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from math import dist, isfinite
from pathlib import Path
from time import get_clock_info, monotonic_ns, perf_counter_ns
from uuid import uuid4

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .drowsiness_decision import (
    DECISION_POLICY,
    STATUS_INDIKASI_KANTUK,
    STATUS_TIDAK_DAPAT_DINILAI,
    STATUS_TIDAK_TERDETEKSI,
    evaluate_drowsiness,
)
from .eye_duration import EyeClosureTracker
from .eye_features import calculate_ear
from .mouth_duration import MouthOpeningTracker
from .raw_video import RawVideoRecorder

EYES = {"R": [33, 160, 158, 133, 153, 144], "L": [362, 385, 387, 263, 373, 380]}
MOUTH = {"C1": 78, "C2": 308, "TOP": 13, "BOTTOM": 14}
ACTIVITIES = {
    ord("1"): "neutral",
    ord("2"): "eyes_closed",
    ord("3"): "mouth_open",
    ord("4"): "speaking",
    ord("5"): "face_out",
}
CSV_FIELDS = [
    "frame_id",
    "timestamp_ms",
    "activity",
    "face_detected",
    "landmark_count",
    "ear_r",
    "ear_l",
    "mar",
    "reason",
    "eye_state",
    "eye_duration_seconds",
    "mouth_state",
    "mouth_duration_seconds",
    "final_status",
    "drowsiness_trigger",
    "decision_reason",
    "decision_policy",
    "processing_ms",
    "snapshot_raw",
    "snapshot_overlay",
]


def calculate_mar(points):
    corner_1, corner_2, top, bottom = points
    width = dist(corner_1, corner_2)
    if width == 0.0:
        raise ValueError("Lebar mulut harus lebih besar dari nol.")
    return dist(top, bottom) / width


class DiagnosticSession:
    def __init__(self, root, metadata, interval_ms=2000, max_snapshots=120, record_video=False):
        self.directory = root / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
        (self.directory / "snapshots").mkdir(parents=True)
        self.metadata = metadata
        self.raw_recorder = RawVideoRecorder(self.directory) if record_video else None
        self.interval_ms = interval_ms
        self.max_snapshots = max_snapshots
        self.frames = 0
        self.no_face_frames = 0
        self.invalid_frames = 0
        self.snapshots = 0
        self.last_snapshot_ms = -interval_ms
        self.last_timestamp_ms = None
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.json_log = (self.directory / "frames.jsonl").open("w", encoding="utf-8")
        self.csv_log = (self.directory / "metrics.csv").open("w", encoding="utf-8", newline="")
        self.csv_writer = csv.DictWriter(self.csv_log, fieldnames=CSV_FIELDS)
        self.csv_writer.writeheader()

    def write(self, timestamp_ms, record, raw, overlay, force_snapshot=False):
        snapshot_raw = ""
        snapshot_overlay = ""
        due = timestamp_ms - self.last_snapshot_ms >= self.interval_ms
        if self.snapshots < self.max_snapshots and (due or force_snapshot):
            name = f"frame_{self.frames:06d}_{timestamp_ms:09d}"
            snapshot_raw = f"snapshots/{name}_raw.png"
            snapshot_overlay = f"snapshots/{name}_overlay.png"
            for relative, image in ((snapshot_raw, raw), (snapshot_overlay, overlay)):
                if not cv2.imwrite(str(self.directory / relative), image):
                    raise OSError(f"Gagal menyimpan snapshot: {relative}")
            self.snapshots += 1
            self.last_snapshot_ms = timestamp_ms
        if self.raw_recorder is not None:
            self.raw_recorder.write(timestamp_ms, raw)
        row = dict(
            record,
            frame_id=self.frames,
            timestamp_ms=timestamp_ms,
            snapshot_raw=snapshot_raw,
            snapshot_overlay=snapshot_overlay,
        )
        self.json_log.write(json.dumps(row, allow_nan=False) + "\n")
        self.csv_writer.writerow({field: row.get(field) for field in CSV_FIELDS})
        self.json_log.flush()
        self.csv_log.flush()
        self.frames += 1
        self.last_timestamp_ms = timestamp_ms
        if not record["face_detected"]:
            self.no_face_frames += 1
        elif record.get("reason"):
            self.invalid_frames += 1

    def close(self, exit_reason):
        if self.json_log.closed:
            return
        try:
            if self.raw_recorder is not None:
                self.metadata["raw_video"] = dict(enabled=True, complete=False)
                try:
                    self.metadata["raw_video"] = self.raw_recorder.close()
                except (OSError, ValueError, RuntimeError) as error:
                    self.metadata["raw_video"]["error"] = str(error)
                    self.metadata["exit_reason_override"] = "raw_verification_failed"
                    raise
        finally:
            self._write_summary(exit_reason)

    def _write_summary(self, exit_reason):
        try:
            summary = dict(
                self.metadata,
                started_utc=self.started_utc,
                ended_utc=datetime.now(timezone.utc).isoformat(),
                exit_reason=self.metadata.get("exit_reason_override", exit_reason),
                frames=self.frames,
                no_face_frames=self.no_face_frames,
                invalid_frames=self.invalid_frames,
                snapshots=self.snapshots,
                last_timestamp_ms=self.last_timestamp_ms,
                snapshot_interval_ms=self.interval_ms,
                max_snapshots=self.max_snapshots,
            )
            (self.directory / "session.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
        finally:
            self.json_log.close()
            self.csv_log.close()


def inspect_frame(frame, result, activity):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    record = dict(
        activity=activity,
        face_detected=bool(result.face_landmarks),
        landmark_count=0,
        ear_r=None,
        ear_l=None,
        mar=None,
        reason="",
        points_px={},
    )
    if not result.face_landmarks:
        record["reason"] = "no_face_detected"
    else:
        landmarks = result.face_landmarks[0]
        record["landmark_count"] = len(landmarks)
        indices = sorted(set(EYES["R"] + EYES["L"] + list(MOUTH.values())))
        if len(landmarks) <= max(indices):
            record["reason"] = "insufficient_landmarks"
        else:
            points = {
                index: (landmarks[index].x * width, landmarks[index].y * height)
                for index in indices
            }
            finite = all(isfinite(value) for point in points.values() for value in point)
            if not finite:
                record["reason"] = "nonfinite_coordinates"
            else:
                record["points_px"] = points
                in_bounds = all(0 <= x < width and 0 <= y < height for x, y in points.values())
                if not in_bounds:
                    record["reason"] = "selected_points_outside_image"
                else:
                    try:
                        right_ear = calculate_ear([points[index] for index in EYES["R"]])
                        left_ear = calculate_ear([points[index] for index in EYES["L"]])
                        mar = calculate_mar([points[index] for index in MOUTH.values()])
                    except ValueError as error:
                        record["reason"] = str(error)
                    else:
                        record.update(ear_r=right_ear, ear_l=left_ear, mar=mar)
                labels = [
                    (f"{eye}{number}", index, (0, 255, 0) if eye == "R" else (255, 255, 0))
                    for eye, indices in EYES.items()
                    for number, index in enumerate(indices, 1)
                ]
                labels += [(label, index, (0, 255, 255)) for label, index in MOUTH.items()]
                for label, index, color in labels:
                    x, y = points[index]
                    if 0 <= x < width and 0 <= y < height:
                        position = (round(x), round(y))
                        cv2.circle(overlay, position, 2, color, -1)
                        cv2.putText(
                            overlay,
                            label,
                            (position[0] + 4, position[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            color,
                            1,
                        )
    values = [
        "N/A" if record[key] is None else f"{record[key]:.3f}" for key in ("ear_r", "ear_l", "mar")
    ]
    lines = [f"EAR R: {values[0]} L: {values[1]} MAR: {values[2]}", f"Activity: {activity}"]
    if record["reason"]:
        lines.append(record["reason"])
    for line_number, text in enumerate(lines, 1):
        cv2.putText(
            overlay, text, (10, line_number * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
    return record, overlay


def process_frame(frame, detector, timestamp_ms, activity, eye_tracker, mouth_tracker=None):
    processing_start = perf_counter_ns()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect_for_video(image, timestamp_ms)
    record, overlay = inspect_frame(frame, result, activity)
    record.update(eye_state=None, eye_duration_seconds=None)
    text = "Eye duration: OFF"
    color = (180, 180, 180)
    if eye_tracker is not None:
        invalid = bool(record["reason"])
        status = eye_tracker.update(
            timestamp_ms / 1000,
            None if invalid else record["ear_r"],
            None if invalid else record["ear_l"],
        )
        record.update(eye_state=status.state, eye_duration_seconds=status.duration_seconds)
        text = f"Eye: {status.state} | {status.duration_seconds:.2f}s"
        color = {
            "no_bilateral_closure": (0, 255, 0),
            "closure_candidate": (0, 215, 255),
            "prolonged_closure": (0, 80, 255),
            "not_assessable": (180, 180, 180),
        }[status.state]
    cv2.putText(overlay, text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    record.update(mouth_state=None, mouth_duration_seconds=None)
    mouth_text = "Mouth duration: OFF"
    mouth_color = (180, 180, 180)
    if mouth_tracker is not None:
        status = mouth_tracker.update(
            timestamp_ms / 1000, None if record["reason"] else record["mar"]
        )
        record.update(mouth_state=status.state, mouth_duration_seconds=status.duration_seconds)
        mouth_text = f"Mouth: {status.state} | {status.duration_seconds:.2f}s"
        mouth_color = {
            "no_large_opening": (0, 255, 0),
            "opening_candidate": (0, 215, 255),
            "prolonged_opening": (0, 80, 255),
            "not_assessable": (180, 180, 180),
        }[status.state]
    cv2.putText(overlay, mouth_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mouth_color, 2)
    decision = evaluate_drowsiness(
        face_detected=record["face_detected"],
        reason=record["reason"],
        eye_state=record["eye_state"],
        mouth_state=record["mouth_state"],
    )
    record.update(
        final_status=decision.final_status,
        drowsiness_trigger=decision.trigger,
        decision_reason=decision.reason,
        decision_policy=DECISION_POLICY,
    )
    if decision.final_status == STATUS_INDIKASI_KANTUK:
        status_display = f"STATUS: INDIKASI KANTUK ({decision.trigger.upper()})"
        status_color = (0, 0, 255)
    elif decision.final_status == STATUS_TIDAK_DAPAT_DINILAI:
        status_display = "STATUS: TIDAK DAPAT DINILAI"
        status_color = (0, 165, 255)
    else:
        status_display = "STATUS: TIDAK TERDETEKSI KANTUK"
        status_color = (0, 255, 0)
    cv2.putText(overlay, status_display, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    cv2.putText(
        overlay,
        "DEV: eye decision; MAR observation only",
        (10, 183),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )
    if decision.reason in ("no_active_channels", "eye_channel_disabled"):
        cv2.putText(
            overlay, decision.reason, (10, 203), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1
        )
    record["processing_ms"] = (perf_counter_ns() - processing_start) / 1_000_000
    return record, overlay


def parse_arguments():
    parser = argparse.ArgumentParser(description="Local landmark diagnostics; recording is opt-in.")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save local silent raw video and synchronized logs.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--input-image", type=Path, help="Process one image without webcam or GUI.")
    parser.add_argument("--snapshot-interval", type=float, default=2.0)
    parser.add_argument("--max-snapshots", type=int, default=120)
    parser.add_argument("--eye-threshold", type=float)
    parser.add_argument("--eye-duration", type=float)
    parser.add_argument("--eye-max-gap", type=float)
    parser.add_argument("--mouth-threshold", type=float)
    parser.add_argument("--mouth-duration", type=float)
    parser.add_argument("--mouth-max-gap", type=float)
    args = parser.parse_args()
    if args.record_video:
        args.record = True
    parameters = (args.eye_threshold, args.eye_duration, args.eye_max_gap)
    if any(value is not None for value in parameters):
        if any(value is None or not isfinite(value) or value <= 0 for value in parameters):
            parser.error("Supply all three finite positive eye parameters.")
    mouth_parameters = (args.mouth_threshold, args.mouth_duration, args.mouth_max_gap)
    if any(value is not None for value in mouth_parameters):
        if any(value is None or not isfinite(value) or value <= 0 for value in mouth_parameters):
            parser.error("Supply all three finite positive mouth parameters.")
    if (
        not isfinite(args.snapshot_interval)
        or args.snapshot_interval < 0.1
        or args.max_snapshots < 0
    ):
        parser.error("Snapshot interval must be finite and >= 0.1; limit must be >= 0.")
    return args


def session_metadata(args, model):
    return dict(
        python=platform.python_version(),
        opencv=cv2.__version__,
        mediapipe=mp.__version__,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        source=str(args.input_image) if args.input_image else f"webcam:{args.camera}",
        eye_indices=EYES,
        mouth_indices=MOUTH,
        mar_formula="distance(13,14) / distance(78,308)",
        coordinate_space="2D pixels, float before drawing",
        mirrored=False,
        activity_labels="user supplied; not independent ground truth",
        eye_duration_rule=dict(
            enabled=args.eye_threshold is not None,
            threshold=args.eye_threshold,
            duration_seconds=args.eye_duration,
            max_gap_seconds=args.eye_max_gap,
            calibrated=False,
            bilateral_rule="both EAR below threshold",
        ),
        mouth_duration_rule=dict(
            enabled=args.mouth_threshold is not None,
            threshold=args.mouth_threshold,
            duration_seconds=args.mouth_duration,
            max_gap_seconds=args.mouth_max_gap,
            calibrated=False,
            meaning="sustained large mouth opening, not confirmed yawn",
        ),
        decision_fusion=dict(
            policy=DECISION_POLICY,
            model="provisional eye-based decision; MAR observation only",
            mouth_evidence_validated=False,
            full_project_acceptance=False,
            states=[STATUS_TIDAK_DAPAT_DINILAI, STATUS_INDIKASI_KANTUK, STATUS_TIDAK_TERDETEKSI],
            triggers=["none", "eye"],
        ),
        processing_clock="perf_counter_ns",
        processing_clock_resolution_seconds=get_clock_info("perf_counter").resolution,
        processing_ms_scope="RGB conversion, inference, feature extraction, drawing; excludes capture, disk and GUI",
        quality_checks="finite coordinates, in-image selected points and nonzero widths only; not full landmark accuracy",
    )


def main():
    args = parse_arguments()
    root = Path(__file__).resolve().parents[1]
    model = root / "models" / "face_landmarker.task"
    if not model.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {model}")
    metadata = session_metadata(args, model)
    eye_tracker = (
        EyeClosureTracker(args.eye_threshold, args.eye_duration, args.eye_max_gap)
        if args.eye_threshold is not None
        else None
    )
    mouth_tracker = (
        MouthOpeningTracker(args.mouth_threshold, args.mouth_duration, args.mouth_max_gap)
        if args.mouth_threshold is not None
        else None
    )
    session = (
        DiagnosticSession(
            root / "outputs" / "diagnostics",
            metadata,
            round(args.snapshot_interval * 1000),
            args.max_snapshots,
            record_video=args.record_video,
        )
        if args.record
        else None
    )
    camera = None
    exit_reason = "error"
    try:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        with vision.FaceLandmarker.create_from_options(options) as detector:
            if not args.input_image:
                camera = cv2.VideoCapture(args.camera)
                if not camera.isOpened():
                    raise RuntimeError("Kamera tidak berhasil dibuka.")
            print(f"Recording: {session.directory}" if session else "Recording OFF")
            print(
                "1 neutral | 2 eyes closed | 3 mouth open | 4 speaking | 5 face out | S snapshot | Q exit"
            )
            activity = "unmarked"
            force_snapshot = False
            start_ns = monotonic_ns()
            previous_ms = -1
            while True:
                if args.input_image:
                    frame = cv2.imread(str(args.input_image))
                    success = frame is not None
                else:
                    success, frame = camera.read()
                if not success or frame is None:
                    raise RuntimeError("Frame tidak dapat dibaca.")
                metadata["frame_size"] = [frame.shape[1], frame.shape[0]]
                timestamp_ms = max((monotonic_ns() - start_ns) // 1_000_000, previous_ms + 1)
                previous_ms = timestamp_ms
                record, overlay = process_frame(
                    frame, detector, timestamp_ms, activity, eye_tracker, mouth_tracker
                )
                if session:
                    session.write(timestamp_ms, record, frame, overlay, force_snapshot)
                force_snapshot = False
                if args.input_image:
                    print(json.dumps(record, allow_nan=False))
                    exit_reason = "single_image_complete"
                    break
                cv2.imshow("KP - Diagnostics (local)", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    exit_reason = "q"
                    break
                if cv2.getWindowProperty("KP - Diagnostics (local)", cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "window_closed"
                    break
                if key in ACTIVITIES:
                    activity = ACTIVITIES[key]
                    force_snapshot = True
                elif key in (ord("s"), ord("S")):
                    force_snapshot = True
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        if session:
            session.close(exit_reason)
            print(f"Saved: {session.directory}")


if __name__ == "__main__":
    main()
