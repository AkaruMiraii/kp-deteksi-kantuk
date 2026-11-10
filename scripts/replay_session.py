import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
from mediapipe.tasks.python import BaseOptions, vision

from .eye_duration import EyeClosureTracker
from .landmark_diagnostics import DiagnosticSession, process_frame
from .mouth_duration import MouthOpeningTracker
from .raw_video import file_sha256, read_raw_session


def replay_analysis(directory, summary):
    model = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
    if file_sha256(model) != summary["model_sha256"]:
        raise ValueError("Model hash differs from source session.")
    eye_rule = summary["eye_duration_rule"]
    mouth_rule = summary["mouth_duration_rule"]
    eyes = (
        EyeClosureTracker(
            eye_rule["threshold"], eye_rule["duration_seconds"], eye_rule["max_gap_seconds"]
        )
        if eye_rule["enabled"]
        else None
    )
    mouth = (
        MouthOpeningTracker(
            mouth_rule["threshold"], mouth_rule["duration_seconds"], mouth_rule["max_gap_seconds"]
        )
        if mouth_rule["enabled"]
        else None
    )
    from .drowsiness_decision import DECISION_POLICY

    metadata = dict(summary)
    for key in ("raw_video", "exit_reason_override"):
        metadata.pop(key, None)
    metadata.update(
        source_session=str(directory.resolve()),
        source="raw_session_replay",
        activity_labels="unmarked during replay; not used as prediction input",
    )
    metadata["decision_fusion"] = dict(
        policy=DECISION_POLICY, mouth_evidence_validated=False, full_project_acceptance=False
    )
    output = DiagnosticSession(directory.parent / "replays", metadata, max_snapshots=0)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    reason = "error"
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            for _, timestamp, raw in read_raw_session(directory):
                record, overlay = process_frame(raw, detector, timestamp, "unmarked", eyes, mouth)
                output.write(timestamp, record, raw, overlay)
        reason = "replay_complete"
    finally:
        output.close(reason)
    return output.directory


def main():
    parser = argparse.ArgumentParser(
        description="Verify or replay local raw video with capture timestamps."
    )
    parser.add_argument("session", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument(
        "--analyze", action="store_true", help="Rerun model headlessly, preserving source files."
    )
    args = parser.parse_args()
    count = 0
    first = None
    last = None
    for _, timestamp, _ in read_raw_session(args.session):
        if first is None:
            first = timestamp
        last = timestamp
        count += 1
    report = dict(
        verified_frames=count,
        first_timestamp_ms=first,
        last_timestamp_ms=last,
        raw_pixels_verified=True,
    )
    if args.verify_only:
        print(json.dumps(report))
        return
    if args.analyze:
        summary = json.loads((args.session / "session.json").read_text(encoding="utf-8"))
        report["replay_output"] = str(replay_analysis(args.session, summary))
        print(json.dumps(report))
        return
    print("Raw playback: Q exit. Capture time comes from video_frames.csv, not AVI duration.")
    start = perf_counter()
    try:
        for _, timestamp, raw in read_raw_session(args.session):
            target = start + (timestamp - first) / 1000
            cv2.imshow("KP - Raw playback (no predictions)", raw)
            delay = max(1, round((target - perf_counter()) * 1000))
            if cv2.waitKey(delay) & 0xFF in (ord("q"), ord("Q")):
                break
            if (
                cv2.getWindowProperty("KP - Raw playback (no predictions)", cv2.WND_PROP_VISIBLE)
                < 1
            ):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
