"""Offline-only multi-face replay. Never opens a webcam."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .detection_config import DetectionConfig
from .face_authenticity import FaceAuthenticity
from .multi_person import MultiPersonPipeline
from .raw_video import file_sha256, read_raw_session


def write_replay(directory, frames, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    count = observations = maximum = 0
    with (directory / "persons.jsonl").open("x", encoding="utf-8") as stream:
        for frame_id, timestamp, rows in frames:
            count += 1
            observations += len(rows)
            maximum = max(maximum, len(rows))
            for row in rows:
                stream.write(json.dumps(dict(row, frame_id=frame_id), allow_nan=False) + "\n")
    elapsed = perf_counter() - started
    summary = dict(
        metadata,
        frames=count,
        observations=observations,
        maximum_faces=maximum,
        elapsed_seconds=elapsed,
        throughput_fps=count / elapsed if elapsed else None,
        validated_accuracy=False,
        authenticity_integrated=True,
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = DetectionConfig()
    engine = FaceAuthenticity(root / "models/anti_spoof/2.7_80x80_MiniFASNetV2.pth")
    pipeline = MultiPersonPipeline(config, engine)
    model = root / "models/face_landmarker.task"
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=config.max_people,
    )
    output = root / "outputs/multiperson_replay" / uuid4().hex[:12]
    with vision.FaceLandmarker.create_from_options(options) as detector:

        def frames():
            source = read_raw_session(args.session)
            try:
                for index, timestamp, frame in source:
                    rgb = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )
                    result = detector.detect_for_video(rgb, timestamp)
                    rows = pipeline.update(
                        timestamp, result, frame.shape[1], frame.shape[0], frame=frame
                    )
                    yield index, timestamp, rows
            finally:
                source.close()

        summary = write_replay(
            output,
            frames(),
            dict(
                source=str(args.session), model_sha256=file_sha256(model), config=config.to_dict()
            ),
        )
    print(json.dumps(dict(output=str(output), **summary)))


if __name__ == "__main__":
    main()
