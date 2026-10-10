import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawVideoRecorder:
    def __init__(self, directory):
        self.directory = directory
        self.video_path = directory / "raw.avi"
        self.map_path = directory / "video_frames.csv"
        self.writer = None
        self.mapping = None
        self.csv_writer = None
        self.size = None
        self.frames = 0
        self.previous_ms = -1
        self.closed = False
        self.metadata = None
        if self.video_path.exists() or self.map_path.exists():
            raise FileExistsError("Raw recording would overwrite existing evidence.")

    def write(self, timestamp_ms, frame):
        if self.closed:
            raise RuntimeError("Raw recorder is closed.")
        if not isinstance(timestamp_ms, int) or timestamp_ms <= self.previous_ms:
            raise ValueError("Raw timestamps must be nonnegative increasing integers.")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Raw frame must be uint8 BGR with three channels.")
        height, width = frame.shape[:2]
        if width % 2 or height % 2:
            raise ValueError("Lossless recording requires even frame dimensions.")
        if self.size is not None and self.size != (width, height):
            raise ValueError("Frame size changed during recording.")
        if self.writer is None:
            self.size = (width, height)
            self.writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, self.size
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise OSError("FFV1 video writer unavailable; raw recording cannot continue.")
            self.mapping = self.map_path.open("x", encoding="utf-8", newline="")
            self.csv_writer = csv.DictWriter(
                self.mapping, fieldnames=["frame_id", "timestamp_ms", "pixel_sha256"]
            )
            self.csv_writer.writeheader()
        self.writer.write(frame)
        self.csv_writer.writerow(
            dict(
                frame_id=self.frames,
                timestamp_ms=timestamp_ms,
                pixel_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
            )
        )
        self.mapping.flush()
        self.frames += 1
        self.previous_ms = timestamp_ms

    def close(self):
        if self.closed:
            if self.metadata is None:
                raise RuntimeError("Previous raw finalization failed.")
            return self.metadata
        self.closed = True
        if self.writer is not None:
            self.writer.release()
        if self.mapping is not None:
            self.mapping.close()
        self.metadata = dict(
            enabled=True,
            complete=False,
            frames=self.frames,
            verified_frames=0,
            codec="FFV1",
            video="raw.avi",
            mapping="video_frames.csv",
            audio=False,
            overlay=False,
            container_fps=30.0,
            timebase="video_frames.csv timestamp_ms; container time is not capture time",
        )
        if self.frames:
            verified = sum(1 for _ in decoded_frames(self.directory))
            self.metadata.update(
                complete=True,
                verified_frames=verified,
                video_sha256=file_sha256(self.video_path),
                mapping_sha256=file_sha256(self.map_path),
            )
        return self.metadata


def decoded_frames(directory):
    capture = cv2.VideoCapture(str(directory / "raw.avi"))
    try:
        if not capture.isOpened():
            raise OSError("Raw video cannot be opened.")
        with (directory / "video_frames.csv").open(encoding="utf-8", newline="") as source:
            previous_ms = -1
            for index, row in enumerate(csv.DictReader(source)):
                timestamp = int(row["timestamp_ms"])
                if int(row["frame_id"]) != index or timestamp <= previous_ms:
                    raise ValueError("Invalid video frame mapping.")
                success, frame = capture.read()
                if not success or frame is None:
                    raise OSError(f"Missing video frame {index}.")
                if hashlib.sha256(frame.tobytes()).hexdigest() != row["pixel_sha256"]:
                    raise ValueError(f"Raw pixel integrity mismatch at frame {index}.")
                previous_ms = timestamp
                yield index, timestamp, frame
        if capture.read()[0]:
            raise ValueError("Video contains extra unmapped frames.")
    finally:
        capture.release()


def read_raw_session(directory):
    directory = Path(directory)
    summary = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    metadata = summary["raw_video"]
    if not metadata["complete"]:
        raise ValueError("Raw session was not finalized successfully.")
    for filename, hash_key in [("raw.avi", "video_sha256"), ("video_frames.csv", "mapping_sha256")]:
        if file_sha256(directory / filename) != metadata[hash_key]:
            raise ValueError(f"Raw session hash mismatch: {filename}")
    count = 0
    with (directory / "frames.jsonl").open(encoding="utf-8") as source:
        for index, timestamp, frame in decoded_frames(directory):
            line = source.readline()
            if not line:
                raise ValueError("Video has no corresponding prediction row.")
            row = json.loads(line)
            if row["frame_id"] != index or row["timestamp_ms"] != timestamp:
                raise ValueError("Video and prediction log are not synchronized.")
            count += 1
            yield index, timestamp, frame
        if source.readline():
            raise ValueError("Prediction log has extra rows.")
    if count != summary["frames"] or count != metadata["verified_frames"]:
        raise ValueError("Session frame count mismatch.")
