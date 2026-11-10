import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

from .raw_video import read_raw_session

LABELS = {
    ord("1"): "eyes_closed_both",
    ord("2"): "eyes_open",
    ord("3"): "mouth_relaxed",
    ord("4"): "mouth_deliberate_opening",
    ord("5"): "mouth_speaking",
    ord("6"): "mouth_yawn_like",
    ord("7"): "eyes_not_visible",
    ord("8"): "mouth_not_visible",
    ord("9"): "eyes_uncertain",
    ord("0"): "mouth_uncertain",
}
FIELDS = [
    "session_id",
    "channel",
    "label",
    "start_frame",
    "end_frame",
    "start_timestamp_ms",
    "end_timestamp_ms",
    "annotator_id",
]
WINDOW = "KP - Raw annotation"


class AnnotationEditor:
    def __init__(self, timestamps, session_id, video_hash, mapping_hash):
        if not timestamps or any(t < 0 for t in timestamps):
            raise ValueError("Annotation requires nonnegative source timestamps.")
        if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("Source timestamps must strictly increase.")
        self.timestamps = timestamps
        self.session_id = session_id
        self.video_hash = video_hash
        self.mapping_hash = mapping_hash
        self.index = 0
        self.start = None
        self.end = None
        self.intervals = []
        self.message = "Pause at first fully closed frame. B=start, E=end, 1=commit label."
        self.dirty = False
        self.last_saved = None

    def seek(self, index):
        self.index = max(0, min(index, len(self.timestamps) - 1))

    def mark_start(self):
        self.start = self.index
        self.end = None
        self.message = f"Start marked at frame {self.start}. Find last frame, then E."

    def mark_end(self):
        self.end = self.index
        self.message = f"End marked at frame {self.end}. Press a label key to commit."

    def commit(self, label):
        if label not in LABELS.values():
            raise ValueError("Unknown annotation label.")
        if self.start is None or self.end is None or self.start > self.end:
            raise ValueError("Mark B at start and E at end in forward order first.")
        channel = "eyes" if label.startswith("eyes_") else "mouth"
        for interval in self.intervals:
            if interval["channel"] == channel:
                if self.start <= interval["end_frame"] and self.end >= interval["start_frame"]:
                    raise ValueError(
                        "Interval overlaps a label in the same channel. U=undo last label."
                    )
        self.intervals.append(
            dict(
                session_id=self.session_id,
                channel=channel,
                label=label,
                start_frame=self.start,
                end_frame=self.end,
                start_timestamp_ms=self.timestamps[self.start],
                end_timestamp_ms=self.timestamps[self.end],
            )
        )
        self.message = f"Added {label}: frames {self.start}..{self.end}. S=save."
        self.start = None
        self.end = None
        self.dirty = True

    def save(self, directory, annotator_id):
        if not annotator_id.strip():
            raise ValueError("Annotator ID is required.")
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
        path = directory / f"intervals_{name}.csv"
        with path.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            for interval in self.intervals:
                writer.writerow(dict(interval, annotator_id=annotator_id))
        metadata = dict(
            session_id=self.session_id,
            video_sha256=self.video_hash,
            mapping_sha256=self.mapping_hash,
            annotator_id=annotator_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
            rubric_version="pilot_visual_intervals_v1",
            interval_bounds="inclusive first and last visible frame",
            annotation_complete=False,
            unreviewed_default="unknown",
            intervals=len(self.intervals),
            pending_selection_not_saved=self.start is not None,
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            labels=list(LABELS.values()),
        )
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.dirty = False
        self.last_saved = path
        self.message = f"Saved {len(self.intervals)} intervals. Unmarked frames remain unknown."
        return path

    def feedback(self):
        if self.start is not None:
            if self.end is None:
                return "SELECTING INTERVAL", "Press E to mark the end", (45, 180, 245)
            if self.end < self.start:
                return "INVALID RANGE", "End is before start. Press E again", (80, 80, 240)
            return "END MARKED", "Choose a label: 1 = eyes closed", (230, 180, 70)
        if self.dirty:
            return "LABEL ADDED - UNSAVED", "Press S to save to disk", (45, 180, 245)
        if self.last_saved is not None:
            return "SAVED TO DISK", "Press B to start another interval", (110, 210, 100)
        return "NOT SELECTING", "Press B to mark the start", (190, 190, 190)

    def render(self, raw, playing=False):
        height, width = raw.shape[:2]
        display = np.full((height + 300, max(width + 320, 960), 3), (25, 20, 16), dtype=np.uint8)
        display[:height, :width] = raw
        title, instruction, color = self.feedback()
        x = width + 16
        cv2.rectangle(display, (x, 18), (display.shape[1] - 16, 66), color, -1)
        cv2.putText(
            display,
            "PLAYING" if playing else "PAUSED",
            (x + 12, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (15, 20, 25),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "ANNOTATION STATUS",
            (x, 103),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (225, 230, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(display, title, (x, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(
            display,
            instruction,
            (x, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (225, 230, 235),
            1,
            cv2.LINE_AA,
        )
        details = [
            f"START: {self.start if self.start is not None else '-'}",
            f"END: {self.end if self.end is not None else '-'}",
            f"CURRENT FRAME: {self.index}",
            f"Labels in draft: {len(self.intervals)}",
            "SAVE: pending"
            if self.dirty
            else "SAVE: up to date"
            if self.last_saved
            else "SAVE: nothing saved",
        ]
        if self.intervals:
            details += ["Last label:", self.intervals[-1]["label"]]
        for line, text in enumerate(details):
            cv2.putText(
                display,
                text,
                (x, 214 + line * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (225, 230, 235),
                1,
                cv2.LINE_AA,
            )
        self.draw_timeline(display, height)
        lines = [
            f"RAW ONLY | Frame {self.index}/{len(self.timestamps) - 1} | Source time {self.timestamps[self.index] / 1000:.3f}s",
            "SPACE play/pause | A/D: -/+1 frame | J/L: -/+30 frames | slider: seek",
            "B: start | E: end | label key: add interval | X: cancel marks | U: undo last",
            "1 eyes closed (both) | 2 eyes open | 7 eyes not visible | 9 eyes uncertain",
            "3 mouth relaxed | 4 deliberate opening | 5 speaking | 6 yawn-like",
            "8 mouth not visible | 0 mouth uncertain | S: save draft | Q: save and exit",
            f"Selection: {self.start}..{self.end} | Intervals: {len(self.intervals)} | Unsaved: {self.dirty}",
            self.message,
            "Unmarked frames are UNKNOWN, never automatic negative labels.",
            "New draft each launch. Earlier annotation files and source evidence are preserved.",
        ]
        for line, text in enumerate(lines):
            cv2.putText(
                display,
                text,
                (10, height + 55 + line * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (225, 230, 235),
                1,
                cv2.LINE_AA,
            )
        return display

    def draw_timeline(self, display, height):
        left, right = 12, display.shape[1] - 12
        y = height + 17

        def position(index):
            return left + round(index * (right - left) / max(1, len(self.timestamps) - 1))

        cv2.line(display, (left, y), (right, y), (70, 70, 70), 8)
        for interval in self.intervals:
            cv2.line(
                display,
                (position(interval["start_frame"]), y),
                (position(interval["end_frame"]), y),
                (110, 210, 100),
                8,
            )
        if self.start is not None:
            endpoint = self.index if self.end is None else self.end
            cv2.line(
                display,
                (position(min(self.start, endpoint)), y),
                (position(max(self.start, endpoint)), y),
                self.feedback()[2],
                8,
            )
            cv2.circle(display, (position(self.start), y), 7, (45, 180, 245), -1)
            if self.end is not None:
                cv2.circle(display, (position(self.end), y), 7, (230, 180, 70), -1)
        cv2.line(
            display,
            (position(self.index), y - 12),
            (position(self.index), y + 12),
            (255, 255, 255),
            2,
        )


def get_frame(capture, index, mapping):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, raw = capture.read()
    if not success or raw is None:
        raise OSError(f"Cannot read source frame {index}.")
    if hashlib.sha256(raw.tobytes()).hexdigest() != mapping[index]["pixel_sha256"]:
        raise ValueError(f"Seek decoded wrong pixels at frame {index}.")
    return raw


def main():
    parser = argparse.ArgumentParser(
        description="Mark raw visual intervals without detector predictions."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--annotator", default="master")
    parser.add_argument(
        "--preview-frame", type=int, help="Headless UI layout check; creates no labels."
    )
    parser.add_argument("--preview-output", type=Path)
    args = parser.parse_args()
    if (args.preview_frame is None) != (args.preview_output is None):
        parser.error("Supply both preview frame and output together.")
    if not args.annotator.strip():
        parser.error("Annotator ID cannot be empty.")
    print("Verifying source frames first. Please wait; no camera is opened.")
    count = sum(1 for _ in read_raw_session(args.session))
    if count == 0:
        raise ValueError("Cannot annotate an empty session.")
    summary = json.loads((args.session / "session.json").read_text(encoding="utf-8"))
    with (args.session / "video_frames.csv").open(encoding="utf-8", newline="") as source:
        mapping = list(csv.DictReader(source))
    timestamps = [int(row["timestamp_ms"]) for row in mapping]
    editor = AnnotationEditor(
        timestamps,
        args.session.name,
        summary["raw_video"]["video_sha256"],
        summary["raw_video"]["mapping_sha256"],
    )
    capture = cv2.VideoCapture(str(args.session / "raw.avi"))
    if args.preview_frame is not None:
        try:
            if not capture.isOpened():
                raise OSError("Cannot open raw video.")
            if not 0 <= args.preview_frame < count:
                parser.error("Preview frame outside source.")
            editor.seek(args.preview_frame)
            display = editor.render(get_frame(capture, editor.index, mapping))
            if args.preview_output.exists():
                raise FileExistsError("Preview would overwrite an existing file.")
            args.preview_output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.preview_output), display):
                raise OSError("Preview image write failed.")
            print(f"Verified {count} frames. Preview: {args.preview_output}")
        finally:
            capture.release()
        return
    run_gui(args, editor, capture, mapping)


def handle_key(key, editor, playing, deadline, directory, annotator):
    saved = None
    if key == ord(" "):
        playing = not playing
        if playing:
            if editor.index == len(editor.timestamps) - 1:
                editor.seek(0)
            deadline = perf_counter()
    elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
        playing = False
        editor.seek(editor.index + {ord("a"): -1, ord("d"): 1, ord("j"): -30, ord("l"): 30}[key])
    elif key == ord("b"):
        playing = False
        editor.mark_start()
    elif key == ord("e"):
        playing = False
        editor.mark_end()
    elif key in LABELS:
        playing = False
        try:
            editor.commit(LABELS[key])
        except ValueError as error:
            editor.message = str(error)
    elif key == ord("x"):
        editor.start = None
        editor.end = None
        editor.message = "Pending selection cancelled; saved intervals unchanged."
    elif key == ord("u"):
        if editor.intervals:
            editor.intervals.pop()
            editor.dirty = True
            editor.message = "Last draft interval undone; previous saved revisions preserved."
    elif key == ord("s"):
        playing = False
        saved = editor.save(directory, annotator)
        print(f"Saved: {saved}")
    return playing, deadline, saved


def run_gui(args, editor, capture, mapping):
    playing = False
    deadline = perf_counter()
    saved = None
    window_open = False
    try:
        if not capture.isOpened():
            raise OSError("Cannot open raw video.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        window_open = True
        cv2.createTrackbar("Frame", WINDOW, 0, len(editor.timestamps) - 1, lambda value: None)
        raw = None
        previous_index = -1
        while True:
            slider = cv2.getTrackbarPos("Frame", WINDOW)
            if slider != editor.index:
                editor.seek(slider)
                playing = False
            if editor.index != previous_index:
                raw = get_frame(capture, editor.index, mapping)
                previous_index = editor.index
            cv2.imshow(WINDOW, editor.render(raw, playing=playing))
            key = cv2.waitKey(10 if playing else 30) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                if editor.dirty:
                    saved = editor.save(args.session / "annotations", args.annotator)
                break
            playing, deadline, new_save = handle_key(
                key, editor, playing, deadline, args.session / "annotations", args.annotator
            )
            if new_save is not None:
                saved = new_save
            if playing:
                if editor.index < len(editor.timestamps) - 1:
                    interval = (
                        editor.timestamps[editor.index + 1] - editor.timestamps[editor.index]
                    ) / 1000
                    if perf_counter() >= deadline + interval:
                        editor.seek(editor.index + 1)
                        deadline += interval
                else:
                    playing = False
            cv2.setTrackbarPos("Frame", WINDOW, editor.index)
    finally:
        capture.release()
        if window_open:
            cv2.destroyAllWindows()
    if editor.start is not None:
        print("Pending start/end selection was not committed and is not an annotation.")
    print(f"Saved: {saved}" if saved else "No annotations saved. Source unchanged.")


if __name__ == "__main__":
    main()
