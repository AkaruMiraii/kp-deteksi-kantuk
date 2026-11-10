"""Portable smoke checks: video model run and minimal camera probe. Saves no camera frames."""
import json, sys, time, tempfile
from pathlib import Path
import numpy as np, cv2, mediapipe
from mediapipe.tasks.python import BaseOptions, vision

APP = Path(__file__).resolve().parent
MODEL = APP / "models" / "face_landmarker.task"


def video_run(frames=120):
    path = Path(tempfile.mkdtemp()) / "synthetic.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 480))
    for i in range(frames):
        img = np.full((480, 640, 3), 40, np.uint8)
        cv2.circle(img, (320 + i % 20, 240), 90, (180, 200, 220), -1)
        w.write(img)
    w.release()
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_faces=6)
    cap = cv2.VideoCapture(str(path)); n = 0
    with vision.FaceLandmarker.create_from_options(opts) as det:
        while True:
            ok, f = cap.read()
            if not ok: break
            det.detect_for_video(mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), n * 33)
            n += 1
    cap.release(); path.unlink(); path.parent.rmdir()
    return {"video_frames_processed": n}


def camera_probe(index=0, backend="dshow", seconds=3):
    be = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[backend]
    cap = cv2.VideoCapture(index, be)
    out = {"backend": backend, "opened": cap.isOpened(), "frames": 0}
    t = time.time()
    while cap.isOpened() and time.time() - t < seconds:
        ok, f = cap.read()
        if ok:
            out["frames"] += 1; out["shape"] = list(f.shape)
    cap.release()
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "video": print(json.dumps(video_run()))
    else: print(json.dumps(camera_probe(backend=sys.argv[2] if len(sys.argv) > 2 else "dshow")))
