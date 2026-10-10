from pathlib import Path
from time import monotonic_ns

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

if __package__:
    from .eye_features import calculate_ear
else:
    from eye_features import calculate_ear


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "face_landmarker.task"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    # Urutan setiap daftar: P1, P2, P3, P4, P5, P6.
    eye_points = {
        "R": [33, 160, 158, 133, 153, 144],
        "L": [362, 385, 387, 263, 373, 380],
    }

    mouth_points = {
        "C1": 78,
        "C2": 308,
        "TOP": 13,
        "BOTTOM": 14,
    }

    camera = cv2.VideoCapture(0)

    try:
        if not camera.isOpened():
            raise RuntimeError("Kamera tidak berhasil dibuka.")

        with vision.FaceLandmarker.create_from_options(options) as detector:
            start_ns = monotonic_ns()
            previous_timestamp_ms = -1

            print("Klik jendela video, lalu tekan Q untuk keluar.")

            while True:
                success, frame = camera.read()

                if not success or frame is None:
                    raise RuntimeError("Frame kamera gagal dibaca.")

                # Timestamp harus meningkat pada setiap pemanggilan.
                elapsed_ms = (monotonic_ns() - start_ns) // 1_000_000
                timestamp_ms = max(elapsed_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                result = detector.detect_for_video(image, timestamp_ms)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    height, width = frame.shape[:2]

                    ear_values = {}

                    for eye_name, indices in eye_points.items():
                        points = [
                            (
                                landmarks[index].x * width,
                                landmarks[index].y * height,
                            )
                            for index in indices
                        ]

                        try:
                            ear_values[eye_name] = calculate_ear(points)
                        except ValueError:
                            ear_values[eye_name] = None

                    for eye_name, indices in eye_points.items():
                        color = (0, 255, 0) if eye_name == "R" else (255, 255, 0)

                        for point_number, landmark_index in enumerate(indices, start=1):
                            landmark = landmarks[landmark_index]

                            x = round(landmark.x * width)
                            y = round(landmark.y * height)

                            if 0 <= x < width and 0 <= y < height:
                                cv2.circle(frame, (x, y), 2, color, -1)

                                cv2.putText(
                                    frame,
                                    f"{eye_name}{point_number}",
                                    (x + 3, y - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3,
                                    color,
                                    1,
                                )

                    for label, landmark_index in mouth_points.items():
                        landmark = landmarks[landmark_index]

                        x = round(landmark.x * width)
                        y = round(landmark.y * height)

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

                            cv2.putText(
                                frame,
                                label,
                                (x + 4, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (0, 255, 255),
                                1,
                            )

                    right_ear = ear_values["R"]
                    left_ear = ear_values["L"]

                    right_text = "N/A" if right_ear is None else f"{right_ear:.3f}"
                    left_text = "N/A" if left_ear is None else f"{left_ear:.3f}"

                    message = f"EAR R: {right_text} | L: {left_text}"

                    color = (
                        (0, 255, 0)
                        if right_ear is not None and left_ear is not None
                        else (0, 165, 255)
                    )

                else:
                    message = "No face detected"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    message,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                cv2.imshow("KP - Landmark Wajah", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
