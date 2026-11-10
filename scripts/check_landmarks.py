from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


def main():
    # Lokasi model dihitung dari lokasi skrip, bukan folder terminal
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "face_landmarker.task"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    camera = cv2.VideoCapture(0)

    try:
        if not camera.isOpened():
            raise RuntimeError("Kamera tidak berhasil dibuka.")

        input("Hadapkan wajah ke kamera, lalu tekan Enter untuk mengambil frame.")

        success, frame = camera.read()

        if not success or frame is None:
            raise RuntimeError("Frame kamera gagal dibaca.")

        # OpenCV menghasilkan BGR, sedangkan input ini memerlukan RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        with vision.FaceLandmarker.create_from_options(options) as detector:
            result = detector.detect(image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            print(f"BERHASIL: terdeteksi {len(landmarks)} landmark pada satu wajah.")
        else:
            print("Frame berhasil diproses, tetapi wajah tidak terdeteksi.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
