import cv2


def main():
    camera = cv2.VideoCapture(0)

    try:
        if not camera.isOpened():
            raise RuntimeError("Kamera tidak berhasil dibuka.")

        print("Klik jendela webcam, lalu tekan Q untuk keluar.")

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                raise RuntimeError("Frame kamera gagal dibaca.")

            cv2.imshow("KP - Tes Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
