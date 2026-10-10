import cv2
import numpy as np

BACKGROUND = (24, 20, 17)
CARD = (39, 32, 26)
TEXT = (240, 235, 229)
MUTED = (179, 163, 143)
GREEN = (158, 209, 112)
AMBER = (106, 193, 246)
RED = (137, 132, 245)
BLUE = (241, 182, 114)

EYE_LABELS = {
    "no_eye_indicator": ("BELUM ADA INDIKATOR", "MATA", GREEN),
    "closure_pending": ("MATA MENUTUP", "Menunggu durasi", AMBER),
    "drowsiness_visual_indicator": ("INDIKASI KANTUK", "Berdasarkan penutupan mata", AMBER),
    "near_sleep_visual_indicator": ("INDIKASI HAMPIR TIDUR", "Indikator visual eksperimental", RED),
    "sleep_like_visual_indicator": ("INDIKASI MENYERUPAI TIDUR", "Bukan konfirmasi tidur", RED),
    "not_assessable": ("TIDAK DAPAT DINILAI", "Periksa posisi / visibilitas mata", MUTED),
    "calibrating_open_eyes": ("KALIBRASI MATA", "Buka mata alami selama 5 detik", BLUE),
    "calibration_failed_open_eyes_required": ("ULANGI KALIBRASI", "Buka mata lalu tekan R", AMBER),
}
LEGACY_LABELS = {
    "tidak_dapat_dinilai": ("TIDAK DAPAT DINILAI", "Data belum memadai", MUTED),
    "indikasi_kantuk": ("INDIKASI KANTUK", "Aturan mata tetap", AMBER),
    "tidak_terdeteksi_indikasi_kantuk": ("BELUM ADA INDIKATOR", "Aturan mata tetap", GREEN),
}
YAWN_LABELS = {
    "no_candidate": ("Belum ada kandidat", GREEN),
    "candidate_pending": ("Menunggu durasi kandidat", AMBER),
    "yawn_candidate": ("Kandidat menguap", AMBER),
    "not_assessable": ("Tidak dapat dinilai", MUTED),
}


def draw_text(canvas, text, x, y, scale=0.5, color=TEXT, thickness=1):
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def draw_landmarks(canvas, record, width, height):
    points = record.get("points_px", {})
    groups = [
        ([33, 160, 158, 133, 153, 144], (80, 230, 110), "ear_r"),
        ([362, 385, 387, 263, 373, 380], (235, 205, 70), "ear_l"),
        ([78, 13, 308, 14], (50, 220, 245), "mar"),
    ]
    for indices, color, feature in groups:
        if record.get(feature) is None:
            continue
        selected = [points.get(index, points.get(str(index))) for index in indices]
        if any(
            p is None
            or len(p) != 2
            or not all(np.isfinite(v) for v in p)
            or not 0 <= p[0] < width
            or not 0 <= p[1] < height
            for p in selected
        ):
            continue
        positions = np.array(
            [(round(p[0]) + 16, round(p[1]) + 16) for p in selected], dtype=np.int32
        )
        cv2.polylines(canvas, [positions], True, color, 1, cv2.LINE_AA)
        for x, y in positions:
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def render_dashboard(raw, record):
    height, width = raw.shape[:2]
    camera_width = max(640, width)
    camera_height = max(480, height)
    panel_x = camera_width + 32
    canvas = np.full((camera_height + 120, camera_width + 468, 3), BACKGROUND, dtype=np.uint8)
    canvas[16 : 16 + height, 16 : 16 + width] = raw
    draw_landmarks(canvas, record, width, height)
    x = panel_x
    right = canvas.shape[1] - 16
    draw_text(canvas, "PEMANTAUAN MATA & MULUT", x, 33, 0.58, TEXT, 2)
    draw_text(canvas, "Lokal | mode pengembangan", x, 57, 0.43, MUTED)
    if "eye_alertness_state" in record:
        title, subtitle, color = EYE_LABELS[record["eye_alertness_state"]]
        duration = record.get("eye_closed_duration_seconds", 0)
    else:
        title, subtitle, color = LEGACY_LABELS[record["final_status"]]
        duration = record.get("eye_duration_seconds")
    cv2.rectangle(canvas, (x, 74), (right, 202), CARD, -1)
    cv2.rectangle(canvas, (x, 74), (x + 4, 202), color, -1)
    draw_text(canvas, "STATUS MATA", x + 16, 97, 0.4, MUTED)
    draw_text(canvas, title, x + 16, 128, 0.52, color, 2)
    draw_text(canvas, subtitle, x + 16, 151, 0.43, MUTED)
    draw_text(canvas, f"Durasi tertutup: {number(duration, 2)} s", x + 16, 183, 0.55, TEXT)
    cv2.rectangle(canvas, (x, 218), (right, 373), CARD, -1)
    label, color = YAWN_LABELS[record["smart_yawn_state"]]
    draw_text(canvas, "PREDIKSI MENGUAP", x + 16, 242, 0.4, MUTED)
    draw_text(canvas, label, x + 16, 273, 0.6, color, 2)
    score = record.get("smart_yawn_score")
    draw_text(canvas, f"Skor: {number(score)}", x + 16, 303, 0.54)
    cv2.rectangle(canvas, (x + 16, 315), (right - 16, 323), (70, 58, 47), -1)
    if score is not None:
        extent = round(max(0, min(1, score)) * (right - x - 32))
        cv2.rectangle(canvas, (x + 16, 315), (x + 16 + extent, 323), color, -1)
    draw_text(canvas, "Skor model, bukan probabilitas", x + 16, 344, 0.4, MUTED)
    draw_text(
        canvas,
        f"Durasi kandidat: {number(record.get('smart_yawn_duration_seconds'), 2)} s",
        x + 16,
        364,
        0.42,
        MUTED,
    )
    draw_text(canvas, "PENGUKURAN", x, 400, 0.4, MUTED)
    ear_r = record.get("independent_ear_r", record.get("ear_r"))
    ear_l = record.get("independent_ear_l", record.get("ear_l"))
    draw_text(canvas, f"EAR kanan  {number(ear_r)}", x, 426, 0.5)
    draw_text(canvas, f"EAR kiri      {number(ear_l)}", x + 210, 426, 0.5)
    draw_text(canvas, f"MAR             {number(record.get('mar'))}", x, 454, 0.5)
    draw_text(
        canvas, f"Proses {number(record.get('processing_ms'), 1)} ms", x + 210, 454, 0.45, MUTED
    )
    limits = (record.get("eye_threshold_right"), record.get("eye_threshold_left"))
    draw_text(
        canvas, f"Batas EAR R/L: {number(limits[0])} / {number(limits[1])}", x, 482, 0.42, MUTED
    )
    footer = camera_height + 32
    cv2.line(canvas, (16, footer - 6), (canvas.shape[1] - 16, footer - 6), (61, 51, 43), 1)
    draw_text(canvas, "Mata kanan: hijau | kiri: biru | mulut: kuning", 16, footer + 17, 0.5)
    draw_text(canvas, "R  Kalibrasi ulang mata     Q  Keluar dan simpan log", x, footer + 17, 0.42)
    draw_text(
        canvas,
        "Indikator visual eksperimental, bukan kepastian kantuk atau tidur.",
        16,
        footer + 47,
        0.43,
        MUTED,
    )
    draw_text(
        canvas,
        "Tutup mata sengaja dan oklusi dapat menyerupai indikator; lihat hasil review.",
        16,
        footer + 73,
        0.41,
        MUTED,
    )
    return canvas
