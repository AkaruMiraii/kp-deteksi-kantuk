"""Contextual monitor UI. Display smoothing never changes detector results."""

from time import monotonic

import cv2
import numpy as np

from .landmark_diagnostics import EYES, MOUTH

FADE_SECONDS = 1.0
CALIBRATION_MAX_GAP_SECONDS = 0.5
STATUS_FRAMES = 6
AUTO_CALIBRATION_SECONDS = 3.0
CRITICAL_SECONDS = 8.0
SOUND_COOLDOWN_SECONDS = 30.0
START_HELP_SECONDS = 6.0
VIDEO_MIN_WIDTH = 960
PANEL_WIDTH = 360
BACKGROUND = (22, 25, 31)
TEXT = (235, 239, 245)
MUTED = (165, 175, 188)
GREEN = (115, 218, 150)
YELLOW = (75, 200, 245)
RED = (100, 100, 245)
PERSON_COLORS = (
    (240, 185, 115),
    (170, 215, 140),
    (200, 150, 240),
    (130, 220, 230),
    (230, 200, 180),
    (180, 170, 245),
)


class MonitorUI:
    def __init__(self, clock=monotonic, sound=False):
        self.clock = clock
        self.started = clock()
        self.tracks = {}
        self.debug = False
        self.help = False
        self.sound = sound
        self.last_sound = float("-inf")

    def handle_key(self, key):
        if key == ord("h"):
            self.debug = not self.debug
        if key == ord("?"):
            self.help = not self.help

    def update(self, rows):
        now = self.clock()
        active = {r["track_id"] for r in rows}
        for row in rows:
            identity = row["track_id"]
            status = self.status(row)
            track = self.tracks.setdefault(
                identity,
                dict(row=row, seen=now, status=status, pending=status, frames=0, stable_since=now),
            )
            if now - track["seen"] > CALIBRATION_MAX_GAP_SECONDS:
                track["stable_since"] = now
            if track.get("missing", False):
                track["frames"] = 0
                track["pending"] = status
                track["stable_since"] = now
            track["missing"] = False
            track["seen"] = now
            track["row"] = row
            if status != track["pending"]:
                track["pending"], track["frames"] = status, 1
            else:
                track["frames"] += 1
            if track["frames"] >= STATUS_FRAMES:
                track["status"] = status
            if (
                self.sound
                and row.get("primary_status") == "indikasi_kantuk"
                and row.get("closed_seconds", 0) >= CRITICAL_SECONDS
                and now - self.last_sound >= SOUND_COOLDOWN_SECONDS
            ):
                import winsound

                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                self.last_sound = now
        for identity, track in self.tracks.items():
            if identity not in active:
                track["missing"] = True
                track["stable_since"] = now
        self.tracks = {
            k: v for k, v in self.tracks.items() if k in active or now - v["seen"] < FADE_SECONDS
        }
        return [
            (
                v["row"],
                1.0 if k in active else max(0, 1 - (now - v["seen"]) / FADE_SECONDS),
                v["status"],
            )
            for k, v in self.tracks.items()
        ]

    @staticmethod
    def status(row):
        if row.get("primary_status") == "indikasi_kantuk":
            return "closed"
        if (
            row.get("primary_status") == "tidak_dapat_dinilai"
            or row.get("observation_quality") == "invalid"
            or (row.get("authenticity") != "live" and row.get("authenticity_required", True))
        ):
            return "attention"
        return "normal"

    def auto_calibrate(self, rows, pipeline):
        now = self.clock()
        events = []
        for row in rows:
            identity = row["track_id"]
            track = self.tracks[identity]
            state = pipeline.states[identity]
            eligible = (
                not row.get("identity_pending", False)
                and row.get("observation_quality") == "valid"
                and (
                    row.get("authenticity") == "live" or not row.get("authenticity_required", True)
                )
                and row.get("head_direction") == "center"
                and row.get("raw_eye_state") == "no_bilateral_closure"
            )
            if not eligible:
                track["stable_since"] = now
                row["ui_calibration_waiting"] = True
            else:
                row["ui_calibration_waiting"] = False
            if state.baseline is not None:
                row["ui_calibration_progress"] = 1.0
                continue
            progress = min(1, (now - track["stable_since"]) / AUTO_CALIBRATION_SECONDS)
            row["ui_calibration_progress"] = progress
            if progress >= 1 and eligible:
                try:
                    state.calibrate_from_history(source="automatic_frontal_open_eye_proxy")
                except ValueError as error:
                    track["stable_since"] = now
                    row["ui_calibration_progress"] = 0.0
                    events.append(
                        dict(
                            track_id=identity, event="auto_calibration_rejected", reason=str(error)
                        )
                    )
                else:
                    row["baseline"] = state.baseline
                    events.append(
                        dict(
                            track_id=identity,
                            event="auto_calibration",
                            provenance="automatic_frontal_open_eye_proxy",
                        )
                    )
        return events

    def render(self, frame, rows, selected_id=None, config=None):
        now = self.clock()
        active = {r["track_id"] for r in rows}
        visible = [
            (
                v["row"],
                1.0 if k in active else max(0, 1 - (now - v["seen"]) / FADE_SECONDS),
                v["status"],
            )
            for k, v in self.tracks.items()
        ]
        h, w = frame.shape[:2]
        scale = max(1, VIDEO_MIN_WIDTH / w)
        vw, vh = round(w * scale), round(h * scale)
        selected = next((r for r, _, _ in visible if r["track_id"] == selected_id), None)
        panel = PANEL_WIDTH if visible else 0
        identity_extra = 80 if selected is not None and "enrollment_remaining" in selected else 0
        height = max(
            vh + 42,
            80
            + len(visible) * 36
            + identity_extra
            + ((530 if self.debug else 410) if selected else 0),
        )
        canvas = np.full((height, vw + panel, 3), BACKGROUND, dtype=np.uint8)
        canvas[:vh, :vw] = cv2.resize(frame, (vw, vh))

        def text(target, value, x, y, color=TEXT, size=0.48, width=330):
            value = str(value)
            while value and cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] > width:
                value = value[:-1]
            cv2.putText(
                target, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA
            )

        canvas = self.draw_faces(canvas, visible, selected_id, (vw, vh, scale), text)
        if not visible:
            text(canvas, "Menunggu wajah terdeteksi...", 24, 40, size=0.65, width=vw - 48)
        else:
            text(canvas, f"Peserta terlihat: {len(rows)}", vw + 16, 27, size=0.56)
        self.draw_details(canvas, selected, (selected_id, len(visible), vw), config, text)
        if self.help or self.clock() - self.started < START_HELP_SECONDS:
            text(
                canvas,
                "1-6 pilih | C kalibrasi | R reset | H detail | T tugas | A/D jawab | Q keluar",
                14,
                vh + 18,
                width=vw - 28,
                size=0.42,
            )
        text(
            canvas,
            "Mode threshold: keaslian tidak diperiksa"
            if any(not r.get("authenticity_required", True) for r in rows)
            else "Parameter pengembangan, bukan diagnosis",
            14,
            height - 14,
            MUTED,
            0.35,
            vw - 140,
        )
        text(canvas, "? bantuan", vw - 110, height - 14, MUTED, 0.35, 105)
        return canvas

    def draw_faces(self, canvas, visible, selected_id, geometry, text):
        vw, vh, scale = geometry
        labels = {
            "normal": ("OK", GREEN),
            "attention": ("Perlu perhatian", YELLOW),
            "closed": ("INDIKASI KANTUK", RED),
        }
        for index, (row, alpha, status) in enumerate(visible):
            layer = canvas.copy()
            identity, x1, y1, x2, y2 = self.draw_geometry(layer, row, vw, vh, scale)
            label, status_color = labels[status]
            if (
                not row.get("identity_pending")
                and row.get("visual_pattern") == "increased_visual_drowsiness"
            ):
                label, status_color = "INDIKASI KANTUK", RED
            if status_color == RED:
                length = min(28, max(8, (x2 - x1) // 4))
                for x, dx in ((x1, length), (x2, -length)):
                    for y, dy in ((y1, length), (y2, -length)):
                        cv2.line(layer, (x, y), (x + dx, y), RED, 4, cv2.LINE_AA)
                        cv2.line(layer, (x, y), (x, y + dy), RED, 4, cv2.LINE_AA)
            progress = row.get("ui_calibration_progress")
            if row.get("identity_pending"):
                label = "Menunggu identitas"
            elif row.get("baseline") is None and progress is not None and status != "closed":
                label = (
                    "Hadap kamera, buka mata"
                    if row.get("ui_calibration_waiting")
                    else f"Kalibrasi {round(progress * 100)}%"
                )
            ly = max(24, y1 - 8)
            lx = min(max(8, x1), max(8, vw - 240))
            shade = layer.copy()
            cv2.rectangle(shade, (lx - 4, ly - 20), (lx + 232, ly + 5), BACKGROUND, -1)
            layer = cv2.addWeighted(shade, 0.8, layer, 0.2, 0)
            text(
                layer,
                f"{'P' + str(identity) if identity > 0 else '?'}  {label}",
                lx + 17,
                ly,
                status_color,
                0.46,
                210,
            )
            if status == "normal":
                cv2.circle(layer, (lx + 5, ly - 6), 4, status_color, -1)
            else:
                text(layer, "!", lx, ly, status_color, 0.55, 16)
            if progress is not None and row.get("baseline") is None:
                cv2.line(
                    layer,
                    (x1, min(vh - 1, y2 + 4)),
                    (x1 + round(max(0, x2 - x1) * progress), min(vh - 1, y2 + 4)),
                    YELLOW,
                    3,
                )
            py = 58 + index * 36
            marker = " >" if identity == selected_id else ""
            text(
                layer,
                f"{index + 1}  {'P' + str(identity) if identity > 0 else '?'}{marker}  {label}",
                vw + 16,
                py,
                status_color,
            )
            canvas = cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0)
        return canvas

    @staticmethod
    def draw_geometry(layer, row, vw, vh, scale):
        identity = row["track_id"]
        color = PERSON_COLORS[(identity - 1) % len(PERSON_COLORS)]
        x1, y1, x2, y2 = [round(v * s) for v, s in zip(row["box"], (vw, vh, vw, vh))]
        x1, x2 = max(0, x1), min(vw - 1, x2)
        y1, y2 = max(0, y1), min(vh - 1, y2)
        cv2.rectangle(layer, (x1, y1), (x2, y2), color, 1)
        points = row.get("points_px", {})
        for contour in list(EYES.values()) + [list(MOUTH.values())]:
            values = [points.get(i, points.get(str(i))) for i in contour]
            if all(v is not None for v in values):
                pts = np.array(
                    [[round(p[0] * scale), round(p[1] * scale)] for p in values], dtype=np.int32
                )
                cv2.polylines(layer, [pts], True, color, 1, cv2.LINE_AA)
        return identity, x1, y1, x2, y2

    def draw_details(self, canvas, selected, layout, config, text):
        selected_id, count, vw = layout
        if selected is not None:
            y = 90 + count * 36
            text(
                canvas,
                f"DETAIL P{selected_id}" if selected_id > 0 else "KONFIRMASI PESERTA",
                vw + 16,
                y,
                size=0.58,
            )
            if selected.get("identity_pending"):
                y += 24
                text(canvas, "[ / ] pilih ID, V konfirmasi", vw + 16, y, YELLOW, 0.4)
                y += 23
                choice = selected.get("confirmation_choice", 0)
                text(
                    canvas,
                    "Pilihan: " + ("peserta baru" if choice == 0 else f"P{choice}"),
                    vw + 16,
                    y,
                    YELLOW,
                    0.4,
                )
            if "enrollment_remaining" in selected:
                y += 23
                text(
                    canvas,
                    f"Pendaftaran: {selected['enrollment_remaining']:.0f}s"
                    if selected["enrollment_remaining"] > 0
                    else "Daftar peserta terkunci",
                    vw + 16,
                    y,
                    MUTED,
                    0.4,
                )
            details = [
                ("EAR kanan", selected.get("ear_r")),
                ("EAR kiri", selected.get("ear_l")),
                ("MAR", selected.get("mar")),
                ("Mata tertutup (detik)", selected.get("closed_seconds")),
                ("Bukaan mulut (detik)", selected.get("mouth_seconds")),
            ]
            for name, value in details:
                if value is not None:
                    y += 25
                    text(canvas, f"{name}: {value:.3f}", vw + 16, y)
            if "eye_events_window" in selected:
                y += 24
                text(
                    canvas,
                    f"5 menit: mata {selected['eye_events_window']} | bukaan {selected['opening_events_window']}",
                    vw + 16,
                    y,
                    YELLOW,
                    0.4,
                )
                y += 22
                text(
                    canvas,
                    f"Teramati {selected['observation_seconds']:.0f}s | mata {selected['eye_coverage']:.0%} / mulut {selected['mouth_coverage']:.0%}",
                    vw + 16,
                    y,
                    MUTED,
                    0.36,
                )
                y += 22
                patterns = {
                    "increased_visual_drowsiness": "Indikasi kantuk visual meningkat",
                    "prolonged_eye_event": "Penutupan mata panjang tercatat",
                    "repeated_opening_candidates": "Kandidat bukaan berulang",
                    "insufficient_observation": "Pengamatan belum cukup",
                    "no_combined_pattern": "Belum ada pola gabungan",
                }
                text(canvas, patterns[selected["visual_pattern"]], vw + 16, y, YELLOW, 0.4)
            if selected.get("challenge_request") not in (None, "none", "complete"):
                y += 28
                text(
                    canvas,
                    "Ikuti arah: "
                    + {
                        "left": "kiri",
                        "right": "kanan",
                        "up": "atas",
                        "down": "bawah",
                        "center": "tengah",
                    }.get(selected["challenge_request"], selected["challenge_request"]),
                    vw + 16,
                    y,
                    YELLOW,
                )
            if selected.get("task_response_request") in ("left", "right"):
                y += 25
                text(
                    canvas,
                    "Jawab A kiri / D kanan: " + selected["task_response_request"],
                    vw + 16,
                    y,
                    YELLOW,
                    0.4,
                )
            if self.debug:
                for name in (
                    "observation_quality",
                    "observation_reason",
                    "primary_reason",
                    "baseline",
                ):
                    if selected.get(name) not in (None, ""):
                        y += 21
                        text(canvas, f"{name}: {selected[name]}", vw + 16, y, MUTED, 0.34)
                if config is not None:
                    y += 21
                    text(
                        canvas,
                        f"EAR {config.ear_threshold} / MAR {config.mar_threshold}",
                        vw + 16,
                        y,
                        MUTED,
                        0.34,
                    )


def render_multi(frame, rows, selected_id=None, config=None):
    monitor = MonitorUI()
    monitor.update(rows)
    return monitor.render(frame, rows, selected_id, config)
