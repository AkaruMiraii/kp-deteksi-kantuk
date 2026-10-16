"""Single-model passive RGB score. Not validated attack coverage."""

from math import isfinite
from pathlib import Path

import cv2

from .face_tracking import iou


def match_detection(detections, box):
    candidates = [
        tuple(map(float, r[3:7]))
        for r in detections
        if r[2] >= 0.6 and iou(tuple(r[3:7]), box) >= 0.25
    ]
    return max(candidates, key=lambda b: iou(b, box)) if candidates else None


class FaceAuthenticity:
    def __init__(self, model_path=None):
        self.model = None
        self.error = ""
        self.detector = None
        if model_path is not None:
            import torch

            from .vendor.minifasnet import MiniFASNetV2

            torch.set_num_threads(2)
            self.model = MiniFASNetV2(conv6_kernel=(5, 5))
            weights = torch.load(model_path, map_location="cpu", weights_only=True)
            weights = {k.removeprefix("module."): v for k, v in weights.items()}
            self.model.load_state_dict(weights)
            self.model.eval()
            directory = Path(model_path).parent
            if hasattr(cv2.dnn, "readNetFromCaffe"):
                self.detector = cv2.dnn.readNetFromCaffe(
                    str(directory / "deploy.prototxt"),
                    str(directory / "Widerface-RetinaFace.caffemodel"),
                )

    def predict(self, frame, box):
        output = dict(
            authenticity="uncertain",
            authenticity_reason="model_unavailable",
            passive_live_score=None,
            validated_attack_coverage=False,
        )
        h, w = frame.shape[:2]
        if len(box) != 4 or any(not isfinite(v) for v in box):
            output["authenticity_reason"] = "invalid_crop"
            return output
        if self.model is not None and self.detector is None:
            output["authenticity_reason"] = "reference_crop_detector_unavailable"
            return output
        if self.detector is not None:
            self.detector.setInput(cv2.dnn.blobFromImage(frame, 1, mean=(104, 117, 123)))
            detections = self.detector.forward().reshape(-1, 7)
            matched = match_detection(detections, box)
            if matched is None:
                output["authenticity_reason"] = "detector_crop_unavailable"
                return output
            box = matched
        x1, y1, x2, y2 = box
        if min((x2 - x1) * w, (y2 - y1) * h) < 80:
            output["authenticity_reason"] = "face_too_small"
            return output
        if self.model is None:
            return output
        import torch

        bw, bh = (x2 - x1) * w, (y2 - y1) * h
        scale = min(2.7, (w - 1) / bw, (h - 1) / bh)
        cw, ch = bw * scale, bh * scale
        cx, cy = (x1 + x2) * w / 2, (y1 + y2) * h / 2
        left = max(0, min(w - cw - 1, cx - cw / 2))
        top = max(0, min(h - ch - 1, cy - ch / 2))
        crop = frame[int(top) : int(top + ch + 1), int(left) : int(left + cw + 1)]
        if crop.size == 0:
            output["authenticity_reason"] = "invalid_crop"
            return output
        patch = cv2.resize(crop, (80, 80)).transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(patch).float().unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].tolist()
        score = probabilities[1]
        # A high passive score is not proof of a live person. Active confirmation is still required.
        output.update(
            passive_live_score=score,
            passive_scores=probabilities,
            authenticity="spoof" if score < 0.2 else "uncertain",
            authenticity_reason="passive_spoof_candidate"
            if score < 0.2
            else "active_challenge_required",
        )
        return output
