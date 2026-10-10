from dataclasses import dataclass

STATUS_INDIKASI_KANTUK = "indikasi_kantuk"
STATUS_TIDAK_TERDETEKSI = "tidak_terdeteksi_indikasi_kantuk"
STATUS_TIDAK_DAPAT_DINILAI = "tidak_dapat_dinilai"
DECISION_POLICY = "v2_eye_evidence_mouth_observation"
EYE_STATES = {"no_bilateral_closure", "closure_candidate", "prolonged_closure", "not_assessable"}
MOUTH_STATES = {"no_large_opening", "opening_candidate", "prolonged_opening", "not_assessable"}


@dataclass(frozen=True)
class DrowsinessDecision:
    final_status: str
    trigger: str
    reason: str


def evaluate_drowsiness(face_detected, reason, eye_state, mouth_state):
    """Provisional eye-based decision; MAR remains an unvalidated mouth observation."""
    if eye_state is not None and eye_state not in EYE_STATES:
        raise ValueError(f"Unknown eye state: {eye_state}")
    if mouth_state is not None and mouth_state not in MOUTH_STATES:
        raise ValueError(f"Unknown mouth state: {mouth_state}")
    if (
        not face_detected
        or reason
        or eye_state == "not_assessable"
        or mouth_state == "not_assessable"
    ):
        failure_reason = reason or ("no_face_detected" if not face_detected else "not_assessable")
        return DrowsinessDecision(STATUS_TIDAK_DAPAT_DINILAI, "none", failure_reason)
    if eye_state is None:
        failure_reason = "no_active_channels" if mouth_state is None else "eye_channel_disabled"
        return DrowsinessDecision(STATUS_TIDAK_DAPAT_DINILAI, "none", failure_reason)
    if eye_state == "prolonged_closure":
        return DrowsinessDecision(STATUS_INDIKASI_KANTUK, "eye", "prolonged_closure")
    if mouth_state == "prolonged_opening":
        return DrowsinessDecision(STATUS_TIDAK_TERDETEKSI, "none", "mouth_opening_not_validated")
    return DrowsinessDecision(STATUS_TIDAK_TERDETEKSI, "none", "no_prolonged_eye_closure")
