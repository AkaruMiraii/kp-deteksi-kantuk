import unittest

from scripts.drowsiness_decision import (
    STATUS_INDIKASI_KANTUK,
    STATUS_TIDAK_DAPAT_DINILAI,
    STATUS_TIDAK_TERDETEKSI,
    evaluate_drowsiness,
)


class TestDrowsinessDecision(unittest.TestCase):
    def test_missing_face_or_invalid_geometry_returns_tidak_dapat_dinilai(self):
        cases = [
            dict(
                face_detected=False,
                reason="no_face_detected",
                eye_state="not_assessable",
                mouth_state="not_assessable",
            ),
            dict(
                face_detected=True,
                reason="selected_points_outside_image",
                eye_state="not_assessable",
                mouth_state="not_assessable",
            ),
            dict(
                face_detected=True,
                reason="",
                eye_state="not_assessable",
                mouth_state="no_large_opening",
            ),
            dict(
                face_detected=True,
                reason="",
                eye_state="no_bilateral_closure",
                mouth_state="not_assessable",
            ),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                decision = evaluate_drowsiness(**kwargs)
                self.assertEqual(decision.final_status, STATUS_TIDAK_DAPAT_DINILAI)
                self.assertEqual(decision.trigger, "none")

    def test_normal_open_eyes_and_closed_mouth_returns_tidak_terdeteksi(self):
        decision = evaluate_drowsiness(
            face_detected=True,
            reason="",
            eye_state="no_bilateral_closure",
            mouth_state="no_large_opening",
        )
        self.assertEqual(decision.final_status, STATUS_TIDAK_TERDETEKSI)
        self.assertEqual(decision.trigger, "none")

    def test_brief_candidates_do_not_trigger_drowsiness(self):
        decision = evaluate_drowsiness(
            face_detected=True,
            reason="",
            eye_state="closure_candidate",
            mouth_state="opening_candidate",
        )
        self.assertEqual(decision.final_status, STATUS_TIDAK_TERDETEKSI)
        self.assertEqual(decision.trigger, "none")

    def test_prolonged_closure_alone_triggers_drowsiness_from_eye(self):
        decision = evaluate_drowsiness(
            face_detected=True,
            reason="",
            eye_state="prolonged_closure",
            mouth_state="no_large_opening",
        )
        self.assertEqual(decision.final_status, STATUS_INDIKASI_KANTUK)
        self.assertEqual(decision.trigger, "eye")

    def test_prolonged_opening_alone_is_not_evidence_of_drowsiness(self):
        decision = evaluate_drowsiness(
            face_detected=True,
            reason="",
            eye_state="no_bilateral_closure",
            mouth_state="prolonged_opening",
        )
        self.assertEqual(decision.final_status, STATUS_TIDAK_TERDETEKSI)
        self.assertEqual(decision.trigger, "none")
        self.assertEqual(decision.reason, "mouth_opening_not_validated")

    def test_both_prolonged_trigger_is_eye_only(self):
        decision = evaluate_drowsiness(
            face_detected=True,
            reason="",
            eye_state="prolonged_closure",
            mouth_state="prolonged_opening",
        )
        self.assertEqual(decision.final_status, STATUS_INDIKASI_KANTUK)
        self.assertEqual(decision.trigger, "eye")

    def test_eye_only_can_decide_but_no_channels_cannot(self):
        decision_eye_only = evaluate_drowsiness(
            face_detected=True,
            reason="",
            eye_state="prolonged_closure",
            mouth_state=None,
        )
        self.assertEqual(decision_eye_only.final_status, STATUS_INDIKASI_KANTUK)
        self.assertEqual(decision_eye_only.trigger, "eye")

        decision_both_disabled = evaluate_drowsiness(
            face_detected=True,
            reason="",
            eye_state=None,
            mouth_state=None,
        )
        self.assertEqual(decision_both_disabled.final_status, STATUS_TIDAK_DAPAT_DINILAI)
        self.assertEqual(decision_both_disabled.trigger, "none")

    def test_unknown_states_fail_loudly(self):
        for eye, mouth in [("typo", "no_large_opening"), ("prolonged_closure", "typo")]:
            with self.subTest(eye=eye, mouth=mouth), self.assertRaises(ValueError):
                evaluate_drowsiness(True, "", eye, mouth)

    def test_mouth_only_cannot_issue_a_negative_drowsiness_decision(self):
        result = evaluate_drowsiness(True, "", None, "prolonged_opening")
        self.assertEqual(result.final_status, STATUS_TIDAK_DAPAT_DINILAI)
        self.assertEqual(result.reason, "eye_channel_disabled")


if __name__ == "__main__":
    unittest.main()
