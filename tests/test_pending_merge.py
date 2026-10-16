import unittest

from scripts.detection_config import DetectionConfig
from scripts.multi_person import MultiPersonPipeline
from scripts.person_state import PersonState


class TestPendingMerge(unittest.TestCase):
    def states(self):
        return PersonState(1, DetectionConfig()), PersonState(-1, DetectionConfig())

    def test_merge_is_idempotent_and_preserves_coverage_origin(self):
        target, pending = self.states()
        for i in range(31):
            pending.update(i * 100, 0.05, 0.05, 0.7, True, "not_checked", threshold_only=True)
        MultiPersonPipeline.merge_pending(target, pending, 3)
        MultiPersonPipeline.merge_pending(target, pending, 3)
        self.assertEqual(target.event_window.total_eye, 1)
        self.assertEqual(target.event_window.total_mouth, 1)
        self.assertEqual(target.event_window.started, 0)
        self.assertEqual(
            target.eyes.started_at if hasattr(target.eyes, "started_at") else target.previous, None
        )

    def test_expired_buffer_never_enters_participant_history(self):
        target, pending = self.states()
        pending.update(0, 0.05, 0.05, 0.7, True, "not_checked", threshold_only=True)
        pending.update(200, 0.05, 0.05, 0.7, True, "not_checked", threshold_only=True)
        MultiPersonPipeline.merge_pending(target, pending, 61)
        self.assertEqual(list(target.event_window.events), [])
        self.assertEqual(list(target.event_window.intervals), [])
