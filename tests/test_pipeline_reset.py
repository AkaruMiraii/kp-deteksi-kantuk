import unittest

from scripts.active_challenge import ActiveChallenge
from scripts.authenticity_window import AuthenticityWindow
from scripts.detection_config import DetectionConfig
from scripts.multi_person import MultiPersonPipeline
from scripts.person_state import PersonState


class TestPipelineReset(unittest.TestCase):
    def test_explicit_reset_removes_auth_support_without_affecting_other_track(self):
        p = MultiPersonPipeline(DetectionConfig())
        p.states = {1: PersonState(1, p.config), 2: PersonState(2, p.config)}
        p.auth_windows = {1: AuthenticityWindow(), 2: AuthenticityWindow()}
        p.challenges = {1: ActiveChallenge(), 2: ActiveChallenge()}
        other = p.states[2]
        p.reset_track(1)
        self.assertNotIn(1, p.auth_windows)
        self.assertNotIn(1, p.challenges)
        self.assertIs(p.states[2], other)
        self.assertIn(2, p.auth_windows)
