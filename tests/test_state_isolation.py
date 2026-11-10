import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scripts.enrolled_identity import EnrolledIdentity
from scripts.multi_person import MultiPersonPipeline
from scripts.person_state import PersonState
from scripts.detection_config import DetectionConfig

class IdentityIsolation(unittest.TestCase):
    def engine(self):
        e=EnrolledIdentity.__new__(EnrolledIdentity)
        e.enrollment_seconds=180;e.started=0;e.templates={1:[np.array([1.,0.])]};e.gallery={1:np.array([1.,0.])};e.pending={};e.bindings={7:1};e.last_active=set();e.restored_from={};e.mismatches={};e.next_id=2
        return e
    def test_first_mismatch_abstains(self):
        e=self.engine()
        self.assertEqual(e.resolve_at(1000,[7],[np.array([0.,1.])]),[-7])
    def test_missing_feature_abstains(self):
        e=self.engine()
        self.assertEqual(e.resolve_at(1000,[7],[None]),[-7])
    def test_pending_change_emits_reset(self):
        e=self.engine();e.bindings={}
        e.resolve_at(1000,[7],[np.array([0.,1.])])
        e.resolve_at(1100,[7],[np.array([1.,0.])])
        self.assertIn(-7,e.reset_pending_ids)
    def test_pipeline_resets_pending_state_on_episode_change(self):
        e=self.engine();e.bindings={};e.reset_pending_ids={-7}
        def resolve(*args):
            e.reset_pending_ids={-7};return [-7]
        e.resolve_at=resolve;e.feature=lambda *args:np.array([0.,1.]);e.gallery={};e.remaining=180
        c=DetectionConfig();p=MultiPersonPipeline(c,threshold_only=True,identity_engine=e)
        previous=PersonState(-7,c);previous.calibrate_open_eyes(.3,.3,'user_confirmed_open_eyes');p.states[-7]=previous
        p.tracker.update=lambda *args:[7]
        points=[SimpleNamespace(x=.3,y=.3),SimpleNamespace(x=.6,y=.6)]
        result=SimpleNamespace(face_landmarks=[points])
        with patch('scripts.multi_person.assess_eyes',return_value=(.2,.2,'')),patch('scripts.multi_person.inspect_frame',return_value=({'reason':'','mar':.1,'points_px':{}},[])),patch('scripts.multi_person.head_direction',return_value='center'):
            p.update(1000,result,640,480,np.zeros((480,640,3),dtype=np.uint8))
        self.assertIsNot(p.states[-7],previous)
        self.assertIsNone(p.states[-7].baseline)
if __name__=='__main__':unittest.main()
