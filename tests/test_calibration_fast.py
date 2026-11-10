import unittest
from types import SimpleNamespace
from scripts.multi_display import MonitorUI
from scripts.person_state import PersonState
from scripts.detection_config import DetectionConfig
class CalibrationTiming(unittest.TestCase):
 def test_calibrates_after_existing_two_second_evidence(self):
  clock=[0.]; ui=MonitorUI(clock=lambda:clock[0]); state=PersonState(1,DetectionConfig()); pipe=SimpleNamespace(states={1:state})
  for n in range(43):
   clock[0]=n*.05
   r=state.update(n*50,.3,.31,.1,True,'not_checked',threshold_only=True)
   r.update(observation_quality='valid',head_direction='center',identity_pending=False)
   ui.update([r]); ui.auto_calibrate([r],pipe)
   if state.baseline is not None:break
  self.assertIsNotNone(state.baseline)
  self.assertGreaterEqual(clock[0],2.)
 def test_short_history_still_rejected(self):
  s=PersonState(1,DetectionConfig())
  for n in range(20):s.update(n*50,.3,.31,.1,True,'not_checked',threshold_only=True)
  with self.assertRaises(ValueError):s.calibrate_from_history()
