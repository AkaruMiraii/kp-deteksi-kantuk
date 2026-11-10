import unittest
from types import SimpleNamespace
from scripts.multi_display import MonitorUI
from scripts.person_state import PersonState
from scripts.detection_config import DetectionConfig

class NewFaceCalibration(unittest.TestCase):
    def test_second_pending_face_calibrates_independently_of_first(self):
        config=DetectionConfig()
        first=PersonState(1,config)
        first.calibrate_open_eyes(.3,.31,'user_confirmed_open_eyes')
        second=PersonState(-2,config)
        pipe=SimpleNamespace(states={1:first,-2:second})
        now=[0.0]
        ui=MonitorUI(clock=lambda:now[0])
        for n in range(50):
            now[0]=n*.05
            rows=[]
            for identity,state,value in [(1,first,.3),(-2,second,.24)]:
                r=state.update(n*50,value,value+.01,.1,True,'not_checked',threshold_only=True)
                r.update(observation_quality='valid',head_direction='center',identity_pending=identity<0)
                if identity<0:r['primary_status']='tidak_dapat_dinilai'
                rows.append(r)
            ui.update(rows)
            ui.auto_calibrate(rows,pipe)
        self.assertIsNotNone(second.baseline)
        self.assertAlmostEqual(second.baseline[0],.24)
        self.assertEqual(first.baseline,(.3,.31))
        self.assertTrue(rows[1]['identity_pending'])
        self.assertEqual(rows[1]['primary_status'],'tidak_dapat_dinilai')
        self.assertEqual(ui.indication_total,0)
        self.assertEqual(rows[1]['ui_calibration_progress'],1)

    def test_all_six_faces_start_their_own_calibration(self):
        config=DetectionConfig(max_people=6)
        states={i:PersonState(i,config) for i in range(-6,0)}
        pipe=SimpleNamespace(states=states)
        now=[0.0];ui=MonitorUI(clock=lambda:now[0])
        for n in range(50):
            now[0]=n*.05
            rows=[]
            for identity,state in states.items():
                value=.20+abs(identity)*.01
                r=state.update(n*50,value,value,.1,True,'not_checked',threshold_only=True)
                r.update(identity_pending=True,observation_quality='valid',head_direction='center')
                rows.append(r)
            ui.update(rows);ui.auto_calibrate(rows,pipe)
        for identity,state in states.items():
            self.assertIsNotNone(state.baseline)
            self.assertAlmostEqual(state.baseline[0],.20+abs(identity)*.01)

    def test_pending_invalid_geometry_does_not_calibrate(self):
        state=PersonState(-2,DetectionConfig())
        pipe=SimpleNamespace(states={-2:state})
        now=[0.0]; ui=MonitorUI(clock=lambda:now[0])
        for n in range(50):
            now[0]=n*.1
            r=state.update(n*100,None,None,.1,True,'not_checked',threshold_only=True)
            r.update(identity_pending=True,observation_quality='invalid',head_direction='center')
            ui.update([r]);ui.auto_calibrate([r],pipe)
        self.assertIsNone(state.baseline)
        self.assertTrue(r['ui_calibration_waiting'])

if __name__=='__main__':unittest.main()
