import unittest
import numpy as np
from scripts.multi_display import MonitorUI
class TestIndicators(unittest.TestCase):
    def test_each_visible_person_has_measurements(self):
        ui=MonitorUI(clock=lambda:0)
        rows=[dict(track_id=i,box=(.1*i,.1,.1*i+.1,.4),primary_status='tidak_terdeteksi_indikasi_kantuk',authenticity_required=False,ear_r=.2,ear_l=.21,mar=.1,closed_seconds=0) for i in (1,2)]
        ui.update(rows)
        calls=[]
        def text(target,value,*args,**kwargs): calls.append(str(value))
        ui.draw_faces(np.zeros((800,1400,3),dtype=np.uint8),[(r,1,'normal') for r in rows],1,(960,720,1.5),text)
        for i in (1,2):
            self.assertTrue(any(f'P{i} EAR' in c for c in calls),calls)
