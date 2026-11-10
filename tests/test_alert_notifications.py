import unittest
import numpy as np
from scripts.multi_display import MonitorUI


def row(identity=1, status='indikasi_kantuk', **extra):
    result = dict(track_id=identity, primary_status=status, box=(.1,.1,.4,.5),
                  points_px={}, authenticity_required=False, identity_pending=False,
                  observation_quality='valid')
    result.update(extra)
    return result


class AlertNotifications(unittest.TestCase):
    def setUp(self):
        self.time = [0.0]
        self.ui = MonitorUI(clock=lambda: self.time[0])

    def test_episode_count_not_frames_and_rearms_only_valid_negative(self):
        positive = row()
        self.ui.update([positive])
        self.ui.update([positive])
        self.assertEqual(self.ui.indication_total, 1)
        self.ui.update([row(status='tidak_dapat_dinilai')])
        self.ui.update([positive])
        self.assertEqual(self.ui.indication_total, 1)
        self.ui.update([row(status='tidak_terdeteksi_indikasi_kantuk')])
        self.ui.update([positive])
        self.assertEqual(self.ui.indication_total, 2)

    def test_face_alert_expires_at_ten_seconds_without_refresh(self):
        r = row(visual_pattern='increased_visual_drowsiness')
        self.ui.update([r])
        self.time[0] = 9.99
        self.ui.update([r])
        self.assertTrue(self.ui.alert_visible(1))
        self.time[0] = 10.0
        self.ui.update([r])
        self.assertFalse(self.ui.alert_visible(1))
        texts = []
        def capture(target, value, *args, **kwargs):
            texts.append(str(value))
        self.ui.draw_faces(np.zeros((800,1320,3), dtype=np.uint8),
                           [(r,1,'closed')],1,(960,720,1.5),capture)
        self.assertFalse(any('INDIKASI KANTUK' in t for t in texts))
        self.assertTrue(any('Mata masih tertutup' in t for t in texts))

    def test_counts_separate_people_and_survive_disappearance(self):
        self.ui.update([row(1),row(2)])
        self.assertEqual(self.ui.indication_total, 2)
        self.time[0] = 12
        self.ui.update([])
        self.ui.update([row(1)])
        self.assertEqual(self.ui.indication_total, 2)
        self.assertEqual(self.ui.indication_counts, {1:1,2:1})
        self.assertFalse(self.ui.alert_visible(1))

    def test_pending_identity_and_historical_pattern_do_not_count(self):
        self.ui.update([row(-1, identity_pending=True)])
        self.ui.update([row(2,status='tidak_terdeteksi_indikasi_kantuk',
                            visual_pattern='increased_visual_drowsiness')])
        self.assertEqual(self.ui.indication_total, 0)

    def test_new_episode_restarts_notification_and_does_not_mutate_rows(self):
        r = row()
        original = r.copy()
        self.ui.update([r])
        self.time[0] = 15
        self.ui.update([row(status='tidak_terdeteksi_indikasi_kantuk')])
        self.ui.update([r])
        self.assertTrue(self.ui.alert_visible(1))
        self.assertEqual(r, original)
        self.assertEqual(self.ui.indication_total, 2)

if __name__ == '__main__':
    unittest.main()
