import unittest
from scripts.face_tracking import FaceTracker
from scripts.multi_person import distinct_faces

class Regression(unittest.TestCase):
    def test_distinct_overlapping_people_keep_tracks(self):
        t=FaceTracker()
        a=(.1,.1,.4,.6)
        b=(.32,.1,.62,.6)
        self.assertEqual(t.update(0,[a,b]),[1,2])
        self.assertEqual(t.update(100,[b,a]),[2,1])
    def test_duplicate_landmarks_filtered_before_tracking(self):
        faces=['left','right','duplicate']
        boxes=[(.08,.28,.30,.60),(.825,.487,1.089,.858),(.832,.519,1.053,.821)]
        f,b=distinct_faces(faces,boxes)
        self.assertEqual(f,['left','right'])
        self.assertEqual(len(b),2)
    def test_separate_people_not_filtered(self):
        self.assertEqual(len(distinct_faces(['a','b'],[(.1,.1,.4,.6),(.32,.1,.62,.6)])[0]),2)
