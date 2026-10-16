"""Ambiguous gallery similarity must not create a new identity/history destination."""
import unittest
import numpy as np
from scripts.enrolled_identity import EnrolledIdentity

class AmbiguityIsolationTests(unittest.TestCase):
    def test_ambiguous_matches_remain_pending_without_history_destination(self):
        engine=EnrolledIdentity.__new__(EnrolledIdentity)
        engine.enrollment_seconds=180
        engine.started=None
        engine.templates={1:[np.array([1.,0.,0.])],2:[np.array([0.,1.,0.])]}
        engine.gallery={k:v[0] for k,v in engine.templates.items()}
        engine.pending={};engine.bindings={};engine.mismatches={}
        engine.last_active=set();engine.restored_from={};engine.next_id=3
        feature=np.array([1.,1.,0.])/np.sqrt(2)
        for i in range(61):
            result=engine.resolve_at(i*100,[7],[feature])
        self.assertEqual(result,[-7], 'ambiguous face must not enroll as third identity')
        self.assertEqual(set(engine.gallery),{1,2})
        self.assertEqual(engine.restored_from,{})
        self.assertEqual(engine.pending[7]['samples'],[])

if __name__=='__main__': unittest.main()
