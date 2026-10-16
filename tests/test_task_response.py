import unittest
from scripts.task_response import TaskResponseProbe


class TaskResponseTests(unittest.TestCase):
    def test_wrong_response_does_not_complete_probe(self):
        probe = TaskResponseProbe()
        probe.start(0, 'left', 1)
        self.assertFalse(probe.respond(500, 'right'))
        self.assertEqual(probe.update(1000, True, True)['zone_out_indicator'], 'task_nonresponse_candidate')

    def test_response_before_deadline_is_recorded(self):
        probe = TaskResponseProbe()
        probe.start(0, 'left', 1)
        self.assertTrue(probe.respond(500, 'left'))
        self.assertEqual(probe.update(1000, True, True)['task_response_status'], 'responded')

    def test_invalid_evidence_cancels_instead_of_zone_out(self):
        probe = TaskResponseProbe()
        probe.start(0, 'left', 1)
        row = probe.update(1000, False, True)
        self.assertEqual(row['zone_out_indicator'], 'unknown')
        self.assertEqual(row['task_response_status'], 'cancelled')

    def test_separate_probes_do_not_share_response(self):
        a, b = TaskResponseProbe(), TaskResponseProbe()
        a.start(0, 'left', 1)
        b.start(0, 'right', 1)
        a.respond(400, 'left')
        self.assertEqual(b.update(1000, True, True)['task_response_status'], 'missed')
