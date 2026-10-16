import unittest


class TestPersonSelection(unittest.TestCase):
    def test_selection_follows_track_id_not_detector_order(self):
        from scripts.person_selection import PersonSelection

        selection = PersonSelection()
        rows = [dict(track_id=10), dict(track_id=20)]
        self.assertEqual(selection.update(rows, ord("2")), 20)
        self.assertEqual(selection.update(list(reversed(rows)), 0), 20)
        self.assertIsNone(selection.update([dict(track_id=30)], 0))
        self.assertEqual(selection.update([dict(track_id=30)], ord("1")), 30)
