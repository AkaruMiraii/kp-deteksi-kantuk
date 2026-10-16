import tempfile
import unittest
from pathlib import Path


class TestMultiReplay(unittest.TestCase):
    def test_headless_replay_writes_separate_predictions_and_summary(self):
        from scripts.multi_replay import write_replay

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [dict(track_id=1, timestamp_ms=100, authenticity="uncertain")]
            write_replay(path, [(0, 100, rows)], {"source": "fixture"})
            self.assertTrue((path / "persons.jsonl").is_file())
            self.assertIn('"authenticity_integrated": true', (path / "summary.json").read_text())
            self.assertIn('"frames": 1', (path / "summary.json").read_text())
            with self.assertRaises(FileExistsError):
                write_replay(path, [], {})
