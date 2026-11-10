import unittest, json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from scripts.multi_live import finalize_session, parse_args
class TestPortableAssets(unittest.TestCase):
    def test_threshold_session_finalizes_without_unused_antispoof(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            model=Path(__file__).resolve().parents[1]/'models/face_landmarker.task'
            args=parse_args(['--threshold-only'])
            summary=finalize_session(None,(None,None,None),(root,0,'source_complete'),args,(model,root/'absent.pth'),perf_counter())
            self.assertIsNone(summary['anti_spoof_sha256'])
            self.assertEqual(json.loads((root/'summary.json').read_text())['exit_reason'],'source_complete')
