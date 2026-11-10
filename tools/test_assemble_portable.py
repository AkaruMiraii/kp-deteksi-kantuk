"""Assembly contract tests using temporary inputs, with no runtime installation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import assemble_portable as builder


class AssemblyTests(unittest.TestCase):
    def test_new_folder_only_and_source_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, runtime, models = (root / n for n in ('source', 'runtime', 'models'))
            for p in (source, runtime, models):
                p.mkdir()
            for n in ('kp_poc_entry.py', 'portable_paths.py', 'smoke_tools.py'):
                (source / n).write_text('# source\n')
            for n in ('scripts', 'tests', 'portable'):
                (source / n).mkdir()
            (source / 'scripts' / 'module.py').write_text('# source\n')
            (source / 'scripts' / '__pycache__').mkdir()
            (source / 'scripts' / '__pycache__' / 'old.pyc').write_bytes(b'cache')
            (source / 'outputs').mkdir()
            (source / 'outputs' / 'private.txt').write_text('not distributable')
            for n in ('KP_Deteksi_Kantuk.bat', 'PANDUAN.txt'):
                (source / 'portable' / n).write_text('launcher or guide\n')
            for n in ('python.exe', 'pythonw.exe'):
                (runtime / n).write_bytes(b'runtime fixture')
            (models / 'model.task').write_bytes(b'model fixture')
            hashes = {'model.task': hashlib.sha256(b'model fixture').hexdigest()}
            out = root / 'new-package'
            with patch.object(builder, 'ROOT', source), patch.object(builder, 'MODEL_HASHES', hashes):
                result = builder.assemble(runtime, models, out)
                manifest = json.loads((out / 'manifest_sha256.json').read_text())
                self.assertEqual(result['files'], len(manifest))
                self.assertNotIn('app/outputs/private.txt', manifest)
                self.assertFalse(any('__pycache__' in n for n in manifest))
                for name, expected in manifest.items():
                    self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), expected)
                with self.assertRaises(FileExistsError):
                    builder.assemble(runtime, models, out)
                with self.assertRaises(ValueError):
                    builder.assemble(runtime, models, source / 'nested')
                (models / 'model.task').write_bytes(b'wrong model')
                with self.assertRaises(ValueError):
                    builder.assemble(runtime, models, root / 'bad')
                self.assertFalse((root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
