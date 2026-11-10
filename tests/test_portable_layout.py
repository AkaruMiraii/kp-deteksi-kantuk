"""Portability contract: the package must not depend on the original build tree."""
import ast, os, subprocess, sys, unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
FORBIDDEN = ("KP_POC_Build", ".venv-build", ".venv-multiperson", "Project KP", "Kerja Praktek", "AkaruMiraii")


class TestPortableLayout(unittest.TestCase):
    def test_entry_has_no_external_runtime_reference(self):
        text = (APP / "kp_poc_entry.py").read_text(encoding="utf-8")
        for token in FORBIDDEN:
            self.assertNotIn(token, text)

    def test_entry_does_not_delegate_to_subprocess(self):
        tree = ast.parse((APP / "kp_poc_entry.py").read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertNotIn("subprocess", names)

    def test_resolve_package_paths_are_relative_to_app(self):
        from portable_paths import resolve_paths
        paths = resolve_paths(APP / "kp_poc_entry.py")
        self.assertEqual(paths.app, APP)
        self.assertEqual(paths.models, APP / "models")
        self.assertEqual(paths.package, APP.parent if (APP.parent / "runtime").exists() else APP.parent)
        self.assertTrue(paths.models.joinpath("face_landmarker.task").is_file())
        self.assertTrue(paths.models.joinpath("sface.onnx").is_file())

    def test_resolve_paths_follow_moved_location(self):
        from portable_paths import resolve_paths
        fake = Path("X:/elsewhere/pkg/app/kp_poc_entry.py")
        paths = resolve_paths(fake)
        self.assertEqual(paths.package, Path("X:/elsewhere/pkg"))
        self.assertEqual(paths.app, Path("X:/elsewhere/pkg/app"))

    def test_runtime_interpreter_is_inside_package_when_bundled(self):
        runtime = APP.parent / "runtime" / "python" / "python.exe"
        if not runtime.exists():
            self.skipTest("source checkout without bundled runtime")
        self.assertTrue(Path(sys.executable).resolve().is_relative_to(APP.parent.resolve())
                        or Path(sys.base_prefix).resolve().is_relative_to(APP.parent.resolve()))

    def test_shipped_scripts_have_no_hardcoded_drive_paths(self):
        import re
        pat = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
        bad = [p.name for p in (APP / "scripts").glob("*.py") if pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
        self.assertEqual(bad, [])

    def test_launcher_uses_relative_runtime(self):
        bat = APP.parent / "KP_Deteksi_Kantuk.bat"
        if not bat.exists():
            self.skipTest("launcher absent in source checkout")
        text = bat.read_text()
        self.assertIn("%~dp0", text)
        self.assertNotRegex(text, r"(?<![A-Za-z])[A-Za-z]:\\")

    def test_shipped_models_exclude_unneeded_antispoof(self):
        self.assertFalse((APP / "models" / "anti_spoof").exists())


if __name__ == "__main__":
    unittest.main()
