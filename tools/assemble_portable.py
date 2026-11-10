"""Assemble a new Windows package from source and an already prepared runtime.

No downloads, environment installation, ZIP creation, or existing-folder changes.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_HASHES = {'face_landmarker.task': '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff', 'sface.onnx': '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79', 'smart_yawn.json': '5b122c41180d70c334130b9ad3ffb8d8ca0507c5fff142e22d8c5267ba9c3d43'}


def assemble(runtime: Path, models: Path, output: Path):
    runtime, models, output = runtime.resolve(), models.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output must be a new folder: {output}")
    if any(output.is_relative_to(p) for p in (ROOT, runtime, models)):
        raise ValueError("Output must be outside source, runtime, and models")
    for name in ("python.exe", "pythonw.exe"):
        if not (runtime / name).is_file():
            raise FileNotFoundError(runtime / name)
    for name, expected in MODEL_HASHES.items():
        actual = hashlib.sha256((models / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Model hash mismatch: {name}")
    app = output / "app"
    app.mkdir(parents=True)
    for name in ("kp_poc_entry.py", "portable_paths.py", "smoke_tools.py"):
        shutil.copy2(ROOT / name, app / name)
    for name in ("scripts", "tests"):
        shutil.copytree(ROOT / name, app / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "audit3.py", "run3.py"))
    (app / "models").mkdir()
    for name in MODEL_HASHES:
        shutil.copy2(models / name, app / "models" / name)
    shutil.copytree(runtime, output / "runtime" / "python",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(ROOT / "portable" / "KP_Deteksi_Kantuk.bat", output)
    shutil.copy2(ROOT / "portable" / "PANDUAN.txt", output)
    manifest = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                manifest[path.relative_to(output).as_posix()] = hashlib.file_digest(
                    handle, "sha256").hexdigest()
    (output / "manifest_sha256.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "files": len(manifest)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True,
                        help="Prepared Windows CPython 3.11 x64 directory including dependencies and Tk")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.runtime_python, args.models, args.output)))
