"""Resolve every runtime location relative to the unpacked package, never to a fixed drive."""
import os
import tempfile
from pathlib import Path
from typing import NamedTuple


class Paths(NamedTuple):
    app: Path
    package: Path
    models: Path
    runtime_python: Path


def resolve_paths(entry_file) -> Paths:
    app = Path(entry_file).resolve().parent if Path(entry_file).exists() else Path(entry_file).parent
    package = app.parent
    return Paths(app, package, app / "models", package / "runtime" / "python" / "python.exe")


def writable_log_dir(app: Path) -> Path:
    """Prefer the app folder; fall back to the user temp dir if the package is read-only."""
    try:
        probe = app / ".write_probe"
        probe.write_text("x")
        probe.unlink()
        return app
    except OSError:
        return Path(tempfile.gettempdir())
