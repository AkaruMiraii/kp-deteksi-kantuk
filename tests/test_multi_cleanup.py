import json
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts import multi_live


class TestMultiCleanup(unittest.TestCase):
    def test_source_close_failure_does_not_skip_finalization(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)

            class Source:
                def __iter__(self):
                    return iter([])

                def close(self):
                    raise OSError("source close failed")

            stack.enter_context(
                patch.object(multi_live, "__file__", str(root / "scripts/multi_live.py"))
            )
            stack.enter_context(
                patch.object(multi_live, "uuid4", return_value=SimpleNamespace(hex="fixture"))
            )
            stack.enter_context(patch.object(multi_live, "FaceAuthenticity"))
            stack.enter_context(patch.object(multi_live, "file_sha256", return_value="fixture"))
            stack.enter_context(
                patch.object(
                    multi_live.vision.FaceLandmarker,
                    "create_from_options",
                    return_value=nullcontext(None),
                )
            )
            stack.enter_context(patch.object(multi_live, "read_raw_session", return_value=Source()))
            destroy = stack.enter_context(patch.object(multi_live.cv2, "destroyAllWindows"))
            stack.enter_context(patch("builtins.print"))
            with self.assertRaisesRegex(OSError, "source close failed"):
                multi_live.run(multi_live.parse_args(["--session", "fixture"]))
            destroy.assert_called_once()
            summary = json.loads((root / "outputs/multiperson/fixture/session.json").read_text())
            self.assertEqual(summary["exit_reason"], "error")
