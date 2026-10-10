"""The shipped experimental model must not depend on local output sessions."""

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import smart_live


class TestModelPath(unittest.TestCase):
    def test_default_model_is_shipped_in_models(self):
        root = Path(smart_live.__file__).resolve().parents[1]
        with patch("sys.argv", ["smart_live"]), patch.object(smart_live, "run") as run:
            smart_live.main()
        self.assertEqual(run.call_args.args[0].model, root / "models" / "smart_yawn.json")


if __name__ == "__main__":
    unittest.main()
