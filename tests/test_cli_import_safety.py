"""Importing diagnostic entrypoints must not access the camera or desktop."""

import importlib
import sys
import unittest
from unittest.mock import patch


class TestCliImportSafety(unittest.TestCase):
    def test_import_does_not_open_camera_or_prompt(self):
        for name in ("check_webcam", "check_landmarks", "check_landmarks_video"):
            module = f"scripts.{name}"
            with self.subTest(module=module):
                sys.modules.pop(module, None)
                with (
                    patch(
                        "cv2.VideoCapture", side_effect=AssertionError("Camera opened on import")
                    ),
                    patch("builtins.input", side_effect=AssertionError("Prompt on import")),
                    patch("cv2.imshow", side_effect=AssertionError("Window opened on import")),
                ):
                    importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
