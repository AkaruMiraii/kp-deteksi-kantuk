import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.raw_video import RawVideoRecorder, read_raw_session


class TestRawVideo(unittest.TestCase):
    def test_lossless_roundtrip_uses_original_timestamps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawVideoRecorder(root)
            images = [np.full((64, 96, 3), value, dtype=np.uint8) for value in (10, 50, 90)]
            timestamps = [120, 157, 243]
            for timestamp, image in zip(timestamps, images):
                recorder.write(timestamp, image)
            metadata = recorder.close()
            (root / "session.json").write_text(json.dumps({"frames": 3, "raw_video": metadata}))
            with (root / "frames.jsonl").open("w") as stream:
                for index, timestamp in enumerate(timestamps):
                    stream.write(json.dumps({"frame_id": index, "timestamp_ms": timestamp}) + "\n")
            decoded = list(read_raw_session(root))
            self.assertEqual([timestamp for _, timestamp, _ in decoded], timestamps)
            self.assertEqual([index for index, _, _ in decoded], [0, 1, 2])
            for (_, _, actual), expected in zip(decoded, images):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(metadata["verified_frames"], 3)
            self.assertEqual(metadata["codec"], "FFV1")

    def test_empty_session_does_not_claim_video_success(self):
        with TemporaryDirectory() as directory:
            metadata = RawVideoRecorder(Path(directory)).close()
            self.assertFalse(metadata["complete"])
            self.assertEqual(metadata["verified_frames"], 0)

    def test_codec_unavailable_fails_loudly(self):
        with (
            TemporaryDirectory() as directory,
            patch("scripts.raw_video.cv2.VideoWriter") as writer,
        ):
            writer.return_value.isOpened.return_value = False
            recorder = RawVideoRecorder(Path(directory))
            with self.assertRaises(OSError):
                recorder.write(10, np.zeros((64, 96, 3), dtype=np.uint8))
            recorder.close()


if __name__ == "__main__":
    unittest.main()
