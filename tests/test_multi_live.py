import unittest


class TestMultiLive(unittest.TestCase):
    def test_offline_option_and_recording_are_explicit_and_bounded(self):
        from scripts.multi_live import parse_args

        args = parse_args(["--session", "sample", "--ear-threshold", ".2", "--max-seconds", "60"])
        self.assertFalse(args.record_video)
        self.assertFalse(args.save_preview)
        self.assertEqual(args.config.ear_threshold, 0.2)
        with self.assertRaises(SystemExit):
            parse_args(["--max-seconds", "0"])
