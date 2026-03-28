from __future__ import annotations

import unittest

from livetalking.providers.tts_segments import normalize_tts_text, split_tts_segments


class TtsSegmentTests(unittest.TestCase):
    def test_normalize_tts_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_tts_text("  hello\tworld\n"), "hello world")

    def test_split_tts_segments_breaks_on_sentence_punctuation(self) -> None:
        self.assertEqual(
            split_tts_segments("Hello world. This is a test!"),
            ["Hello world.", "This is a test!"],
        )

    def test_split_tts_segments_handles_chinese_punctuation(self) -> None:
        self.assertEqual(
            split_tts_segments("你好\u3002Hello\uff01"),
            ["你好。", "Hello!"],
        )

    def test_split_tts_segments_keeps_short_plain_text_as_one_segment(self) -> None:
        self.assertEqual(split_tts_segments("Short input"), ["Short input"])

    def test_split_tts_segments_splits_long_plain_text(self) -> None:
        text = (
            "This sentence is intentionally long enough to be split into multiple chunks "
            "without relying on punctuation markers"
        )
        segments = split_tts_segments(text, max_chars=40)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment) <= 40 for segment in segments))


if __name__ == "__main__":
    unittest.main()
