"""Note filenames never overwrite an existing file."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from voice2notes import Transcript, write_markdown_note

SECTIONS = {
    "Title": "Milk run",
    "Summary": "Buy milk.",
    "Action items": "- [ ] Buy milk",
    "Key points": "- Milk",
}
TRANSCRIPT = Transcript(text="Buy milk.", language="en", duration=3)
DAY = date(2026, 10, 20)


class FilenameCollisionTests(unittest.TestCase):
    def test_second_write_uses_a_numeric_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / "memo.wav"
            first = write_markdown_note(
                SECTIONS,
                TRANSCRIPT,
                source=source,
                out_dir=out,
                whisper_model="base",
                llm="qwen2.5:3b",
                day=DAY,
            )
            second = write_markdown_note(
                SECTIONS,
                TRANSCRIPT,
                source=source,
                out_dir=out,
                whisper_model="base",
                llm="qwen2.5:3b",
                day=DAY,
            )
            third = write_markdown_note(
                SECTIONS,
                TRANSCRIPT,
                source=source,
                out_dir=out,
                whisper_model="base",
                llm="qwen2.5:3b",
                day=DAY,
            )
            names = (first.name, second.name, third.name)
            first_text = first.read_text(encoding="utf-8")
            second_text = second.read_text(encoding="utf-8")
        self.assertEqual(names, ("2026-10-20_memo.md", "2026-10-20_memo-2.md", "2026-10-20_memo-3.md"))
        self.assertTrue(first_text.startswith("# Milk run"))
        self.assertTrue(second_text.startswith("# Milk run"))


if __name__ == "__main__":
    unittest.main()
