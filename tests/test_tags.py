"""Optional hashtags supplied with --tags."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voice2notes import Transcript, main, normalize_tags, parse_args


class TagNormalizationTests(unittest.TestCase):
    def test_commas_spaces_and_hashes(self) -> None:
        self.assertEqual(
            normalize_tags(["#meeting, priya", "project notes"]),
            ["meeting", "priya", "project-notes"],
        )

    def test_duplicates_and_blanks_are_dropped(self) -> None:
        self.assertEqual(normalize_tags(["meeting", "#meeting", "  ", "!!!"]), ["meeting"])

    def test_parser_accepts_repeated_flags(self) -> None:
        args = parse_args(["--tags", "meeting,priya", "--tags", "lecture", "--web"])
        self.assertEqual(args.tags, ["meeting", "priya", "lecture"])


class TagNoteTests(unittest.TestCase):
    def test_tags_are_written_at_the_end_of_the_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "memo.wav"
            audio.write_bytes(b"RIFF")
            out = root / "notes"

            def fake_generate(transcript: str, model: str, *, url: str = "", style: str = "both") -> str:
                del transcript, model, url, style
                return (
                    "## Title\nMilk\n## Summary\nBuy milk.\n"
                    "## Action items\n- [ ] Buy milk\n## Key points\n- milk\n"
                )

            with (
                patch("voice2notes.load_whisper_model", return_value=object()),
                patch(
                    "voice2notes.transcribe_file",
                    return_value=Transcript(text="Buy milk tomorrow.", language="en", duration=3),
                ),
                patch("voice2notes.generate_notes", side_effect=fake_generate),
            ):
                main([str(audio), "--out", str(out), "--tags", "shopping, #errands"])
            note = next(out.glob("*.md")).read_text(encoding="utf-8")
        self.assertTrue(note.endswith("#shopping #errands\n"))
        self.assertIn("## Full transcript\nBuy milk tomorrow.\n", note)
        self.assertNotIn("#Buy", note)


if __name__ == "__main__":
    unittest.main()
