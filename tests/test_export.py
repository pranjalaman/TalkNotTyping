"""Action-item export. Dates are kept only when the memo said them."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from voice2notes import (
    Transcript,
    due_date_for_item,
    main,
    parse_args,
    render_ics,
    render_todo_txt,
    spoken_dates,
)

TRANSCRIPT = "Book the room on October 20 and email Priya tomorrow."


class SpokenDateTests(unittest.TestCase):
    def test_month_day_and_iso_dates(self) -> None:
        text = "Meet on October 20, 2026, or on 3 March, or on 2026-11-02."
        self.assertEqual(
            spoken_dates(text, default_year=2026),
            [date(2026, 10, 20), date(2026, 3, 3), date(2026, 11, 2)],
        )

    def test_missing_year_uses_the_note_year(self) -> None:
        self.assertEqual(
            spoken_dates("on 20 October", default_year=2026),
            [date(2026, 10, 20)],
        )

    def test_relative_words_are_not_dates(self) -> None:
        self.assertEqual(spoken_dates("tomorrow and next week", default_year=2026), [])

    def test_impossible_days_are_ignored(self) -> None:
        self.assertEqual(spoken_dates("February 31", default_year=2026), [])

    def test_item_date_must_also_be_in_the_transcript(self) -> None:
        self.assertEqual(
            due_date_for_item("Book the room on October 20", TRANSCRIPT, default_year=2026),
            date(2026, 10, 20),
        )
        self.assertIsNone(
            due_date_for_item("Meet Sam on March 9", TRANSCRIPT, default_year=2026)
        )
        self.assertIsNone(
            due_date_for_item("Email Priya tomorrow", TRANSCRIPT, default_year=2026)
        )


class ExportFileTests(unittest.TestCase):
    def test_render_skips_dates_that_were_not_spoken(self) -> None:
        items = [
            ("Book the room on October 20", date(2026, 10, 20)),
            ("Email Priya tomorrow", None),
            ("Meet Sam on March 9", None),
        ]
        todo = render_todo_txt(items)
        self.assertEqual(
            todo,
            "Book the room on October 20 due:2026-10-20\n"
            "Email Priya tomorrow\n"
            "Meet Sam on March 9\n",
        )
        calendar = render_ics(items, stamp=datetime(2026, 10, 20, tzinfo=timezone.utc))
        self.assertEqual(calendar.count("DTSTART;VALUE=DATE:20261020"), 1)
        self.assertNotIn("20260309", calendar)
        self.assertNotIn("DTSTART", calendar.replace("DTSTART;VALUE=DATE:20261020", "", 1))

    def test_cli_writes_both_exports(self) -> None:
        self.assertEqual(parse_args(["a.wav", "--export", "todo", "--export", "ics"]).export, ["todo", "ics"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "memo.wav"
            audio.write_bytes(b"RIFF")
            out = root / "notes"

            def fake_generate(transcript: str, model: str, *, url: str = "", style: str = "both") -> str:
                del transcript, model, url, style
                return (
                    "## Title\nVenue\n## Summary\nBook the room.\n"
                    "## Action items\n"
                    "- [ ] Book the room on October 20\n"
                    "- [ ] Email Priya tomorrow\n"
                    "- [ ] Meet Sam on March 9\n"
                    "## Key points\n- room\n"
                )

            with (
                patch("voice2notes.load_whisper_model", return_value=object()),
                patch(
                    "voice2notes.transcribe_file",
                    return_value=Transcript(text=TRANSCRIPT, language="en", duration=12),
                ),
                patch("voice2notes.generate_notes", side_effect=fake_generate),
            ):
                main([str(audio), "--out", str(out), "--export", "todo", "--export", "ics"])
            year = date.today().year
            todo = (out / f"{date.today().isoformat()}_memo.todo.txt").read_text(encoding="utf-8")
            calendar = (out / f"{date.today().isoformat()}_memo.ics").read_text(encoding="utf-8")
        self.assertIn(f"due:{year}-10-20", todo)
        self.assertIn("Email Priya tomorrow\n", todo)
        self.assertNotIn("tomorrow due:", todo)
        self.assertNotIn("March 9 due:", todo)
        self.assertIn(f"DTSTART;VALUE=DATE:{year}1020", calendar)
        self.assertNotIn(f"{year}0309", calendar)


if __name__ == "__main__":
    unittest.main()
