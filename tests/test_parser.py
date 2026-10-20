"""Section parser: a good reply splits, a bad one stays usable."""

from __future__ import annotations

import unittest

from voice2notes import parse_sections, render_markdown_note, Transcript
from datetime import date

WELL_FORMED = """\
## Title
Milk run

## Summary
Buy milk and call Priya.

## Action items
- [ ] Buy milk
- [ ] Call Priya

## Key points
- Milk
- Priya
"""

MISSING_HEADING = """\
## Title
Hi

## Summary
Text only.
"""

NO_HEADINGS = "The model wrote a paragraph and forgot the headings."


class ParserTests(unittest.TestCase):
    def test_well_formed_reply_has_four_sections(self) -> None:
        parsed = parse_sections(WELL_FORMED)
        self.assertEqual(
            set(parsed),
            {"Title", "Summary", "Action items", "Key points"},
        )
        self.assertEqual(parsed["Title"], "Milk run")
        self.assertEqual(parsed["Summary"], "Buy milk and call Priya.")
        self.assertIn("- [ ] Buy milk", parsed["Action items"])
        self.assertIn("- Priya", parsed["Key points"])

    def test_missing_heading_falls_back_to_notes(self) -> None:
        parsed = parse_sections(MISSING_HEADING)
        self.assertEqual(parsed, {"Notes": MISSING_HEADING})

    def test_prose_without_headings_falls_back_to_notes(self) -> None:
        parsed = parse_sections(NO_HEADINGS)
        self.assertEqual(parsed, {"Notes": NO_HEADINGS})

    def test_malformed_reply_still_renders_a_note(self) -> None:
        note = render_markdown_note(
            parse_sections(NO_HEADINGS),
            Transcript(text="Buy milk.", language="en", duration=3),
            source_name="memo.wav",
            day=date(2026, 10, 20),
            whisper_model="base",
            llm="qwen2.5:3b",
        )
        self.assertIn("## Notes", note)
        self.assertIn(NO_HEADINGS, note)
        self.assertIn("## Full transcript", note)
        self.assertIn("Buy milk.", note)


if __name__ == "__main__":
    unittest.main()
