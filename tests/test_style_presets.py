"""Style presets, checked with fixture model replies. Ollama is not called."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from voice2notes import (
    NOTES_SYSTEM_PROMPT,
    STYLE_PROMPTS,
    SUMMARY_SYSTEM_PROMPT,
    TODO_SYSTEM_PROMPT,
    Transcript,
    build_parser,
    generate_notes,
    main,
    parse_sections,
    render_markdown_note,
)

TRANSCRIPT = Transcript(
    text="Email Priya about the venue tomorrow.",
    language="en",
    duration=192.4,
)
DAY = date(2026, 10, 20)

# One fixture reply that contains every section. Each preset must keep only its own.
FIXTURE_REPLY = """\
## Title
Venue follow-up

## Summary
Priya needs an email about the venue.

## Action items
- [ ] Email Priya about the venue

## Key points
- The venue is still open
"""

BOTH_NOTE = """\
# Venue follow-up

- Source: memo.m4a
- Date: 2026-10-20
- Duration: 03:12
- Language: en
- Models: whisper-base, qwen2.5:3b

## Summary
Priya needs an email about the venue.

## Action items
- [ ] Email Priya about the venue

## Key points
- The venue is still open

## Full transcript
Email Priya about the venue tomorrow.
"""

TODO_NOTE = """\
# Notes

- Source: memo.m4a
- Date: 2026-10-20
- Duration: 03:12
- Language: en
- Models: whisper-base, qwen2.5:3b

## Action items
- [ ] Email Priya about the venue

## Full transcript
Email Priya about the venue tomorrow.
"""

SUMMARY_NOTE = """\
# Venue follow-up

- Source: memo.m4a
- Date: 2026-10-20
- Duration: 03:12
- Language: en
- Models: whisper-base, qwen2.5:3b

## Summary
Priya needs an email about the venue.

## Full transcript
Email Priya about the venue tomorrow.
"""


class FrozenDate(date):
    @classmethod
    def today(cls) -> FrozenDate:
        return cls(2026, 10, 20)


def heading_lines(prompt: str) -> list[str]:
    return [line[3:] for line in prompt.splitlines() if line.startswith("## ")]


def rendered(sections: dict[str, str]) -> str:
    return render_markdown_note(
        sections,
        TRANSCRIPT,
        source_name="memo.m4a",
        day=DAY,
        whisper_model="base",
        llm="qwen2.5:3b",
    )


class StylePresetTests(unittest.TestCase):
    def test_style_defaults_to_both(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).style, "both")
        self.assertEqual(parser.parse_args(["--style", "both"]).style, "both")
        self.assertEqual(parser.parse_args(["--style", "todo"]).style, "todo")
        self.assertEqual(parser.parse_args(["--style", "summary"]).style, "summary")

    def test_each_style_has_its_own_prompt(self) -> None:
        self.assertIs(STYLE_PROMPTS["both"], NOTES_SYSTEM_PROMPT)
        self.assertIs(STYLE_PROMPTS["todo"], TODO_SYSTEM_PROMPT)
        self.assertIs(STYLE_PROMPTS["summary"], SUMMARY_SYSTEM_PROMPT)
        self.assertEqual(
            heading_lines(NOTES_SYSTEM_PROMPT),
            ["Title", "Summary", "Action items", "Key points"],
        )
        self.assertEqual(heading_lines(TODO_SYSTEM_PROMPT), ["Action items"])
        self.assertEqual(heading_lines(SUMMARY_SYSTEM_PROMPT), ["Title", "Summary"])

    def test_generate_notes_sends_the_style_prompt(self) -> None:
        sent: list[tuple[str, list[dict[str, str]]]] = []

        def fake_chat(model, messages, *, url, timeout=300):
            sent.append((model, messages))
            return FIXTURE_REPLY

        with patch("voice2notes.ollama_chat", side_effect=fake_chat):
            self.assertEqual(
                generate_notes("hello", "qwen2.5:3b", style="todo"),
                FIXTURE_REPLY,
            )
            self.assertEqual(generate_notes("hello", "qwen2.5:3b"), FIXTURE_REPLY)
            self.assertEqual(
                generate_notes("hello", "qwen2.5:3b", style="summary"),
                FIXTURE_REPLY,
            )

        self.assertEqual(
            [(model, messages[0]["content"], messages[1]["content"]) for model, messages in sent],
            [
                ("qwen2.5:3b", TODO_SYSTEM_PROMPT, "hello"),
                ("qwen2.5:3b", NOTES_SYSTEM_PROMPT, "hello"),
                ("qwen2.5:3b", SUMMARY_SYSTEM_PROMPT, "hello"),
            ],
        )

    def test_parse_keeps_only_requested_headings(self) -> None:
        self.assertEqual(
            parse_sections(FIXTURE_REPLY),
            {
                "Title": "Venue follow-up",
                "Summary": "Priya needs an email about the venue.",
                "Action items": "- [ ] Email Priya about the venue",
                "Key points": "- The venue is still open",
            },
        )
        self.assertEqual(
            parse_sections(FIXTURE_REPLY, ("Action items",)),
            {"Action items": "- [ ] Email Priya about the venue"},
        )
        self.assertEqual(
            parse_sections(FIXTURE_REPLY, ("Title", "Summary")),
            {
                "Title": "Venue follow-up",
                "Summary": "Priya needs an email about the venue.",
            },
        )

    def test_missing_heading_falls_back_to_notes(self) -> None:
        reply = "## Title\nHi\n## Summary\nText\n"
        self.assertEqual(parse_sections(reply), {"Notes": reply})

    def test_both_renders_the_full_note(self) -> None:
        self.assertEqual(rendered(parse_sections(FIXTURE_REPLY)), BOTH_NOTE)

    def test_todo_renders_action_items_only(self) -> None:
        note = rendered(parse_sections(FIXTURE_REPLY, ("Action items",)))
        self.assertEqual(note, TODO_NOTE)
        self.assertNotIn("## Summary", note)
        self.assertNotIn("## Key points", note)
        self.assertNotIn("Priya needs an email", note)
        self.assertNotIn("The venue is still open", note)

    def test_summary_renders_title_and_summary_only(self) -> None:
        note = rendered(parse_sections(FIXTURE_REPLY, ("Title", "Summary")))
        self.assertEqual(note, SUMMARY_NOTE)
        self.assertNotIn("## Action items", note)
        self.assertNotIn("## Key points", note)
        self.assertNotIn("The venue is still open", note)

    def test_main_writes_the_preset_note(self) -> None:
        cases = (
            (None, NOTES_SYSTEM_PROMPT, BOTH_NOTE),
            ("todo", TODO_SYSTEM_PROMPT, TODO_NOTE),
            ("summary", SUMMARY_SYSTEM_PROMPT, SUMMARY_NOTE),
        )
        for style, prompt, expected in cases:
            with self.subTest(style=style or "both"):
                self.assertEqual(self._run_main(style, prompt), expected)

    def _run_main(self, style: str | None, prompt: str) -> str:
        seen: dict[str, object] = {}

        def fake_chat(model, messages, *, url, timeout=300):
            seen["messages"] = messages
            return FIXTURE_REPLY

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "memo.m4a"
            audio.write_bytes(b"not real audio")
            out = root / "notes"
            argv = [str(audio), "--out", str(out)]
            if style is not None:
                argv.extend(["--style", style])
            with (
                patch("voice2notes.load_whisper_model", return_value=object()),
                patch("voice2notes.transcribe_file", return_value=TRANSCRIPT),
                patch("voice2notes.ollama_chat", side_effect=fake_chat),
                patch("voice2notes.date", FrozenDate),
            ):
                main(argv)
            messages = seen["messages"]
            assert isinstance(messages, list)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[0]["content"], prompt)
            self.assertEqual(messages[1], {"role": "user", "content": TRANSCRIPT.text})
            written = (out / "2026-10-20_memo.md").read_text(encoding="utf-8")
            return written


if __name__ == "__main__":
    unittest.main()
