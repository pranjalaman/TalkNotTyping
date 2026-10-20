#!/usr/bin/env python3
"""TalkNotTyping: turn a rambling voice memo into a tidy markdown note."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NoReturn

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TEMPERATURE = 0.2
CHUNK_WORD_LIMIT = 2500


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    duration: float


def main(argv: list[str] | None = None) -> None:
    """Run the voice-to-notes pipeline."""
    args = parse_args(argv)
    if not args.files:
        return
    model = load_whisper_model(args.whisper_model)
    out_dir = Path(args.out)
    for path in args.files:
        try:
            transcript = transcribe_file(model, path, args.lang)
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue
        if not transcript.text:
            print(f"Warning: no speech detected in {path}", file=sys.stderr)
            continue
        sections = generate_notes_for_transcript(transcript.text, args.llm)
        written = write_markdown_note(
            sections,
            transcript,
            source=path,
            out_dir=out_dir,
            whisper_model=args.whisper_model,
            llm=args.llm,
        )
        print(written)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice2notes.py",
        description="Turn a rambling voice recording into a tidy markdown note.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="one or more audio files (mp3, m4a, wav, ogg, flac, webm)",
    )
    parser.add_argument(
        "--watch",
        metavar="DIR",
        help="watch a folder and process new audio automatically",
    )
    parser.add_argument(
        "--out",
        default="notes",
        metavar="DIR",
        help="output folder (default: ./notes)",
    )
    parser.add_argument(
        "--style",
        choices=("both", "todo", "summary"),
        default="both",
        help="note style: both, todo, or summary (default: both)",
    )
    parser.add_argument(
        "--whisper-model",
        choices=("tiny", "base", "small", "medium"),
        default="base",
        metavar="NAME",
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--llm",
        default="qwen2.5:3b",
        metavar="NAME",
        help="Ollama model tag (default: qwen2.5:3b)",
    )
    parser.add_argument(
        "--lang",
        default=None,
        metavar="CODE",
        help="force language code, e.g. en, hi (default: auto-detect)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10,
        metavar="SECONDS",
        help="poll interval in watch mode (default: 10)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="do not move processed audio in watch mode",
    )
    return parser


def validate_audio_files(paths: list[str]) -> list[Path]:
    """Keep supported files that exist. Print a reason and skip the rest."""
    valid: list[Path] = []
    for raw in paths:
        path = Path(raw)
        suffix = path.suffix.lower()
        if not path.exists():
            print(f"Skipping {raw}: file not found", file=sys.stderr)
            continue
        if not path.is_file():
            print(f"Skipping {raw}: not a file", file=sys.stderr)
            continue
        if suffix not in SUPPORTED_EXTENSIONS:
            shown = suffix or "(no extension)"
            print(f"Skipping {raw}: unsupported type {shown}", file=sys.stderr)
            continue
        valid.append(path)
    return valid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.files and args.watch is None:
        parser.print_usage(sys.stderr)
        raise SystemExit(2)
    args.files = validate_audio_files(args.files)
    return args


def load_whisper_model(name: str):
    """Load faster-whisper on CPU with int8 weights."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "faster-whisper is not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        return WhisperModel(name, device="cpu", compute_type="int8")
    except Exception as exc:
        print(f"Could not load Whisper model '{name}': {exc}", file=sys.stderr)
        raise SystemExit(1)


def transcribe_file(model, path: Path, language: str | None) -> Transcript:
    """Transcribe one file. VAD drops silence. Language is auto-detected when omitted."""
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
    )
    text = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )
    return Transcript(text=text, language=info.language, duration=info.duration)


def ollama_setup_message(model: str) -> str:
    """Hint printed when Ollama is down or the requested model is not pulled."""
    return (
        f"Start Ollama (`ollama serve`) and pull the model (`ollama pull {model}`)"
    )


def ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    url: str = OLLAMA_CHAT_URL,
    timeout: float = 300,
) -> str:
    """Return the assistant text from Ollama `/api/chat`.

    A connection failure and a missing model each print the setup hint and exit 1.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if _ollama_model_missing(exc.code, detail):
            _exit_model_not_pulled(model)
        print(
            f"Ollama request failed ({exc.code}): {detail or exc.reason}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except urllib.error.URLError:
        _exit_ollama_not_running(model)

    try:
        body = json.loads(raw.decode("utf-8"))
        content = body["message"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"Ollama returned an unexpected response: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(content, str):
        print(
            "Ollama returned an unexpected response: message content missing",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return content


def _ollama_model_missing(status: int, detail: str) -> bool:
    """True for HTTP 404 or an Ollama body that says the model was not found."""
    if status == 404:
        return True
    lowered = detail.lower()
    return "model" in lowered and "not found" in lowered


def _exit_ollama_not_running(model: str) -> NoReturn:
    print(ollama_setup_message(model), file=sys.stderr)
    raise SystemExit(1)


def _exit_model_not_pulled(model: str) -> NoReturn:
    print(ollama_setup_message(model), file=sys.stderr)
    raise SystemExit(1)


# NOTES_SYSTEM_PROMPT is the only notes prompt. Edit this constant.
NOTES_SYSTEM_PROMPT = """\
You turn messy spoken transcripts into clean notes. Use only what is in the transcript. Do not invent tasks, names, dates, or facts. If there are no action items, write None.

The user message is the transcript. Return markdown with exactly these headings:
## Title
## Summary
## Action items
## Key points

Spoken-language cleanup:
- Remove filler words.
- Resolve "um, I mean" self-corrections. Keep the corrected meaning and drop the abandoned attempt.
- Keep the user's own wording for names and technical terms.
"""

SECTION_HEADINGS = ("Title", "Summary", "Action items", "Key points")
_ATX_HEADING = re.compile(r"^#{1,6}\s+(.*?)(?:\s+#+)?\s*$")


def parse_sections(model_output: str) -> dict[str, str]:
    """Split a model reply into Title, Summary, Action items, and Key points.

    If any expected heading is missing, return the raw reply under Notes
    instead of raising.
    """
    if not isinstance(model_output, str):
        model_output = "" if model_output is None else str(model_output)

    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in model_output.splitlines():
        heading = _match_section_heading(line)
        if heading is not None:
            if heading not in found:
                found[heading] = []
                current = heading
            continue
        if current is not None:
            found[current].append(line)

    if any(name not in found for name in SECTION_HEADINGS):
        return {"Notes": model_output}

    return {name: "\n".join(found[name]).strip() for name in SECTION_HEADINGS}


def _match_section_heading(line: str) -> str | None:
    match = _ATX_HEADING.match(line.strip())
    if match is None:
        return None
    text = match.group(1).strip().rstrip(":").strip()
    folded = text.casefold()
    for name in SECTION_HEADINGS:
        if folded == name.casefold():
            return name
    return None


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as MM:SS."""
    if seconds < 0:
        seconds = 0
    total = int(seconds + 0.5)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def render_markdown_note(
    sections: dict[str, str],
    transcript: Transcript,
    *,
    source_name: str,
    day: date,
    whisper_model: str,
    llm: str,
) -> str:
    """Build the SPEC section 7 note, including a Notes fallback."""
    title = " ".join(sections.get("Title", "").split()) or "Notes"
    lines = [
        f"# {title}",
        "",
        f"- Source: {source_name}",
        f"- Date: {day.isoformat()}",
        f"- Duration: {format_duration(transcript.duration)}",
        f"- Language: {transcript.language}",
        f"- Models: whisper-{whisper_model}, {llm}",
        "",
    ]
    if "Notes" in sections:
        lines.append("## Notes")
        body = sections["Notes"].strip()
        if body:
            lines.append(body)
        lines.append("")
    else:
        for heading in ("Summary", "Action items", "Key points"):
            lines.append(f"## {heading}")
            body = sections.get(heading, "").strip()
            if body:
                lines.append(body)
            lines.append("")
    lines.append("## Full transcript")
    transcript_text = transcript.text.strip()
    if transcript_text:
        lines.append(transcript_text)
    lines.append("")
    return "\n".join(lines)


def unique_note_path(out_dir: Path, audio_stem: str, day: date) -> Path:
    """Return YYYY-MM-DD_<stem>.md, or the same name with -2, -3, ... appended."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{day.isoformat()}_{audio_stem}"
    candidate = out_dir / f"{base}.md"
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = out_dir / f"{base}-{suffix}.md"
        if not candidate.exists():
            return candidate
        suffix += 1


def write_markdown_note(
    sections: dict[str, str],
    transcript: Transcript,
    source: Path,
    out_dir: Path,
    whisper_model: str,
    llm: str,
    *,
    day: date | None = None,
) -> Path:
    """Write one markdown note. Create out_dir and never overwrite an existing file."""
    written_on = day or date.today()
    text = render_markdown_note(
        sections,
        transcript,
        source_name=source.name,
        day=written_on,
        whisper_model=whisper_model,
        llm=llm,
    )
    path = unique_note_path(out_dir, source.stem, written_on)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def generate_notes(
    transcript: str,
    model: str,
    *,
    url: str = OLLAMA_CHAT_URL,
) -> str:
    """Send a transcript to the Ollama client and return the notes reply."""
    messages = [
        {"role": "system", "content": NOTES_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    return ollama_chat(model, messages, url=url)


_NONE_ITEM = re.compile(
    r"^(?:[-*+]\s+)?(?:\[[ xX]\]\s+)?none\.?$",
    re.IGNORECASE,
)


def chunk_transcript(text: str, *, max_words: int = CHUNK_WORD_LIMIT) -> list[str]:
    """Split a transcript into chunks of at most max_words words.

    Splits on whitespace only, so a word is never cut in half. A transcript
    of max_words words or fewer is returned unchanged as a single chunk.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    words = text.split() if text else []
    if not words:
        return []
    if len(words) <= max_words:
        return [text]
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), max_words)
    ]


def merge_note_sections(parts: list[dict[str, str]]) -> dict[str, str]:
    """Merge per-chunk notes into one document.

    A single part is returned unchanged, with no Part heading. Several parts
    keep one summary each and combine action items and key points into one list.
    """
    if not parts:
        return {
            "Title": "",
            "Summary": "None",
            "Action items": "None",
            "Key points": "None",
        }
    if len(parts) == 1:
        return dict(parts[0])
    if any("Notes" in part for part in parts):
        return {"Notes": _join_fallback_notes(parts)}
    return {
        "Title": _first_title(parts),
        "Summary": _merge_summaries(parts),
        "Action items": _merge_lists(parts, "Action items"),
        "Key points": _merge_lists(parts, "Key points"),
    }


def generate_notes_for_transcript(
    transcript: str,
    model: str,
    *,
    url: str = OLLAMA_CHAT_URL,
) -> dict[str, str]:
    """Generate notes for one transcript, chunking when it is longer than the limit.

    A short transcript is one model call and one note, with no Part heading.
    Longer transcripts get notes per chunk, then those notes are merged.
    """
    chunks = chunk_transcript(transcript)
    if len(chunks) <= 1:
        text = chunks[0] if chunks else transcript
        return parse_sections(generate_notes(text, model, url=url))
    parsed = [
        parse_sections(generate_notes(chunk, model, url=url)) for chunk in chunks
    ]
    return merge_note_sections(parsed)


def _list_lines(body: str) -> list[str]:
    """Keep content lines and drop a standalone None placeholder."""
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or _NONE_ITEM.match(stripped):
            continue
        lines.append(line.rstrip())
    return lines


def _first_title(parts: list[dict[str, str]]) -> str:
    for part in parts:
        title = " ".join(part.get("Title", "").split())
        if title:
            return title
    return ""


def _merge_lists(parts: list[dict[str, str]], heading: str) -> str:
    lines: list[str] = []
    for part in parts:
        lines.extend(_list_lines(part.get(heading, "")))
    if not lines:
        return "None"
    return "\n".join(lines)


def _merge_summaries(parts: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for index, part in enumerate(parts, start=1):
        body = part.get("Summary", "").strip() or "None"
        blocks.append(f"### Part {index}\n{body}")
    return "\n\n".join(blocks)


def _join_fallback_notes(parts: list[dict[str, str]]) -> str:
    """One Notes body when a chunk could not be parsed into sections."""
    blocks: list[str] = []
    for part in parts:
        if "Notes" in part:
            body = part["Notes"].strip()
        else:
            body = "\n\n".join(
                f"## {name}\n{part[name].strip()}"
                for name in SECTION_HEADINGS
                if part.get(name, "").strip()
            )
        if body:
            blocks.append(body)
    return "\n\n".join(blocks)


if __name__ == "__main__":
    main()
