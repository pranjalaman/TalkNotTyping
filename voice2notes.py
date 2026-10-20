#!/usr/bin/env python3
"""TalkNotTyping: turn a rambling voice memo into a tidy markdown note."""

from __future__ import annotations

import argparse
import errno
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
        sections = generate_notes_for_transcript(
            transcript.text,
            args.llm,
            style=args.style,
        )
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


# Hub and socket failures that mean the Whisper weights never made it onto disk.
_WHISPER_DOWNLOAD_ERROR_NAMES = frozenset(
    {
        "OfflineModeIsEnabled",
        "LocalEntryNotFoundError",
        "IncompleteSnapshotError",
        "RevisionResolutionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "NetworkError",
        "ProxyError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "ConnectionResetError",
        "TimeoutError",
        "URLError",
        "NewConnectionError",
        "MaxRetryError",
        "HfHubHTTPError",
    }
)
_WHISPER_DOWNLOAD_MARKERS = (
    "outgoing traffic has been disabled",
    "cannot find an appropriate cached snapshot",
    "cannot find the requested files in the disk cache",
    "hf_hub_offline",
    "cannot find the appropriate snapshot folder",
    "getaddrinfo failed",
    "network is unreachable",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
)
_NETWORK_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "ENETDOWN", None),
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "ENETRESET", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "ECONNREFUSED", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "EHOSTDOWN", None),
        getattr(errno, "EHOSTUNREACH", None),
        10050,  # WSAENETDOWN
        10051,  # WSAENETUNREACH
        10052,  # WSAENETRESET
        10053,  # WSAECONNABORTED
        10054,  # WSAECONNRESET
        10060,  # WSAETIMEDOUT
        10061,  # WSAECONNREFUSED
        10064,  # WSAEHOSTDOWN
        10065,  # WSAEHOSTUNREACH
        11001,  # WSAHOST_NOT_FOUND
        11002,  # WSATRY_AGAIN
    )
    if isinstance(code, int)
)


def whisper_weights_message(model: str) -> str:
    """Name the missing Whisper model and how to download it once."""
    return (
        f"Whisper model '{model}' is not downloaded. "
        "The weights are a one-time download. "
        "Connect to the internet and run this command again so they can be fetched. "
        "After the download, you can force offline mode with HF_HUB_OFFLINE=1."
    )


def is_missing_whisper_weights(exc: BaseException) -> bool:
    """True when loading failed because the weights are offline, the hub, or a cache miss.

    Walks the exception chain so a wrapper around a hub or socket error still matches.
    """
    for current in _iter_exception_chain(exc):
        names = {cls.__name__ for cls in type(current).mro()}
        if names & _WHISPER_DOWNLOAD_ERROR_NAMES:
            return True
        if isinstance(current, OSError) and current.errno in _NETWORK_ERRNOS:
            return True
        text = str(current).lower()
        if any(marker in text for marker in _WHISPER_DOWNLOAD_MARKERS):
            return True
    return False


def whisper_load_failure_message(model: str, exc: BaseException) -> str:
    """Stderr text for a WhisperModel failure. Missing weights stay free of a traceback."""
    if is_missing_whisper_weights(exc):
        return whisper_weights_message(model)
    return f"Could not load Whisper model '{model}': {exc}"


def _iter_exception_chain(exc: BaseException):
    """Yield an exception and the causes, contexts, and grouped exceptions under it."""
    stack: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.append(current.__cause__)
        stack.append(current.__context__)
        grouped = getattr(current, "exceptions", None)
        if isinstance(grouped, (tuple, list)):
            stack.extend(grouped)


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
        print(whisper_load_failure_message(name, exc), file=sys.stderr)
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


# Edit the prompt for each --style value here. `both` is the default.
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

TODO_SYSTEM_PROMPT = """\
You turn messy spoken transcripts into action items. Use only what is in the transcript. Do not invent tasks, names, dates, or facts. If there are no action items, write None.

The user message is the transcript. Return markdown with exactly this heading:
## Action items

Spoken-language cleanup:
- Remove filler words.
- Resolve "um, I mean" self-corrections. Keep the corrected meaning and drop the abandoned attempt.
- Keep the user's own wording for names and technical terms.
"""

SUMMARY_SYSTEM_PROMPT = """\
You turn messy spoken transcripts into a short summary. Use only what is in the transcript. Do not invent names, dates, or facts.

The user message is the transcript. Return markdown with exactly these headings:
## Title
## Summary

Spoken-language cleanup:
- Remove filler words.
- Resolve "um, I mean" self-corrections. Keep the corrected meaning and drop the abandoned attempt.
- Keep the user's own wording for names and technical terms.
"""

STYLE_PROMPTS = {
    "both": NOTES_SYSTEM_PROMPT,
    "todo": TODO_SYSTEM_PROMPT,
    "summary": SUMMARY_SYSTEM_PROMPT,
}

# Title is the note's H1. The other names are the body sections written for that style.
STYLE_HEADINGS = {
    "both": ("Title", "Summary", "Action items", "Key points"),
    "todo": ("Action items",),
    "summary": ("Title", "Summary"),
}

SECTION_HEADINGS = ("Title", "Summary", "Action items", "Key points")
_ATX_HEADING = re.compile(r"^#{1,6}\s+(.*?)(?:\s+#+)?\s*$")


def parse_sections(
    model_output: str,
    headings: tuple[str, ...] = SECTION_HEADINGS,
) -> dict[str, str]:
    """Split a model reply into the requested headings.

    If any requested heading is missing, return the raw reply under Notes
    instead of raising. Only the requested headings are returned.
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

    if any(name not in found for name in headings):
        return {"Notes": model_output}

    return {name: "\n".join(found[name]).strip() for name in headings}


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
    """Build the note, including a Notes fallback.

    Summary, Action items, and Key points are written only when that
    heading was parsed. The title, metadata block, and full transcript
    are always included.
    """
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
            if heading not in sections:
                continue
            lines.append(f"## {heading}")
            body = sections[heading].strip()
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
    style: str = "both",
) -> str:
    """Send a transcript to the Ollama client and return the notes reply.

    ``style`` selects the system prompt. The default is ``both``.
    """
    messages = [
        {"role": "system", "content": STYLE_PROMPTS[style]},
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


def merge_note_sections(
    parts: list[dict[str, str]],
    headings: tuple[str, ...] = SECTION_HEADINGS,
) -> dict[str, str]:
    """Merge per-chunk notes into one document.

    A single part is returned unchanged, with no Part heading. Several parts
    keep one summary each and combine action items and key points into one list.
    Only the requested headings are included.
    """
    if not parts:
        return {name: "" if name == "Title" else "None" for name in headings}
    if len(parts) == 1:
        return dict(parts[0])
    if any("Notes" in part for part in parts):
        return {"Notes": _join_fallback_notes(parts)}
    merged: dict[str, str] = {}
    if "Title" in headings:
        merged["Title"] = _first_title(parts)
    if "Summary" in headings:
        merged["Summary"] = _merge_summaries(parts)
    for name in ("Action items", "Key points"):
        if name in headings:
            merged[name] = _merge_lists(parts, name)
    return merged


def generate_notes_for_transcript(
    transcript: str,
    model: str,
    *,
    url: str = OLLAMA_CHAT_URL,
    style: str = "both",
) -> dict[str, str]:
    """Generate notes for one transcript, chunking when it is longer than the limit.

    A short transcript is one model call and one note, with no Part heading.
    Longer transcripts get notes per chunk, then those notes are merged.
    ``style`` selects the prompt and which sections are kept.
    """
    headings = STYLE_HEADINGS[style]
    chunks = chunk_transcript(transcript)
    if len(chunks) <= 1:
        text = chunks[0] if chunks else transcript
        return parse_sections(
            generate_notes(text, model, url=url, style=style),
            headings,
        )
    parsed = [
        parse_sections(
            generate_notes(chunk, model, url=url, style=style),
            headings,
        )
        for chunk in chunks
    ]
    return merge_note_sections(parsed, headings)


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
