#!/usr/bin/env python3
"""TalkNotTyping: turn a rambling voice memo into a tidy markdown note."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}


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
    printed = False
    for path in args.files:
        try:
            transcript = transcribe_file(model, path, args.lang)
        except Exception as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
            continue
        if not transcript.text:
            print(f"Warning: no speech detected in {path}", file=sys.stderr)
            continue
        if printed:
            print()
        print(transcript.text)
        printed = True


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


if __name__ == "__main__":
    main()
