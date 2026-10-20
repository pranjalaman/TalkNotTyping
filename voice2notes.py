#!/usr/bin/env python3
"""TalkNotTyping: turn a rambling voice memo into a tidy markdown note."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"}


def main(argv: list[str] | None = None) -> None:
    """Run the voice-to-notes pipeline."""
    parse_args(argv)


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


if __name__ == "__main__":
    main()
