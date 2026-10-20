#!/usr/bin/env python3
"""TalkNotTyping: turn a rambling voice memo into a tidy markdown note.

Everything after this stub lands in later issues. Running with --help
is the scaffold check.
"""

import argparse


def main() -> None:
    """Run the voice-to-notes pipeline."""


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="voice2notes.py",
        description="Turn a rambling voice recording into a tidy markdown note.",
    )


if __name__ == "__main__":
    build_parser().parse_args()
    main()
