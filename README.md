# TalkNotTyping

A lazy person's note taker. Ramble into a voice memo and get a clean markdown note back. Speech-to-text and the notes model both run on your machine.

This is the project scaffold. Transcription, note generation, and watch mode are not wired up yet.

## Setup

```bash
pip install -r requirements.txt
python voice2notes.py --help
```

Requires Python 3.10 or newer.

## Languages

The note header includes the language code on its own line (`- Language: hi`). Leave `--lang` unset and that code is the one Whisper auto-detected. Pass `--lang hi` to force Hindi; the note then shows `hi` instead of the auto-detected code.

Small Whisper models (`tiny` and `base`) are weaker on Hindi and Hinglish. They handle code-switching poorly and often miss names. The notes model may also miss nuance on a short non-English transcript.
