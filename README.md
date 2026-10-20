# TalkNotTyping

Talk into a voice memo. Get a markdown note back: a short title, a summary, action items, key points, and the full transcript. Speech-to-text and the notes model both run on your machine. After the one-time model downloads, the pipeline does not need the network.

## Install

You need Python 3.10 or newer, plus [Ollama](https://ollama.com/).

```bash
git clone https://github.com/pranjalaman/TalkNotTyping.git
cd TalkNotTyping
pip install -r requirements.txt
ollama serve
ollama pull qwen2.5:3b
```

Leave `ollama serve` running in its own terminal. The first transcription also downloads the Whisper `base` weights once. That download is the slow step. A first note from the sample below is a few minutes on a normal connection, then seconds after that.

`requirements.txt` pins PyAV below version 19. The current faster-whisper release cannot decode audio with PyAV 19.

## First note

```bash
python voice2notes.py tests/fixtures/sample.wav
```

The recording says: "Remember to buy milk and call Priya tomorrow."

The note is written to `notes/`, named with the date and the audio file, for example `notes/2026-10-20_sample.md`. Running the same file again does not overwrite. The next file ends in `-2`.

A note looks like this:

```markdown
# Milk run

- Source: sample.wav
- Date: 2026-10-20
- Duration: 00:04
- Language: en
- Models: whisper-base, qwen2.5:3b

## Summary
Buy milk and call Priya tomorrow.

## Action items
- [ ] Buy milk
- [ ] Call Priya

## Key points
- Milk
- Priya

## Full transcript
Remember to buy milk and call Priya tomorrow.
```

The transcript is always included so you can check that the notes model did not invent anything. Wording will vary a little with the model you pull.

## Usage

One file, or several:

```bash
python voice2notes.py memo.m4a
python voice2notes.py memo.m4a interview.wav --out ~/Notes
```

Supported audio: mp3, m4a, wav, ogg, flac, webm. A missing or unsupported file is skipped. The rest of the run continues.

Style:

```bash
python voice2notes.py memo.m4a --style todo      # action items, plus the transcript
python voice2notes.py memo.m4a --style summary   # title and summary, plus the transcript
python voice2notes.py memo.m4a --style both      # default
```

Export action items. A due date is written only when that calendar date was spoken in the memo. "Tomorrow" is not turned into a date.

```bash
python voice2notes.py memo.m4a --export todo
python voice2notes.py memo.m4a --export todo --export ics
```

Hashtags for note apps:

```bash
python voice2notes.py memo.m4a --tags meeting,priya
```

Local page, bound to `127.0.0.1` only:

```bash
python voice2notes.py --web
```

Open http://127.0.0.1:8765 and drop a file. Pass `--web 9000` to change the port.

Smaller or larger Whisper models, another Ollama model, or a forced language:

```bash
python voice2notes.py memo.m4a --whisper-model tiny
python voice2notes.py memo.m4a --llm llama3.2:3b
python voice2notes.py memo.m4a --lang hi
```

## CLI

```
voice2notes.py [files ...] [options]

positional:
  files                 one or more audio files (mp3, m4a, wav, ogg, flac, webm)

options:
  --out DIR             output folder (default: ./notes)
  --style {both,todo,summary}
                        note style (default: both)
  --whisper-model NAME  tiny, base, small, or medium (default: base)
  --llm NAME            Ollama model tag (default: qwen2.5:3b)
  --lang CODE           force a language code, for example en or hi (default: auto-detect)
  --tags TAGS           comma-separated hashtags added to the note
  --export {todo,ics}   also write action items; repeat the flag to write both
  --web [PORT]          local drag-and-drop page (default port: 8765)
```

`--watch`, `--interval`, and `--keep-audio` are accepted and are not wired up yet. A folder is not polled.

With no files and no `--web`, the command prints usage and exits.

## Offline

Two downloads happen once: the Whisper weights, and the Ollama model (`ollama pull qwen2.5:3b`). After that, disconnect and run as usual. Ollama is already local. To stop Whisper from trying the network:

```bash
# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python voice2notes.py memo.m4a

# macOS or Linux
HF_HUB_OFFLINE=1 python voice2notes.py memo.m4a
```

If the Whisper weights are missing and the machine is offline, the command names the model and tells you to connect once, then use `HF_HUB_OFFLINE=1` afterwards.

If Ollama is not running, or the model was not pulled, the command exits with:

```
Start Ollama (`ollama serve`) and pull the model (`ollama pull qwen2.5:3b`)
```

The model name in that line is whatever you passed to `--llm`.

## Languages

The note header includes the language code on its own line (`- Language: hi`). Leave `--lang` unset and that code is the one Whisper auto-detected. Pass `--lang hi` to force Hindi. The note then shows `hi` instead of the auto-detected code.

Small Whisper models (`tiny` and `base`) are weaker on Hindi and Hinglish. They handle code-switching poorly and often miss names. The notes model may also miss nuance on a short non-English transcript.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The smoke test transcribes `tests/fixtures/sample.wav` with the real Whisper model and a local stand-in for Ollama on port 11434. That port needs to be free. Parser tests cover a well-formed reply and a reply that is missing headings. Filename tests check that a second run writes `-2`.

## Why open source

Voice memos are personal. This stack never uploads the audio. faster-whisper runs on the CPU. The notes model is an open weight you choose in Ollama, called only at `http://localhost:11434`. There is no account and no per-minute transcription bill, so a long memo costs the same as a short one: nothing.

The pieces are also replaceable. `--whisper-model` swaps tiny, base, small, and medium. `--llm` swaps the notes model (`qwen2.5:3b`, `llama3.2:3b`, or anything else you have pulled). The prompt and the note headings are plain text in `voice2notes.py`, so the output format is yours to change.

A closed transcription API would charge for every minute, would receive the recording, and would stop working when the network does. Here the recording, the transcript, and the note stay in the folder you picked.
