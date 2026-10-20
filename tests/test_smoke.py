"""End-to-end smoke test: real Whisper, local stand-in for the notes model."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(__file__).resolve().parent / "fixtures" / "sample.wav"

REPLY = """\
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


class _OllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length:
            self.rfile.read(length)
        body = json.dumps({"message": {"role": "assistant", "content": REPLY}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


class SmokeTests(unittest.TestCase):
    def test_sample_recording_writes_a_note(self) -> None:
        self.assertTrue(SAMPLE.is_file(), f"missing sample recording: {SAMPLE}")
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 11434), _OllamaHandler)
        except OSError as exc:
            self.skipTest(f"port 11434 is already in use: {exc}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server() -> None:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.addCleanup(stop_server)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "voice2notes.py"),
                    str(SAMPLE),
                    "--out",
                    str(out),
                    "--whisper-model",
                    "tiny",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            notes = list(out.glob("*.md"))
            self.assertEqual(len(notes), 1, result.stderr or result.stdout)
            text = notes[0].read_text(encoding="utf-8")
            note_name = notes[0].name
        self.assertEqual(
            result.returncode,
            0,
            result.stderr or result.stdout,
        )
        self.assertIn("# Milk run", text)
        self.assertIn("## Summary", text)
        self.assertIn("## Action items", text)
        self.assertIn("## Key points", text)
        self.assertIn("## Full transcript", text)
        self.assertIn("- Language:", text)
        transcript = text.split("## Full transcript", 1)[1].strip()
        self.assertTrue(transcript)
        self.assertIn(note_name, result.stdout)


if __name__ == "__main__":
    unittest.main()
