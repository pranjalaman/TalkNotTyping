"""Local drag-and-drop page. Whisper and Ollama are stubbed."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from unittest.mock import patch

from voice2notes import Transcript, main, make_web_server, parse_args

NOTE = """\
# Milk

- Source: memo.wav
- Date: 2026-10-20
- Duration: 00:03
- Language: en
- Models: whisper-base, qwen2.5:3b

## Summary
Buy milk.

## Action items
- [ ] Buy milk

## Key points
- milk

## Full transcript
Buy milk tomorrow.
"""

TRANSCRIPT = Transcript(text="Buy milk tomorrow.", language="en", duration=3)


class FrozenDate(date):
    @classmethod
    def today(cls) -> FrozenDate:
        return cls(2026, 10, 20)


def fake_generate(transcript: str, model: str, *, url: str = "", style: str = "both") -> str:
    del transcript, model, url, style
    return (
        "## Title\nMilk\n"
        "## Summary\nBuy milk.\n"
        "## Action items\n- [ ] Buy milk\n"
        "## Key points\n- milk\n"
    )


class WebArgumentTests(unittest.TestCase):
    def test_web_alone_is_accepted(self) -> None:
        args = parse_args(["--web"])
        self.assertEqual(args.web, 8765)
        self.assertEqual(args.files, [])

    def test_web_port_is_accepted(self) -> None:
        self.assertEqual(parse_args(["--web", "9000"]).web, 9000)

    def test_no_files_and_no_web_prints_usage(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            parse_args([])
        self.assertEqual(caught.exception.code, 2)


class WebServerTests(unittest.TestCase):
    def _start(self, out: Path) -> int:
        args = parse_args(["--web", "0", "--out", str(out)])
        for item in (
            patch("voice2notes.load_whisper_model", return_value=object()),
            patch("voice2notes.transcribe_file", return_value=TRANSCRIPT),
            patch("voice2notes.generate_notes", side_effect=fake_generate),
            patch("voice2notes.date", FrozenDate),
        ):
            item.start()
            self.addCleanup(item.stop)
        server = make_web_server(args, 0)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_index_is_the_drop_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            port = self._start(Path(tmp))
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
                page = response.read().decode("utf-8")
        self.assertIn("Drop an audio file", page)
        self.assertIn("/notes", page)

    def test_upload_returns_the_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            port = self._start(out)
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/notes",
                data=b"RIFFfake",
                headers={"X-Filename": "memo.wav"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["note"], NOTE)
            written = Path(payload["path"])
            self.assertTrue(written.is_file())
            self.assertEqual(written.parent, out)
            self.assertTrue(written.name.startswith("2026-10-20_memo"))

    def test_unsupported_upload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            port = self._start(Path(tmp))
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/notes",
                data=b"hello",
                headers={"X-Filename": "notes.txt"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            body = json.loads(caught.exception.read().decode("utf-8"))
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("unsupported type .txt", body["error"])

    def test_main_starts_the_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            started = threading.Event()

            def serve_once(args, *, port: int) -> None:
                del args, port
                started.set()

            with patch("voice2notes.serve_web", side_effect=serve_once):
                main(["--web", "--out", str(tmp)])
            self.assertTrue(started.is_set())


if __name__ == "__main__":
    unittest.main()
