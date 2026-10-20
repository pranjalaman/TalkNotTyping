"""Error-table checks for offline Whisper weights and the messages that already existed."""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import voice2notes
from voice2notes import (
    Transcript,
    generate_notes,
    is_missing_whisper_weights,
    load_whisper_model,
    main,
    ollama_chat,
    ollama_setup_message,
    validate_audio_files,
    whisper_load_failure_message,
    whisper_weights_message,
)


class _NotFoundHandler(BaseHTTPRequestHandler):
    """One POST that answers the way Ollama does when the model was never pulled."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = b'{"error":"model not found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def _closed_port_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/api/chat"


class WhisperOfflineMessageTests(unittest.TestCase):
    def test_message_names_the_model_and_the_offline_switch(self) -> None:
        text = whisper_weights_message("small")
        self.assertIn("Whisper model 'small' is not downloaded.", text)
        self.assertIn("one-time download", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertNotIn("Traceback", text)

    def test_local_cache_miss_uses_the_friendly_message(self) -> None:
        from huggingface_hub.errors import LocalEntryNotFoundError

        exc = LocalEntryNotFoundError(
            "Cannot find an appropriate cached snapshot folder and "
            "outgoing traffic has been disabled."
        )
        text = whisper_load_failure_message("base", exc)
        self.assertEqual(text, whisper_weights_message("base"))
        self.assertNotIn("outgoing traffic", text)
        self.assertNotIn("Traceback", text)

    def test_offline_cause_is_classified(self) -> None:
        from huggingface_hub.errors import OfflineModeIsEnabled

        outer = RuntimeError("Could not load Whisper model")
        outer.__cause__ = OfflineModeIsEnabled("HF_HUB_OFFLINE is set")
        self.assertTrue(is_missing_whisper_weights(outer))
        self.assertEqual(
            whisper_load_failure_message("medium", outer),
            whisper_weights_message("medium"),
        )

    def test_hub_and_connection_errors_are_classified(self) -> None:
        import httpx2

        class HfHubHTTPError(Exception):
            """Stand-in with the hub error's class name. Nothing is downloaded."""

        hub = HfHubHTTPError("504 Gateway Timeout")
        self.assertTrue(is_missing_whisper_weights(hub))
        self.assertTrue(
            is_missing_whisper_weights(httpx2.ConnectError("connection refused"))
        )
        self.assertTrue(
            is_missing_whisper_weights(httpx2.ProxyError("proxy unreachable"))
        )
        self.assertTrue(is_missing_whisper_weights(OSError(10065, "No route to host")))
        wrapped = RuntimeError("snapshot_download failed")
        wrapped.__cause__ = httpx2.ConnectError("timed out")
        self.assertEqual(
            whisper_load_failure_message("tiny", wrapped),
            whisper_weights_message("tiny"),
        )

    def test_unrelated_load_errors_keep_the_original_message(self) -> None:
        corrupt = RuntimeError("Unable to open model.bin")
        missing_file = FileNotFoundError(2, "model.bin")
        denied = OSError(13, "Permission denied")
        self.assertFalse(is_missing_whisper_weights(corrupt))
        self.assertFalse(is_missing_whisper_weights(missing_file))
        self.assertFalse(is_missing_whisper_weights(denied))
        self.assertEqual(
            whisper_load_failure_message("base", corrupt),
            "Could not load Whisper model 'base': Unable to open model.bin",
        )

    def test_cache_miss_wording_is_classified_without_a_hub_type(self) -> None:
        exc = Exception(
            "Cannot find the requested files in the disk cache and "
            "outgoing traffic has been disabled."
        )
        self.assertTrue(is_missing_whisper_weights(exc))

    def test_load_prints_the_friendly_message_and_exits(self) -> None:
        from huggingface_hub.errors import LocalEntryNotFoundError

        fake = ModuleType("faster_whisper")

        class WhisperModel:
            def __init__(self, *args, **kwargs):
                raise LocalEntryNotFoundError(
                    "outgoing traffic has been disabled"
                )

        fake.WhisperModel = WhisperModel
        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, {"faster_whisper": fake}):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    load_whisper_model("small")
        self.assertEqual(caught.exception.code, 1)
        text = stderr.getvalue().strip()
        self.assertEqual(text, whisper_weights_message("small"))
        self.assertNotIn("Traceback", text)

    def test_live_offline_cache_miss_names_the_missing_model(self) -> None:
        """HF_HUB_OFFLINE with an empty cache. Does not use or delete the user cache."""
        script = r"""
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(root)
os.environ["HF_HUB_CACHE"] = str(root / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
sys.path.insert(0, sys.argv[2])

import huggingface_hub.constants as constants

cache = Path(constants.HF_HUB_CACHE)
if not str(cache).startswith(str(root)):
    print(f"REFUSING cache={cache}", file=sys.stderr)
    raise SystemExit(2)

from voice2notes import load_whisper_model

load_whisper_model("small")
"""
        with tempfile.TemporaryDirectory(prefix="tnt-whisper-offline-") as tmp:
            completed = subprocess.run(
                [sys.executable, "-c", script, tmp, str(ROOT)],
                capture_output=True,
                text=True,
                timeout=90,
                cwd=ROOT,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            1,
            msg=f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
        )
        self.assertIn("Whisper model 'small' is not downloaded.", completed.stderr)
        self.assertIn("one-time download", completed.stderr)
        self.assertIn("HF_HUB_OFFLINE=1", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertNotIn("REFUSING", completed.stderr)


class ExistingErrorTableTests(unittest.TestCase):
    def test_ollama_not_running_names_the_configured_model(self) -> None:
        model = "custom-llm:9b"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                ollama_chat(
                    model,
                    [{"role": "user", "content": "hello"}],
                    url=_closed_port_url(),
                    timeout=2,
                )
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(stderr.getvalue().strip(), ollama_setup_message(model))
        self.assertIn("ollama serve", stderr.getvalue())
        self.assertIn(f"ollama pull {model}", stderr.getvalue())

    def test_generate_notes_uses_the_llm_name_when_the_port_is_closed(self) -> None:
        model = "qwen-test:3b"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                generate_notes("buy milk", model, url=_closed_port_url())
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(stderr.getvalue().strip(), ollama_setup_message(model))

    def test_model_not_pulled_uses_the_same_message(self) -> None:
        model = "not-pulled:7b"
        server = HTTPServer(("127.0.0.1", 0), _NotFoundHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    ollama_chat(
                        model,
                        [{"role": "user", "content": "hello"}],
                        url=f"http://127.0.0.1:{port}/api/chat",
                        timeout=5,
                    )
        finally:
            thread.join(timeout=5)
            server.server_close()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(stderr.getvalue().strip(), ollama_setup_message(model))

    def test_unsupported_audio_is_skipped_and_the_rest_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            bad = folder / "notes.txt"
            bad.write_text("not audio", encoding="utf-8")
            good = folder / "memo.wav"
            good.write_bytes(b"RIFF")
            missing = folder / "gone.wav"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                valid = validate_audio_files(
                    [str(bad), str(missing), str(good)]
                )
            text = stderr.getvalue()
            self.assertEqual(valid, [good])
            self.assertIn(f"Skipping {bad}: unsupported type .txt", text)
            self.assertIn(f"Skipping {missing}: file not found", text)
            self.assertNotIn(f"Skipping {good}", text)

    def test_unsupported_only_does_not_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "clip.txt"
            bad.write_text("nope", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                main([str(bad), "--llm", "unused:1b"])
            self.assertIn("unsupported type .txt", stderr.getvalue())

    def test_corrupt_audio_and_silence_skip_without_a_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            out = folder / "notes"
            bad = folder / "bad.wav"
            quiet = folder / "quiet.wav"
            spoken = folder / "spoken.wav"
            for path in (bad, quiet, spoken):
                path.write_bytes(b"RIFF")

            def transcribe(_model, path: Path, _language):
                if path.name == "bad.wav":
                    raise RuntimeError("Invalid data found when processing input")
                if path.name == "quiet.wav":
                    return Transcript(text="", language="en", duration=1.0)
                return Transcript(text="buy milk tomorrow", language="en", duration=3.0)

            seen: list[str] = []

            def generate(transcript: str, model: str, url: str = "") -> str:
                seen.append(model)
                self.assertEqual(transcript, "buy milk tomorrow")
                return (
                    "## Title\nMilk\n"
                    "## Summary\nBuy milk.\n"
                    "## Action items\n- buy milk\n"
                    "## Key points\n- milk\n"
                )

            stderr = io.StringIO()
            stdout = io.StringIO()
            with (
                mock.patch.object(voice2notes, "load_whisper_model", return_value=object()),
                mock.patch.object(voice2notes, "transcribe_file", side_effect=transcribe),
                mock.patch.object(voice2notes, "generate_notes", side_effect=generate),
                redirect_stderr(stderr),
                redirect_stdout(stdout),
            ):
                main(
                    [
                        str(bad),
                        str(quiet),
                        str(spoken),
                        "--out",
                        str(out),
                        "--llm",
                        "notes-model:3b",
                    ]
                )
            err = stderr.getvalue()
            self.assertIn(
                f"Skipping {bad}: Invalid data found when processing input",
                err,
            )
            self.assertIn(f"Warning: no speech detected in {quiet}", err)
            self.assertEqual(seen, ["notes-model:3b"])
            written = list(out.glob("*.md"))
            self.assertEqual(len(written), 1)
            self.assertIn("spoken", written[0].name)
            self.assertIn(written[0].name, stdout.getvalue())

    def test_watch_flag_still_does_not_move_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            watch = Path(tmp) / "inbox"
            watch.mkdir()
            audio = watch / "memo.wav"
            audio.write_bytes(b"RIFF")
            with redirect_stderr(io.StringIO()):
                main(["--watch", str(watch)])
            self.assertFalse((watch / "failed").exists())
            self.assertEqual(list(watch.iterdir()), [audio])


if __name__ == "__main__":
    unittest.main()
