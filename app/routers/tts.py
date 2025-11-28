# app/routers/tts.py

import asyncio, json

from pathlib import Path
from typing import AsyncGenerator, Dict, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


# ---- Voice registry -------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

VOICES: Dict[str, Tuple[str, str]] = {
    "en-gb-male": (
        str(MODELS_DIR / "en_GB-alan-medium.onnx"),
        str(MODELS_DIR / "en_GB-alan-medium.onnx.json"),
    ),
    "en-gb-female": (
        str(MODELS_DIR / "en_GB-alba-medium.onnx"),
        str(MODELS_DIR / "en_GB-alba-medium.onnx.json"),
    ),
    "en-gb-neutral": (
        str(MODELS_DIR / "en_GB-alan-medium.onnx"),
        str(MODELS_DIR / "en_GB-alan-medium.onnx.json"),
    ),
}


def _piper_cmd(voice_id: str):
    model, cfg = VOICES.get(voice_id, VOICES["en-gb-neutral"])
    return ["piper", "--model", model, "--config", cfg, "--raw-output", "-"]

router = APIRouter()

@router.post("/tts/stream")
async def tts_stream(request: Request, voice: str = "en-gb-neutral"):
    """
    Accepts NDJSON lines: {"text": "..."} on the request body (streaming).
    Streams back 'audio/webm;codecs=opus' as soon as audio is produced.
    """
    # 1) Start Piper (PCM out)
    piper = await asyncio.create_subprocess_exec(
        *_piper_cmd(voice),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    # 2) Transcode to WebM/Opus for browser playback
    ffmpeg = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-f", "s16le", "-ar", "22050", "-ac", "1",  # match Piper default PCM
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-f", "webm",
        "pipe:1",
        stdin=piper.stdout,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def producer() -> AsyncGenerator[bytes, None]:
        """
        - Read NDJSON from client (sentences).
        - Write each sentence to Piper (newline-terminated).
        - Concurrently read encoded bytes from ffmpeg and yield them.
        """
        try:
            async for line in request.stream():
                # Each line must be JSON: {"text": "..."}
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                txt = (obj.get("text") or "").strip()
                if not txt:
                    continue

                # Piper expects newline-terminated utterances
                piper.stdin.write((txt + "\n").encode("utf-8"))
                await piper.stdin.drain()

                # Emit whatever audio is ready right now with minimal latency
                while True:
                    chunk = await ffmpeg.stdout.read(4096)
                    if not chunk:
                        break
                    yield chunk

            # Client finished sending; close Piper stdin so it flushes
            if not piper.stdin.at_eof():
                piper.stdin.write_eof()

            # Drain remaining audio
            while True:
                chunk = await ffmpeg.stdout.read(4096)
                if not chunk:
                    break
                yield chunk

        finally:
            for proc in (piper, ffmpeg):
                try:
                    if proc and proc.returncode is None:
                        proc.terminate()
                except Exception:
                    pass

    return StreamingResponse(producer(), media_type="audio/webm;codecs=opus")
