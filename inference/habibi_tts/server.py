"""
OpenAI-compatible TTS server for Habibi-TTS.

Implements:
  POST /v1/audio/speech   — same interface as OpenAI TTS API
  GET  /v1/models         — model list (for health checks)
  GET  /health            — simple health probe

LiveKit agent uses:
  openai.TTS(base_url="http://habibi_tts:8000/v1", model="habibi", voice="SAU", ...)
"""

import io
import os
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("habibi-server")

app = FastAPI(title="Habibi-TTS OpenAI-compatible server")

# ── Dialect / model config ────────────────────────────────────────────────────
# Supported dialect IDs: MSA SAU UAE ALG IRQ EGY MAR OMN TUN LEV SDN LBY
DEFAULT_DIALECT = os.getenv("HABIBI_DIALECT", "SAU")
# Reference audio file path (optional — zero-shot voice cloning)
# If not set the model uses its built-in default prompt for the dialect.
REF_AUDIO = os.getenv("HABIBI_REF_AUDIO", "")
REF_TEXT  = os.getenv("HABIBI_REF_TEXT",  "")

_pipeline = None

def get_pipeline():
    """Lazy-load the Habibi pipeline once."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading Habibi-TTS pipeline (dialect=%s)…", DEFAULT_DIALECT)
        try:
            from habibi_tts import HabibiTTS  # type: ignore
            _pipeline = HabibiTTS(dialect=DEFAULT_DIALECT)
            logger.info("Habibi-TTS pipeline ready.")
        except ImportError:
            # Fallback: use the f5-tts CLI wrapper if HabibiTTS class not available
            logger.warning("habibi_tts.HabibiTTS not found — using CLI wrapper")
            _pipeline = "cli"
    return _pipeline


def synthesize(text: str, dialect: str) -> bytes:
    """Generate WAV bytes for the given Arabic text."""
    pipeline = get_pipeline()

    if pipeline == "cli":
        # CLI fallback — calls habibi-tts_infer-cli subprocess
        import subprocess, shlex
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        cmd = [
            "habibi-tts_infer-cli",
            "--gen_text", text,
            "--dialect_id", dialect,
            "--output_file", out_path,
        ]
        if REF_AUDIO:
            cmd += ["--ref_audio", REF_AUDIO]
        if REF_TEXT:
            cmd += ["--ref_text", REF_TEXT]

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())

        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        Path(out_path).unlink(missing_ok=True)
        return wav_bytes

    else:
        # HabibiTTS class path
        audio_data = pipeline.synthesize(text)          # returns np.ndarray or bytes
        if isinstance(audio_data, (bytes, bytearray)):
            return bytes(audio_data)

        # numpy array → WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio_data, samplerate=24000, format="WAV")
        return buf.getvalue()


# ── Request model ─────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    model: str = "habibi"
    input: str
    voice: str = DEFAULT_DIALECT      # dialect ID used as "voice" in OpenAI API
    response_format: str = "wav"
    speed: float = 1.0


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "habibi", "object": "model", "owned_by": "habibi-tts"},
        ],
    }


@app.post("/v1/audio/speech")
async def text_to_speech(req: TTSRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input text is empty")

    dialect = req.voice.upper() if req.voice else DEFAULT_DIALECT
    logger.info("TTS request: dialect=%s len=%d", dialect, len(req.input))

    try:
        wav_bytes = synthesize(req.input, dialect)
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )
