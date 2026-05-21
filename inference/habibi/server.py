"""
OpenAI-compatible TTS server for habibi-tts.
Exposes POST /v1/audio/speech — same interface as OpenAI TTS API.
"""

import io
import logging
import os
import tempfile

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("habibi-tts-server")

app = FastAPI(title="Habibi TTS Server", version="1.0.0")

# ── Config ────────────────────────────────────────────────────────────────────

HABIBI_MODEL   = os.getenv("HABIBI_MODEL", "Unified")        # Unified | SAU | EGY | MSA ...
HABIBI_VOICE   = os.getenv("HABIBI_VOICE", "SAU_male_1")     # ref audio name (no extension)
NFE_STEPS      = int(os.getenv("HABIBI_NFE", "16"))          # 16 = faster, 32 = better quality
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

ASSETS_DIR     = os.path.join(os.path.dirname(__file__), "assets")

# ── Model loading ─────────────────────────────────────────────────────────────

tts_model = None

def load_model():
    global tts_model
    logger.info("Loading habibi-tts model=%s on device=%s ...", HABIBI_MODEL, DEVICE)
    from habibi_tts import HabibiTTS
    tts_model = HabibiTTS(model=HABIBI_MODEL, device=DEVICE)
    logger.info("habibi-tts model loaded ✓")

@app.on_event("startup")
async def startup():
    load_model()

# ── Schemas ───────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    model: str = "habibi-tts"
    input: str
    voice: str = HABIBI_VOICE          # maps to ref audio filename
    response_format: str = "wav"       # only wav supported for now
    speed: float = 1.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ref_audio(voice: str) -> tuple[str, str]:
    """
    Returns (ref_audio_path, ref_text) for the requested voice.
    Looks for <voice>.wav + <voice>.txt in the assets dir.
    Falls back to the default voice if not found.
    """
    wav_path = os.path.join(ASSETS_DIR, f"{voice}.wav")
    txt_path = os.path.join(ASSETS_DIR, f"{voice}.txt")

    if not os.path.exists(wav_path):
        # fall back to default voice bundled with habibi-tts
        logger.warning("Voice '%s' not found in assets, using built-in default.", voice)
        from habibi_tts.assets import get_asset_path
        wav_path = get_asset_path(f"{HABIBI_VOICE}.wav")
        txt_path = get_asset_path(f"{HABIBI_VOICE}.txt")

    with open(txt_path, encoding="utf-8") as f:
        ref_text = f.read().strip()

    return wav_path, ref_text

# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/v1/audio/speech")
async def text_to_speech(req: TTSRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input text is empty")

    try:
        ref_audio, ref_text = get_ref_audio(req.voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        tts_model.infer(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=req.input,
            output_path=out_path,
            nfe_step=NFE_STEPS,
            speed=req.speed,
        )

        with open(out_path, "rb") as f:
            audio_bytes = f.read()

        os.unlink(out_path)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except Exception as e:
        logger.exception("TTS inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "habibi-tts", "object": "model", "owned_by": "habibi"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": HABIBI_MODEL, "device": DEVICE}
