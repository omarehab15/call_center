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

HABIBI_MODEL   = os.getenv("HABIBI_MODEL", "Unified")
HABIBI_VOICE   = os.getenv("HABIBI_VOICE", "SAU_male_1")
NFE_STEPS      = int(os.getenv("HABIBI_NFE", "16"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
ASSETS_DIR     = os.path.join(os.path.dirname(__file__), "assets")

# ── Model loading ─────────────────────────────────────────────────────────────

tts_model   = None
tts_vocoder = None

# Map model name → HuggingFace paths
MODEL_CONFIG = {
    "Unified": {
        "vocab":  "hf://SWivid/Habibi-TTS/Unified/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors",
    },
    "MSA": {
        "vocab":  "hf://SWivid/Habibi-TTS/Specialized/MSA/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Specialized/MSA/model_200000.safetensors",
    },
    "SAU": {
        "vocab":  "hf://SWivid/Habibi-TTS/Specialized/SAU/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Specialized/SAU/model_200000.safetensors",
    },
    "EGY": {
        "vocab":  "hf://SWivid/Habibi-TTS/Specialized/EGY/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Specialized/EGY/model_100000.safetensors",
    },
    "IRQ": {
        "vocab":  "hf://SWivid/Habibi-TTS/Specialized/IRQ/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Specialized/IRQ/model_100000.safetensors",
    },
    "MAR": {
        "vocab":  "hf://SWivid/Habibi-TTS/Specialized/MAR/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Specialized/MAR/model_100000.safetensors",
    },
    "ALG": {
        "vocab":  "hf://SWivid/Habibi-TTS/Specialized/ALG/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Specialized/ALG/model_100000.safetensors",
    },
    "UAE": {
        "vocab":  "hf://SWivid/Habibi-TTS/Specialized/UAE/vocab.txt",
        "ckpt":   "hf://SWivid/Habibi-TTS/Specialized/UAE/model_100000.safetensors",
    },
}

def load_model():
    global tts_model, tts_vocoder

    logger.info("Loading habibi-tts model=%s on device=%s ...", HABIBI_MODEL, DEVICE)

    from cached_path import cached_path
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_model as f5_load_model, load_vocoder

    cfg = MODEL_CONFIG.get(HABIBI_MODEL, MODEL_CONFIG["Unified"])

    vocab_path = str(cached_path(cfg["vocab"]))
    ckpt_path  = str(cached_path(cfg["ckpt"]))

    base_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

    tts_model   = f5_load_model(DiT, base_cfg, ckpt_path, vocab_file=vocab_path)
    tts_vocoder = load_vocoder()

    logger.info("habibi-tts model loaded ✓")


@app.on_event("startup")
async def startup():
    load_model()

# ── Schemas ───────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    model: str = "habibi-tts"
    input: str
    voice: str = HABIBI_VOICE
    response_format: str = "wav"
    speed: float = 1.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ref_audio(voice: str) -> tuple[str, str]:
    wav_path = os.path.join(ASSETS_DIR, f"{voice}.wav")
    txt_path = os.path.join(ASSETS_DIR, f"{voice}.txt")

    if not os.path.exists(wav_path):
        logger.warning("Voice '%s' not found in assets, using built-in default.", voice)
        from importlib.resources import files
        assets_pkg = files("habibi_tts").joinpath("assets")
        wav_path = str(assets_pkg.joinpath(f"{HABIBI_VOICE}.wav"))
        txt_path = str(assets_pkg.joinpath(f"{HABIBI_VOICE}.txt"))

    with open(txt_path, encoding="utf-8") as f:
        ref_text = f.read().strip()

    return wav_path, ref_text

# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/v1/audio/speech")
async def text_to_speech(req: TTSRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input text is empty")

    try:
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process

        ref_audio, ref_text = get_ref_audio(req.voice)

        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        audio, sample_rate, _ = infer_process(
            ref_audio     = ref_audio_proc,
            ref_text      = ref_text_proc,
            gen_text      = req.input,
            model_obj     = tts_model,
            vocoder       = tts_vocoder,
            nfe_step      = NFE_STEPS,
            speed         = req.speed,
            device        = DEVICE,
        )

        sf.write(out_path, audio, sample_rate)

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
