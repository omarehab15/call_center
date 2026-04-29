"""
lahgtna-chatterbox TTS server — OpenAI-compatible /v1/audio/speech

Uses the oddadmix lahgtna-chatterbox fork (not PyPI chatterbox-tts).
Loads model with from_checkpoint() exactly as the Colab notebook does.
"""

import glob
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel

# Import from the cloned fork (PYTHONPATH=/app so src.chatterbox resolves)
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lahgtna")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HF_REPO_ID   = os.getenv("HF_MODEL_REPO", "oddadmix/lahgtna-chatterbox-v1")
VOICES_DIR   = os.getenv("VOICES_DIR", "/voices")
LANGUAGE_ID  = os.getenv("LANGUAGE_ID", "sa")   # Saudi Arabic

EXAGGERATION       = float(os.getenv("EXAGGERATION", "0.5"))
CFG_WEIGHT         = float(os.getenv("CFG_WEIGHT", "0.5"))
TEMPERATURE        = float(os.getenv("TEMPERATURE", "0.8"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "2.0"))

# ── Global model state ────────────────────────────────────────────────────────
MODEL    = None
MODEL_SR = 24000


def load_model():
    global MODEL, MODEL_SR

    logger.info("Downloading checkpoint from %s …", HF_REPO_ID)
    t0 = time.time()

    # Download only the files the notebook specifies
    ckpt_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        revision="main",
        allow_patterns=[
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ],
    )
    logger.info("Checkpoint downloaded to %s (%.1fs)", ckpt_dir, time.time() - t0)

    logger.info("Loading model on device=%s …", DEVICE)
    t1 = time.time()
    MODEL = ChatterboxMultilingualTTS.from_checkpoint(str(ckpt_dir) + "/", DEVICE)
    if hasattr(MODEL, "to") and str(getattr(MODEL, "device", "")) != DEVICE:
        MODEL.to(DEVICE)

    MODEL_SR = MODEL.sr
    logger.info(
        "Model ready. sr=%dHz  load_time=%.1fs  total=%.1fs",
        MODEL_SR, time.time() - t1, time.time() - t0,
    )


# ── Voice file resolution ─────────────────────────────────────────────────────

def resolve_voice_file(voice_name: str) -> str | None:
    """
    Map voice name (e.g. 'fahad') → WAV file in VOICES_DIR.
    Matches: exact filename, prefix (Fasseh-fahad.wav), substring, any wav.
    """
    candidates = glob.glob(os.path.join(VOICES_DIR, "**", "*.wav"), recursive=True)
    if not candidates:
        return None
    vn = (voice_name or "").lower()
    for c in candidates:
        if os.path.splitext(os.path.basename(c))[0].lower() == vn:
            return c
    for c in candidates:
        if vn in os.path.basename(c).lower():
            return c
    return candidates[0]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def tensor_to_audio_bytes(
    wav_tensor: torch.Tensor, sample_rate: int, fmt: str = "mp3"
) -> tuple[bytes, str]:
    """Convert waveform tensor → (audio_bytes, mime_type). Supports mp3/wav/pcm."""
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = wav_tensor.cpu()
    fmt = fmt.lower()

    if fmt == "pcm":
        pcm = (wav_tensor * 32767).clamp(-32768, 32767).short()
        return pcm.numpy().tobytes(), "audio/pcm"

    buf = io.BytesIO()
    if fmt == "mp3":
        ta.save(buf, wav_tensor, sample_rate, format="mp3")
        mime = "audio/mpeg"
    else:
        ta.save(buf, wav_tensor, sample_rate, format="wav")
        mime = "audio/wav"

    buf.seek(0)
    return buf.read(), mime


# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="lahgtna-chatterbox TTS",
    description="OpenAI-compatible TTS — oddadmix/lahgtna-chatterbox-v1",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    model: str = "tts-1-hd"
    input: str
    voice: str = "fahad"
    response_format: str = "mp3"   # matches LiveKit openai.TTS default
    speed: float = 1.0
    # Chatterbox generation overrides (optional)
    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None


@app.get("/health")
@app.get("/v1/health")
async def health():
    return {"status": "ok", "model": HF_REPO_ID, "device": DEVICE, "loaded": MODEL is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "tts-1-hd", "object": "model", "created": 1700000000, "owned_by": "lahgtna"}],
    }


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input text is empty")

    voice_file = resolve_voice_file(req.voice)
    logger.info(
        "TTS request: chars=%d  voice=%s  file=%s  fmt=%s",
        len(text), req.voice, voice_file, req.response_format,
    )

    exaggeration  = req.exaggeration       if req.exaggeration       is not None else EXAGGERATION
    cfg_weight    = req.cfg_weight         if req.cfg_weight         is not None else CFG_WEIGHT
    temperature   = req.temperature        if req.temperature        is not None else TEMPERATURE
    rep_penalty   = req.repetition_penalty if req.repetition_penalty is not None else REPETITION_PENALTY

    # ── Generate — exactly as the Colab notebook does ──────────────────────────
    t0 = time.time()
    try:
        generate_kwargs = dict(
            language_id=LANGUAGE_ID,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=rep_penalty,
        )
        if voice_file:
            generate_kwargs["audio_prompt_path"] = voice_file

        wav = MODEL.generate(text, **generate_kwargs)

    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}")

    elapsed = time.time() - t0
    audio_duration = wav.shape[-1] / MODEL_SR
    logger.info(
        "Done: %.2fs audio in %.2fs (RTF %.2f)",
        audio_duration, elapsed, elapsed / max(audio_duration, 0.001),
    )

    audio_bytes, mime_type = tensor_to_audio_bytes(wav, MODEL_SR, req.response_format)
    return Response(
        content=audio_bytes,
        media_type=mime_type,
        headers={"Content-Disposition": f"inline; filename=speech.{req.response_format}"},
    )