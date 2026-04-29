"""
lahgtna-chatterbox TTS server
OpenAI-compatible /v1/audio/speech endpoint for LiveKit agents.

The server loads oddadmix/lahgtna-chatterbox-v1 at startup and keeps it
warm in GPU memory. Each request generates audio for a sentence-length
chunk (LiveKit already handles sentence splitting upstream), so per-call
latency is the only thing that matters here — no internal streaming needed.

Voice cloning: pass ?voice=<name> or include "voice" in the JSON body.
The server looks for /voices/<name>.wav (case-insensitive glob match).
Mount your existing inference/xtts/voices/ directory into /voices.
"""

import glob
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lahgtna")

# ─────────────────────────────────────────────
# Configuration (all via environment variables)
# ─────────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "oddadmix/lahgtna-chatterbox-v1")
VOICES_DIR = os.getenv("VOICES_DIR", "/voices")
DEFAULT_VOICE_FILE = os.getenv("DEFAULT_VOICE_FILE", "")  # fallback if no match

# Generation parameters (tunable via env vars)
EXAGGERATION = float(os.getenv("EXAGGERATION", "0.5"))
CFG_WEIGHT = float(os.getenv("CFG_WEIGHT", "0.5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.2"))  # lahgtna needs this
LANGUAGE_ID = os.getenv("LANGUAGE_ID", "ar")

# ─────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────
MODEL = None
MODEL_SR = 24000  # chatterbox default sample rate


def load_model():
    """
    Load lahgtna-chatterbox-v1.

    Strategy:
    1. Try ChatterboxMultilingualTTS.from_pretrained(repo_id=HF_MODEL_REPO)
       This works if the chatterbox library accepts a custom repo_id.
    2. Fall back to: load base multilingual model, then patch T3 weights
       from the fine-tune repo (same pattern as other chatterbox fine-tunes).
    """
    global MODEL, MODEL_SR
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    logger.info("Loading model: %s  device: %s", HF_MODEL_REPO, DEVICE)
    t0 = time.time()

    # ── Attempt 1: direct repo_id argument ──────────────────────────────────
    try:
        MODEL = ChatterboxMultilingualTTS.from_pretrained(
            device=DEVICE, repo_id=HF_MODEL_REPO
        )
        logger.info("Model loaded via repo_id in %.1fs", time.time() - t0)
    except TypeError:
        # Older chatterbox versions don't have repo_id param → fall through
        logger.info("repo_id not supported, using base + fine-tune weight strategy")
        MODEL = None

    # ── Attempt 2: base model + patch fine-tune weights ──────────────────────
    if MODEL is None:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file as load_safetensors

        logger.info("Downloading base ChatterboxMultilingualTTS…")
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

        logger.info("Downloading lahgtna weights from %s…", HF_MODEL_REPO)
        model_dir = snapshot_download(HF_MODEL_REPO)

        # Look for fine-tuned T3 component (language model backbone)
        t3_path = os.path.join(model_dir, "t3.safetensors")
        if os.path.exists(t3_path):
            logger.info("Patching T3 weights from %s", t3_path)
            state = load_safetensors(t3_path, device="cpu")
            MODEL.t3.load_state_dict(state, strict=False)
            MODEL.t3.to(DEVICE).eval()
            logger.info("T3 weights applied")
        else:
            # Some fine-tunes ship the whole model as a single checkpoint
            ckpt_candidates = glob.glob(os.path.join(model_dir, "*.safetensors"))
            if ckpt_candidates:
                logger.info("Loading full checkpoint: %s", ckpt_candidates[0])
                state = load_safetensors(ckpt_candidates[0], device="cpu")
                # Try T3 first, then full model
                try:
                    MODEL.t3.load_state_dict(state, strict=False)
                    MODEL.t3.to(DEVICE).eval()
                except Exception:
                    MODEL.load_state_dict(state, strict=False)
                    MODEL.to(DEVICE).eval()
            else:
                logger.warning(
                    "No .safetensors found in %s — using base multilingual model. "
                    "Arabic dialect quality may be limited.",
                    model_dir,
                )

    MODEL_SR = MODEL.sr
    logger.info(
        "Model ready. Sample rate: %dHz  Total load time: %.1fs",
        MODEL_SR,
        time.time() - t0,
    )


# ─────────────────────────────────────────────
# Voice file resolution
# ─────────────────────────────────────────────

def resolve_voice_file(voice_name: str) -> str | None:
    """
    Map a voice name (e.g. 'fahad') to a .wav file in VOICES_DIR.

    Search order:
    1. Exact match:       /voices/fahad.wav
    2. Prefix match:      /voices/Fasseh-fahad.wav  (XTTS naming convention)
    3. Any file containing the voice name (case-insensitive)
    4. DEFAULT_VOICE_FILE env var
    5. Any .wav in VOICES_DIR
    """
    if not voice_name:
        voice_name = ""

    candidates = glob.glob(os.path.join(VOICES_DIR, "**", "*.wav"), recursive=True)

    # Exact
    for c in candidates:
        if os.path.splitext(os.path.basename(c))[0].lower() == voice_name.lower():
            return c

    # Contains (handles Fasseh-fahad.wav → "fahad")
    for c in candidates:
        if voice_name.lower() in os.path.basename(c).lower():
            return c

    # Env default
    if DEFAULT_VOICE_FILE and os.path.exists(DEFAULT_VOICE_FILE):
        return DEFAULT_VOICE_FILE

    # Any wav as last resort
    return candidates[0] if candidates else None


# ─────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────

def tensor_to_audio_bytes(
    wav_tensor: torch.Tensor, sample_rate: int, fmt: str = "mp3"
) -> tuple[bytes, str]:
    """
    Convert a torchaudio waveform tensor to audio bytes in the requested format.

    Returns (audio_bytes, mime_type).

    Supported formats:
      mp3  — default; what LiveKit openai.TTS requests
      wav  — lossless, larger
      pcm  — raw signed 16-bit little-endian samples, no header
    """
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


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="lahgtna-chatterbox TTS",
    description="OpenAI-compatible TTS endpoint powered by lahgtna-chatterbox-v1",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    model: str = "tts-1-hd"
    input: str
    voice: str = "fahad"
    response_format: str = "mp3"  # matches LiveKit openai.TTS default
    speed: float = 1.0
    # Chatterbox-specific overrides (optional)
    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None


@app.get("/health")
@app.get("/v1/health")
async def health():
    return {"status": "ok", "model": HF_MODEL_REPO, "device": DEVICE}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1700000000,
                "owned_by": "lahgtna",
            }
        ],
    }


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input text is empty")

    voice_file = resolve_voice_file(req.voice)
    if voice_file:
        logger.info("TTS: voice=%s  file=%s  chars=%d", req.voice, voice_file, len(text))
    else:
        logger.info("TTS: no voice file found for '%s', generating without prompt", req.voice)

    exaggeration = req.exaggeration if req.exaggeration is not None else EXAGGERATION
    cfg_weight = req.cfg_weight if req.cfg_weight is not None else CFG_WEIGHT
    temperature = req.temperature if req.temperature is not None else TEMPERATURE
    rep_penalty = req.repetition_penalty if req.repetition_penalty is not None else REPETITION_PENALTY

    t0 = time.time()
    try:
        generate_kwargs = dict(
            text=text,
            language_id=LANGUAGE_ID,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        # Voice cloning: pass reference audio if available
        if voice_file:
            generate_kwargs["audio_prompt_path"] = voice_file

        # repetition_penalty: supported in mtl_tts.generate if the model exposes it
        try:
            generate_kwargs["repetition_penalty"] = rep_penalty
            wav = MODEL.generate(**generate_kwargs)
        except TypeError:
            # Older chatterbox versions may not have repetition_penalty
            del generate_kwargs["repetition_penalty"]
            wav = MODEL.generate(**generate_kwargs)

    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}")

    elapsed = time.time() - t0
    audio_duration = wav.shape[-1] / MODEL_SR
    logger.info(
        "Generated %.2fs audio in %.2fs (RTF %.2f)",
        audio_duration,
        elapsed,
        elapsed / max(audio_duration, 0.001),
    )

    audio_bytes, mime_type = tensor_to_audio_bytes(wav, MODEL_SR, req.response_format)

    return Response(
        content=audio_bytes,
        media_type=mime_type,
        headers={
            "Content-Disposition": f"inline; filename=speech.{req.response_format}",
            "X-RTF": f"{elapsed / max(audio_duration, 0.001):.3f}",
        },
    )


# ─────────────────────────────────────────────
# Streaming endpoint (sentence-chunked)
# LiveKit doesn't use this directly, but useful
# for testing and future integrations.
# ─────────────────────────────────────────────

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?،؟])\s+')

import re


def split_sentences(text: str) -> list[str]:
    """Split Arabic/mixed text on sentence boundaries."""
    parts = SENTENCE_ENDINGS.split(text.strip())
    # Filter empty, recombine very short fragments
    result = []
    buf = ""
    for part in parts:
        buf = (buf + " " + part).strip() if buf else part
        if len(buf) >= 20:  # only emit chunks long enough to sound natural
            result.append(buf)
            buf = ""
    if buf:
        result.append(buf)
    return result or [text]


@app.post("/v1/audio/speech/stream")
async def text_to_speech_stream(req: SpeechRequest):
    """
    Sentence-chunked streaming: generates audio sentence by sentence
    and streams raw WAV bytes. Reduces time-to-first-audio significantly
    for long inputs.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    voice_file = resolve_voice_file(req.voice)
    exaggeration = req.exaggeration if req.exaggeration is not None else EXAGGERATION
    cfg_weight = req.cfg_weight if req.cfg_weight is not None else CFG_WEIGHT
    temperature = req.temperature if req.temperature is not None else TEMPERATURE
    rep_penalty = req.repetition_penalty if req.repetition_penalty is not None else REPETITION_PENALTY

    sentences = split_sentences(req.input)
    logger.info("Streaming %d sentence chunks for %d chars", len(sentences), len(req.input))

    def generate_chunks():
        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                kwargs = dict(
                    text=sentence,
                    language_id=LANGUAGE_ID,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                if voice_file:
                    kwargs["audio_prompt_path"] = voice_file
                try:
                    kwargs["repetition_penalty"] = rep_penalty
                    wav = MODEL.generate(**kwargs)
                except TypeError:
                    del kwargs["repetition_penalty"]
                    wav = MODEL.generate(**kwargs)

                audio_bytes, _ = tensor_to_audio_bytes(wav, MODEL_SR, req.response_format)
                yield audio_bytes
            except Exception as exc:
                logger.error("Chunk generation failed: %s", exc)
                # skip failed chunk rather than aborting stream
                continue

    return StreamingResponse(generate_chunks(), media_type="audio/wav")