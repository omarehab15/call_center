"""
Provider factory — switch STT / TTS / LLM / VAD implementations from `.env`
=============================================================================

This module is the single place that decides *which* concrete plugin backs
each stage of the voice pipeline (STT → LLM → TTS, plus VAD). Everything is
driven by environment variables, so switching providers never requires
touching `agent.py`.

Quick reference (see `.env.local.example` for full details + provider-specific keys)
--------------------------------------------------------------------------
    STT_PROVIDER = groq | openai | deepgram | google | whisper_local
    TTS_PROVIDER = gemini | groq | openai | elevenlabs | cartesia
    LLM_PROVIDER = groq | openai | anthropic | google | openai_compatible
    VAD_PROVIDER = silero   (only backend currently supported by LiveKit)

Backward compatibility
-----------------------
This project originally shipped with two hardcoded env vars: GROQ_LLM_MODEL
(the LLM model name) and TTS_VOICE_NAME (the Gemini voice). Existing
`.env.local` files keep working unchanged — those are still read as
fallbacks whenever the new, provider-agnostic LLM_MODEL / TTS_VOICE aren't
set.

Design notes
------------
* Every optional plugin (deepgram / elevenlabs / cartesia / anthropic) is
  imported lazily, inside the branch that needs it — exactly like the
  `ai_coustics` pattern already used in `agent.py`. If the package isn't
  installed we raise a clear `RuntimeError` telling the user which `pip`/`uv`
  extra to install, instead of crashing with a cryptic ImportError deep in
  someone else's code.
* Some STT backends are *streaming-native* (Deepgram, Google) and some are
  *request/response* (Groq Whisper, OpenAI Whisper, any local
  OpenAI-compatible whisper server). Non-streaming STT engines must be
  wrapped in `stt.StreamAdapter` + a VAD instance to work inside
  `AgentSession`. `build_stt()` handles that automatically per provider.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from livekit.agents import stt as stt_module
from livekit.agents import vad as vad_module
from livekit.plugins import openai, silero
from livekit.plugins import groq as groq_plugin

logger = logging.getLogger("agent.providers")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _missing_plugin(package: str, extra: str) -> RuntimeError:
    return RuntimeError(
        f"'{package}' is not installed. Install it with:\n"
        f'    uv add "livekit-agents[{extra}]"   # or: pip install {package}'
    )


# ══════════════════════════════════════════════════════════════════════════
# VAD
# ══════════════════════════════════════════════════════════════════════════

def build_vad() -> vad_module.VAD:
    """Build the VAD instance used for turn/speech detection.

    Only Silero is supported by LiveKit Agents today, but the provider
    switch is kept here (VAD_PROVIDER) so a future backend only needs a
    new branch in this function — nothing else in the codebase changes.
    """
    provider = _env("VAD_PROVIDER", "silero").lower()
    if provider not in {"", "silero"}:
        logger.warning(
            "VAD_PROVIDER=%r is not supported yet; falling back to silero.",
            provider,
        )

    return silero.VAD.load(
        min_silence_duration=float(_env("VAD_MIN_SILENCE_DURATION", "0.8")),
        prefix_padding_duration=float(_env("VAD_PREFIX_PADDING_DURATION", "0.3")),
        activation_threshold=float(_env("VAD_ACTIVATION_THRESHOLD", "0.6")),
        deactivation_threshold=float(_env("VAD_DEACTIVATION_THRESHOLD", "0.4")),
    )


# ══════════════════════════════════════════════════════════════════════════
# STT
# ══════════════════════════════════════════════════════════════════════════

# Providers whose plugin already streams natively — must NOT be wrapped in
# StreamAdapter (that would double-buffer and add latency).
_STREAMING_STT_PROVIDERS = {"deepgram", "google"}


def build_stt(vad: vad_module.VAD) -> Any:
    """Build the STT engine selected by STT_PROVIDER.

    `vad` is required because non-streaming engines (groq, openai,
    whisper_local) are wrapped in `stt.StreamAdapter`, which needs a VAD
    instance to know when to cut audio into utterances.
    """
    provider = _env("STT_PROVIDER", "groq").lower()
    language = _env("STT_LANGUAGE", "ar")
    model = _env("STT_MODEL", "whisper-large-v3")
    prompt = _env(
        "STT_PROMPT",
        "محادثة خدمة عملاء باللهجة السعودية النجدية. "
        "كلمات شائعة: وش، كيفك، إيش، زين، ابشر، تمام، والله، يعني، "
        "حياك، شلونك، عندي مشكلة، رقم الطلب، الحساب، خدمة العملاء.",
    )

    if provider == "groq":
        engine = groq_plugin.STT(
            model=model,
            language=language,
            prompt=prompt,
            api_key=_env("STT_API_KEY") or _env("GROQ_API_KEY"),
        )

    elif provider == "openai":
        engine = openai.STT(
            model=model or "gpt-4o-transcribe",
            language=language,
            api_key=_env("STT_API_KEY") or _env("OPENAI_API_KEY"),
        )

    elif provider == "whisper_local":
        # Any OpenAI-compatible whisper server (faster-whisper, whisper.cpp
        # server, vLLM, etc). Point STT_BASE_URL at it.
        engine = openai.STT(
            model=model,
            language=language,
            base_url=_env("STT_BASE_URL", "http://localhost:11435/v1"),
            api_key=_env("STT_API_KEY", "no-key-needed"),
        )

    elif provider == "deepgram":
        try:
            from livekit.plugins import deepgram
        except ImportError as exc:
            raise _missing_plugin("livekit-plugins-deepgram", "deepgram") from exc
        engine = deepgram.STT(
            model=model or "nova-3",
            language=language,
            api_key=_env("STT_API_KEY") or _env("DEEPGRAM_API_KEY"),
        )

    elif provider == "google":
        try:
            from livekit.plugins import google
        except ImportError as exc:
            raise _missing_plugin("livekit-plugins-google", "google") from exc
        engine = google.STT(
            languages=[_env("STT_GOOGLE_LOCALE", "ar-SA")],
        )

    else:
        raise ValueError(
            f"Unknown STT_PROVIDER={provider!r}. "
            "Supported: groq, openai, whisper_local, deepgram, google."
        )

    logger.info("STT provider=%s model=%s language=%s", provider, model, language)

    if provider in _STREAMING_STT_PROVIDERS:
        return engine

    # Wrap request/response engines so they behave like a streaming STT.
    return stt_module.StreamAdapter(stt=engine, vad=vad)


# ══════════════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════════════

def build_tts() -> Any:
    """Build the TTS engine selected by TTS_PROVIDER."""
    provider = _env("TTS_PROVIDER", "gemini").lower()
    # TTS_VOICE_NAME is the legacy var name this project shipped with —
    # keep honoring it so existing .env.local files don't need to change.
    voice = _env("TTS_VOICE") or _env("TTS_VOICE_NAME", "")

    if provider == "gemini":
        from gemini_tts import GeminiAIStudioTTS

        engine = GeminiAIStudioTTS(
            api_key=_env("GOOGLE_AI_API_KEY"),
            voice_name=voice or "Puck",
            language_code=_env("TTS_LANGUAGE_CODE", "ar-SA"),
            model=_env("TTS_MODEL", "gemini-2.5-flash-preview-tts"),
        )

    elif provider == "groq":
        engine = groq_plugin.TTS(
            model=_env("TTS_MODEL", "canopylabs/orpheus-v1-english"),
            voice=voice or "autumn",
            api_key=_env("TTS_API_KEY") or _env("GROQ_API_KEY"),
        )

    elif provider == "openai":
        engine = openai.TTS(
            model=_env("TTS_MODEL", "tts-1"),
            voice=voice or "alloy",
            api_key=_env("TTS_API_KEY") or _env("OPENAI_API_KEY"),
        )

    elif provider == "elevenlabs":
        try:
            from livekit.plugins import elevenlabs
        except ImportError as exc:
            raise _missing_plugin("livekit-plugins-elevenlabs", "elevenlabs") from exc
        engine = elevenlabs.TTS(
            voice_id=voice or _env("ELEVENLABS_VOICE_ID"),
            model=_env("TTS_MODEL", "eleven_multilingual_v2"),
            # ElevenLabs' own plugin default env var is ELEVEN_API_KEY (no "LABS") —
            # accept all three spellings so nothing surprises you later.
            api_key=(
                _env("TTS_API_KEY")
                or _env("ELEVENLABS_API_KEY")
                or _env("ELEVEN_API_KEY")
            ),
        )

    elif provider == "cartesia":
        try:
            from livekit.plugins import cartesia
        except ImportError as exc:
            raise _missing_plugin("livekit-plugins-cartesia", "cartesia") from exc
        engine = cartesia.TTS(
            voice=voice or _env("CARTESIA_VOICE_ID"),
            model=_env("TTS_MODEL", "sonic-2"),
            api_key=_env("TTS_API_KEY") or _env("CARTESIA_API_KEY"),
        )

    else:
        raise ValueError(
            f"Unknown TTS_PROVIDER={provider!r}. "
            "Supported: gemini, groq, openai, elevenlabs, cartesia."
        )

    logger.info("TTS provider=%s voice=%s", provider, voice or "(default)")
    return engine


# ══════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════

def build_llm() -> Any:
    """Build the LLM engine selected by LLM_PROVIDER."""
    provider = _env("LLM_PROVIDER", "groq").lower()
    # GROQ_LLM_MODEL is the legacy var name this project shipped with —
    # keep honoring it (for the groq provider) so existing .env.local files
    # don't need to change just to pick up this refactor.
    model = (
        _env("LLM_MODEL")
        or (_env("GROQ_LLM_MODEL") if provider == "groq" else "")
        or "openai/gpt-oss-20b"
    )

    if provider == "groq":
        engine = openai.LLM(
            base_url=_env("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            model=model,
            api_key=_env("LLM_API_KEY") or _env("GROQ_API_KEY", ""),
        )

    elif provider == "openai":
        engine = openai.LLM(
            model=model or "gpt-4o-mini",
            api_key=_env("LLM_API_KEY") or _env("OPENAI_API_KEY", ""),
        )

    elif provider == "openai_compatible":
        # Any self-hosted / third-party OpenAI-compatible chat endpoint
        # (vLLM, Ollama, Together, Fireworks, local llama.cpp server, ...).
        engine = openai.LLM(
            base_url=_env("LLM_BASE_URL", "http://localhost:11436/v1"),
            model=model,
            api_key=_env("LLM_API_KEY", "no-key-needed"),
        )

    elif provider == "anthropic":
        try:
            from livekit.plugins import anthropic
        except ImportError as exc:
            raise _missing_plugin("livekit-plugins-anthropic", "anthropic") from exc
        engine = anthropic.LLM(
            model=model or "claude-sonnet-4-5",
            api_key=_env("LLM_API_KEY") or _env("ANTHROPIC_API_KEY"),
        )

    elif provider == "google":
        try:
            from livekit.plugins import google
        except ImportError as exc:
            raise _missing_plugin("livekit-plugins-google", "google") from exc
        engine = google.LLM(
            model=model or "gemini-2.5-flash",
            api_key=_env("LLM_API_KEY") or _env("GOOGLE_AI_API_KEY"),
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            "Supported: groq, openai, openai_compatible, anthropic, google."
        )

    logger.info("LLM provider=%s model=%s", provider, model)
    return engine
