"""
gemini_tts.py
─────────────
Custom LiveKit TTS plugin for Gemini 2.5 Flash TTS via Google AI Studio.

Uses the google-genai SDK (GOOGLE_AI_API_KEY) instead of Google Cloud
service-account credentials, so it works with a plain AI Studio key.

Usage:
    from gemini_tts import GeminiAIStudioTTS

    tts = GeminiAIStudioTTS(
        api_key=os.getenv("GOOGLE_AI_API_KEY"),
        voice_name="Puck",           # Aoede, Puck, Charon, Kore, Fenrir …
        language_code="ar-SA",
        model="gemini-2.5-flash-preview-tts",
    )
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from typing import AsyncGenerator

from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("gemini_tts")

# ── Audio constants ────────────────────────────────────────────────────────────
# Gemini TTS returns 16-bit PCM at 24 kHz mono by default.
SAMPLE_RATE = 24_000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2  # int16


# ── Options dataclass ──────────────────────────────────────────────────────────
@dataclass
class _TTSOptions:
    model: str
    voice_name: str
    language_code: str
    api_key: str


# ══════════════════════════════════════════════════════════════════════════════
# Main TTS class
# ══════════════════════════════════════════════════════════════════════════════
class GeminiAIStudioTTS(tts.TTS):
    """
    LiveKit TTS plugin backed by Gemini 2.5 Flash TTS (Google AI Studio).

    Authentication: plain GOOGLE_AI_API_KEY — no service account needed.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_name: str = "Puck",
        language_code: str = "ar-SA",
        model: str = "gemini-2.5-flash-preview-tts",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        resolved_key = api_key or os.getenv("GOOGLE_AI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GeminiAIStudioTTS: no API key found. "
                "Set GOOGLE_AI_API_KEY or pass api_key=..."
            )

        self._opts = _TTSOptions(
            model=model,
            voice_name=voice_name,
            language_code=language_code,
            api_key=resolved_key,
        )

    # ── Public factory (matches LiveKit TTS interface) ─────────────────────────
    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self, input_text=text, opts=self._opts, conn_options=conn_options
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SynthesizeStream":
        return SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)


# ══════════════════════════════════════════════════════════════════════════════
# ChunkedStream  (non-streaming synthesis)
# ══════════════════════════════════════════════════════════════════════════════
class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: GeminiAIStudioTTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self) -> None:
        try:
            import google.genai as genai
            import google.genai.types as genai_types
        except ImportError:
            raise RuntimeError(
                "google-genai package is required. "
                "Run: pip install google-genai"
            )

        client = genai.Client(api_key=self._opts.api_key)

        config = genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=self._opts.voice_name,
                    )
                ),
                language_code=self._opts.language_code,
            ),
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self._opts.model,
                    contents=self._input_text,
                    config=config,
                ),
            )
        except Exception as e:
            logger.error("GeminiTTS API error: %s", e)
            raise

        # Extract audio bytes from response
        audio_bytes = self._extract_audio(response)
        if not audio_bytes:
            logger.warning("GeminiTTS: empty audio response for text: %.60s", self._input_text)
            return

        # Convert raw PCM bytes → numpy → LiveKit AudioFrame chunks
        import numpy as np

        pcm = np.frombuffer(audio_bytes, dtype=np.int16)

        # Split into ~100ms chunks (2400 samples at 24kHz)
        chunk_samples = SAMPLE_RATE // 10
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()

        for i in range(0, len(pcm), chunk_samples):
            chunk = pcm[i : i + chunk_samples]
            frame = tts.SynthesizedAudio(
                request_id=request_id,
                segment_id=segment_id,
                frame=self._make_audio_frame(chunk),
            )
            self._event_ch.send_nowait(frame)

    def _extract_audio(self, response) -> bytes | None:
        """Pull raw PCM bytes out of the Gemini response."""
        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    # data can be bytes or base64 string
                    if isinstance(data, (bytes, bytearray)):
                        return bytes(data)
                    if isinstance(data, str):
                        return base64.b64decode(data)
        except (AttributeError, IndexError, KeyError) as e:
            logger.error("GeminiTTS: failed to parse response: %s | response=%s", e, response)
        return None

    def _make_audio_frame(self, pcm: "np.ndarray"):
        from livekit import rtc
        frame = rtc.AudioFrame(
            data=pcm.tobytes(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(pcm),
        )
        return frame


# ══════════════════════════════════════════════════════════════════════════════
# SynthesizeStream  (streaming interface — buffers internally)
# ══════════════════════════════════════════════════════════════════════════════
class SynthesizeStream(tts.SynthesizeStream):
    """
    LiveKit agents call stream() for incremental text pushes.
    We buffer all text then synthesize once on flush.
    """

    def __init__(
        self,
        *,
        tts: GeminiAIStudioTTS,
        opts: _TTSOptions,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._conn_options = conn_options
        self._text_buf: list[str] = []

    async def _run(self) -> None:
        async for data in self._input_ch:
            if isinstance(data, self.FlushSentinel):
                text = "".join(self._text_buf).strip()
                self._text_buf.clear()
                if text:
                    await self._synthesize(text)
            else:
                self._text_buf.append(data)

    async def _synthesize(self, text: str) -> None:
        chunked = ChunkedStream(
            tts=self._tts,
            input_text=text,
            opts=self._opts,
            conn_options=self._conn_options,
        )
        async with chunked:
            async for ev in chunked:
                self._event_ch.send_nowait(ev)