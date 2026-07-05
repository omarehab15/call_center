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
import time
from dataclasses import dataclass
from typing import Callable, Optional

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


@dataclass
class TimingEvent:
    """Reported to the on_timing callback after every Gemini TTS generation call."""
    success: bool
    elapsed_sec: float
    char_count: int
    audio_bytes: int = 0
    error: Optional[str] = None
    text_preview: str = ""


# Signature: (event: TimingEvent) -> None
TimingCallback = Callable[[TimingEvent], None]


# ── Lazily-created, reused genai.Client (avoids a fresh TLS handshake / ─────
# connection setup on every single TTS call, which was adding latency to
# every request regardless of text length).
_client_cache: dict[str, "genai.Client"] = {}


def _get_genai_client(api_key: str):
    import google.genai as genai

    client = _client_cache.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key)
        _client_cache[api_key] = client
    return client


# Google's own docs (Limitations section) document two known failure modes
# for Gemini TTS that are not really "errors" in our code:
#   1. "Occasional text token returns" — the model occasionally returns text
#      instead of audio for a tiny % of requests, even for a perfectly normal
#      transcript. Google's recommended fix is automated retry.
#   2. "Prompt classifier false rejections" — vague prompts can fail to
#      trigger the speech classifier. Google's recommended fix is a clear
#      preamble instructing the model to synthesize speech, with the actual
#      transcript clearly labeled.
# Both fixes are applied below.
_TTS_PREAMBLE = (
    "TTS the following transcript exactly as written, output audio only, "
    "do not reply with text:\n"
    "TRANSCRIPT: "
)
_MAX_TTS_ATTEMPTS = 3

# How many sentence-chunks we'll synthesize concurrently (pipelined) for one
# agent turn. Keeps a turn with many sentences from firing 10+ simultaneous
# API calls and tripping rate limits, while still overlapping latency.
_MAX_CONCURRENT_CHUNKS = 3

# Arabic + Latin sentence-ending punctuation. We split on these so each
# chunk is a natural prosodic unit (a full sentence/clause), not an arbitrary
# character cut — cutting mid-sentence would produce audio with a dead stop
# and no clause-final intonation.
_SENTENCE_END_CHARS = ".!?؟!۔"


def _split_into_chunks(text: str, max_chunk_chars: int = 120) -> list[str]:
    """Split text into sentence-sized chunks for pipelined TTS synthesis.

    Splits on sentence-ending punctuation (Arabic and Latin). If a single
    "sentence" is still longer than max_chunk_chars (no punctuation for a
    while), it's left as one chunk anyway — we never break mid-word/mid-clause,
    since that produces unnatural audio.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in _SENTENCE_END_CHARS:
            stripped = current.strip()
            if stripped:
                chunks.append(stripped)
            current = ""
    if current.strip():
        chunks.append(current.strip())

    # Merge tiny trailing fragments (e.g. a lone "..") into the previous chunk
    # so we don't fire an API call for 1-2 leftover characters.
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < 8:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)

    return merged or [text]


# ── Shared Gemini call + audio extraction (used by both stream classes) ───────
async def _call_gemini_tts(
    opts: _TTSOptions,
    text: str,
    on_timing: Optional[TimingCallback] = None,
) -> bytes | None:
    """Call the Gemini AI Studio TTS API and return raw PCM bytes, or None.

    If on_timing is provided, it is called once per attempt with a TimingEvent
    describing whether generation succeeded, how long it took, and how much
    audio came back. Retries automatically (per Google's documented guidance)
    on the known "model returned text instead of audio" / empty-audio failure
    mode, which happens randomly on a small percentage of otherwise normal
    requests.
    """
    try:
        import google.genai as genai
        import google.genai.types as genai_types
    except ImportError as e:
        if on_timing:
            on_timing(TimingEvent(
                success=False, elapsed_sec=0.0, char_count=len(text),
                error=f"google-genai not installed: {e}", text_preview=text[:60],
            ))
        raise RuntimeError(
            "google-genai package is required. Run: pip install google-genai"
        )

    client = _get_genai_client(opts.api_key)
    prompted_text = _TTS_PREAMBLE + text

    config = genai_types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                    voice_name=opts.voice_name,
                )
            ),
            language_code=opts.language_code,
        ),
    )

    last_error: str | None = None

    for attempt in range(1, _MAX_TTS_ATTEMPTS + 1):
        start = time.monotonic()
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=opts.model,
                    contents=prompted_text,
                    config=config,
                ),
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "GeminiTTS API error (attempt %d/%d): %s", attempt, _MAX_TTS_ATTEMPTS, e
            )
            if on_timing:
                on_timing(TimingEvent(
                    success=False, elapsed_sec=time.monotonic() - start,
                    char_count=len(text), error=last_error, text_preview=text[:60],
                ))
            continue  # retry — covers the documented "occasional text token" 400/500s

        audio_bytes = _extract_audio(response)
        if not audio_bytes:
            last_error = "empty audio response (model returned text instead of audio)"
            logger.warning(
                "GeminiTTS: %s (attempt %d/%d) for text: %.60s",
                last_error, attempt, _MAX_TTS_ATTEMPTS, text,
            )
            if on_timing:
                on_timing(TimingEvent(
                    success=False, elapsed_sec=time.monotonic() - start,
                    char_count=len(text), error=last_error, text_preview=text[:60],
                ))
            continue  # retry

        # Success
        if on_timing:
            on_timing(TimingEvent(
                success=True, elapsed_sec=time.monotonic() - start,
                char_count=len(text), audio_bytes=len(audio_bytes),
                text_preview=text[:60],
            ))
        return audio_bytes

    # All attempts exhausted — surface the last error to the caller.
    logger.error(
        "GeminiTTS: all %d attempts failed for text: %.60s | last_error=%s",
        _MAX_TTS_ATTEMPTS, text, last_error,
    )
    return None


def _extract_audio(response) -> bytes | None:
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
            # We implement our own stream() below (it buffers text and calls the
            # Gemini API per flushed segment), so we advertise streaming=True.
            # Leaving this False makes the LiveKit AgentSession silently wrap us
            # in a StreamAdapter / call synthesize() per-sentence instead of
            # using our stream() implementation directly.
            capabilities=tts.TTSCapabilities(streaming=True),
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

        # Settable after construction — agent.py wires this once the
        # CallLogger (which lives on the Assistant) actually exists, since
        # tts_instance is created before Assistant() in agent.py.
        self.on_timing: Optional[TimingCallback] = None

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
# ChunkedStream  (non-streaming synthesis — one full text in, one full audio out)
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

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        chunks = _split_into_chunks(self._input_text)

        if len(chunks) <= 1:
            audio_bytes = await _call_gemini_tts(
                self._opts, self._input_text, on_timing=self._tts.on_timing
            )
        else:
            # Pipelined: fire off all chunk synthesis calls concurrently
            # (bounded), then concatenate the resulting PCM in original
            # sentence order. This still waits for everything before the
            # single output_emitter.push() below (ChunkedStream has no
            # incremental output), but it cuts the *total* wait roughly from
            # sum(chunk latencies) down to ~max(chunk latencies), since the
            # chunk API calls overlap instead of running one after another.
            sem = asyncio.Semaphore(_MAX_CONCURRENT_CHUNKS)

            async def _synthesize_chunk(chunk_text: str) -> bytes | None:
                async with sem:
                    return await _call_gemini_tts(
                        self._opts, chunk_text, on_timing=self._tts.on_timing
                    )

            results = await asyncio.gather(*(_synthesize_chunk(c) for c in chunks))
            parts = [r for r in results if r]
            audio_bytes = b"".join(parts) if parts else None

        if not audio_bytes:
            logger.warning(
                "GeminiTTS: empty audio response for text: %.60s", self._input_text
            )
            return

        # Hand raw PCM bytes to the AudioEmitter (current livekit-agents API).
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        output_emitter.push(audio_bytes)
        output_emitter.flush()


# ══════════════════════════════════════════════════════════════════════════════
# SynthesizeStream  (streaming interface — buffers text per segment, synthesizes
# the whole segment on flush, since the Gemini AI Studio TTS endpoint itself is
# not incremental/streaming)
# ══════════════════════════════════════════════════════════════════════════════
class SynthesizeStream(tts.SynthesizeStream):
    """
    LiveKit agents call stream() for incremental text pushes.
    We buffer text per segment and synthesize once on each flush.
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
        self._text_buf: list[str] = []

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                await self._flush_buffer(output_emitter)
            else:
                self._text_buf.append(data)

        # Catch any trailing text that never received an explicit flush.
        await self._flush_buffer(output_emitter)

    async def _flush_buffer(self, output_emitter: tts.AudioEmitter) -> None:
        text = "".join(self._text_buf).strip()
        self._text_buf.clear()
        if not text:
            return

        chunks = _split_into_chunks(text)
        if len(chunks) <= 1:
            # Single chunk — no pipelining benefit, just synthesize directly.
            audio_bytes = await _call_gemini_tts(
                self._opts, text, on_timing=self._tts.on_timing
            )
            if not audio_bytes:
                logger.warning("GeminiTTS: empty audio response for text: %.60s", text)
                return
            output_emitter.start_segment(segment_id=utils.shortuuid())
            output_emitter.push(audio_bytes)
            output_emitter.end_segment()
            return

        # Pipelined multi-chunk path: kick off TTS calls for several chunks
        # concurrently (bounded), but push audio to the emitter strictly in
        # original sentence order, so playback never sounds shuffled — only
        # the *latency to start* improves, since later chunks' API calls are
        # already in flight by the time earlier chunks finish.
        sem = asyncio.Semaphore(_MAX_CONCURRENT_CHUNKS)

        async def _synthesize_chunk(chunk_text: str) -> bytes | None:
            async with sem:
                return await _call_gemini_tts(
                    self._opts, chunk_text, on_timing=self._tts.on_timing
                )

        tasks = [asyncio.create_task(_synthesize_chunk(c)) for c in chunks]

        for chunk_text, task in zip(chunks, tasks):
            audio_bytes = await task
            if not audio_bytes:
                logger.warning(
                    "GeminiTTS: empty audio response for chunk: %.60s", chunk_text
                )
                continue
            output_emitter.start_segment(segment_id=utils.shortuuid())
            output_emitter.push(audio_bytes)
            output_emitter.end_segment()