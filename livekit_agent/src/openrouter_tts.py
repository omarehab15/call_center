"""
openrouter_tts.py
─────────────────
Custom LiveKit TTS plugin for Google Gemini TTS via OpenRouter's
/api/v1/audio/speech endpoint.

Why not just use livekit-plugins-openai pointed at OpenRouter?
----------------------------------------------------------------
OpenRouter's TTS endpoint always returns a *raw audio byte stream*
(mp3/pcm/wav) — never JSON, never Server-Sent-Events — no matter what
`stream_format` you ask for. livekit-plugins-openai's TTS class picks
its internal stream implementation based on the model name: any model
other than "tts-1"/"tts-1-hd" (which includes our Gemini model) gets
routed through SSEChunkedStream. That class requests stream_format=
"sse" and then parses "data: {...}" JSON lines out of the response
body. Since OpenRouter never actually sends that format for this
model, zero lines ever match, and the plugin silently pushes zero
audio frames — which is exactly the

    "no audio frames were pushed for text: ..."

error. This plugin sidesteps that routing bug entirely: it talks to
OpenRouter directly and treats the response as what it actually is —
raw PCM bytes.

Usage:
    from openrouter_tts import OpenRouterGeminiTTS

    tts = OpenRouterGeminiTTS(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="google/gemini-3.1-flash-tts-preview",
        voice="Kore",
        speed=1.0,
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("openrouter_tts")

# Gemini TTS returns 16-bit PCM at 24 kHz mono natively.
SAMPLE_RATE = 24_000
NUM_CHANNELS = 1

_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"
_MAX_ATTEMPTS = 3

# How many sentence-chunks we'll synthesize concurrently (pipelined) for
# one agent turn. Overlaps network latency across sentences instead of
# paying it once per sentence sequentially.
_MAX_CONCURRENT_CHUNKS = 3

# Arabic + Latin sentence-ending punctuation — split on these so each
# request is a natural prosodic unit, never a mid-sentence cut.
_SENTENCE_END_CHARS = ".!?؟!۔"


@dataclass
class _TTSOptions:
    model: str
    voice: str
    speed: float
    api_key: str


def _split_into_chunks(text: str, max_chunk_chars: int = 120) -> list[str]:
    """Split text into sentence-sized chunks for pipelined TTS synthesis."""
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

    # Merge tiny trailing fragments into the previous chunk so we don't
    # fire a whole API call for 1-2 leftover characters.
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < 8:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)

    return merged or [text]


# Reused, lazily-created httpx client (keepalive avoids a fresh TLS
# handshake on every single TTS call).
_client_cache: dict[str, httpx.AsyncClient] = {}


def _get_http_client(api_key: str) -> httpx.AsyncClient:
    client = _client_cache.get(api_key)
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=50),
        )
        _client_cache[api_key] = client
    return client


async def _call_openrouter_tts(opts: _TTSOptions, text: str) -> Optional[bytes]:
    """POST to OpenRouter's /audio/speech endpoint and return raw PCM bytes, or None.

    Reads the response as raw bytes (never assumes JSON/SSE), and retries
    on transient network errors or an unexpectedly empty body.
    """
    client = _get_http_client(opts.api_key)
    headers = {
        "Authorization": f"Bearer {opts.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": opts.model,
        "input": text,
        "voice": opts.voice,
        "response_format": "pcm",
        "speed": opts.speed,
    }

    last_error: str | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        start = time.monotonic()
        try:
            async with client.stream("POST", _ENDPOINT, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    err_body = await resp.aread()
                    last_error = f"HTTP {resp.status_code}: {err_body[:300]!r}"
                    logger.warning(
                        "OpenRouter TTS error (attempt %d/%d): %s",
                        attempt, _MAX_ATTEMPTS, last_error,
                    )
                else:
                    chunks = [c async for c in resp.aiter_bytes()]
                    audio_bytes = b"".join(chunks)
                    if audio_bytes:
                        return audio_bytes
                    last_error = "empty audio response"
                    logger.warning(
                        "OpenRouter TTS: empty audio (attempt %d/%d) for text: %.60s",
                        attempt, _MAX_ATTEMPTS, text,
                    )
        except httpx.TimeoutException as e:
            last_error = f"timeout: {e}"
            logger.warning(
                "OpenRouter TTS timeout (attempt %d/%d): %s", attempt, _MAX_ATTEMPTS, e
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "OpenRouter TTS error (attempt %d/%d): %s", attempt, _MAX_ATTEMPTS, e
            )

        if attempt < _MAX_ATTEMPTS:
            await asyncio.sleep(0.3 * attempt)

    logger.error(
        "OpenRouter TTS: all %d attempts failed for text: %.60s (%s)",
        _MAX_ATTEMPTS, text, last_error,
    )
    return None


async def _synthesize_pipelined(text: str, opts: _TTSOptions) -> Optional[bytes]:
    """Synthesize (possibly multi-sentence) text, pipelining chunk requests."""
    chunks = _split_into_chunks(text)
    if len(chunks) <= 1:
        return await _call_openrouter_tts(opts, text)

    sem = asyncio.Semaphore(_MAX_CONCURRENT_CHUNKS)

    async def _synth(chunk_text: str) -> Optional[bytes]:
        async with sem:
            return await _call_openrouter_tts(opts, chunk_text)

    results = await asyncio.gather(*(_synth(c) for c in chunks))
    parts = [r for r in results if r]
    return b"".join(parts) if parts else None


# ══════════════════════════════════════════════════════════════════════════
# Main TTS class
# ══════════════════════════════════════════════════════════════════════════
class OpenRouterGeminiTTS(tts.TTS):
    """LiveKit TTS plugin: Google Gemini TTS via OpenRouter's /audio/speech endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "google/gemini-3.1-flash-tts-preview",
        voice: str = "Kore",
        speed: float = 1.0,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenRouterGeminiTTS: no API key found. "
                "Set OPENROUTER_API_KEY or pass api_key=..."
            )

        self._opts = _TTSOptions(model=model, voice=voice, speed=speed, api_key=resolved_key)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self, input_text=text, opts=self._opts, conn_options=conn_options
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)

    async def aclose(self) -> None:
        client = _client_cache.pop(self._opts.api_key, None)
        if client is not None:
            await client.aclose()


# ══════════════════════════════════════════════════════════════════════════
# ChunkedStream — one full text in, one full audio out (pipelined internally)
# ══════════════════════════════════════════════════════════════════════════
class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: OpenRouterGeminiTTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        audio_bytes = await _synthesize_pipelined(self._input_text, self._opts)

        if not audio_bytes:
            logger.warning(
                "OpenRouter TTS: empty audio response for text: %.60s", self._input_text
            )
            return

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        output_emitter.push(audio_bytes)
        output_emitter.flush()


# ══════════════════════════════════════════════════════════════════════════
# SynthesizeStream — buffers text per segment (per LiveKit flush), synthesizes
# each segment's sentences pipelined, and pushes audio in original order.
# ══════════════════════════════════════════════════════════════════════════
class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: OpenRouterGeminiTTS,
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
            audio_bytes = await _call_openrouter_tts(self._opts, text)
            if not audio_bytes:
                logger.warning("OpenRouter TTS: empty audio response for text: %.60s", text)
                return
            output_emitter.start_segment(segment_id=utils.shortuuid())
            output_emitter.push(audio_bytes)
            output_emitter.end_segment()
            return

        # Pipelined multi-chunk path: fire concurrent requests (bounded), but
        # push audio strictly in original sentence order.
        sem = asyncio.Semaphore(_MAX_CONCURRENT_CHUNKS)

        async def _synth(chunk_text: str) -> Optional[bytes]:
            async with sem:
                return await _call_openrouter_tts(self._opts, chunk_text)

        tasks = [asyncio.create_task(_synth(c)) for c in chunks]

        for chunk_text, task in zip(chunks, tasks):
            audio_bytes = await task
            if not audio_bytes:
                logger.warning(
                    "OpenRouter TTS: empty audio response for chunk: %.60s", chunk_text
                )
                continue
            output_emitter.start_segment(segment_id=utils.shortuuid())
            output_emitter.push(audio_bytes)
            output_emitter.end_segment()
