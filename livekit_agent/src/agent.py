import asyncio
import logging
import os
from collections import deque
from typing import Any, Optional

import numpy as np
import pyloudnorm as pyln
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    room_io,
)
from livekit import rtc as lk_audio
from livekit.agents.metrics import TTSMetrics
from livekit.agents.voice.recorder_io import RecorderIO
from livekit.plugins import noise_cancellation, openai, silero
from livekit.plugins import cartesia as cartesia_plugin
from livekit.plugins.cartesia.models import TTSDefaultVoiceId as CARTESIA_DEFAULT_VOICE_ID
from livekit.plugins import deepgram as deepgram_plugin
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ai_coustics requires LiveKit Cloud — import only if available
try:
    from livekit.plugins import ai_coustics
    _AI_COUSTICS_AVAILABLE = True
except ImportError:
    _AI_COUSTICS_AVAILABLE = False


from call_logger import CallLogger
from intent_classifier import needs_rag
from rag import RagRetriever, build_rag_from_env

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# ══════════════════════════════════════════════════════════════════════════════
# Audio loudness normalizer
# ══════════════════════════════════════════════════════════════════════════════

class LoudnessNormalizer:
    """
    EBU R128 loudness normalizer for LiveKit AudioFrames.

    Buffers incoming frames until it has enough samples to measure
    integrated loudness (≥ 400 ms recommended by the standard), then
    normalizes the buffer to TARGET_LUFS and yields normalized frames.

    Attributes
    ----------
    TARGET_LUFS   : target integrated loudness  (-23 LUFS is broadcast standard;
                    -18 LUFS is more suitable for call-centre speech where
                    headroom is less important than clarity)
    MAX_GAIN_DB   : safety ceiling — prevents over-amplification of very
                    quiet/silence-only buffers (e.g. hold music, dead air)
    BUFFER_FRAMES : number of LiveKit frames to accumulate before processing
                    (each frame is typically 10 ms → 40 frames ≈ 400 ms)
    """

    TARGET_LUFS   = float(os.getenv("LOUDNORM_TARGET_LUFS",  "-18"))
    MAX_GAIN_DB   = float(os.getenv("LOUDNORM_MAX_GAIN_DB",   "15"))
    BUFFER_FRAMES = int(os.getenv("LOUDNORM_BUFFER_FRAMES",   "40"))

    def __init__(self, sample_rate: int = 16_000, num_channels: int = 1) -> None:
        self.sample_rate  = sample_rate
        self.num_channels = num_channels
        self._meter       = pyln.Meter(sample_rate)
        self._buffer: deque[lk_audio.AudioFrame] = deque()

    # ------------------------------------------------------------------
    def push(self, frame: lk_audio.AudioFrame) -> list[lk_audio.AudioFrame]:
        """
        Accept one LiveKit AudioFrame and return a (possibly empty) list of
        normalized frames once the internal buffer is full.
        """
        self._buffer.append(frame)
        if len(self._buffer) < self.BUFFER_FRAMES:
            return []

        frames     = list(self._buffer)
        self._buffer.clear()

        # Concatenate raw int16 samples → float64 in [-1, 1]
        raw    = b"".join(f.data for f in frames)
        pcm_i  = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        # pyloudnorm expects shape (samples,) for mono or (samples, channels) for stereo
        if self.num_channels > 1:
            pcm_i = pcm_i.reshape(-1, self.num_channels)

        try:
            loudness = self._meter.integrated_loudness(pcm_i)
        except Exception:
            # If measurement fails (e.g. silent buffer) pass frames through unchanged
            logger.debug("LoudnessNormalizer: measurement failed, passing through")
            return frames

        if not np.isfinite(loudness):
            return frames

        gain_db    = self.TARGET_LUFS - loudness
        gain_db    = min(gain_db, self.MAX_GAIN_DB)   # safety ceiling
        gain_linear = 10 ** (gain_db / 20.0)

        pcm_norm   = np.clip(pcm_i * gain_linear, -1.0, 1.0)
        pcm_out    = (pcm_norm.flatten() * 32768.0).astype(np.int16)

        logger.debug(
            "LoudnessNormalizer: measured=%.1f LUFS  gain=%.1f dB  frames=%d",
            loudness, gain_db, len(frames),
        )

        # Split the normalized array back into per-frame chunks
        samples_per_frame = len(frames[0].data) // 2  # int16 → 2 bytes each
        out_frames: list[lk_audio.AudioFrame] = []
        for i, orig in enumerate(frames):
            start = i * samples_per_frame
            end   = start + samples_per_frame
            chunk = pcm_out[start:end].tobytes()
            out_frames.append(
                lk_audio.AudioFrame(
                    data=chunk,
                    sample_rate=orig.sample_rate,
                    num_channels=orig.num_channels,
                    samples_per_channel=orig.samples_per_channel,
                )
            )
        return out_frames

    def flush(self) -> list[lk_audio.AudioFrame]:
        """Drain any remaining buffered frames without normalization."""
        frames = list(self._buffer)
        self._buffer.clear()
        return frames


# ══════════════════════════════════════════════════════════════════════════════
# Local call recorder
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Local call recorder
# ══════════════════════════════════════════════════════════════════════════════

class LocalCallRecorder:
    """
    Records every call to a local .ogg file (Opus stereo) using the
    built-in livekit-agents RecorderIO.

    Output format — stereo Opus in an OGG container:
        Left  channel → caller audio  (session.input.audio)
        Right channel → agent audio   (session.output.audio / TTS)

    File location:
        <CALL_RECORDINGS_DIR>/<room_name>_<timestamp>.ogg

    The directory is configured via the CALL_RECORDINGS_DIR env var
    (default: /app/call_recordings).

    ── How it avoids the race condition ──────────────────────────────────────
    session.start() calls RoomIO.start() which sets session.output.audio
    synchronously, then immediately fires _on_audio_output_changed().
    session.start() then returns and spawns on_enter() as a background task.

    If we wrap output.audio AFTER session.start() returns, there is a race:
    on_enter() may grab a reference to the old (unwrapped) output.audio before
    our wrap runs, so TTS frames never pass through RecorderAudioOutput.

    The fix: we patch session._on_audio_output_changed BEFORE session.start().
    When RoomIO sets session.output.audio, our hook fires synchronously
    (no await) in the same call stack, wraps the stream with RecorderIO,
    and starts the file writer — all before the event loop has a chance to
    run on_enter().  This guarantees every TTS frame is captured.

    The input side (caller audio) is handled the same way via
    _on_audio_input_changed.
    """

    def __init__(self, session: AgentSession, room_name: str) -> None:
        self._session    = session
        self._room_name  = room_name
        self._recorder: Optional[RecorderIO]  = None
        self._output_path: Optional[str]      = None
        self._enabled                         = (
            os.getenv("CALL_RECORDING_ENABLED", "true").strip().lower()
            in {"true", "1", "yes", "on"}
        )
        self._installed  = False   # guard: install hooks only once

    # ------------------------------------------------------------------
    def install(self) -> None:
        """
        Patch session._on_audio_input_changed and _on_audio_output_changed
        BEFORE session.start() is called.

        When RoomIO wires up input.audio / output.audio during session.start(),
        both callbacks fire synchronously.  We use them to wrap the streams
        with RecorderIO at exactly the right moment — before on_enter() can
        grab a stale reference.
        """
        if not self._enabled or self._installed:
            return

        self._installed = True
        _orig_in  = self._session._on_audio_input_changed
        _orig_out = self._session._on_audio_output_changed

        def _on_audio_input_changed() -> None:
            _orig_in()
            self._try_attach()

        def _on_audio_output_changed() -> None:
            _orig_out()
            self._try_attach()

        # Patch directly — AgentSession stores these as bound methods via
        # AgentInput/AgentOutput callbacks, so we need to update the
        # underlying _audio_changed references on the io containers.
        self._session._input._audio_changed  = _on_audio_input_changed   # type: ignore[attr-defined]
        self._session._output._audio_changed = _on_audio_output_changed  # type: ignore[attr-defined]
        logger.debug("LocalCallRecorder: hooks installed")

    def _try_attach(self) -> None:
        """Called synchronously when either audio stream is set. Attaches RecorderIO
        as soon as BOTH input and output are non-None."""
        if self._recorder is not None:
            return  # already attached

        audio_input  = getattr(self._session.input,  "audio", None)
        audio_output = getattr(self._session.output, "audio", None)

        if audio_input is None or audio_output is None:
            return  # wait until both are ready

        recordings_dir = os.getenv(
            "CALL_RECORDINGS_DIR",
            os.path.join(os.path.dirname(__file__), "..", "call_recordings"),
        )
        os.makedirs(recordings_dir, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in self._room_name
        ).strip("_") or "call"
        self._output_path = os.path.join(recordings_dir, f"{safe_name}_{timestamp}.ogg")

        try:
            self._recorder = RecorderIO(agent_session=self._session)

            # Wrap both streams synchronously — no await needed here.
            # RecorderAudioInput wraps __anext__ transparently.
            # RecorderAudioOutput wraps capture_frame/flush and listens to
            # playback_finished events from _ParticipantAudioOutput to flush
            # each segment to disk.
            self._session.input.audio  = self._recorder.record_input(audio_input)
            self._session.output.audio = self._recorder.record_output(audio_output)

            # Start the file writer (needs an event loop — schedule as a task).
            asyncio.ensure_future(self._start_writer())
            logger.info("LocalCallRecorder: streams wrapped → %s", self._output_path)

        except Exception:
            logger.exception("LocalCallRecorder: failed to attach")
            self._recorder     = None
            self._output_path  = None

    async def _start_writer(self) -> None:
        """Start the background encoder thread that writes frames to disk."""
        if self._recorder is None or self._output_path is None:
            return
        try:
            await self._recorder.start(output_path=self._output_path)
            logger.info("📹 Local recording started → %s  (L=caller  R=agent)", self._output_path)
        except Exception:
            logger.exception("LocalCallRecorder: failed to start writer")

    async def stop(self) -> None:
        """Flush and finalise the .ogg file."""
        if self._recorder is None:
            return
        try:
            await self._recorder.aclose()
            logger.info("📹 Local recording saved  → %s", self._output_path)
        except Exception:
            logger.exception(
                "LocalCallRecorder: error while stopping (path=%s)", self._output_path
            )
        finally:
            self._recorder = None

    @property
    def output_path(self) -> Optional[str]:
        return self._output_path


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════

class Assistant(Agent):
    def __init__(
        self,
        call_id: str = "local_call",
        rag_retriever: Optional[RagRetriever] = None,
    ) -> None:
        self.call_id       = call_id
        self.rag_retriever = rag_retriever

        logs_dir = os.getenv(
            "CALL_LOGS",
            os.path.join(os.path.dirname(__file__), "..", "call_logs"),
        )
        self.call_logger     = CallLogger(call_id, logs_dir=logs_dir)
        self._summary_written = False
        self._last_query: str = ""

        self.base_instructions = """أنت مساعدة ذكاء اصطناعي صوتي اسمك لينا لمركز اتصالات. يتفاعل المستخدم معك عبر الصوت.

        القاعدة الأولى — حفظ المعلومات فوراً:
        في كل مرة يذكر فيها المستخدم اسمه أو مشكلته أو أي معلومة مهمة، استدعِ أداة add_note فوراً قبل أي رد آخر.
        أمثلة على متى تستخدم add_note:
        - قال المستخدم اسمه → استدعِ add_note باسمه
        - ذكر مشكلة أو شكوى → استدعِ add_note بتفاصيل المشكلة
        - أعطى رقم طلب أو حساب → استدعِ add_note بالرقم
        - ذكر أي معلومة تحتاج إليها لاحقاً → استدعِ add_note بها

        قواعد المحادثة:
        أجب دائماً بلهجة سعودية نجدية بشكل مباشر وواضح.
        قصّر إجاباتك قدر الإمكان — جملة أو جملتين كحد أقصى في معظم الأحيان.
        إذا وصلت لك معلومات من قاعدة المعرفة في سياق المحادثة، استخدمها فقط إذا كانت مرتبطة بسؤال العميل ولا تخترع تفاصيل غير موجودة فيها.
        لا تستخدم تنسيقات أو رموز أو مقدمات فارغة مثل بالتأكيد أو حسناً.
        كن ودوداً ومباشراً."""

        super().__init__(instructions=self.base_instructions)
        self.notes: list[str] = []

    async def on_enter(self) -> None:
        import time as _time
        self.call_logger.log_system_event("Call started - Agent initialized")
        _t = self.call_logger.start_timer()
        await self.session.generate_reply(
            user_input="...",
            instructions=(
                "ابدأ المكالمة بتحية الشخص المتصل بلهجة سعودية ودية قول التالى "
                "[هلا بيك معك لينا ممكن اعرف اسمك الكريم] "
            ),
        )
        self.call_logger.log_timing("LLM", "Opening greeting generated", _time.monotonic() - _t)

    def log_agent_message(self, message: str) -> None:
        self.call_logger.log_agent_message(message)

    def finalize_call(self) -> None:
        """Write the call summary once, when the session/job is actually closing."""
        if self._summary_written:
            return
        self._summary_written = True
        self.call_logger.log_call_summary(self.notes)

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        import time as _time
        _t_turn_start = _time.monotonic()
        query = _message_text(new_message)
        if query:
            self.call_logger.log_user_message(query)
            self.call_logger.log_timing("STT", "User turn transcription delivered", _time.monotonic() - _t_turn_start,
                                        extra=f"chars={len(query)}")

        if self.rag_retriever is None or not query:
            return

        # ── Intent classifier timing ───────────────────────────────────────────
        import time as _time
        _t_intent = self.call_logger.start_timer()
        intent_result = await needs_rag(query, previous_query=self._last_query)
        self.call_logger.log_timing("INTENT", "Intent classifier", _time.monotonic() - _t_intent,
                                    extra=f"decision={'YES→RAG' if intent_result else 'NO→skip'}")

        if not intent_result:
            self.call_logger.log_system_event("RAG skipped — intent classifier: no knowledge lookup needed")
            self._last_query = query
            return

        self._last_query = query
        try:
            self.call_logger.log_system_event(f"RAG lookup started — query: {query[:120]}")
            # ── RAG retrieval timing ───────────────────────────────────────────
            _t_rag = self.call_logger.start_timer()
            # retrieve_with_chunks بيرجع الـ context + list من الـ chunks للـ log
            if hasattr(self.rag_retriever, "retrieve_with_chunks"):
                rag_content, rag_chunks = await self.rag_retriever.retrieve_with_chunks(query)
            else:
                rag_content = await self.rag_retriever.retrieve(query)
                rag_chunks = []
            self.call_logger.log_timing("RAG", "RAG retrieval", _time.monotonic() - _t_rag,
                                        extra=f"chunks={len(rag_chunks)} chars={len(rag_content)}")
        except Exception as exc:
            msg = f"RAG lookup failed: {exc}"
            logger.exception("RAG lookup failed")
            self.call_logger.log_error(msg)
            return

        rag_content = rag_content.strip()
        if not rag_content:
            self.call_logger.log_system_event("RAG returned no results for this query")
            return

        # ── لوّج كل chunk بشكل واضح في الـ call log ──────────────────────────
        self.call_logger.log_rag_event(
            f"retrieved {len(rag_chunks)} chunks | total {len(rag_content)} chars"
        )
        for i, chunk in enumerate(rag_chunks, 1):
            preview = chunk.text[:120].replace("\n", " ")
            self.call_logger.log_rag_event(
                f"chunk {i}/{len(rag_chunks)} | distance={chunk.distance:.3f} | "
                f"source={chunk.source} | chunk_idx={chunk.chunk_index}\n"
                f"    └─ {preview}{'...' if len(chunk.text) > 120 else ''}"
            )

        self.call_logger.log_system_event(f"RAG injected {len(rag_content)} chars into context")
        turn_ctx.add_message(
            role="assistant",
            content=(
                "معلومات من قاعدة المعرفة قد تساعد في الرد التالي. "
                "استخدمها فقط إذا كانت مرتبطة بسؤال العميل، ولا تذكرها كمصدر داخلي:\n"
                f"{rag_content}"
            ),
        )

    @function_tool()
    async def add_note(self, context: RunContext, note: str) -> str:
        """Save an important note about the caller to remember throughout the call.

        Call this tool IMMEDIATELY whenever the caller mentions:
        - their name
        - a problem or complaint
        - an order number, account number, or any reference
        - any detail you will need to remember later

        Args:
            note: The note to save. Write it clearly and concisely in Arabic.
        """
        logger.info("🟢 LLM CALLED add_note TOOL! Note: %s", note)
        self.notes.append(note)
        self.call_logger.log_note(note)
        notes_text        = "\n".join(f"- {n}" for n in self.notes)
        new_instructions  = f"{self.base_instructions}\n\nالملاحظات الحالية:\n{notes_text}"
        await self.update_instructions(new_instructions)
        return "تم حفظ الملاحظة."


# ══════════════════════════════════════════════════════════════════════════════
# Worker lifecycle
# ══════════════════════════════════════════════════════════════════════════════

def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.8,
        prefix_padding_duration=0.3,
        # Tighter thresholds: reduces false triggers from background noise
        # while remaining responsive to natural Saudi Arabic speech patterns.
        activation_threshold=0.6,
        deactivation_threshold=0.4,
    )
    try:
        rag_retriever = build_rag_from_env()
        if rag_retriever is not None and hasattr(rag_retriever, "warmup"):
            rag_retriever.warmup()
        proc.userdata["rag_retriever"] = rag_retriever
    except Exception:
        logger.exception("Failed to initialize RAG; continuing without it")
        proc.userdata["rag_retriever"] = None


async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(
        "Job received: room=%s job_id=%s agent_name=%s",
        ctx.room.name,
        ctx.job.id,
        ctx.job.agent_name,
    )

    # ── STT ── Deepgram Nova-3 (Saudi Arabic) ─────────────────────────────────
    stt_model = os.getenv("STT_MODEL", "nova-3")
    stt_language = os.getenv("STT_LANGUAGE", "ar-SA")
    logger.info("Starting agent with STT provider=deepgram model=%s language=%s", stt_model, stt_language)

    # ── TTS ── Cartesia Sonic 3.5 ──────────────────────────────────────────────
    tts_provider = "cartesia"
    cartesia_model = os.getenv("CARTESIA_MODEL", "sonic-3.5")
    cartesia_voice_id = os.getenv("CARTESIA_VOICE_ID") or CARTESIA_DEFAULT_VOICE_ID
    cartesia_language = os.getenv("CARTESIA_LANGUAGE", "ar")
    tts_instance = cartesia_plugin.TTS(
        api_key=os.getenv("CARTESIA_API_KEY"),
        model=cartesia_model,
        voice=cartesia_voice_id,
        language=cartesia_language,
    )
    logger.info(
        "TTS provider=cartesia model=%s voice=%s language=%s",
        cartesia_model, cartesia_voice_id, cartesia_language,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_llm_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")

    # ── Session ───────────────────────────────────────────────────────────────
    turn_detector = MultilingualModel()
    session = AgentSession(
        stt=deepgram_plugin.STT(
            model=stt_model,
            language=stt_language,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            keyterm=[
                "فهد", "رقم الطلب", "الحساب", "خدمة العملاء",
            ],
        ),
        llm=openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            model=groq_llm_model,
            api_key=os.getenv("GROQ_API_KEY", ""),
        ),
        tts=tts_instance,
        turn_detection=turn_detector,
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await ctx.connect()

    # ── Build assistant ───────────────────────────────────────────────────────
    assistant = Assistant(
        call_id=ctx.room.name,
        rag_retriever=ctx.proc.userdata.get("rag_retriever"),
    )

    session_room_options = _build_room_options()

    # ── Loudness normalization ────────────────────────────────────────────────
    # Attached AFTER ctx.connect() so session.input.audio is guaranteed
    # to be non-None before we try to wrap it.
    # We poll briefly (max 10 × 50 ms = 500 ms) in case the audio track
    # arrives slightly after the room connection is established.
    if os.getenv("LOUDNORM_ENABLED", "true").strip().lower() in {"true", "1", "yes", "on"}:
        _loudnorm_attached = False
        for _attempt in range(10):
            if getattr(session.input, "audio", None) is not None:
                _attach_loudness_normalizer(session)
                _loudnorm_attached = True
                break
            await asyncio.sleep(0.05)

        if _loudnorm_attached:
            assistant.call_logger.log_system_event(
                f"Loudness normalization enabled "
                f"(target={LoudnessNormalizer.TARGET_LUFS} LUFS, "
                f"max_gain={LoudnessNormalizer.MAX_GAIN_DB} dB)"
            )
        else:
            assistant.call_logger.log_system_event(
                "Loudness normalization skipped — audio input not ready after 500 ms"
            )
    else:
        assistant.call_logger.log_system_event("Loudness normalization disabled")

    # ── Local recorder ────────────────────────────────────────────────────────
    # install() patches session._on_audio_input_changed and
    # _on_audio_output_changed BEFORE session.start().  When RoomIO wires up
    # input.audio / output.audio inside session.start(), our hooks fire
    # synchronously and wrap the streams with RecorderIO immediately —
    # before on_enter() has a chance to grab a stale reference.
    recorder = LocalCallRecorder(session, ctx.room.name)
    recorder.install()

    # ── Event handlers ────────────────────────────────────────────────────────
    _llm_reply_start: dict[str, float] = {}  # track per-item start time

    _tts_start: dict[str, float] = {}

    @session.on("agent_speaking_started")
    def on_agent_speaking_started(event: Any) -> None:
        import time as _time
        _now = _time.monotonic()
        _llm_reply_start["t"] = _now
        _tts_start["t"] = _now
        assistant.call_logger.log_system_event("TTS playback started", stage="TTS")

    @session.on("agent_speaking_stopped")
    def on_agent_speaking_stopped(event: Any) -> None:
        import time as _time
        if "t" in _tts_start:
            elapsed = _time.monotonic() - _tts_start.pop("t")
            assistant.call_logger.log_timing("TTS", "TTS playback duration", elapsed)

    # Provider-agnostic TTS success/timing log, using the standard
    # livekit-agents metrics event emitted by every TTS plugin (Cartesia
    # included) after each synthesis call.
    @session.on("metrics_collected")
    def on_metrics_collected(event: Any) -> None:
        m = getattr(event, "metrics", None)
        if isinstance(m, TTSMetrics):
            assistant.call_logger.log_timing(
                "TTS",
                f"{tts_provider} TTS generation",
                m.duration,
                extra=(
                    f"chars={m.characters_count} ttfb={m.ttfb:.3f}s "
                    f"audio_duration={m.audio_duration:.2f}s"
                    + (" CANCELLED" if m.cancelled else "")
                ),
            )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: Any) -> None:
        import time as _time
        message = getattr(event, "item", None)
        if getattr(message, "role", None) != "assistant":
            return
        text = _message_text(message)
        if text:
            assistant.log_agent_message(text)
            # LLM→TTS pipeline: measure from when speaking started (if captured)
            if "t" in _llm_reply_start:
                elapsed = _time.monotonic() - _llm_reply_start.pop("t")
                assistant.call_logger.log_timing("LLM", "LLM reply generated", elapsed,
                                                 extra=f"chars={len(text)}")

    @session.on("close")
    def on_session_close(event: Any) -> None:
        reason = getattr(event, "reason", "unknown")
        error  = getattr(event, "error", None)
        assistant.call_logger.log_system_event(f"Session closed: {reason}")
        if error:
            assistant.call_logger.log_error(f"Session closed with error: {error}")
        asyncio.ensure_future(recorder.stop())
        assistant.finalize_call()

    async def on_job_shutdown(reason: str) -> None:
        assistant.call_logger.log_system_event(f"Job shutdown: {reason}")
        await recorder.stop()
        assistant.finalize_call()

    ctx.add_shutdown_callback(on_job_shutdown)

    # ── Start session ─────────────────────────────────────────────────────────
    try:
        start_kwargs: dict[str, Any] = {
            "agent": assistant,
            "room":  ctx.room,
        }
        if session_room_options is not None:
            start_kwargs["room_options"] = session_room_options
        await session.start(**start_kwargs)

    except Exception as exc:
        logger.exception("Agent session failed")
        assistant.call_logger.log_error(f"Agent session failed: {exc}")
        await recorder.stop()
        assistant.finalize_call()
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _message_text(message: Any) -> str:
    text_content = getattr(message, "text_content", "")
    if callable(text_content):
        text_content = text_content()
    if isinstance(text_content, list):
        return "\n".join(str(part) for part in text_content if part).strip()
    return str(text_content or "").strip()


def _attach_loudness_normalizer(session: AgentSession) -> None:
    """
    Monkey-patch the session's audio input to run every incoming frame
    through LoudnessNormalizer before it reaches VAD / STT.

    LiveKit AgentSession exposes the audio input stream via
    ``session.input.audio``.  We wrap ``__aiter__`` so the normalizer
    is transparent to all downstream consumers.

    Must be called AFTER ctx.connect() — session.input.audio is None
    until the room connection is established and the audio track arrives.
    """
    try:
        audio_input = session.input.audio
        # Guard against None: audio track not yet subscribed
        if audio_input is None:
            logger.warning(
                "LoudnessNormalizer: session.input.audio is None — "
                "normalization skipped (call after ctx.connect())"
            )
            return
    except AttributeError:
        logger.warning(
            "LoudnessNormalizer: session.input.audio not available — "
            "normalization skipped"
        )
        return

    normalizer = LoudnessNormalizer()
    original_aiter = audio_input.__aiter__

    async def _normalized_aiter():
        async for frame in original_aiter():
            for norm_frame in normalizer.push(frame):
                yield norm_frame
        # Flush remaining frames when the stream closes
        for norm_frame in normalizer.flush():
            yield norm_frame

    audio_input.__aiter__ = _normalized_aiter
    logger.info("LoudnessNormalizer attached to session audio input")


def _build_room_options() -> Optional[room_io.RoomOptions]:
    is_sip = os.getenv("SIP_ENABLED", "false").strip().lower() in {"true", "1", "yes", "on"}
    if is_sip:
        # SIP lines carry 8 kHz / 16 kHz narrowband audio with codec artifacts.
        # BVCTelephony is tuned specifically for this degraded signal profile —
        # it outperforms the wideband models (QUAIL, NC) on telephone audio.
        logger.info("SIP mode: applying BVCTelephony noise cancellation")
        return room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    # WebRTC path — pick provider from env or default to QUAIL_VF_L
    noise_filter = _build_noise_cancellation()
    if noise_filter is None:
        return None
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(noise_cancellation=noise_filter),
    )


def _build_noise_cancellation() -> Any | None:
    provider = os.getenv("AGENT_NOISE_CANCELLATION", "off").strip().lower()
    if provider in {"", "off", "false", "0", "no", "none"}:
        return None
    if provider in {"ai_coustics", "ai-coustics", "quail", "quail_l"}:
        if not _AI_COUSTICS_AVAILABLE:
            logger.warning("ai_coustics not available; noise cancellation disabled")
            return None
        logger.info("Agent noise cancellation enabled: ai_coustics QUAIL_L")
        return ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_L)
    if provider in {"ai_coustics_voice_focus", "ai-coustics-voice-focus", "quail_vf_l", "voice_focus"}:
        if not _AI_COUSTICS_AVAILABLE:
            logger.warning("ai_coustics not available; noise cancellation disabled")
            return None
        logger.info("Agent noise cancellation enabled: ai_coustics QUAIL_VF_L")
        return ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_L)
    if provider in {"krisp", "noise_cancellation", "noise-cancellation", "nc"}:
        logger.info("Agent noise cancellation enabled: Krisp NC")
        return noise_cancellation.NC()
    if provider in {"bvc", "krisp_bvc", "background_voice"}:
        logger.info("Agent noise cancellation enabled: Krisp BVC")
        return noise_cancellation.BVC()
    if provider in {"bvc_telephony", "krisp_bvc_telephony", "telephony"}:
        logger.info("Agent noise cancellation enabled: Krisp BVCTelephony")
        return noise_cancellation.BVCTelephony()
    logger.warning("Unknown AGENT_NOISE_CANCELLATION=%r; noise cancellation disabled", provider)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=my_agent,
            prewarm_fnc=prewarm,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "").strip(),
        )
    )