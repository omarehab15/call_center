import asyncio
import logging
import os
from collections import deque
from typing import Any, Optional

import numpy as np
import pyloudnorm as pyln
from dotenv import load_dotenv
from livekit import api as lkapi_module
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
from livekit.agents import audio as lk_audio
from livekit.agents import stt as stt_module
from livekit.plugins import ai_coustics, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from call_logger import CallLogger
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
# Egress helpers
# ══════════════════════════════════════════════════════════════════════════════

async def start_call_recording(room_name: str) -> Optional[str]:
    """
    Start a RoomCompositeEgress (audio-only) for the given room.

    Audio is saved as an .mp3 file to S3.  The path inside the bucket is:
        recordings/<room_name>.mp3

    Required env vars:
        LIVEKIT_URL          — e.g. https://myproject.livekit.cloud
        LIVEKIT_API_KEY
        LIVEKIT_API_SECRET
        S3_RECORDING_BUCKET  — target S3 bucket name
        S3_RECORDING_REGION  — AWS region  (default: us-east-1)
        S3_ACCESS_KEY        — AWS access key id
        S3_SECRET_KEY        — AWS secret access key

    Optional:
        S3_ENDPOINT          — custom endpoint for S3-compatible stores (MinIO, etc.)

    Returns the egress_id on success, or None if recording is disabled /
    configuration is missing.
    """
    bucket     = os.getenv("S3_RECORDING_BUCKET", "").strip()
    access_key = os.getenv("S3_ACCESS_KEY", "").strip()
    secret_key = os.getenv("S3_SECRET_KEY", "").strip()

    if not all([bucket, access_key, secret_key]):
        logger.info(
            "Recording disabled: S3_RECORDING_BUCKET / S3_ACCESS_KEY / "
            "S3_SECRET_KEY not set."
        )
        return None

    region   = os.getenv("S3_RECORDING_REGION", "us-east-1").strip()
    endpoint = os.getenv("S3_ENDPOINT", "").strip()  # leave empty for AWS

    s3_upload = lkapi_module.S3Upload(
        bucket=bucket,
        region=region,
        access_key=access_key,
        secret=secret_key,
        **({"endpoint": endpoint} if endpoint else {}),
    )

    file_output = lkapi_module.EncodedFileOutput(
        file_type=lkapi_module.EncodedFileType.MP3,
        filepath=f"recordings/{room_name}.mp3",
        s3=s3_upload,
    )

    req = lkapi_module.RoomCompositeEgressRequest(
        room_name=room_name,
        audio_only=True,
        # Opus 96 kbps — good quality for voice; tiny file size
        preset=lkapi_module.EncodingOptionsPreset.OPUS_96,
        file_outputs=[file_output],
    )

    try:
        lk = lkapi_module.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL", ""),
            api_key=os.getenv("LIVEKIT_API_KEY", ""),
            api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        )
        info      = await lk.egress.start_room_composite_egress(req)
        egress_id = info.egress_id
        logger.info("Recording started: egress_id=%s  file=recordings/%s.mp3", egress_id, room_name)
        return egress_id
    except Exception:
        logger.exception("Failed to start call recording")
        return None


async def stop_call_recording(egress_id: str) -> None:
    """Stop a running egress by its ID."""
    if not egress_id:
        return
    try:
        lk = lkapi_module.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL", ""),
            api_key=os.getenv("LIVEKIT_API_KEY", ""),
            api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        )
        await lk.egress.stop_egress(
            lkapi_module.StopEgressRequest(egress_id=egress_id)
        )
        logger.info("Recording stopped: egress_id=%s", egress_id)
    except Exception:
        logger.exception("Failed to stop call recording (egress_id=%s)", egress_id)


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

        self.base_instructions = """أنت مساعد ذكاء اصطناعي صوتي اسمك فهد لمركز اتصالات. يتفاعل المستخدم معك عبر الصوت.

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
        self.call_logger.log_system_event("Call started - Agent initialized")
        await self.session.generate_reply(
            user_input="...",
            instructions=(
                "ابدأ المكالمة بتحية الشخص المتصل بلهجة سعودية ودية قول التالى "
                "[هلا بيك معك فهد ممكن اعرف اسمك الكريم] "
            ),
        )

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
        query = _message_text(new_message)
        if query:
            self.call_logger.log_user_message(query)

        if self.rag_retriever is None or not query:
            return

        try:
            self.call_logger.log_system_event(f"RAG lookup started — query: {query[:120]}")
            rag_content = await self.rag_retriever.retrieve(query)
        except Exception as exc:
            msg = f"RAG lookup failed: {exc}"
            logger.exception("RAG lookup failed")
            self.call_logger.log_error(msg)
            return

        rag_content = rag_content.strip()
        if not rag_content:
            self.call_logger.log_system_event("RAG returned no results for this query")
            return

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
        proc.userdata["rag_retriever"] = build_rag_from_env()
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

    # ── STT ──────────────────────────────────────────────────────────────────
    stt_provider = os.getenv("STT_PROVIDER", "whisper").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://whisper:80/v1"
        default_stt_model    = "whisper-large-v3"
    else:
        default_stt_base_url = "http://nemotron:8000/v1"
        default_stt_model    = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model    = os.getenv("STT_MODEL",    default_stt_model)
    stt_api_key  = os.getenv("STT_API_KEY",  "no-key-needed")

    logger.info(
        "Starting agent with STT provider=%s model=%s base_url=%s",
        stt_provider, stt_model, stt_base_url,
    )

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_voice    = os.getenv("TTS_VOICE", "fahad")
    tts_instance = openai.TTS(
        base_url="https://api.groq.com/openai/v1",
        model="canopylabs/orpheus-arabic-saudi",
        voice=tts_voice,
        api_key=os.getenv("GROQ_API_KEY", ""),
        response_format="wav",
    )
    logger.info("TTS voice=%s", tts_voice)

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_llm_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")

    # ── Session ───────────────────────────────────────────────────────────────
    turn_detector = MultilingualModel()
    session = AgentSession(
        stt=stt_module.StreamAdapter(
            stt=openai.STT(
                base_url=stt_base_url,
                model=stt_model,
                api_key=stt_api_key,
                language="ar",
                # Improved prompt: seeds Whisper with Saudi dialect vocabulary
                # and common call-centre phrases so the model biases toward
                # the correct transcription path from the first token.
                prompt=(
                    "محادثة خدمة عملاء باللهجة السعودية النجدية. "
                    "كلمات شائعة: وش، كيفك، إيش، زين، ابشر، تمام، والله، يعني، "
                    "حياك، شلونك، عندي مشكلة، رقم الطلب، الحساب، خدمة العملاء."
                ),
                # Language is fixed to Arabic — disable auto-detection to
                # save ~20 ms of per-utterance latency.
                detect_language=False,
            ),
            vad=ctx.proc.userdata["vad"],
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

    # ── Start audio recording ─────────────────────────────────────────────────
    egress_id: Optional[str] = await start_call_recording(ctx.room.name)

    # ── Build assistant ───────────────────────────────────────────────────────
    assistant = Assistant(
        call_id=ctx.room.name,
        rag_retriever=ctx.proc.userdata.get("rag_retriever"),
    )

    if egress_id:
        assistant.call_logger.log_system_event(
            f"Audio recording started (egress_id={egress_id})"
        )
    else:
        assistant.call_logger.log_system_event(
            "Audio recording disabled (S3 credentials not configured)"
        )

    session_room_options = _build_room_options()

    # ── Loudness normalization ────────────────────────────────────────────────
    # Wraps the session's audio input stream so every caller's audio is
    # normalized to TARGET_LUFS before it reaches VAD + STT.
    # Only active when LOUDNORM_ENABLED=true (default: true).
    if os.getenv("LOUDNORM_ENABLED", "true").strip().lower() in {"true", "1", "yes", "on"}:
        _attach_loudness_normalizer(session)
        assistant.call_logger.log_system_event(
            f"Loudness normalization enabled "
            f"(target={LoudnessNormalizer.TARGET_LUFS} LUFS, "
            f"max_gain={LoudnessNormalizer.MAX_GAIN_DB} dB)"
        )
    else:
        assistant.call_logger.log_system_event("Loudness normalization disabled")

    # ── Event handlers ────────────────────────────────────────────────────────
    @session.on("conversation_item_added")
    def on_conversation_item_added(event: Any) -> None:
        message = getattr(event, "item", None)
        if getattr(message, "role", None) != "assistant":
            return
        text = _message_text(message)
        if text:
            assistant.log_agent_message(text)

    @session.on("close")
    def on_session_close(event: Any) -> None:
        reason = getattr(event, "reason", "unknown")
        error  = getattr(event, "error", None)
        assistant.call_logger.log_system_event(f"Session closed: {reason}")
        if error:
            assistant.call_logger.log_error(f"Session closed with error: {error}")
        # Stop the audio recording then finalize the text log
        if egress_id:
            asyncio.ensure_future(stop_call_recording(egress_id))
        assistant.finalize_call()

    async def on_job_shutdown(reason: str) -> None:
        assistant.call_logger.log_system_event(f"Job shutdown: {reason}")
        if egress_id:
            await stop_call_recording(egress_id)
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
        if egress_id:
            await stop_call_recording(egress_id)
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
    """
    try:
        audio_input = session.input.audio
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
    # Default changed to quail_vf_l — best voice isolation for WebRTC calls
    # in diverse environments (car, street, open office).
    provider = os.getenv("AGENT_NOISE_CANCELLATION", "quail_vf_l").strip().lower()
    if provider in {"", "off", "false", "0", "no", "none"}:
        return None
    if provider in {"ai_coustics", "ai-coustics", "quail", "quail_l"}:
        logger.info("Agent noise cancellation enabled: ai_coustics QUAIL_L")
        return ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_L)
    if provider in {"ai_coustics_voice_focus", "ai-coustics-voice-focus", "quail_vf_l", "voice_focus"}:
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