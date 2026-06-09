import logging
import os
from typing import Any, Optional

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
from livekit.agents import stt as stt_module
from livekit.plugins import ai_coustics, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from call_logger import CallLogger
from rag import RagRetriever, build_rag_from_env

logger = logging.getLogger("agent")

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(
        self,
        call_id: str = "local_call",
        rag_retriever: Optional[RagRetriever] = None,
    ) -> None:
        self.call_id = call_id
        self.rag_retriever = rag_retriever

        # Initialize CallLogger with correct path
        # Use environment variable or default to relative path
        logs_dir = os.getenv("CALL_LOGS", os.path.join(os.path.dirname(__file__), "..", "call_logs"))
        self.call_logger = CallLogger(call_id, logs_dir=logs_dir)
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
        super().__init__(
            instructions=self.base_instructions,
        )
        self.notes = []

    async def on_enter(self) -> None:
        self.call_logger.log_system_event("Call started - Agent initialized")
        await self.session.generate_reply(
            user_input="...",
            instructions=(
                "ابدأ المكالمة بتحية الشخص المتصل بلهجة سعودية ودية قول التالى [هلا بيك معك فهد ممكن اعرف اسمك الكريم] "
            ),
        )

    def log_agent_message(self, message: str):
        """Log agent message to call logger."""
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
        # Log user message
        query = _message_text(new_message)
        if query:
            self.call_logger.log_user_message(query)

        if self.rag_retriever is None:
            return

        if not query:
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
    async def add_note(
        self,
        context: RunContext,
        note: str,
    ) -> str:
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
        notes_text = "\n".join(f"- {n}" for n in self.notes)
        new_instructions = f"{self.base_instructions}\n\nالملاحظات الحالية:\n{notes_text}"
        await self.update_instructions(new_instructions)

        return "تم حفظ الملاحظة."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # MultilingualModel requires a JobContext, so create it inside my_agent.
    # The Dockerfile pre-downloads its model files to keep runtime startup fast.
    try:
        proc.userdata["rag_retriever"] = build_rag_from_env()
    except Exception:
        logger.exception("Failed to initialize RAG; continuing without it")
        proc.userdata["rag_retriever"] = None


async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    logger.info(
        "Job received: room=%s job_id=%s agent_name=%s",
        ctx.room.name,
        ctx.job.id,
        ctx.job.agent_name,
    )

    groq_llm_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")

    stt_provider = os.getenv("STT_PROVIDER", "whisper").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://whisper:80/v1"
        default_stt_model = "whisper-large-v3"
    else:
        default_stt_base_url = "http://nemotron:8000/v1"
        default_stt_model = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")

    logger.info(
        "Starting agent with STT provider=%s model=%s base_url=%s",
        stt_provider,
        stt_model,
        stt_base_url,
    )

    tts_voice = os.getenv("TTS_VOICE", "fahad")

    tts_instance = openai.TTS(
        base_url="https://api.groq.com/openai/v1",
        model="canopylabs/orpheus-arabic-saudi",
        voice=tts_voice,
        api_key=os.getenv("GROQ_API_KEY", ""),
        response_format="wav",
    )

    logger.info("TTS voice=%s", tts_voice)

    turn_detector = MultilingualModel()

    session = AgentSession(
        stt=stt_module.StreamAdapter(
            stt=openai.STT(
                base_url=stt_base_url,
                model=stt_model,
                api_key=stt_api_key,
                language="ar"
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

    assistant = Assistant(
        call_id=ctx.room.name,
        rag_retriever=ctx.proc.userdata.get("rag_retriever"),
    )
    session_room_options = _build_room_options()

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: Any) -> None:
        """Log assistant messages from the session conversation history."""
        message = getattr(event, "item", None)
        if getattr(message, "role", None) != "assistant":
            return
        text = _message_text(message)
        if text:
            assistant.log_agent_message(text)

    @session.on("close")
    def on_session_close(event: Any) -> None:
        reason = getattr(event, "reason", "unknown")
        error = getattr(event, "error", None)
        assistant.call_logger.log_system_event(f"Session closed: {reason}")
        if error:
            assistant.call_logger.log_error(f"Session closed with error: {error}")
        assistant.finalize_call()

    async def on_job_shutdown(reason: str) -> None:
        assistant.call_logger.log_system_event(f"Job shutdown: {reason}")
        assistant.finalize_call()

    ctx.add_shutdown_callback(on_job_shutdown)

    try:
        start_kwargs = {
            "agent": assistant,
            "room": ctx.room,
        }
        if session_room_options is not None:
            start_kwargs["room_options"] = session_room_options
        await session.start(**start_kwargs)
    except Exception as exc:
        logger.exception("Agent session failed")
        assistant.call_logger.log_error(f"Agent session failed: {exc}")
        assistant.finalize_call()
        raise


def _message_text(message: Any) -> str:
    text_content = getattr(message, "text_content", "")
    if callable(text_content):
        text_content = text_content()
    if isinstance(text_content, list):
        return "\n".join(str(part) for part in text_content if part).strip()
    return str(text_content or "").strip()


def _build_room_options() -> Optional[room_io.RoomOptions]:
    noise_filter = _build_noise_cancellation()
    if noise_filter is None:
        return None
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=noise_filter,
        ),
    )


def _build_noise_cancellation() -> Any | None:
    provider = os.getenv("AGENT_NOISE_CANCELLATION", "off").strip().lower()
    if provider in {"", "off", "false", "0", "no", "none"}:
        return None

    if provider in {"ai_coustics", "ai-coustics", "quail", "quail_l"}:
        logger.info("Agent noise cancellation enabled: ai_coustics QUAIL_L")
        return ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_L)

    if provider in {
        "ai_coustics_voice_focus",
        "ai-coustics-voice-focus",
        "quail_vf_l",
        "voice_focus",
    }:
        logger.info("Agent noise cancellation enabled: ai_coustics QUAIL_VF_L")
        return ai_coustics.audio_enhancement(
            model=ai_coustics.EnhancerModel.QUAIL_VF_L
        )

    if provider in {"krisp", "noise_cancellation", "noise-cancellation", "nc"}:
        logger.info("Agent noise cancellation enabled: Krisp NC")
        return noise_cancellation.NC()

    if provider in {"bvc", "krisp_bvc", "background_voice"}:
        logger.info("Agent noise cancellation enabled: Krisp BVC")
        return noise_cancellation.BVC()

    if provider in {"bvc_telephony", "krisp_bvc_telephony", "telephony"}:
        logger.info("Agent noise cancellation enabled: Krisp BVCTelephony")
        return noise_cancellation.BVCTelephony()

    logger.warning(
        "Unknown AGENT_NOISE_CANCELLATION=%r; noise cancellation disabled",
        provider,
    )
    return None


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=my_agent,
        prewarm_fnc=prewarm,
        agent_name=os.getenv("LIVEKIT_AGENT_NAME", "").strip(),
    ))
