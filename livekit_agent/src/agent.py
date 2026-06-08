import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import stt as stt_module
from rag import RagRetriever, build_rag_from_env
from call_logger import CallLogger

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

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Load turn-detection model once at startup (not per call)
    proc.userdata["turn_detector"] = MultilingualModel()
    try:
        proc.userdata["rag_retriever"] = build_rag_from_env()
    except Exception:
        logger.exception("Failed to initialize RAG; continuing without it")
        proc.userdata["rag_retriever"] = None

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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
        turn_detection=ctx.proc.userdata["turn_detector"],
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await ctx.connect()

    assistant = Assistant(
        call_id=ctx.room.name,
        rag_retriever=ctx.proc.userdata.get("rag_retriever"),
    )

    @session.on("agent_message")
    def on_agent_message(message: ChatMessage) -> None:
        """Called whenever the agent produces a reply — gives us the real text."""
        text = _message_text(message)
        if text:
            assistant.log_agent_message(text)

    try:
        await session.start(
            agent=assistant,
            room=ctx.room,
        )
    finally:
        # Save call summary when call ends
        assistant.call_logger.log_call_summary(assistant.notes)

def _message_text(message: Any) -> str:
    text_content = getattr(message, "text_content", "")
    if callable(text_content):
        text_content = text_content()
    if isinstance(text_content, list):
        return "\n".join(str(part) for part in text_content if part).strip()
    return str(text_content or "").strip()

if __name__ == "__main__":
    cli.run_app(server)
