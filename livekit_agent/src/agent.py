import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    AudioConfig,
    JobContext,
    JobProcess,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import stt as stt_module

logger = logging.getLogger("agent")

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self, call_id: str = "local_call") -> None:
        self.call_id = call_id
        self.base_instructions = """أنت مساعد ذكاء اصطناعي صوتي اسمك فهد لمركز اتصالات. يتفاعل المستخدم معك عبر الصوت.
        أجب دائماً بلهجة سعودية نجدية بشكل مباشر وواضح.
        قصّر إجاباتك قدر الإمكان — جملة أو جملتين كحد أقصى في معظم الأحيان.
        لا تستخدم تنسيقات أو رموز أو نجمات أو مقدمات فارغة مثل "بالتأكيد" أو "حسناً".
        كن ودوداً ومباشراً.
        
        تعليمات هامة جداً:
        إذا ذكر المستخدم اسمه أو مشكلته، **يجب** عليك استخدام أداة `add_note` فوراً لحفظ هذه المعلومات في ذاكرتك."""
        super().__init__(
            instructions=self.base_instructions,
        )
        self.notes = []

    @function_tool()
    async def add_note(
        self,
        context: RunContext,
        note: str,
    ) -> str:
        """استخدم هذه الأداة لحفظ ملاحظة مهمة (مثلاً اسم المتصل، أو المشاكل التي يواجهها) لتتذكرها طوال المكالمة.
        
        Args:
            note: الملاحظة المراد حفظها. يجب أن تكون واضحة ومباشرة.
        """
        logger.info("🟢 LLM CALLED add_note TOOL! Note: %s", note)
        self.notes.append(note)
        
        # Debug: write to file in the same directory as agent.py
        debug_path = os.path.join(os.path.dirname(__file__), f"{self.call_id}_notes.txt")
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                for n in self.notes:
                    f.write(f'"{n}"\n')
            logger.info("Saved call notes to %s", debug_path)
        except Exception as e:
            logger.error("Failed to save debug notes: %s", e)
        
        notes_text = "\n".join(f"- {n}" for n in self.notes)
        new_instructions = f"{self.base_instructions}\n\nالملاحظات الحالية:\n{notes_text}"
        await context.agent.update_instructions(new_instructions)

        return "تم حفظ الملاحظة."

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    llama_model = os.getenv("LLAMA_MODEL", "allam-7b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://llama_cpp:11434/v1")

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
        response_format="wav",  # Groq Orpheus only supports wav
    )

    logger.info("TTS voice=%s", tts_voice)

    session = AgentSession(
        stt=stt_module.StreamAdapter(
            stt=openai.STT(
                base_url=stt_base_url,
                # base_url="http://localhost:11435/v1", # uncomment for local testing
                model=stt_model,
                api_key=stt_api_key,
                language="ar"
            ),
            vad=ctx.proc.userdata["vad"],
        ),
        llm=openai.LLM(
            base_url=llama_base_url,
            # base_url="http://localhost:11436/v1", # uncomment for local testing
            model=llama_model,
            api_key="no-key-needed"
        ),
        tts=tts_instance,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        
    )

    await ctx.connect()

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.5),
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    await session.start(
        agent=Assistant(call_id=ctx.room.name),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(server)
