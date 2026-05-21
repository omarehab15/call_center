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
        لا تستخدم تنسيقات أو رموز أو مقدمات فارغة مثل بالتأكيد أو حسناً.
        كن ودوداً ومباشراً."""
        super().__init__(
            instructions=self.base_instructions,
        )
        self.notes = []

    async def on_enter(self) -> None:
        """Fires when the agent becomes active — agent speaks first.

        We must pass user_input alongside instructions so that the chat context
        contains at least one user-turn before calling the LLM.  Without it the
        peg-native chat template formats an empty prompt (0 tokens) and the
        llama.cpp slot is released immediately — producing silence.
        """
        await self.session.generate_reply(
            user_input="...",  # synthetic trigger — never spoken, just seeds the chat context
            instructions=(
                "ابدأ المكالمة بتحية الشخص المتصل بلهجة سعودية ودية "
                "ثم اسأله عن اسمه وعن سبب اتصاله بطريقة محترمة. "
                "يمكنك قول شيء مثل تحية الاسلام او اي تحية اخرى"
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
        await self.update_instructions(new_instructions)

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

    groq_llm_model = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

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

    tts_voice = os.getenv("TTS_VOICE", "SAU_male_1")
    tts_base_url = os.getenv("TTS_BASE_URL", "http://habibi_tts:8000/v1")

    tts_instance = openai.TTS(
        base_url=tts_base_url,
        model="habibi-tts",
        voice=tts_voice,
        api_key="no-key-needed",
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
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        
    )

    await ctx.connect()

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8)
        # thinking_sound=[
        #     AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5),
        #     AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5),
        # ],
    )
    

    await session.start(
        agent=Assistant(call_id=ctx.room.name),
        room=ctx.room,
         room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_S),
        ),
         ),
    )
    
    await background_audio.start(room=ctx.room, agent_session=session)

if __name__ == "__main__":
    cli.run_app(server)
