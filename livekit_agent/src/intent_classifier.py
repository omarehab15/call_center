from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger("agent.intent_classifier")

_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

_CLASSIFIER_MODEL = os.getenv("INTENT_CLASSIFIER_MODEL", "llama-3.1-8b-instant")

# ─── Prompt مبني مباشرة على الـ KB ────────────────────────────────────────────
_CLASSIFIER_SYSTEM = """أنت نظام تصنيف لمركز اتصالات شركة Seven Hunderds Apps (سبعمية تطبيق).
مهمتك الوحيدة: قرر إذا كان سؤال العميل يحتاج بحثاً في قاعدة المعرفة.

قاعدة مهمة — الجواب دائماً yes إذا ذكر العميل أياً من هذه الأسماء أو سأل عنها:
Seven Hunderds Apps، سبعمية، سبعمية تطبيق، Seven Hunderds، 700 apps، سبعمية آبس

الجواب yes أيضاً لأي سؤال عن:
- الشركة: تأسيسها، مقرها، حجمها، فروعها، انتشارها الجغرافي
- الخدمات: التحول الرقمي، المنصات الرقمية، الذكاء الاصطناعي، البيانات، الابتكار، أتمتة الإجراءات
- العملاء: الجهات الحكومية، الشركاء، المشاريع المنفذة
- الشهادات: ISO، CMMI، The Open Group
- التواصل: الموقع، الهاتف، ممثل خدمة العملاء
- الرؤية والرسالة والقيم

الجواب no فقط في هذه الحالات:
- تحيات (هلا، مرحبا، كيف حالك، صباح الخير)
- ردود قصيرة (أيوه، لا، تمام، شكراً، ماشي)
- أسئلة شخصية عن المتصل نفسه

رد بكلمة واحدة فقط: yes أو no"""

# ─── Heuristic ────────────────────────────────────────────────────────────────
_TRIVIAL_MAX_CHARS = 8
_TRIVIAL_WORDS = {
    "أيوه", "آه", "اه", "ايوه", "نعم", "صح", "تمام", "ماشي", "اوكيه", "أوكيه",
    "اوك", "أوك", "ok", "okay", "yes",
    "لا", "لأ", "لأه", "no",
    "شكرا", "شكراً", "شكرا جزيلا", "شكراً جزيلاً", "متشكر", "مشكور",
    "مع السلامة", "سلامه", "باي", "bye", "وداعاً",
    "حلو", "كويس", "ممتاز", "عظيم", "جميل", "زين", "بسيطة", "عادي",
}


def _is_trivial(query: str) -> bool:
    stripped = query.strip()
    if len(stripped) <= _TRIVIAL_MAX_CHARS:
        return True
    if stripped.rstrip("!؟?.,،") in _TRIVIAL_WORDS:
        return True
    return False


async def needs_rag(query: str, previous_query: str = "") -> bool:
    """
    يرجع True لو السؤال يحتاج RAG lookup، False لو ممكن نتخطاه.

    ثلاث طبقات:
      1. Heuristic محلي (0ms)   — تحيات وردود قصيرة
      2. Context merge           — لو الجملة قصيرة نضمها مع السابقة
      3. LLM micro-call (~100ms) — التصنيف الفعلي
    """
    if not query or not query.strip():
        return False

    if _is_trivial(query):
        logger.debug("Intent heuristic → skip RAG | query=%r", query[:60])
        return False

    combined = query.strip()
    if len(combined) < 15 and previous_query:
        combined = f"{previous_query.strip()} {combined}"
        logger.debug("Intent context merge | combined=%r", combined[:80])

    try:
        response = await _client.chat.completions.create(
            model=_CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": combined},
            ],
            max_tokens=1,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip().lower()
        result = answer == "yes"
        logger.debug(
            "Intent LLM → %s RAG | model=%s query=%r",
            "use" if result else "skip",
            _CLASSIFIER_MODEL,
            combined[:60],
        )
        return result
    except Exception as exc:
        logger.warning("Intent classifier failed (%s) — falling back to RAG", exc)
        return True
