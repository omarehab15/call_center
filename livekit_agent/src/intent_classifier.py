from __future__ import annotations

import json
import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger("agent.intent_classifier")

_client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

_CLASSIFIER_MODEL = os.getenv("INTENT_CLASSIFIER_MODEL", "llama-3.1-8b-instant")

# ─── Prompt ───────────────────────────────────────────────────────────────────
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

رد بـ JSON فقط بدون أي نص خارجه، بالشكل ده:
{"decision": "yes", "reason": "سبب مختصر بالعربي جملة واحدة"}
أو
{"decision": "no", "reason": "سبب مختصر بالعربي جملة واحدة"}"""

# ─── Heuristic ────────────────────────────────────────────────────────────────
_TRIVIAL_MAX_CHARS = 2  # كان 8 — عدد كبير جداً يطنش كلمات عربية مفيدة زي "خدماتكم"
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
      3. LLM micro-call (~150ms) — التصنيف الفعلي مع السبب
    """
    if not query or not query.strip():
        return False

    if _is_trivial(query):
        logger.info("Intent heuristic → skip RAG | reason=trivial_input | query=%r", query[:60])
        return False

    combined = query.strip()
    if len(combined) < 25 and previous_query:
        combined = f"{previous_query.strip()} {combined}"
        logger.debug("Intent context merge | combined=%r", combined[:80])

    try:
        response = await _client.chat.completions.create(
            model=_CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": combined},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # ─── Parse JSON ───────────────────────────────────────────────────────
        # نشيل backticks لو الموديل حطهم رغم التعليمات
        clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        decision = str(parsed.get("decision", "")).strip().lower()
        reason   = str(parsed.get("reason", "")).strip()
        result   = decision.startswith("yes")

        logger.info(
            "Intent LLM → %s RAG | reason=%r | model=%s | query=%r",
            "USE" if result else "SKIP",
            reason,
            _CLASSIFIER_MODEL,
            combined[:80],
        )
        return result

    except json.JSONDecodeError:
        # الموديل ما ردش بـ JSON — نقرا الرد كـ yes/no fallback
        logger.warning(
            "Intent classifier returned non-JSON (%r) — falling back to startswith check",
            raw[:80],
        )
        return raw.lower().startswith("yes")

    except Exception as exc:
        logger.warning("Intent classifier failed (%s) — falling back to RAG", exc)
        return True