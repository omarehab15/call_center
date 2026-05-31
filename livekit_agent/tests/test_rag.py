import pytest

from agent import Assistant
from rag import chunk_text, format_rag_results, stable_chunk_id


class FakeRetriever:
    def __init__(self, context: str) -> None:
        self.context = context
        self.queries: list[str] = []

    async def retrieve(self, query: str) -> str:
        self.queries.append(query)
        return self.context


class FakeTurnContext:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add_message(self, *, role: str, content: str) -> None:
        self.messages.append((role, content))


class FakeMessage:
    text_content = "وش سياسة الاسترجاع؟"


@pytest.mark.asyncio
async def test_agent_injects_rag_context_for_user_turn() -> None:
    retriever = FakeRetriever("سياسة الاسترجاع خلال 14 يوم.")
    turn_ctx = FakeTurnContext()

    await Assistant(rag_retriever=retriever).on_user_turn_completed(
        turn_ctx,
        FakeMessage(),
    )

    assert retriever.queries == ["وش سياسة الاسترجاع؟"]
    assert turn_ctx.messages == [
        (
            "assistant",
            (
                "معلومات من قاعدة المعرفة قد تساعد في الرد التالي. "
                "استخدمها فقط إذا كانت مرتبطة بسؤال العميل، ولا تذكرها كمصدر داخلي:\n"
                "سياسة الاسترجاع خلال 14 يوم."
            ),
        )
    ]


@pytest.mark.asyncio
async def test_agent_skips_rag_context_when_no_result() -> None:
    turn_ctx = FakeTurnContext()

    await Assistant(rag_retriever=FakeRetriever("")).on_user_turn_completed(
        turn_ctx,
        FakeMessage(),
    )

    assert turn_ctx.messages == []


def test_chunk_text_uses_overlap() -> None:
    text = "A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80

    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 3
    assert chunks[0] == "A" * 80
    assert chunks[1].startswith("A" * 10)
    assert chunks[1].endswith("B" * 80)


def test_format_rag_results_includes_source_and_document() -> None:
    results = {
        "documents": [["سياسة الاسترجاع خلال 14 يوم.", "الشحن مجاني داخل الرياض."]],
        "metadatas": [[{"source": "returns.md"}, {"source": "shipping.md"}]],
        "distances": [[0.12, 0.34]],
    }

    context = format_rag_results(results, max_chars=500)

    assert "returns.md" in context
    assert "shipping.md" in context
    assert "سياسة الاسترجاع" in context
    assert "الشحن مجاني" in context


def test_stable_chunk_id_is_deterministic() -> None:
    assert stable_chunk_id("policy.md", 2) == stable_chunk_id("policy.md", 2)
    assert stable_chunk_id("policy.md", 2) != stable_chunk_id("policy.md", 3)
