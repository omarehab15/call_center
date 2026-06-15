from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

from dotenv import load_dotenv

logger = logging.getLogger("agent.rag")

DEFAULT_COLLECTION_PREFIX = "call_center_knowledge"
DEFAULT_EMBEDDING_PROVIDER = "chroma"
DEFAULT_CHROMA_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHROMA_PATH = "rag/chroma"
DEFAULT_KNOWLEDGE_DIR = "knowledge_base"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".json", ".csv"}
LOCAL_EMBEDDING_PROVIDERS = {"local", "sentence-transformers", "sentence_transformers"}


class RagRetriever(Protocol):
    async def retrieve(self, query: str) -> str:
        """Return context relevant to the query, or an empty string."""


@dataclass(frozen=True)
class RagConfig:
    enabled: bool
    chroma_path: Path
    collection_name: str
    embedding_provider: str
    embedding_model: str
    top_k: int
    max_context_chars: int
    device: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None

    @classmethod
    def from_env(cls, *, force_enabled: bool = False) -> "RagConfig":
        provider = os.getenv(
            "RAG_EMBEDDING_PROVIDER",
            DEFAULT_EMBEDDING_PROVIDER,
        ).strip().lower()
        return cls(
            enabled=force_enabled or _env_bool("RAG_ENABLED", False),
            chroma_path=Path(os.getenv("RAG_CHROMA_PATH", DEFAULT_CHROMA_PATH)),
            collection_name=os.getenv("RAG_COLLECTION_NAME")
            or _default_collection_name(provider),
            embedding_provider=provider,
            embedding_model=_embedding_model_from_env(provider),
            top_k=max(1, _env_int("RAG_TOP_K", 4)),
            max_context_chars=max(300, _env_int("RAG_MAX_CONTEXT_CHARS", 1800)),
            device=os.getenv("RAG_EMBEDDING_DEVICE") or None,
            embedding_api_key=os.getenv("RAG_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            embedding_base_url=os.getenv("RAG_EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
        )


@dataclass(frozen=True)
class RagChunk:
    text: str
    source: str
    distance: float
    chunk_index: int


class LocalSentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "RAG_EMBEDDING_PROVIDER=sentence-transformers requires the "
                "optional rag-local dependencies."
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            list(input),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class OpenAIEmbeddingFunction:
    def __init__(
        self,
        model_name: str,
        *,
        api_key: Optional[str],
        base_url: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "RAG_EMBEDDING_PROVIDER=openai requires RAG_EMBEDDING_API_KEY "
                "or OPENAI_API_KEY."
            )

        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._model_name = model_name

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model_name,
            input=list(input),
        )
        return [item.embedding for item in response.data]


def build_embedding_function(config: RagConfig) -> Any:
    provider = config.embedding_provider
    if provider in {"chroma", "default", "onnx", "mini"}:
        from chromadb.utils import embedding_functions

        return embedding_functions.DefaultEmbeddingFunction()

    if provider in LOCAL_EMBEDDING_PROVIDERS:
        return LocalSentenceTransformerEmbeddingFunction(
            config.embedding_model,
            device=config.device,
        )

    if provider == "openai":
        return OpenAIEmbeddingFunction(
            config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
        )

    raise ValueError(
        "Unsupported RAG_EMBEDDING_PROVIDER="
        f"{provider!r}. Use chroma, openai, or sentence-transformers."
    )


class ChromaRagRetriever:
    def __init__(self, config: RagConfig, collection: Any) -> None:
        self._config = config
        self._collection = collection

    @classmethod
    def from_config(cls, config: RagConfig) -> "ChromaRagRetriever":
        import chromadb

        config.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(config.chroma_path))
        collection = client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=build_embedding_function(config),
            metadata={"hnsw:space": "cosine"},
        )
        return cls(config=config, collection=collection)

    async def retrieve(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""
        return await asyncio.to_thread(self._retrieve_sync, query)

    async def retrieve_with_chunks(self, query: str) -> tuple[str, list[RagChunk]]:
        """مثل retrieve() بس بيرجع كمان list من RagChunk للـ call log."""
        query = query.strip()
        if not query:
            return "", []
        return await asyncio.to_thread(self._retrieve_sync_with_chunks, query)

    def _retrieve_sync(self, query: str) -> str:
        context, _ = self._retrieve_sync_with_chunks(query)
        return context

    def _retrieve_sync_with_chunks(self, query: str) -> tuple[str, list[RagChunk]]:
        results = self._collection.query(
            query_texts=[query],
            n_results=self._config.top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs      = _first_result_list(results.get("documents"))
        metadatas = _first_result_list(results.get("metadatas"))
        distances = _first_result_list(results.get("distances"))

        chunks: list[RagChunk] = []
        for i, doc in enumerate(docs):
            doc = str(doc).strip() if doc else ""
            if not doc:
                continue
            meta     = metadatas[i] if i < len(metadatas) else {}
            dist     = distances[i] if i < len(distances) else 1.0
            source   = meta.get("source", "unknown") if isinstance(meta, dict) else "unknown"
            chunk_idx = int(meta.get("chunk", i)) if isinstance(meta, dict) else i
            chunks.append(RagChunk(text=doc, source=source, distance=float(dist), chunk_index=chunk_idx))

        hits = len(chunks)
        if hits:
            dist_str = ", ".join(f"{c.distance:.3f}" for c in chunks)
            logger.debug(
                "RAG hits=%d/%d distances=[%s] query=%r",
                hits, self._config.top_k, dist_str, query[:80],
            )
        else:
            logger.debug("RAG no hits for query=%r", query[:80])

        context = format_rag_results(results, max_chars=self._config.max_context_chars)
        return context, chunks


def build_rag_from_env() -> Optional[RagRetriever]:
    config = RagConfig.from_env()
    if not config.enabled:
        logger.info("RAG disabled. Set RAG_ENABLED=true to enable it.")
        return None

    logger.info(
        "Starting RAG with Chroma collection=%s path=%s embedding_provider=%s embedding_model=%s",
        config.collection_name,
        config.chroma_path,
        config.embedding_provider,
        config.embedding_model,
    )
    return ChromaRagRetriever.from_config(config)


def format_rag_results(results: dict[str, Any], *, max_chars: int) -> str:
    documents = _first_result_list(results.get("documents"))
    metadatas = _first_result_list(results.get("metadatas"))
    distances = _first_result_list(results.get("distances"))
    if not documents:
        return ""

    chunks: list[str] = []
    total_chars = 0
    for index, document in enumerate(documents):
        document = str(document).strip()
        if not document:
            continue

        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
        source_title = metadata.get("title") if isinstance(metadata, dict) else None
        label = source_title or source
        score_text = f" distance={distance:.3f}" if isinstance(distance, float) else ""
        chunk = f"Source: {label}{score_text}\n{document}"
        remaining_chars = max_chars - total_chars
        if remaining_chars <= 0:
            break
        if len(chunk) > remaining_chars:
            chunk = chunk[:remaining_chars].rstrip()
        chunks.append(chunk)
        total_chars += len(chunk)

    return "\n\n---\n\n".join(chunks)


def ingest_directory(
    config: RagConfig,
    source_dir: Path,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    reset: bool = False,
) -> int:
    import chromadb

    source_dir = source_dir.resolve()
    config.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chroma_path))
    if reset:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            logger.info("No existing Chroma collection named %s", config.collection_name)

    collection = client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=build_embedding_function(config),
        metadata={"hnsw:space": "cosine"},
    )

    count = 0
    for path in iter_knowledge_files(source_dir):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        relative_source = path.relative_to(source_dir).as_posix()
        collection.delete(where={"source": relative_source})
        chunks = chunk_text_smart(text, chunk_size=chunk_size, overlap=chunk_overlap)
        ids = [
            stable_chunk_id(relative_source, chunk_index)
            for chunk_index in range(len(chunks))
        ]
        metadatas = [
            {
                "source": relative_source,
                "title": path.stem,
                "chunk": chunk_index,
            }
            for chunk_index in range(len(chunks))
        ]
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        count += len(chunks)
        logger.info("Indexed %s chunks from %s", len(chunks), relative_source)

    return count


def iter_knowledge_files(source_dir: Path) -> Iterable[Path]:
    if not source_dir.exists():
        return []

    return (
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def detect_qa_format(text: str) -> bool:
    """هل النص فيه Q&A بصيغة س:/ج: ؟"""
    import re
    qa_lines = sum(1 for line in text.splitlines() if re.match(r"^\*?\*?س:", line.strip()))
    return qa_lines >= 3


def chunk_qa_individual(text: str) -> list[str]:
    """كل سؤال + جوابه → chunk منفصل (نفس منطق chunk_preview.py)."""
    import re

    text = re.sub(r"\*\*\s*(س:)\s*", r"\1 ", text)
    text = re.sub(r"\*\*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+$", "", text, flags=re.MULTILINE)

    chunks: list[str] = []
    current_q: str | None = None
    current_a_lines: list[str] = []
    state = "idle"

    def flush() -> None:
        nonlocal current_q, current_a_lines, state
        if current_q and current_a_lines:
            chunks.append(f"س: {current_q.strip()}\nج: {' '.join(current_a_lines).strip()}")
        current_q = None
        current_a_lines = []
        state = "idle"

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("---") or stripped.startswith("# "):
            if state == "in_answer":
                flush()
            continue
        if not stripped:
            if state == "in_answer":
                flush()
            continue
        q_match = re.match(r"^س:\s*(.+)", stripped)
        if q_match:
            flush()
            current_q = q_match.group(1).strip()
            state = "in_question"
            continue
        a_match = re.match(r"^ج:\s*(.*)", stripped)
        if a_match:
            rest = a_match.group(1).strip()
            current_a_lines = [rest] if rest else []
            state = "in_answer"
            continue
        if state == "in_answer":
            current_a_lines.append(stripped)

    flush()
    return chunks


def chunk_text_smart(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """
    Chunking ذكي: لو النص فيه Q&A بصيغة س:/ج: يعمل Q&A chunking،
    غير كده يرجع للـ character chunking العادي.
    """
    if detect_qa_format(text):
        return chunk_qa_individual(text)
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(normalized) <= chunk_size:
        return [normalized] if normalized else []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        min_split_at = start + overlap + 1
        split_search_start = min(end, min_split_at + 1)
        split_at = normalized.rfind("\n\n", split_search_start, end)
        if split_at <= min_split_at:
            split_at = normalized.rfind("\n", split_search_start, end)
        if split_at <= min_split_at:
            split_at = end

        chunk = normalized[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= len(normalized):
            break
        next_start = max(0, split_at - overlap)
        if next_start <= start:
            next_start = split_at
        if next_start <= start:
            break
        start = next_start

    return chunks


def stable_chunk_id(source: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}".encode("utf-8")).hexdigest()
    return digest[:32]


def run_ingest_cli() -> None:
    load_dotenv(".env.local")
    parser = argparse.ArgumentParser(description="Index local knowledge files into Chroma.")
    parser.add_argument(
        "--source",
        default=os.getenv("RAG_KNOWLEDGE_DIR", DEFAULT_KNOWLEDGE_DIR),
        help="Directory containing .md, .txt, .html, .json, or .csv knowledge files.",
    )
    parser.add_argument("--reset", action="store_true", help="Recreate the collection.")
    parser.add_argument("--chunk-size", type=int, default=_env_int("RAG_CHUNK_SIZE", 900))
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=_env_int("RAG_CHUNK_OVERLAP", 150),
    )
    args = parser.parse_args()

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = RagConfig.from_env(force_enabled=True)
    count = ingest_directory(
        config,
        Path(args.source),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )
    print(
        f"Indexed {count} chunks into Chroma collection "
        f"'{config.collection_name}' at {config.chroma_path}"
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s; using %s", name, raw, default)
        return default


def _embedding_model_from_env(provider: str) -> str:
    raw = os.getenv("RAG_EMBEDDING_MODEL")
    if raw:
        return raw
    if provider == "openai":
        return DEFAULT_OPENAI_EMBEDDING_MODEL
    if provider in LOCAL_EMBEDDING_PROVIDERS:
        return DEFAULT_LOCAL_EMBEDDING_MODEL
    return DEFAULT_CHROMA_EMBEDDING_MODEL


def _default_collection_name(provider: str) -> str:
    if provider in {"chroma", "default", "onnx", "mini"}:
        suffix = "chroma"
    elif provider in LOCAL_EMBEDDING_PROVIDERS:
        suffix = "local"
    else:
        suffix = provider.replace("-", "_")
    return f"{DEFAULT_COLLECTION_PREFIX}_{suffix}"


def ingest_from_preview(
    config: RagConfig,
    preview_path: Path,
    *,
    reset: bool = False,
) -> int:
    """
    اقرأ chunks_preview.json اللي أنتجه chunk_preview.py واعمل upsert في Chroma.
    بدل ما تعمل chunking من الأول، بتستخدم الـ chunks اللي راجعتها وعدّلتها.
    """
    import chromadb

    if not preview_path.exists():
        raise FileNotFoundError(f"ملف الـ preview مش موجود: {preview_path}")

    with preview_path.open(encoding="utf-8") as f:
        data = json.load(f)

    chunks_data: list[dict[str, Any]] = data.get("chunks", [])
    if not chunks_data:
        logger.warning("ملف الـ preview فارغ أو مافيش chunks فيه.")
        return 0

    config.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chroma_path))

    if reset:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            logger.info("No existing Chroma collection named %s", config.collection_name)

    collection = client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=build_embedding_function(config),
        metadata={"hnsw:space": "cosine"},
    )

    # جمّع الـ chunks حسب source عشان نعمل delete قبل upsert لكل ملف
    from collections import defaultdict
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks_data:
        by_source[chunk.get("source", "unknown")].append(chunk)

    total = 0
    for source, source_chunks in by_source.items():
        collection.delete(where={"source": source})
        ids = [stable_chunk_id(source, i) for i in range(len(source_chunks))]
        documents = [c["text"] for c in source_chunks]
        metadatas = [
            {
                "source": source,
                "title": Path(source).stem,
                "chunk": i,
                "type": c.get("type", "plain"),
                **({"section": c["section"]} if c.get("section") else {}),
            }
            for i, c in enumerate(source_chunks)
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        total += len(source_chunks)
        logger.info("Indexed %d chunks from %s (from preview)", len(source_chunks), source)

    return total


def run_ingest_from_preview_cli() -> None:
    """CLI لـ ingest من chunks_preview.json."""
    import argparse

    load_dotenv(".env.local")
    parser = argparse.ArgumentParser(
        description="Index pre-chunked knowledge from a chunks_preview.json file into Chroma."
    )
    parser.add_argument(
        "--from-preview",
        required=True,
        metavar="CHUNKS_JSON",
        help="مسار ملف chunks_preview.json اللي أنتجه chunk_preview.py",
    )
    parser.add_argument("--reset", action="store_true", help="احذف الـ collection وابدأ من الأول.")
    args = parser.parse_args()

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = RagConfig.from_env(force_enabled=True)
    preview_path = Path(args.from_preview)

    print(f"\n📂 قراءة الـ chunks من: {preview_path}")
    count = ingest_from_preview(config, preview_path, reset=args.reset)
    print(
        f"✅ تم الـ indexing: {count} chunk في Chroma collection "
        f"'{config.collection_name}' في {config.chroma_path}\n"
    )


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []