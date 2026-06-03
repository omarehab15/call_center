from __future__ import annotations

import argparse
import asyncio
import hashlib
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

    def _retrieve_sync(self, query: str) -> str:
        results = self._collection.query(
            query_texts=[query],
            n_results=self._config.top_k,
            include=["documents", "metadatas", "distances"],
        )
        return format_rag_results(results, max_chars=self._config.max_context_chars)


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
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
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


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []
