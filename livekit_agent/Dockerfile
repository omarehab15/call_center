# syntax=docker/dockerfile:1

# Use Python slim base + uv from Docker Hub (avoids ghcr.io DNS issues)
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Copy uv binary from Docker Hub (no ghcr.io needed)
COPY --from=docker.io/astral/uv:latest /uv /usr/local/bin/uv

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install runtime deps only — gcc/g++/python3-dev removed intentionally.
# chromadb, grpcio, and all other packages have pre-built wheels for
# Python 3.13 / linux/amd64, so no compilation is needed.
# libglib2.0-0 is a shared-library runtime dep for some wheels.
# curl is needed for the docker-compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

# Copy dependency files first for better layer caching
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN mkdir -p src

# Install Python dependencies
RUN --mount=type=cache,target=/app/.cache/uv,uid=10001,gid=10001 \
    UV_CACHE_DIR=/app/.cache/uv \
    uv sync --locked --no-dev

# FIX: Pre-download model files needed at runtime.
#
# silero.VAD.load()
#   Downloads the silero ONNX model (~2 MB) into the HF cache.
#
# _EUORunnerMultilingual._download_files()
#   Downloads the turn-detector multilingual model files:
#     - model_q8.onnx   (~45 MB)
#     - languages.json  (~2 KB)
#     - tokenizer files (~1 MB)
#   This is the internal classmethod that downloads without needing a
#   job context (unlike MultilingualModel() which calls get_job_context()
#   and raises RuntimeError at build time).
#
# ai-coustics and noise-cancellation are intentionally excluded — they
# contact external licensing APIs at build time and hang in CI/offline envs.
RUN uv run python -c "\
from livekit.plugins import silero; \
from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual; \
silero.VAD.load(); \
_EUORunnerMultilingual._download_files(); \
print('All model files pre-downloaded successfully.')"

# Chroma's default embedding function downloads all-MiniLM-L6-v2 on first use.
# Do it at build time with a no-read-timeout stream so startup/ingest do not
# fail midway through a 79 MB runtime download.
RUN uv run python - <<'PY'
import os

import httpx
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
    ONNXMiniLM_L6_V2,
    _verify_sha256,
)


def _download_without_read_timeout(
    self: ONNXMiniLM_L6_V2,
    url: str,
    fname: str,
    chunk_size: int = 1024 * 1024,
) -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with httpx.stream("GET", url, timeout=None, follow_redirects=True) as response:
        response.raise_for_status()
        with open(fname, "wb") as file:
            for data in response.iter_bytes(chunk_size=chunk_size):
                file.write(data)

    if not _verify_sha256(fname, self._MODEL_SHA256):
        os.remove(fname)
        raise RuntimeError("Downloaded Chroma ONNX archive failed SHA256 check.")


ONNXMiniLM_L6_V2._download = _download_without_read_timeout
model = ONNXMiniLM_L6_V2()
model._download_model_if_not_exists()
model(["warmup"])
print("Chroma all-MiniLM-L6-v2 model pre-downloaded successfully.")
PY

# Copy all remaining application files
COPY --chown=appuser:appuser . .

CMD ["uv", "run", "src/agent.py", "start"]
