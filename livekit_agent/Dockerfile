# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.13 on Debian Bookworm
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

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

# Install build deps + curl (needed for the healthcheck)
# FIX: added curl so the docker-compose healthcheck doesn't fail
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    python3-dev \
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

# FIX: Pre-download only the silero VAD ONNX model file.
# - silero.VAD.load() downloads the ONNX model so it's baked into the image.
# - MultilingualModel() is intentionally excluded: it requires a live job
#   context (get_job_context()) and raises RuntimeError outside one. Its
#   model files are downloaded automatically on first job startup instead.
# - The old Plugin.registered_plugins loop is also removed because it called
#   download_files() on livekit-plugins-ai-coustics and
#   livekit-plugins-noise-cancellation, which contact external APIs at build
#   time and can hang or fail in offline / CI environments.
RUN uv run python -c "\
from livekit.plugins import silero; \
silero.VAD.load(); \
print('Silero VAD model pre-download complete.')"

# Copy all remaining application files
COPY --chown=appuser:appuser . .

CMD ["uv", "run", "src/agent.py", "start"]
