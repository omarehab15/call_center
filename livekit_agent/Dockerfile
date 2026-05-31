# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.13 on Debian Bookworm
# UV is a fast Python package manager that provides better performance than pip
# We use the slim variant to keep the image size smaller while still having essential tools
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install build dependencies required for Python packages with native extensions
# gcc: C compiler needed for building Python packages with C extensions
# python3-dev: Python development headers needed for compilation
# We clean up the apt cache after installation to keep the image size down
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Create a new directory for our application code
# And set it as the working directory
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

# Copy just the dependency files first, for more efficient layer caching
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN mkdir -p src

# Install Python dependencies using UV's lock file
# --locked ensures we use exact versions from uv.lock for reproducible builds
# --no-dev keeps test/lint tooling out of the runtime image
# This creates a virtual environment and installs all dependencies
# Ensure your uv.lock file is checked in for consistency across environments
RUN --mount=type=cache,target=/app/.cache/uv,uid=10001,gid=10001 \
    UV_CACHE_DIR=/app/.cache/uv \
    uv sync --locked --no-dev

# Pre-download any ML models or files the agent needs
# We do this before copying the application code to cache the download step
# so that it only runs when dependencies (uv.lock) change.
RUN uv run python -c "from livekit.agents.plugin import Plugin; \
    from livekit.plugins import silero, openai; \
    from livekit.plugins.turn_detector.multilingual import MultilingualModel; \
    [p.download_files() for p in Plugin.registered_plugins]"

# Copy all remaining application files into the container
# This includes source code, configuration files, and dependency specifications
# (Excludes files specified in .dockerignore)
COPY --chown=appuser:appuser . .

# Run the application using UV
# UV will activate the virtual environment and run the agent.
# The "start" command tells the worker to connect to LiveKit and begin waiting for jobs.
CMD ["uv", "run", "src/agent.py", "start"]
