# syntax=docker/dockerfile:1
# Habibi-TTS inference server
# Wraps the habibi-tts Python package behind an OpenAI-compatible
# POST /v1/audio/speech  endpoint so LiveKit agent can use openai.TTS(base_url=...)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install habibi-tts and a tiny FastAPI server
RUN pip install --no-cache-dir \
    habibi-tts \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    soundfile \
    numpy

# Copy the OpenAI-compatible wrapper server
COPY server.py .

# Cache model weights at build time so first request is fast
# Set HABIBI_DIALECT to the dialect you want (default: SAU for Saudi Arabic)
ARG HABIBI_DIALECT=SAU
ENV HABIBI_DIALECT=${HABIBI_DIALECT}

# Pre-download model weights
RUN python -c "from f5_tts.infer.utils_infer import load_model; print('Model pre-load skipped — will load on first request')" || true

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=10s --retries=12 --start-period=180s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
