# inference/lahgtna/Dockerfile
# Builds the lahgtna-chatterbox TTS server.
# Inherits CUDA 12.4 + cuDNN for GPU acceleration.
# Python 3.11 is required — chatterbox does NOT support 3.12+ (no prebuilt wheels).

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# ── System deps ────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        git \
        git-lfs \
        curl \
        ffmpeg \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip

# ── Python deps ────────────────────────────────────────────────────────────────
# Install PyTorch with CUDA 12.4 first (chatterbox is strict about torch version)
RUN pip install --no-cache-dir \
    torch==2.4.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# chatterbox-tts pins its own deps — install after torch to avoid conflicts
RUN pip install --no-cache-dir \
    chatterbox-tts \
    huggingface_hub \
    safetensors \
    fastapi \
    uvicorn[standard] \
    pydantic

# ── App ────────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY server.py /app/server.py

# Voice files are mounted at runtime: -v ./inference/xtts/voices:/voices
# Model cache is persisted via a named volume mounted at /root/.cache/huggingface
ENV VOICES_DIR=/voices
ENV HF_HOME=/model-cache
ENV HUGGINGFACE_HUB_CACHE=/model-cache

# ── Health check ───────────────────────────────────────────────────────────────
# Model download takes a few minutes on first boot; generous start_period accounts
# for that. The /health endpoint returns 200 only after the model is fully loaded.
HEALTHCHECK --interval=15s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]