# inference/lahgtna/Dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip

# ── Clone the lahgtna-chatterbox fork ─────────────────────────────────────────
# This is NOT the PyPI chatterbox-tts package — it's oddadmix's fork that adds
# Saudi Arabic (language_id="sa") and ships from_checkpoint() loader.
RUN git clone https://github.com/Oddadmix/lahgtna-chatterbox.git /app

WORKDIR /app

# ── Python deps (from notebook cell 1) ────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.4.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    resampy==0.4.3 \
    librosa==0.10.0 \
    s3tokenizer \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    omegaconf==2.3.0 \
    "resemble-perth==1.0.1" \
    "silero-vad==5.1.2" \
    "conformer==0.3.2" \
    safetensors \
    "numpy==1.26.0" \
    huggingface_hub \
    fastapi \
    "uvicorn[standard]"

# ── Copy server ────────────────────────────────────────────────────────────────
COPY server.py /app/server.py

# PYTHONPATH must include /app so `from src.chatterbox.mtl_tts import ...` works
ENV PYTHONPATH=/app
ENV VOICES_DIR=/voices
ENV HF_HOME=/model-cache
ENV HUGGINGFACE_HUB_CACHE=/model-cache

HEALTHCHECK --interval=15s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]