# syntax=docker/dockerfile:1.5
ARG WHISPER_BASE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM ${WHISPER_BASE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip "setuptools<70.0.0" wheel \
    && python3 -m pip install "openai-whisper==20240930" --no-build-isolation \
    && python3 -m pip install vox-box

WORKDIR /app

ENV DATA_DIR=/data

CMD ["sh", "-c", "vox-box start --huggingface-repo-id ${VOXBOX_HF_REPO_ID:-Systran/faster-whisper-small} --data-dir ${DATA_DIR} --device ${VOXBOX_DEVICE:-cpu}"]
