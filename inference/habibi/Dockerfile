FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    habibi-tts \
    fastapi \
    uvicorn[standard] \
    soundfile \
    pydantic

# Copy server + assets
COPY server.py .
COPY assets/ ./assets/

EXPOSE 8000

# Pre-download model weights at build time so first request is fast
ARG HABIBI_MODEL=Unified
ENV HABIBI_MODEL=${HABIBI_MODEL}
RUN python -c "from habibi_tts import HabibiTTS; HabibiTTS(model='${HABIBI_MODEL}')" || true

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
