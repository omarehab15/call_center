FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — pin habibi-tts to latest known working version
RUN pip install --no-cache-dir \
    "habibi-tts==0.1.1" \
    "f5-tts>=1.0.0" \
    fastapi \
    "uvicorn[standard]" \
    soundfile \
    pydantic \
    cached-path

# Copy server + assets
COPY server.py .
COPY assets/ ./assets/

EXPOSE 8000

# Pre-download model weights at build time so first request is fast
ARG HABIBI_MODEL=Unified
ENV HABIBI_MODEL=${HABIBI_MODEL}
RUN python -c "
from cached_path import cached_path
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder

models = {
    'Unified': ('hf://SWivid/Habibi-TTS/Unified/vocab.txt', 'hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors'),
}
import os
m = os.environ.get('HABIBI_MODEL', 'Unified')
cfg = models.get(m, models['Unified'])
vocab = str(cached_path(cfg[0]))
ckpt  = str(cached_path(cfg[1]))
base  = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
load_model(DiT, base, ckpt, vocab_file=vocab)
load_vocoder()
print('Model pre-downloaded OK')
" || echo "Pre-download failed (non-fatal, will download on first run)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]