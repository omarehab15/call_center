<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="App Icon" width="80" />
  <h1>Local Voice AI</h1>
  <p>A fully open-source, real-time voice AI call center agent built for Saudi Arabic dialect.</p>
  <p>Built on <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a> with WebRTC audio, local STT + LLM inference, and cloud TTS.</p>
</div>

## Overview

A real-time AI voice assistant for Saudi Arabic call centers, using:

- **LiveKit** for WebRTC realtime audio + rooms.
- **LiveKit Agents (Python)** to orchestrate the STT → LLM → TTS pipeline.
- **Whisper (via VoxBox)** for Arabic speech-to-text.
- **llama.cpp** for running local LLMs (OpenAI-compatible API).
- **Groq Orpheus** for Saudi Arabic text-to-speech (cloud API).
- **Next.js + Tailwind** frontend UI.
- Fully containerized via Docker Compose.

## Deployment Options

### Option A — Single Machine (monolithic)

Run everything on one machine. Requires a GPU for acceptable performance.

```bash
# Copy and configure the env file
cp .env.example .env
# Edit .env — set your GROQ_API_KEY

# CPU mode
docker compose up --build

# GPU mode (recommended)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Then open [http://localhost:3000](http://localhost:3000).

### Option B — Split Deployment (local + remote GPU)

Run LiveKit + Agent + Frontend locally, with STT + LLM on a remote GPU machine (e.g. vast.ai) connected via Tailscale. See [SPLIT_DEPLOYMENT.md](SPLIT_DEPLOYMENT.md) for full setup.

```bash
# On the remote GPU machine:
./start-remote.sh

# On your local machine:
./start-local.sh
```

## Architecture

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│     LOCAL MACHINE           │      │   REMOTE / SAME MACHINE      │
│                             │      │                              │
│  Browser ←──WebRTC──→ LiveKit│      │   Whisper STT    (:11435)   │
│                    (:7880)  │      │   llama.cpp LLM  (:11436)   │
│  Frontend          (:3000)  │      │                              │
│                             │      └──────────────────────────────┘
│  Agent ──── HTTP ──────────────→   (OpenAI-compatible APIs)
│       └──── HTTPS ──────────────→  Groq Orpheus TTS (cloud)
└─────────────────────────────┘
```

### Pipeline per voice turn
```
User speaks
  → Whisper STT (dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8)
  → LLM (via llama.cpp)
  → Groq Orpheus TTS (canopylabs/orpheus-arabic-saudi)
  → Audio back to user via WebRTC
```

## Agent

The agent entrypoint is `livekit_agent/src/agent.py`. It uses LiveKit Agents OpenAI-compatible plugins:

- `openai.STT` → Whisper (configurable via `STT_PROVIDER` / `STT_BASE_URL` / `STT_MODEL`)
- `openai.LLM` → llama.cpp (`llama-server`)
- `openai.TTS` → Groq Orpheus (cloud API)
- `silero.VAD` for voice activity detection
- `MultilingualModel` for Arabic turn detection

The agent instructs the LLM to respond in **Saudi Najdi dialect** (`لهجة سعودية نجدية`).

## Environment Variables

### `.env` (monolithic deployment)

| Variable | Default | Description |
|---|---|---|
| `STT_PROVIDER` | `whisper` | STT backend |
| `STT_MODEL` | `dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8` | Whisper model |
| `LLAMA_HF_REPO` | `bartowski/ALLaM-AI_ALLaM-7B-Instruct-preview-GGUF` | LLM model repo |
| `LLAMA_HF_FILE` | `ALLaM-AI_ALLaM-7B-Instruct-preview-Q6_K_L.gguf` | LLM model file |
| `LLAMA_MODEL` | `allam-7b` | Model alias used by agent |
| `LLAMA_CTX_SIZE` | `8192` | Context window size |
| `TTS_VOICE` | `fahad` | Groq Orpheus voice |
| `GROQ_API_KEY` | — | **Required** — Groq API key for TTS |

### Split deployment env files

- `.env.local` — local machine config (see `.env.local.example`)
- `.env.remote` — remote GPU machine config (see `.env.remote.example`)

## Project Structure

```
.
├─ frontend/                    # Next.js UI client
├─ inference/
│   ├─ whisper/                 # STT (VoxBox + Whisper)
│   └─ llama/                   # LLM model cache volume
├─ livekit_agent/               # Python voice agent (LiveKit Agents)
│   └─ src/agent.py             # Main agent — STT→LLM→TTS pipeline
├─ docker-compose.yml           # Monolithic single-machine deployment
├─ docker-compose.gpu.yml       # GPU overlay for monolithic deployment
├─ docker-compose.local.yml     # Split deployment: local side
├─ docker-compose.remote.yml    # Split deployment: remote GPU side
├─ docker-compose.remote-gpu.yml # GPU overlay for remote side
├─ start-local.sh               # Helper: start local stack
└─ start-remote.sh              # Helper: start remote stack
```

## Notes

- The LLM auto-downloads from Hugging Face on first boot (no manual model download needed).
- The first run downloads several GB of model weights. GPU-enabled images are bigger and take longer.
- `llama_cpp` returns 503s while the model is loading. The Compose stack includes healthchecks, and `livekit_agent` waits for `llama_cpp` to be healthy before starting.

## Development

Use `.env.local` files in both `frontend` and `livekit_agent` dirs for local (non-Docker) development:

```bash
# Agent
cd livekit_agent && uv run python src/agent.py dev

# Frontend
cd frontend && pnpm dev
```

## Requirements

- Docker + Docker Compose
- **Groq API key** (for TTS) — get one at [console.groq.com](https://console.groq.com)
- GPU recommended for STT + LLM (CPU works but is slow)

## Credits

- Built with [LiveKit](https://livekit.io/) and [LiveKit Agents](https://docs.livekit.io/agents/)
- STT via Whisper: [VoxBox](https://pypi.org/project/vox-box/)
- LLM via [llama.cpp](https://github.com/ggml-org/llama.cpp)
- TTS via [Groq Orpheus](https://console.groq.com/)
