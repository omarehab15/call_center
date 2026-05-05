# Split Deployment Guide

Run the voice AI stack across two machines: GPU models on a vast.ai machine, and LiveKit + frontend on your local machine. Connected via Tailscale.

## Architecture

```
┌─────────────────────────────┐      ┌──────────────────────────────────────┐
│     LOCAL MACHINE           │      │   VAST.AI MACHINE (100.x.x.x)       │
│                             │      │                                      │
│  Browser ←──WebRTC──→ LiveKit│      │   lahgtna-chatterbox TTS  (:8880)  │
│                    (:7880)  │      │   Whisper STT             (:11435)  │
│  Frontend          (:3000)  │      │   llama.cpp LLM           (:11436)  │
│                             │      │                                      │
│  Agent ──── HTTP/Tailscale ──────→ │   (all OpenAI-compatible APIs)      │
└─────────────────────────────┘      └──────────────────────────────────────┘
```

## Prerequisites

- **Tailscale** installed on both machines and connected
- **Docker** and **Docker Compose** on both machines
- **NVIDIA GPU + Container Runtime** on the vast.ai machine (recommended: RTX 3090 / A4000 or better, ≥16GB VRAM for all three models)

## Quick Start

### Step 1 — Start models on the vast.ai machine

SSH into your vast.ai machine and clone the repo:

```bash
git clone <your-repo-url>
cd local-voice-ai

# Start the GPU model containers
chmod +x start-remote.sh
./start-remote.sh
```

Wait for all health checks to pass (lahgtna downloads ~4GB on first boot):
```bash
docker compose -f docker-compose.remote.yml -f docker-compose.remote-gpu.yml ps
```

All three services should show `healthy`:
- `lahgtna` — Saudi Arabic TTS on port 8880
- `whisper` — STT on port 11435
- `llama_cpp` — LLM on port 11436

### Step 2 — Configure your local machine

Clone the repo on your local machine:

```bash
git clone <your-repo-url>
cd local-voice-ai
```

Edit `.env.local` and replace `100.x.x.x` with your vast.ai machine's Tailscale IP:

```bash
tailscale status   # find the vast.ai machine IP
nano .env.local
```

### Step 3 — Start the local stack

```bash
chmod +x start-local.sh
./start-local.sh
```

### Step 4 — Use it

Open your browser at **http://localhost:3000** and click **Start call**.

---

## TTS: lahgtna-chatterbox-v1

The TTS engine is [oddadmix/lahgtna-chatterbox-v1](https://huggingface.co/oddadmix/lahgtna-chatterbox-v1), a Saudi Arabic fine-tune of ResembleAI's Chatterbox flow-matching diffusion TTS model.

### Voice cloning
The agent uses `Fasseh-fahad.wav` from `inference/xtts/voices/` as the reference voice (same file that XTTS was using). To add more voices:

1. Place a clean 5–15 second WAV recording at `inference/xtts/voices/<name>.wav`
2. In `agent.py`, change `voice="fahad"` to `voice="<name>"`

The server resolves the voice name by matching the filename (e.g. `voice="fahad"` → `Fasseh-fahad.wav`).

### Tuning generation quality

Set these in `.env.remote` to tune the output for your call-center use case:

| Variable | Default | Effect |
|---|---|---|
| `LAHGTNA_EXAGGERATION` | `0.5` | Emotional intensity — try `0.3` for calm professional tone |
| `LAHGTNA_CFG_WEIGHT` | `0.5` | Voice adherence — higher = closer to reference file |
| `LAHGTNA_TEMPERATURE` | `0.8` | Randomness — lower = more consistent delivery |
| `LAHGTNA_REPETITION_PENALTY` | `2` | **Don't lower below 1.1** — prevents syllable looping |

### First-boot model download

On the first `docker compose up`, the lahgtna container downloads ~4GB of model weights from HuggingFace. The healthcheck has a 300s `start_period` to account for this. Subsequent starts are fast (weights are cached in the `lahgtna-model-cache` Docker volume).

---

## Manual Commands

**On vast.ai:**
```bash
docker compose -f docker-compose.remote.yml -f docker-compose.remote-gpu.yml --env-file .env.remote up --build
```

**On local:**
```bash
docker compose -f docker-compose.local.yml --env-file .env.local up --build
```

---

## Troubleshooting

### Models not reachable from local machine

```bash
curl http://<TAILSCALE_IP>:8880/v1/models   # lahgtna TTS
curl http://<TAILSCALE_IP>:11435/v1/models  # Whisper STT
curl http://<TAILSCALE_IP>:11436/v1/models  # llama.cpp LLM
```

If these fail:
1. Check Tailscale is running on both machines: `tailscale status`
2. Check the model containers are running: `docker ps` on the vast.ai machine
3. Verify ports aren't blocked: `ufw status` on the vast.ai machine

### lahgtna still loading (503 on /v1/audio/speech)

The model downloads and warms up on first boot. Watch progress:
```bash
docker compose -f docker-compose.remote.yml logs -f lahgtna
```

The `/health` endpoint returns `{"status": "ok"}` once the model is ready.

### Audio has looping/repetition artefacts

Increase `LAHGTNA_REPETITION_PENALTY` to `2` in `.env.remote` and restart:
```bash
docker compose -f docker-compose.remote.yml restart lahgtna
```

### Audio sounds robotic or flat

Lower `LAHGTNA_EXAGGERATION` to `0.3` and lower `LAHGTNA_CFG_WEIGHT` to `0.3` — this gives the model more freedom to express natural prosody.

### CUDA OOM on the vast.ai machine

All three models (lahgtna ~4GB, Whisper ~3GB, llama.cpp ~18GB) together need ~25GB VRAM. If you're on a smaller GPU:
- Reduce `LLAMA_N_GPU_LAYERS` in `.env.remote` to offload some LLM layers to CPU RAM
- Or run a smaller LLM (e.g. `unsloth/gemma-4-26B-A4B-it-GGUF`)

### WebRTC not connecting

LiveKit runs locally so WebRTC should just work. If you see ICE failures:
```bash
docker compose -f docker-compose.local.yml logs livekit_agent
```

---

## File Reference

| File | Purpose |
|---|---|
| `docker-compose.remote.yml` | Model containers for vast.ai (lahgtna, whisper, llama.cpp) |
| `docker-compose.remote-gpu.yml` | GPU overlay for remote compose |
| `.env.remote` | Environment vars for vast.ai |
| `docker-compose.local.yml` | LiveKit + Agent + Frontend for local machine |
| `.env.local` | Environment vars for local (set your Tailscale IP here) |
| `inference/lahgtna/server.py` | FastAPI TTS server (OpenAI-compatible) |
| `inference/lahgtna/Dockerfile` | Container for lahgtna-chatterbox |
| `inference/xtts/voices/` | Voice reference WAV files (shared with lahgtna) |
| `start-remote.sh` | Helper script to start models on vast.ai |
| `start-local.sh` | Helper script to start local stack |