# Split Deployment Guide

Run the voice AI stack across two machines: GPU models on a vast.ai machine, and LiveKit + frontend on your local machine. Connected via Tailscale.

## Architecture

```
┌─────────────────────────────┐      ┌──────────────────────────────────────┐
│     LOCAL MACHINE           │      │   VAST.AI MACHINE (100.x.x.x)       │
│                             │      │                                      │
│  Browser ←──WebRTC──→ LiveKit│      │   Whisper STT             (:11435)  │
│                    (:7880)  │      │   llama.cpp LLM           (:11436)  │
│  Frontend          (:3000)  │      │                                      │
│                             │      │   (all OpenAI-compatible APIs)       │
│  Agent ──── HTTP/Tailscale ──────→ │                                      │
│       └──── HTTPS ──────────────→  Groq Orpheus TTS (cloud)              │
└─────────────────────────────┘      └──────────────────────────────────────┘
```

TTS is handled by **Groq cloud** (no self-hosted TTS container). Only STT and LLM run on the remote GPU machine.

## Prerequisites

- **Tailscale** installed on both machines and connected
- **Docker** and **Docker Compose** on both machines
- **NVIDIA GPU + Container Runtime** on the vast.ai machine
- **Groq API key** for TTS — get one at [console.groq.com](https://console.groq.com)

## Quick Start

### Step 1 — Start models on the vast.ai machine

SSH into your vast.ai machine and clone the repo:

```bash
git clone <your-repo-url>
cd local-voice-ai
cp .env.remote.example .env.remote
# Edit .env.remote if needed (defaults work for ALLaM-7B + Arabic Whisper)

chmod +x start-remote.sh
./start-remote.sh
```

Wait for health checks to pass:
```bash
docker compose -f docker-compose.remote.yml -f docker-compose.remote-gpu.yml ps
```

Both services should show `healthy`:
- `whisper` — STT on port 11435
- `llama_cpp` — LLM on port 11436

### Step 2 — Configure your local machine

```bash
git clone <your-repo-url>
cd local-voice-ai
cp .env.local.example .env.local
```

Edit `.env.local` and replace `100.x.x.x` with your vast.ai machine's Tailscale IP:

```bash
tailscale status   # find the vast.ai machine IP
nano .env.local    # set the IP and your GROQ_API_KEY
```

### Step 3 — Start the local stack

```bash
chmod +x start-local.sh
./start-local.sh
```

### Step 4 — Use it

Open your browser at **http://localhost:3000** and click **Start call**.

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
curl http://<TAILSCALE_IP>:11435/v1/models  # Whisper STT
curl http://<TAILSCALE_IP>:11436/v1/models  # llama.cpp LLM
```

If these fail:
1. Check Tailscale is running on both machines: `tailscale status`
2. Check the model containers are running: `docker ps` on the vast.ai machine
3. Verify ports aren't blocked: `ufw status` on the vast.ai machine

### CUDA OOM on the vast.ai machine

Both models (Whisper ~3GB, llama.cpp ~6-20GB depending on model) run on GPU. If you're on a smaller GPU:
- Reduce `LLAMA_CTX_SIZE` in `.env.remote`
- Or use a smaller/more quantized LLM model

### WebRTC not connecting

LiveKit runs locally so WebRTC should just work. If you see ICE failures:
```bash
docker compose -f docker-compose.local.yml logs livekit_agent
```

---

## File Reference

| File | Purpose |
|---|---|
| `docker-compose.remote.yml` | Model containers for vast.ai (whisper, llama.cpp) |
| `docker-compose.remote-gpu.yml` | GPU overlay for remote compose |
| `.env.remote` / `.env.remote.example` | Environment vars for vast.ai |
| `docker-compose.local.yml` | LiveKit + Agent + Frontend for local machine |
| `.env.local` / `.env.local.example` | Environment vars for local (set Tailscale IP + Groq key) |
| `start-remote.sh` | Helper script to start models on vast.ai |
| `start-local.sh` | Helper script to start local stack |