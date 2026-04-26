# Split Deployment Guide

Run the voice AI stack across two machines: GPU models on a vast.ai machine, and LiveKit + frontend on your local machine. Connected via Tailscale.

## Architecture

```
┌─────────────────────────────┐      ┌─────────────────────────────────┐
│     LOCAL MACHINE           │      │   VAST.AI MACHINE (100.x.x.x)  │
│                             │      │                                 │
│  Browser ←──WebRTC──→ LiveKit│      │   Kokoro TTS      (:8880)     │
│                    (:7880)  │      │   Nemotron STT    (:11435)     │
│  Frontend          (:3000)  │      │   llama.cpp LLM   (:11436)     │
│                             │      │                                 │
│  Agent ──── HTTP/Tailscale ──────→ │   (all OpenAI-compatible APIs) │
└─────────────────────────────┘      └─────────────────────────────────┘
```

## Prerequisites

- **Tailscale** installed on both machines and connected
- **Docker** and **Docker Compose** on both machines
- **NVIDIA GPU + Container Runtime** on the vast.ai machine

## Quick Start

### Step 1 — Start models on the vast.ai machine

SSH into your vast.ai machine and clone the repo:

```bash
git clone <your-repo-url>
cd local-voice-ai
git checkout split-deployment

# Start the GPU model containers
chmod +x start-remote.sh
./start-remote.sh
```

Wait for all health checks to pass:
```bash
docker compose -f docker-compose.remote.yml -f docker-compose.remote-gpu.yml ps
```

All three services should show `healthy`:
- `kokoro` — TTS on port 8880
- `nemotron` — STT on port 11435
- `llama_cpp` — LLM on port 11436

### Step 2 — Configure your local machine

Clone the repo on your local machine:

```bash
git clone <your-repo-url>
cd local-voice-ai
git checkout split-deployment
```

Edit `.env.local` and replace `100.x.x.x` with your vast.ai machine's Tailscale IP:

```bash
# Find the Tailscale IP of your vast.ai machine
tailscale status

# Edit the env file
nano .env.local
```

Replace all occurrences of `100.x.x.x` with your actual Tailscale IP.

### Step 3 — Start the local stack

```bash
chmod +x start-local.sh
./start-local.sh
```

The script will:
1. Verify connectivity to the remote model endpoints
2. Start LiveKit server, the agent, and the frontend

### Step 4 — Use it

Open your browser at **http://localhost:3000** and click **Start call**.

## Manual Commands

If you prefer not to use the helper scripts:

**On vast.ai:**
```bash
docker compose -f docker-compose.remote.yml -f docker-compose.remote-gpu.yml --env-file .env.remote up --build
```

**On local:**
```bash
docker compose -f docker-compose.local.yml --env-file .env.local up --build
```

## Troubleshooting

### Models not reachable from local machine

```bash
# Test connectivity from your local machine
curl http://<TAILSCALE_IP>:8880/v1/models   # Kokoro TTS
curl http://<TAILSCALE_IP>:11435/v1/models  # Nemotron STT
curl http://<TAILSCALE_IP>:11436/v1/models  # llama.cpp LLM
```

If these fail:
1. Check Tailscale is running on both machines: `tailscale status`
2. Check the model containers are running: `docker ps` on the vast.ai machine
3. Verify the ports aren't blocked by a firewall

### WebRTC not connecting

The whole point of this split setup is that LiveKit runs locally, so WebRTC should just work. If you see ICE failures:
1. Make sure the LiveKit container is running: `docker ps | grep livekit`
2. Check you can reach it: `curl http://localhost:7880`

### Agent can't connect to LiveKit

Check the agent logs:
```bash
docker compose -f docker-compose.local.yml logs livekit_agent
```

## File Reference

| File | Purpose |
|---|---|
| `docker-compose.remote.yml` | Model containers for vast.ai |
| `docker-compose.remote-gpu.yml` | GPU overlay for remote compose |
| `.env.remote` | Environment vars for vast.ai |
| `docker-compose.local.yml` | LiveKit + Agent + Frontend for local |
| `.env.local` | Environment vars for local (set your Tailscale IP here) |
| `start-remote.sh` | Helper script to start models on vast.ai |
| `start-local.sh` | Helper script to start local stack |
