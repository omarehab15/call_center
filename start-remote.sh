#!/usr/bin/env bash
# start-remote.sh — Run this on the vast.ai GPU machine
# Starts model containers: Whisper (STT) — LLM + TTS are handled by Groq cloud
set -euo pipefail

echo "========================================"
echo "  Starting model containers (GPU mode)"
echo "========================================"
echo ""
echo "Services:"
echo "  • Whisper STT   → port 11435"
echo ""

docker compose \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-gpu.yml \
  --env-file .env.remote \
  up --build "$@"
