#!/usr/bin/env bash
# start-remote.sh — Run this on the vast.ai GPU machine
# Starts model containers: Whisper (STT) + habibi-tts (TTS)
# LLM is handled by Groq cloud — no local container needed
set -euo pipefail

echo "========================================"
echo "  Starting model containers (GPU mode)"
echo "========================================"
echo ""
echo "Services:"
echo "  • Whisper STT   → port 11435"
echo "  • habibi-TTS    → port 11437"
echo "  • LLM           → Groq cloud (no local container)"
echo ""

docker compose \
  -f docker-compose.remote.yml \
  --env-file .env.remote \
  up --build "$@"
