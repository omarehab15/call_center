#!/usr/bin/env bash
# start-remote.sh — Run this on the vast.ai GPU machine
# Starts model containers: Kokoro (TTS), Nemotron (STT), llama.cpp (LLM)
set -euo pipefail

echo "========================================"
echo "  Starting model containers (GPU mode)"
echo "========================================"
echo ""
echo "Services:"
echo "  • Kokoro TTS    → port 8880"
echo "  • Nemotron STT  → port 11435"
echo "  • llama.cpp LLM → port 11436"
echo ""

docker compose \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-gpu.yml \
  --env-file .env.remote \
  up --build "$@"
