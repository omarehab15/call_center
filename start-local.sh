#!/usr/bin/env bash
# start-local.sh — Run this on your local machine
# Starts LiveKit server, Habibi-TTS, agent, and frontend
# STT connects to the remote vast.ai machine via Tailscale
set -euo pipefail

ENV_FILE=".env.local"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found."
  echo "Copy .env.local.example to .env.local and set your vast.ai Tailscale IP."
  exit 1
fi

# Extract the remote host IP from STT_BASE_URL in .env.local
REMOTE_HOST=$(grep -oP 'STT_BASE_URL=http://\K[^:]+' "$ENV_FILE" 2>/dev/null || echo "")

if [ -z "$REMOTE_HOST" ] || [ "$REMOTE_HOST" = "100.x.x.x" ]; then
  echo "ERROR: Please update $ENV_FILE with your vast.ai Tailscale IP."
  echo "Replace 100.x.x.x with the actual IP (e.g., 100.64.0.5)"
  exit 1
fi

echo "========================================"
echo "  Starting local stack"
echo "========================================"
echo ""
echo "Remote STT host : $REMOTE_HOST"
echo "TTS             : Habibi-TTS (local container — port 8002)"
echo ""

# Check connectivity to remote STT only (TTS is now local)
echo "Checking remote STT connectivity..."
FAILED=0
if curl -sf --connect-timeout 5 "http://$REMOTE_HOST:11435/v1/models" > /dev/null 2>&1; then
  echo "  ✓ Whisper STT (port 11435) reachable"
else
  echo "  ✗ Whisper STT (port 11435) NOT reachable"
  FAILED=1
fi

if [ "$FAILED" -eq 1 ]; then
  echo ""
  echo "WARNING: Remote STT endpoint is not reachable."
  echo "Make sure Whisper is running on the vast.ai machine (./start-remote.sh)"
  echo ""
  read -r -p "Continue anyway? (y/N): " choice
  case "$choice" in
    y|Y) echo "Continuing..." ;;
    *) exit 1 ;;
  esac
fi

echo ""
echo "Services:"
echo "  • Frontend      → http://localhost:3000"
echo "  • LiveKit       → ws://localhost:7880"
echo "  • Habibi-TTS    → http://localhost:8002/v1"
echo "  • Whisper STT   → $REMOTE_HOST:11435 (remote)"
echo ""

docker compose \
  -f docker-compose.local.yml \
  --env-file .env.local \
  up --build "$@"
