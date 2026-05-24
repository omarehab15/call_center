#!/usr/bin/env bash
# start-local.sh — Run this on your local machine
# Starts: LiveKit + Agent + Frontend
# Both STT (Whisper) and TTS (Habibi-TTS) connect to the remote vast.ai machine via Tailscale
set -euo pipefail

ENV_FILE=".env.local"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found."
  echo "Copy .env.local.example to .env.local and set your vast.ai Tailscale IP."
  exit 1
fi

# Extract remote host from STT_BASE_URL (both STT and TTS live on same remote machine)
REMOTE_HOST=$(grep -oP 'STT_BASE_URL=http://\K[^:]+' "$ENV_FILE" 2>/dev/null || echo "")

if [ -z "$REMOTE_HOST" ] || [ "$REMOTE_HOST" = "100.x.x.x" ]; then
  echo "ERROR: Please update $ENV_FILE with your vast.ai Tailscale IP."
  echo "Replace 100.x.x.x in both STT_BASE_URL and HABIBI_TTS_BASE_URL"
  exit 1
fi

echo "========================================"
echo "  Starting local stack"
echo "========================================"
echo ""
echo "Remote machine  : $REMOTE_HOST (vast.ai via Tailscale)"
echo "STT             : Whisper    → $REMOTE_HOST:11435"
echo "TTS             : Habibi-TTS → $REMOTE_HOST:8002"
echo "LLM             : Groq cloud"
echo ""

# Check connectivity to both remote services
echo "Checking remote connectivity..."
FAILED=0

if curl -sf --connect-timeout 5 "http://$REMOTE_HOST:11435/v1/models" > /dev/null 2>&1; then
  echo "  ✓ Whisper STT    (port 11435) reachable"
else
  echo "  ✗ Whisper STT    (port 11435) NOT reachable"
  FAILED=1
fi

if curl -sf --connect-timeout 5 "http://$REMOTE_HOST:8002/health" > /dev/null 2>&1; then
  echo "  ✓ Habibi-TTS     (port 8002)  reachable"
else
  echo "  ✗ Habibi-TTS     (port 8002)  NOT reachable"
  FAILED=1
fi

if [ "$FAILED" -eq 1 ]; then
  echo ""
  echo "WARNING: One or more remote services are not reachable."
  echo "Make sure the remote machine is running: ./start-remote.sh"
  echo ""
  read -r -p "Continue anyway? (y/N): " choice
  case "$choice" in
    y|Y) echo "Continuing..." ;;
    *) exit 1 ;;
  esac
fi

echo ""
echo "Local services:"
echo "  • Frontend   → http://localhost:3000"
echo "  • LiveKit    → ws://localhost:7880"
echo ""

docker compose \
  -f docker-compose.local.yml \
  --env-file .env.local \
  up --build "$@"
