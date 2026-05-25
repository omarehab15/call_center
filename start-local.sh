#!/usr/bin/env bash
# start-local.sh — Run this on your local machine
# Starts self-hosted LiveKit stack: Redis + LiveKit + SIP + agent + frontend
# The agent connects to remote STT on the vast.ai machine via Tailscale
set -euo pipefail

# Load .env.local to read STT endpoint and runtime config
ENV_FILE=".env.local"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found."
  echo "Copy .env.local.example to .env.local and set your vast.ai Tailscale STT URL."
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [ -z "${STT_BASE_URL:-}" ]; then
  echo "ERROR: STT_BASE_URL is missing in $ENV_FILE."
  echo "Set it to your vast.ai/Tailscale STT endpoint (example: http://100.64.0.5:11435/v1)."
  exit 1
fi

if [[ "$STT_BASE_URL" == *"100.x.x.x"* ]]; then
  echo "ERROR: Please update STT_BASE_URL in $ENV_FILE with your actual vast.ai Tailscale IP."
  echo "Example: STT_BASE_URL=http://100.64.0.5:11435/v1"
  exit 1
fi

# Normalize STT models endpoint to check connectivity
STT_MODELS_ENDPOINT="${STT_BASE_URL%/}/models"

echo "========================================"
echo "  Starting local stack"
echo "========================================"
echo ""
echo "Remote STT endpoint: $STT_BASE_URL"
echo ""

echo "Checking remote STT connectivity..."
FAILED=0
if curl -sf --connect-timeout 5 "$STT_MODELS_ENDPOINT" > /dev/null 2>&1; then
  echo "  ✓ STT endpoint reachable ($STT_MODELS_ENDPOINT)"
else
  echo "  ✗ STT endpoint NOT reachable ($STT_MODELS_ENDPOINT)"
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

COMPOSE_FILES=(-f docker-compose.local.yml)
if [ "${SIP_TEST_PROFILE:-}" = "vast" ]; then
  COMPOSE_FILES+=(-f docker-compose.local.vast-sip.yml)
  echo ""
  echo "SIP test profile: vast (reduced RTP range for limited port budgets)"
  echo "  • RTP range → ${SIP_TEST_RTP_PORT_START:-12000}-${SIP_TEST_RTP_PORT_END:-12031}"
fi

echo ""
echo "Services:"
echo "  • Frontend      → http://localhost:3000"
echo "  • LiveKit       → ws://localhost:7880"
echo "  • SIP signaling → ${SIP_PUBLIC_HOST:-<set SIP_PUBLIC_HOST>}:${SIP_SIGNALING_PORT:-5060}"
echo "  • Agent         → connecting to remote STT + Groq LLM/TTS"
echo ""

docker compose \
  "${COMPOSE_FILES[@]}" \
  --env-file .env.local \
  up --build "$@"
