#!/usr/bin/env bash
# start-local.sh — Run this on your local machine
# Starts LiveKit server, agent, and frontend
# The agent connects to remote model APIs on the vast.ai machine via Tailscale
set -euo pipefail

# Load the .env.local to read REMOTE_HOST-related URLs
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
echo "Remote models host: $REMOTE_HOST"
echo ""

# Check connectivity to remote models
echo "Checking remote model connectivity..."
FAILED=0
for endpoint in "$REMOTE_HOST:8880/v1/models" "$REMOTE_HOST:11435/v1/models" "$REMOTE_HOST:11436/v1/models"; do
  PORT=$(echo "$endpoint" | grep -oP ':\K[0-9]+')
  if curl -sf --connect-timeout 5 "http://$endpoint" > /dev/null 2>&1; then
    echo "  ✓ Port $PORT reachable"
  else
    echo "  ✗ Port $PORT NOT reachable"
    FAILED=1
  fi
done

if [ "$FAILED" -eq 1 ]; then
  echo ""
  echo "WARNING: Some remote model endpoints are not reachable."
  echo "Make sure the models are running on the vast.ai machine (./start-remote.sh)"
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
echo "  • Agent          → connecting to remote models"
echo ""

docker compose \
  -f docker-compose.local.yml \
  --env-file .env.local \
  up --build "$@"
