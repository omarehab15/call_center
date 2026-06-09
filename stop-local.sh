#!/usr/bin/env bash
# stop-local.sh — Stop local containers without deleting them.
#
# This keeps containers, networks, and cached images in place so the next
# ./start-local.sh can resume quickly.
set -euo pipefail

ENV_FILE=".env.local"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found."
  echo "Copy .env.local.example to .env.local first."
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

COMPOSE_FILES=(-f docker-compose.local.yml)
COMPOSE_ARGS=()
if [ "${SIP_ENABLED:-false}" = "true" ]; then
  COMPOSE_ARGS+=(--profile sip)
fi

if [ "${SIP_TEST_PROFILE:-}" = "vast" ]; then
  COMPOSE_FILES+=(-f docker-compose.local.vast-sip.yml)
  COMPOSE_ARGS+=(--profile sip)
fi

docker compose \
  "${COMPOSE_FILES[@]}" \
  "${COMPOSE_ARGS[@]}" \
  --env-file "$ENV_FILE" \
  stop "$@"

echo ""
echo "Stack stopped. Containers were not deleted."
echo "Start again with: ./start-local.sh"
