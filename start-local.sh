#!/usr/bin/env bash
# start-local.sh — Run this on your local machine
# Starts local web stack: Redis + LiveKit + agent + frontend
# SIP is optional and can be enabled via SIP_ENABLED=true.
#
# FIX: Now starts in detached mode (-d) by default so Ctrl+C does NOT
#      kill the containers or corrupt your terminal.
#
# To watch live logs after starting:
#   docker compose -f docker-compose.local.yml logs -f
#
# To stop cleanly:
#   docker compose -f docker-compose.local.yml --profile sip down
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
COMPOSE_ARGS=()
if [ "${SIP_ENABLED:-false}" = "true" ]; then
  COMPOSE_ARGS+=(--profile sip)
fi

if [ "${SIP_TEST_PROFILE:-}" = "vast" ]; then
  COMPOSE_FILES+=(-f docker-compose.local.vast-sip.yml)
  COMPOSE_ARGS+=(--profile sip)
  echo ""
  echo "SIP test profile: vast (reduced RTP range for limited port budgets)"
  echo "  • RTP range → ${SIP_TEST_RTP_PORT_START:-12000}-${SIP_TEST_RTP_PORT_END:-12031}"
fi

compose_cmd() {
  docker compose \
    "${COMPOSE_FILES[@]}" \
    "${COMPOSE_ARGS[@]}" \
    --env-file .env.local \
    "$@"
}

build_images_if_needed() {
  local build_mode="${BUILD_IMAGES:-missing}"
  local build_mode_normalized="${build_mode,,}"
  local build_progress="${BUILD_PROGRESS:-plain}"
  local build_services=(livekit_agent frontend)
  local missing_services=()
  local service
  local image_id

  case "$build_mode_normalized" in
    always|true|1|yes)
      echo "Building images..."
      compose_cmd build --progress "$build_progress" "${build_services[@]}"
      ;;
    missing|auto)
      for service in "${build_services[@]}"; do
        image_id="$(compose_cmd images -q "$service" 2>/dev/null || true)"
        if [ -z "$image_id" ]; then
          missing_services+=("$service")
        fi
      done

      if [ "${#missing_services[@]}" -gt 0 ]; then
        echo "Building missing images: ${missing_services[*]}"
        compose_cmd build --progress "$build_progress" "${missing_services[@]}"
      else
        echo "Images already exist; skipping build."
        echo "  Set BUILD_IMAGES=always to force a rebuild."
        echo "  Set BUILD_PROGRESS=plain to show full build steps."
      fi
      ;;
    never|false|0|no|skip)
      echo "Skipping image build (BUILD_IMAGES=$build_mode)."
      ;;
    *)
      echo "ERROR: BUILD_IMAGES must be one of: always, missing, never."
      exit 1
      ;;
  esac
}

rag_ingest_signature() {
  printf 'provider=%s\nmodel=%s\ncollection=%s\nknowledge_dir=%s\nchroma_dir=%s\nchunk_size=%s\nchunk_overlap=%s\n' \
    "${RAG_EMBEDDING_PROVIDER:-chroma}" \
    "${RAG_EMBEDDING_MODEL:-}" \
    "${RAG_COLLECTION_NAME:-}" \
    "${RAG_KNOWLEDGE_DIR_HOST:-./livekit_agent/knowledge_base}" \
    "${RAG_CHROMA_PATH_HOST:-./rag/chroma}" \
    "${RAG_CHUNK_SIZE:-900}" \
    "${RAG_CHUNK_OVERLAP:-150}"
}

should_ingest_rag() {
  local ingest_mode="${RAG_INGEST_MODE:-changed}"
  local ingest_mode_normalized="${ingest_mode,,}"
  local knowledge_dir="${RAG_KNOWLEDGE_DIR_HOST:-./livekit_agent/knowledge_base}"
  local chroma_dir="${RAG_CHROMA_PATH_HOST:-./rag/chroma}"
  local marker_path="${RAG_INGEST_MARKER:-./rag/.last_ingest}"
  local current_signature
  current_signature="$(rag_ingest_signature)"

  case "$ingest_mode_normalized" in
    always|true|1|yes)
      return 0
      ;;
    never|false|0|no|skip)
      return 1
      ;;
    changed|auto)
      if [ ! -f "$marker_path" ]; then
        return 0
      fi
      if [ "$(cat "$marker_path" 2>/dev/null || true)" != "$current_signature" ]; then
        return 0
      fi
      if [ ! -d "$chroma_dir" ] || [ -z "$(find "$chroma_dir" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        return 0
      fi
      if [ ! -d "$knowledge_dir" ]; then
        echo "Knowledge directory not found on host: $knowledge_dir"
        echo "  Set RAG_KNOWLEDGE_DIR_HOST if you use a custom mounted directory."
        return 1
      fi
      if find "$knowledge_dir" -type f \( \
        -name '*.md' -o \
        -name '*.txt' -o \
        -name '*.html' -o \
        -name '*.htm' -o \
        -name '*.json' -o \
        -name '*.csv' \
      \) -newer "$marker_path" -print -quit | grep -q .; then
        return 0
      fi
      return 1
      ;;
    *)
      echo "ERROR: RAG_INGEST_MODE must be one of: always, changed, never."
      exit 1
      ;;
  esac
}

echo ""
echo "Services:"
echo "  • Frontend      → http://localhost:3000"
echo "  • LiveKit       → ws://localhost:7880"
echo "  • Agent         → connecting to remote STT + Groq LLM/TTS"
if [ "${SIP_ENABLED:-false}" = "true" ] || [ "${SIP_TEST_PROFILE:-}" = "vast" ]; then
  echo "  • SIP signaling → ${SIP_PUBLIC_HOST:-<set SIP_PUBLIC_HOST>}:${SIP_SIGNALING_PORT:-5060}"
else
  echo "  • SIP           → disabled (set SIP_ENABLED=true to enable)"
fi
if [ "${RAG_ENABLED:-true}" = "true" ]; then
  echo "  • RAG           → enabled (ingest mode: ${RAG_INGEST_MODE:-changed})"
else
  echo "  • RAG           → disabled"
fi
echo "  • Build mode    → ${BUILD_IMAGES:-missing}"
echo "  • Build logs    → ${BUILD_PROGRESS:-plain}"
echo ""

echo "Preparing containers..."
build_images_if_needed

# Ingest RAG knowledge base if enabled
if [ "${RAG_ENABLED:-true}" = "true" ]; then
  if should_ingest_rag; then
    echo ""
    echo "Ingesting knowledge base into Chroma..."
    compose_cmd run --rm livekit_agent uv run python src/ingest_rag.py
    mkdir -p "$(dirname "${RAG_INGEST_MARKER:-./rag/.last_ingest}")"
    rag_ingest_signature > "${RAG_INGEST_MARKER:-./rag/.last_ingest}"
    echo ""
  else
    echo ""
    echo "RAG knowledge base unchanged; skipping ingest."
    echo "  Set RAG_INGEST_MODE=always to force ingestion."
    echo ""
  fi
fi

# ── FIX: Start detached so Ctrl+C doesn't kill containers or break terminal ──
echo "Starting containers in detached mode (background)..."
compose_cmd up --build -d "$@"

echo ""
echo "========================================"
echo "  Stack is running in the background"
echo "========================================"
echo ""
echo "  Watch logs:   docker compose -f docker-compose.local.yml logs -f"
echo "  Stop cleanly: docker compose -f docker-compose.local.yml$([ "${SIP_ENABLED:-false}" = "true" ] && echo " --profile sip") down"
echo ""
