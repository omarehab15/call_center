# Project Handover: Local Voice AI — Saudi Arabic Call Center Agent

## What This Project Is

A fully open-source, real-time voice AI call center agent built for Saudi Arabic dialect.
Built on top of [LiveKit Agents](https://docs.livekit.io/agents/), running across two machines:

- **Local machine** — LiveKit server, LiveKit agent, Next.js frontend
- **vast.ai GPU machine** — all inference: STT (Whisper), LLM (llama.cpp), TTS (lahgtna-chatterbox)

The two machines are connected via **Tailscale**. The agent on the local machine calls the GPU models over HTTP through the Tailscale tunnel.

---

## Architecture

```
┌─────────────────────────────┐      ┌──────────────────────────────────────┐
│     LOCAL MACHINE           │      │   VAST.AI MACHINE (100.x.x.x)       │
│                             │      │                                      │
│  Browser ←──WebRTC──→ LiveKit│      │   lahgtna-chatterbox TTS  (:8880)  │
│                    (:7880)  │      │   Whisper STT             (:11435)  │
│  Frontend          (:3000)  │      │   llama.cpp LLM           (:11436)  │
│                             │      │                                      │
│  Agent ──── HTTP/Tailscale ──────→ │   (all OpenAI-compatible APIs)      │
└─────────────────────────────┘      └──────────────────────────────────────┘
```

### Pipeline per voice turn
```
User speaks
  → Whisper STT (dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8)
  → gemma-4-26B LLM (via llama.cpp)
  → lahgtna-chatterbox TTS (oddadmix/lahgtna-chatterbox-v1)
  → Audio back to user via WebRTC
```

---

## Repository Structure

```
.
├── frontend/                        # Next.js UI client
├── inference/
│   ├── lahgtna/                     # ← NEW: lahgtna-chatterbox TTS service
│   │   ├── Dockerfile
│   │   └── server.py                # FastAPI, OpenAI-compatible /v1/audio/speech
│   ├── whisper/                     # STT (VoxBox wrapper)
│   └── llama/                       # LLM model cache volume
├── livekit/                         # LiveKit server config
├── livekit_agent/
│   └── src/
│       └── agent.py                 # Main agent — STT→LLM→TTS pipeline
├── docker-compose.yml               # Base local stack
├── docker-compose.gpu.yml           # GPU overlay for local stack
├── docker-compose.local.yml         # Split deployment: local side
├── docker-compose.remote.yml        # Split deployment: vast.ai side
├── docker-compose.remote-gpu.yml    # GPU overlay for remote side
├── .env.local.example               # Local env template
├── .env.remote.example              # Remote env template
├── start-local.sh                   # Helper: start local stack
└── start-remote.sh                  # Helper: start remote stack
```

---

## Key Files To Know

### `livekit_agent/src/agent.py`

The agent brain. Defines the Arabic voice assistant persona, STT/LLM/TTS connections, and tool functions.

Key env vars it reads:
- `STT_PROVIDER` — `whisper` or `nemotron`
- `STT_BASE_URL`, `STT_MODEL`, `STT_API_KEY`
- `LLAMA_BASE_URL`, `LLAMA_MODEL`
- `XTTS_BASE_URL` — points at lahgtna despite the name (kept for backward compat)

The agent instructs the LLM to respond in **Saudi Najdi dialect** (`لهجة سعودية نجدية`).

### `inference/lahgtna/server.py`

FastAPI wrapper around lahgtna-chatterbox. OpenAI-compatible `/v1/audio/speech` endpoint.

Critical implementation details:
- Clones `https://github.com/Oddadmix/lahgtna-chatterbox.git` at build time (NOT PyPI `chatterbox-tts`)
- Import path: `from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS`
- Loads via `ChatterboxMultilingualTTS.from_checkpoint(ckpt_dir, device)` — NOT `from_pretrained()`
- Model downloaded from HF: `oddadmix/lahgtna-chatterbox-v1`
- **`language_id="sa"`** — Saudi Arabic. NOT `"ar"` (Egyptian Arabic in base model)
- **`repetition_penalty=2.0`** — critical, prevents syllable looping artifacts
- Returns MP3 by default (matches what LiveKit's `openai.TTS` plugin requests)
- Voice cloning: pass reference WAV path via `audio_prompt_path`

### `inference/xtts/voices/`

Voice reference WAV files. Mounted read-only into the lahgtna container at `/voices`.
The server resolves `voice="fahad"` → `Fasseh-fahad.wav` by substring filename match.

**To add a voice:** drop a `.wav` file here. No rebuild needed.

---

## How To Run

### Remote (vast.ai) — GPU models

```bash
# SSH into vast.ai machine
git pull
docker compose \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-gpu.yml \
  --env-file .env.remote \
  up --build
```

Services and ports:
| Service | Port | What |
|---|---|---|
| `lahgtna` | 8880 | TTS (lahgtna-chatterbox) |
| `whisper` | 11435 | STT (faster-whisper) |
| `llama_cpp` | 11436 | LLM (gemma-4-26B) |

### Local machine

```bash
# Edit .env.local — replace 100.x.x.x with vast.ai Tailscale IP
./start-local.sh
# or manually:
docker compose -f docker-compose.local.yml --env-file .env.local up --build
```

Then open `http://localhost:3000`.

### Rebuild only lahgtna (most common)

```bash
docker compose \
  -f docker-compose.remote.yml \
  -f docker-compose.remote-gpu.yml \
  --env-file .env.remote \
  up --build lahgtna
```

---

## Environment Variables

### `.env.local` (local machine)

```env
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

STT_PROVIDER=whisper
STT_BASE_URL=http://100.x.x.x:11435/v1
STT_MODEL=dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8
STT_API_KEY=no-key-needed

LLAMA_BASE_URL=http://100.x.x.x:11436/v1
LLAMA_MODEL=gemma-4-26b

XTTS_BASE_URL=http://100.x.x.x:8880/v1   # points at lahgtna
```

### `.env.remote` (vast.ai machine)

```env
LLAMA_HF_REPO=unsloth/gemma-4-26B-A4B-it-GGUF
LLAMA_HF_FILE=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
LLAMA_MODEL_ALIAS=gemma-4-26b
LLAMA_HOST_PORT=11436
LLAMA_CTX_SIZE=16384

VOXBOX_HF_REPO_ID=dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8

LAHGTNA_HF_REPO=oddadmix/lahgtna-chatterbox-v1
LAHGTNA_HOST_PORT=8880
LAHGTNA_DEVICE=cuda
LAHGTNA_EXAGGERATION=0.5
LAHGTNA_CFG_WEIGHT=0.5
LAHGTNA_TEMPERATURE=0.8
LAHGTNA_REPETITION_PENALTY=2.0   # do NOT lower below 1.8
```

---

## TTS: lahgtna-chatterbox — Things That Burned Us

This section exists so the next engineer doesn't repeat the same debugging.

### 1. It's a fork, not a pip package
The PyPI package `chatterbox-tts` is the base ResembleAI model. It does NOT support `language_id="sa"` or `from_checkpoint()`. You must clone `https://github.com/Oddadmix/lahgtna-chatterbox.git` and set `PYTHONPATH=/app`.

### 2. language_id must be "sa", not "ar"
`"ar"` is Egyptian Arabic in the base multilingual model. `"sa"` is Saudi Arabic added by the lahgtna fork. Using `"ar"` either raises `ValueError: Unsupported language_id` or produces Egyptian-accented output.

### 3. repetition_penalty=2.0 is not optional
Values below ~1.8 cause the model to loop syllables mid-sentence. The Colab notebook hardcodes 2. Keep it there.

### 4. Never split text before passing to generate()
Chatterbox needs full sentence context for natural prosody. Splitting on Arabic commas `،` and generating per-chunk causes interruptions, mismatched intonation, and chunks getting dropped. Pass the full response text as one string.

### 5. LiveKit expects MP3, not WAV
LiveKit's `openai.TTS` plugin sends `response_format: "mp3"` in the request body. If the server ignores this and returns WAV, the LiveKit audio decoder initializes an MP3 decoder, receives WAV bytes, fails, and throws `ValueError: I/O operation on closed file`. The server respects `response_format` and defaults to `"mp3"`.

### 6. from_checkpoint(), not from_pretrained()
The lahgtna model uses a non-standard loader. `from_pretrained()` either fails or loads the base model without the Saudi Arabic fine-tune weights. Always use `from_checkpoint(ckpt_dir + "/", device)` where `ckpt_dir` is the result of `snapshot_download()` with the specific `allow_patterns` list.

---

## Known Issues / Next Steps

### Latency (not yet optimized)

Observed latency breakdown:
- **Whisper STT**: ~3 seconds — too slow, needs investigation
- **LLM first token**: ~1.5 seconds — acceptable but improvable
- **TTS (lahgtna)**: ~1–2 seconds — chatterbox is a diffusion model, not autoregressive

**Whisper investigation starting points:**
- Check if it's cold-starting each request (model should stay warm in memory)
- Try `faster-whisper-small` or `faster-whisper-medium` for lower latency at the cost of accuracy
- Check if `VOXBOX_DEVICE=cuda` is actually being used vs falling back to CPU
- Consider switching STT to a streaming model (e.g. Whisper streaming via `whisper-live`)

**LLM starting points:**
- `LLAMA_ARG_FLASH_ATTN=1` is already set (good)
- `LLAMA_N_GPU_LAYERS=-1` offloads all layers to GPU (good)
- Consider speculative decoding if the model supports it
- Consider a smaller model for faster first token

**TTS starting points:**
- Sentence-level streaming: split on `.` `!` `?` `؟` only (not `،`), generate per sentence, stream audio chunks as they complete. This reduces perceived latency by getting audio out before the full response is synthesized.
- This requires a custom LiveKit TTS plugin (subclassing `livekit.agents.tts.TTS`) rather than using `openai.TTS` which waits for the full response.

### Adding voices
Drop a clean 5–15 second WAV file of the speaker into `inference/xtts/voices/`. No rebuild needed. Use `voice="filename-without-extension"` in `agent.py`.

### Switching LLM models
Change `LLAMA_HF_REPO` and `LLAMA_HF_FILE` in `.env.remote` and `LLAMA_MODEL` / `LLAMA_MODEL_ALIAS` to match. The llama.cpp container auto-downloads from HuggingFace on first boot.

---

## VRAM Budget (vast.ai machine)

| Model | Approx VRAM |
|---|---|
| lahgtna-chatterbox | ~4–5 GB |
| faster-whisper-large-v3-turbo | ~3 GB |
| gemma-4-26B Q4_K_M | ~18–20 GB |
| **Total** | **~25–28 GB** |

Minimum recommended GPU: RTX 3090 (24GB) with some CPU offload for LLM, or A6000/A100 (48GB) for full GPU offload.

If on a smaller GPU, reduce `LLAMA_N_GPU_LAYERS` to a fixed number (e.g. 20) to offload remaining layers to CPU RAM.

---

## LiveKit Agent Details

- Framework: `livekit-agents~=1.3` with `openai`, `silero`, `turn-detector` plugins
- VAD: Silero VAD (pre-warmed in `prewarm()`)
- Turn detection: `MultilingualModel` (handles Arabic)
- `preemptive_generation=True` — starts TTS generation before LLM finishes, reduces latency
- The agent has one registered tool: `multiply_numbers` (placeholder — replace with real call center tools)

---

## Useful Commands

```bash
# Watch lahgtna logs
docker compose -f docker-compose.remote.yml logs -f lahgtna

# Test TTS endpoint directly
curl -X POST http://100.x.x.x:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "مرحبا، كيف يمكنني مساعدتك؟", "voice": "fahad"}' \
  --output test.mp3

# Test STT endpoint
curl http://100.x.x.x:11435/v1/models

# Test LLM endpoint
curl http://100.x.x.x:11436/v1/models

# Check all remote services health
for port in 8880 11435 11436; do
  echo -n "Port $port: "
  curl -sf http://100.x.x.x:$port/health || \
  curl -sf http://100.x.x.x:$port/v1/models || \
  echo "not reachable"
done
```