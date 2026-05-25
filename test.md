# Local Voice AI + Self-hosted SIP (Vast.ai + Local) Runbook

This is the current setup flow used in this project:

- Remote machine (Vast.ai): Whisper STT only
- Local machine: Redis + LiveKit + LiveKit SIP + Agent + Frontend
- Telephony: Self-hosted LiveKit SIP
- Optional test mode: "Vast-friendly SIP" overlay with a smaller RTP range

---

## 1) Remote (Vast.ai) machine setup

### 1.1 Prerequisites

- Ubuntu VM on Vast.ai (not Docker template)
- Docker + Docker Compose
- NVIDIA driver + NVIDIA container toolkit
- Tailscale connected to the same tailnet as local machine

### 1.2 Start remote STT

```bash
git clone <your-repo-url>
cd local_call_center
cp .env.remote.example .env.remote
chmod +x start-remote.sh
./start-remote.sh
```

### 1.3 Verify remote STT

```bash
curl http://localhost:11435/v1/models
tailscale ip -4
```

Save the Tailscale IP (example: `100.88.77.66`).

---

## 2) Local machine setup

### 2.1 Prepare `.env.local`

```bash
cp .env.local.example .env.local
```

Edit `.env.local`:

- Set STT endpoint to Vast.ai Tailscale IP:
  - `STT_BASE_URL=http://100.88.77.66:11435/v1`
- Set Groq key/model for LLM/TTS:
  - `GROQ_API_KEY=...`
  - `GROQ_LLM_MODEL=...`
- Set SIP values:
  - `SIP_PROVIDER_NUMBER=+E164_PROVIDER_NUMBER`
  - `SIP_OUTBOUND_HOST=sip.provider.com` (must be provider host/domain, not phone number)
  - `SIP_DESTINATION_COUNTRY=US` (or your destination country code)
  - `SIP_AUTH_USERNAME` / `SIP_AUTH_PASSWORD` (if provider requires auth)
  - `SIP_PUBLIC_HOST=<public-dns-or-ip>`

---

## 3) Choose SIP port profile

### Option A (default): full RTP range

Uses `10000-20000` UDP RTP range from `docker-compose.local.yml`.

### Option B (testing): Vast-friendly smaller RTP range

Use overlay file:

- `docker-compose.local.vast-sip.yml`
- RTP range defaults to `12000-12031`

You can tune it in `.env.local`:

- `SIP_TEST_RTP_PORT_START=12000`
- `SIP_TEST_RTP_PORT_END=12031`

---

## 4) Start local stack

### 4.1 Normal mode

```bash
./start-local.sh
```

### 4.2 Vast-friendly SIP test mode

```bash
SIP_TEST_PROFILE=vast ./start-local.sh
```

PowerShell equivalent:

```powershell
$env:SIP_TEST_PROFILE="vast"
bash ./start-local.sh
```

What starts locally:

- Redis
- LiveKit Server
- LiveKit SIP
- LiveKit Agent
- Frontend (`http://localhost:3000`)

---

## 5) Provision SIP resources in LiveKit

In another terminal:

```bash
cd livekit_agent
uv run python sip_setup.py setup
```

Optional outbound call test:

```bash
uv run python sip_setup.py setup --call-now --call-to +15559876543
```

---

## 6) SIP provider origination URI

Point provider inbound/origination URI to your self-hosted SIP endpoint:

- UDP: `sip:<SIP_PUBLIC_HOST>:<SIP_SIGNALING_PORT>;transport=udp`
- TCP: `sip:<SIP_PUBLIC_HOST>:<SIP_SIGNALING_PORT>;transport=tcp`
- TLS: `sip:<SIP_PUBLIC_HOST>:<SIP_TLS_PORT>;transport=tls`

Examples:

- `sip:sip.example.com:5060;transport=udp`
- `sip:203.0.113.10:5060;transport=tcp`

---

## 7) Public reachability checklist

Provider must reach these ports on `SIP_PUBLIC_HOST`:

- SIP signaling: `5060` (UDP/TCP)
- Optional SIP TLS: `5061` (TCP)
- RTP media:
  - default: `10000-20000` UDP
  - Vast-friendly test overlay: `12000-12031` UDP (or your custom test range)

---

## 8) Troubleshooting

### Remote STT unreachable

```bash
curl http://<TAILSCALE_IP>:11435/v1/models
```

If it fails:

- confirm `tailscale status` on both machines
- confirm `./start-remote.sh` is running on Vast.ai

### SIP trunks created but no audio

- verify RTP UDP ports are truly open and forwarded on public host
- confirm provider is using the same transport/port as your origination URI
- in test mode, ensure provider/firewall allows the smaller RTP range you configured

### Outbound trunk creation fails

- verify `SIP_OUTBOUND_HOST` is a SIP host/domain (not `+phone_number`)

