# WebRTC smoke-test server

Step 0 from `docs/web-rtc.md`. Minimal FastAPI + aiortc app that proves the
WebRTC transport works against a real browser before any Pipecat / STT /
LLM / TTS wiring lands.

What it does:

- Accepts an SDP offer at `POST /api/offer`, returns an answer.
- Relays the inbound audio track straight back out, so the mic loops to the
  browser speakers after one round trip.
- Accepts a `control` DataChannel. Logs every JSON message and echoes it as
  `{"type":"echo","original":<msg>}`. A `hello` message also triggers a
  synthesized `{"type":"ready","session_id":...}` reply.

What it does **not** do: STT, LLM, TTS, persona routing, idle timeout, auth,
TLS. Localhost / LAN dev only.

## Install

From the repo root, in the project venv (the same `.venv` `install_mac.sh`
sets up):

```
.venv/bin/pip install -r webrtc_smoke/requirements.txt
```

Or in a throwaway venv:

```
python3 -m venv webrtc_smoke/.venv
webrtc_smoke/.venv/bin/pip install -r webrtc_smoke/requirements.txt
```

## Run

Local only (mic works on `localhost` over plain HTTP):

```
make run-webrtc-smoke
# then open http://localhost:8080
```

LAN-reachable (other computers / phones on the same Wi-Fi):

```
make run-webrtc-smoke-lan
# then open https://<your-lan-ip>:8080 on the other device
```

`run-webrtc-smoke-lan` generates a self-signed cert under `webrtc_smoke/.certs/`
the first time it's run, with `subjectAltName` covering `localhost`,
`127.0.0.1`, and every IPv4 address bound to `en0…en3`. The cert is reused
on subsequent runs — delete `webrtc_smoke/.certs/` to regenerate (e.g.
after changing networks).

### First-time setup for LAN clients

1. **Accept the cert warning.** Self-signed certs aren't in any trust
   store. Chrome shows "Your connection is not private" — click *Advanced*
   → *Proceed*. Safari needs *Show Details* → *visit this website*. iOS
   Safari additionally requires installing the cert in *Settings → General
   → VPN & Device Management* before trusting it for HTTPS.
2. **Allow inbound connections.** macOS firewall (System Settings →
   Network → Firewall) may pop up the first time Python tries to listen on
   `0.0.0.0:8080`. Allow it.
3. **Same subnet.** Both machines must be on the same Wi-Fi / LAN.
   `192.168.x.y` ↔ `192.168.x.z` works; corporate VLANs that isolate
   clients do not.

Env vars: `WEBRTC_HOST` (default `0.0.0.0`), `WEBRTC_PORT` (default `8080`),
`WEBRTC_SSL_CERT` / `WEBRTC_SSL_KEY` (set by `run-webrtc-smoke-lan`).

## What to check

1. **Audio loopback.** Click **Talk**, grant mic access, speak. Your own
   voice plays back through the browser speakers within ~100 ms. Use
   headphones to avoid the obvious mic→speaker→mic feedback loop.
2. **DataChannel.** The log pane shows a `hello` going out and matching
   `echo` + `ready` messages coming back. Type any JSON into the input box
   to send an arbitrary control message.
3. **chrome://webrtc-internals.** Inbound and outbound audio packet
   counters should both be increasing. ICE state reaches `connected`.

## Why it stays in the repo

Once the real backend lands (`server.py` at the repo root, `web/` for the
real browser client), this directory stays as a regression harness. Any
time the transport itself is suspected — codec negotiation broken, ICE
failing on a new network, firmware unable to connect — point the suspect
client at `webrtc_smoke/` to isolate the transport from the pipeline.
