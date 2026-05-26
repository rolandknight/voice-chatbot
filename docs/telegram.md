# Telegram Bridge

## Context

We want an iOS app that lets you **send text messages to the chatbot** and **send/receive voice calls** with it. The chatbot today is a Pipecat-based local pipeline (`app.py`) that uses a custom WebRTC signaling protocol (HTTP `POST /api/offer` + SDP, Opus 16 kHz uplink / 24 kHz downlink, JSON over DataChannel). That custom protocol is what the ESP32-S3-BOX-3 firmware speaks — **no third-party iOS app speaks it**, so the realistic question isn't "which app understands my server" but "which client ecosystem do I bridge into."

Decisions taken up front:

- **Client:** zero iOS development — must be an existing App Store app
- **Calling/messaging surface:** wrap the bot as a **Telegram bot**
- **Messaging:** text chat with the bot

The recommended iOS app is therefore **Telegram for iOS** (`apps.apple.com/app/telegram-messenger/id686449807`). All the real work is server-side: a Telegram bot bridge that hands user input to the existing Pipecat pipeline pieces and ships responses back.

---

## Why Telegram (vs. other "zero-build" options)

| Option   | Text             | Voice messages          | Real-time calls                                          | Notes                                                                                |
|----------|------------------|-------------------------|----------------------------------------------------------|--------------------------------------------------------------------------------------|
| Telegram | ✅ Bot API        | ✅ OGG/Opus, native UI   | ⚠️ Only via userbot in a group voice chat (pytgcalls)    | Best bot API of any consumer chat app; native voice notes are first-class.           |
| Signal   | ❌ no bot API     | ❌                       | ❌                                                        | No third-party automation. Eliminated.                                               |
| WhatsApp | ⚠️ Cloud API, template-message rules, paid | ❌ no bot voice notes | ❌                              | Heavy compliance overhead for personal use.                                          |
| iMessage | ❌ no bot API     | ❌                       | ❌                                                        | Apple does not allow bot accounts.                                                   |
| Discord  | ✅                | ❌ (no PTT voice messages) | ✅ via voice channels                                  | DM voice notes aren't first-class; voice is channel-based.                           |

Telegram wins on the combination of: (1) open bot API, (2) native push-to-talk voice messages that feel conversational, (3) iOS app polish, (4) optional escape hatch to real-time audio via pytgcalls later.

---

## Important constraint to know up front

The **official Telegram Bot API does not expose real-time 1-on-1 voice calls.** What it *does* expose:

- `sendMessage` / `sendVoice` — text and OGG/Opus voice notes ("push to talk" style)
- `sendChatAction: record_voice` — typing-style indicator while the bot generates a reply
- Webhooks / long-polling for incoming text + voice + voice notes

If you want **real-time streaming voice calls**, the path is:

- Run a **userbot** (a regular Telegram user account, not a bot) with `pytgcalls` joining a **group voice chat**. Bot accounts cannot join voice chats.
- This works, is the same mechanism every "music bot" uses, but requires a phone number, an MTProto session, and lives in a grey area for ToS-strict deployments.

**Recommendation: ship phase 1 as text + voice-notes only.** It feels conversational on iOS (hold-to-talk in Telegram is muscle-memory), is fully ToS-clean, and reuses the existing pipeline almost as-is. Phase 2 (real-time calls via userbot + pytgcalls) is an optional follow-up.

---

## Server-side architecture

The Pipecat pipeline is already transport-agnostic — `docs/web-rtc.md:107-113` explicitly calls out that `BackendRouter`, `PersonaTagRouter`, `SkillFilterProcessor`, the skills, the Whisper MLX STT block, and Chatterbox TTS are all reusable across transports. The Telegram bridge is a **new frontend**, not a rewrite.

### High-level flow

```
Telegram iOS ── text msg ────────► aiogram handler ─► BackendRouter ─► LLM ─► text reply ─► sendMessage
Telegram iOS ── voice note (OGG) ► aiogram handler ─► ffmpeg→PCM ─► Whisper MLX ─► BackendRouter ─► LLM ─► PersonaTagRouter ─► TTS (PCM) ─► ffmpeg→OGG/Opus ─► sendVoice
```

### Components to add

1. **`telegram_bot.py`** — new top-level entry point (sibling of `app.py`). Uses `aiogram` (modern async, matches Pipecat's asyncio model; consensus winner over `python-telegram-bot` for new projects).
2. **A non-Pipecat invocation path for the existing services.** `app.py` wires services into a Pipecat `Pipeline` for the live mic loop. The Telegram bridge needs to call the *same service objects* (`WhisperSTTServiceMLX`, the Ollama/Anthropic LLM clients, the persona router, the TTS service) directly on request/response boundaries — one Telegram message = one round-trip, not a continuous stream. Factor the service construction out of `app.py` into a small `services.py` module so both entry points share it.
3. **Audio format conversion.** Telegram voice notes are OGG/Opus; Whisper MLX wants 16 kHz PCM. TTS output is 24 kHz PCM; `sendVoice` requires OGG/Opus. Use `ffmpeg` subprocess (already a likely dependency) for both directions. Keep it as a thin `audio_codec.py` helper.
4. **Auth / allowlist.** Telegram bots are reachable by anyone who finds their handle. Gate handlers on a `TELEGRAM_ALLOWED_USER_IDS` env var (comma-separated `chat.id` values). Drop anything else silently. This is the only "auth" needed for personal use.
5. **Wake-word bypass.** In voice mode the chatbot uses Silero VAD + microWakeWord to gate the LLM. In Telegram every voice note is an explicit invocation — bypass wake-phrase logic entirely. The text message path bypasses it naturally.
6. **Backend selection.** No wake-phrase to disambiguate Ollama vs Claude on Telegram. Use a `/claude` and `/ollama` slash command (aiogram supports these directly) that sets a per-chat backend preference; default to Ollama. Mirrors the DataChannel selection pattern from `docs/web-rtc.md:91-98`.
7. **Persona tags in text replies.** `PersonaTagRouter` reads inline `[persona:name]` tags from LLM output to switch TTS voice. For text replies, **strip** those tags before `sendMessage` (they're meaningless without TTS). For voice replies, parse them and pass to the TTS service as usual.

### Files to create

- `telegram_bot.py` — aiogram entry point with `/start`, `/claude`, `/ollama`, text handler, voice-note handler
- `services.py` — extracted service factory (`build_stt()`, `build_llm(backend)`, `build_tts()`, `build_persona_router()`) consumed by both `app.py` and `telegram_bot.py`
- `audio_codec.py` — `ogg_opus_to_pcm16k(bytes) -> bytes` and `pcm24k_to_ogg_opus(bytes) -> bytes` (ffmpeg subprocess wrappers)
- `run_telegram.sh` — sibling of `run.sh`; loads `.env`, runs `python telegram_bot.py`
- `.env.example` additions: `TELEGRAM_BOT_TOKEN=`, `TELEGRAM_ALLOWED_USER_IDS=`

### Files to modify

- `app.py` — extract service construction into `services.py`; have `app.py` import from it. No behavior change for the existing mic loop.
- `README.md` — add a "Telegram bridge" section pointing to `run_telegram.sh` and the BotFather setup steps.
- `requirements.txt` (or pyproject) — add `aiogram` (~3.x).

### Slash commands to register with BotFather

- `/start` — greeting + persona list
- `/claude` — route this chat's next turns to Anthropic
- `/ollama` — route this chat's next turns to local Ollama (default)
- `/persona <name>` — pin a specific persona for voice replies (optional polish)

---

## Optional phase 2: real-time voice calls

If hold-to-talk later feels limiting, add **pytgcalls** in a private group voice chat:

- Create a private Telegram group with just you + the userbot
- Userbot (Pyrogram session, your own phone number) joins voice chat with pytgcalls
- pytgcalls exposes raw PCM streams in both directions — wire them into a *Pipecat* `Pipeline` similar to `app.py`'s mic loop, with the userbot's PCM streams replacing the local Jabra
- This is the path called out in `docs/comparison.md:87` as the LiveKit-Agents SIP equivalent for Telegram

**Don't build this in phase 1.** Validate the text + voice-notes UX first.

---

## Verification

Phase 1 is done when:

1. `python telegram_bot.py` starts cleanly, registers commands with BotFather, and logs the bot's username.
2. From the **Telegram iOS app** on a phone whose user-id is in `TELEGRAM_ALLOWED_USER_IDS`:
   - Send `/start` → receive greeting reply
   - Send text "what's the weather in Columbus?" → receive a coherent text reply (proves text → BackendRouter → LLM → text out works, and a skill round-trip works)
   - Send `/claude`, then send text → reply comes from Anthropic (proves backend switching)
   - Hold-to-talk a voice note "set a five minute timer" → bot replies with an OGG/Opus voice note in a recognizable persona voice within ~3-5 s (proves OGG→PCM→Whisper→LLM→TTS→OGG round-trip and persona tag handling)
3. A message from a non-allowlisted user-id is silently dropped (check logs).
4. `python app.py` still works unchanged for the local Jabra mic loop (regression check on the `services.py` extraction).

If phase 2 (real-time calls) is ever built, verification adds: joining the userbot to a group voice chat and having a back-and-forth conversation with audible interruption handling.
