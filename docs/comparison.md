# Open-Source Local Alexa Alternatives — Deep Research & Comparison to voice-chatbot

## Context

You're building **voice-chatbot**, a Mac-native, Pipecat-based, LLM-first local voice assistant. The recent arc of commits (v2 → skills → bbc → voice cloning → whoosh) shows it's growing from a chat prototype into a full Alexa-class device: skills system, live radio + on-demand podcasts, voice cloning, sound design, persona switching. This research surveys the broader open-source local-voice landscape so you can (a) understand where voice-chatbot sits, (b) decide what to borrow vs build, and (c) decide whether to stay on the current trajectory or pivot toward a more standard ecosystem like Home Assistant Voice / Wyoming.

---

## 1. Your current stack — one-line snapshot

| Layer | voice-chatbot |
|---|---|
| Orchestrator | Pipecat (Python async pipeline) |
| STT | Whisper MLX (Apple Silicon, ~300–500ms) |
| LLM | **Ollama Gemma 4 26B MoE** primary + **Claude API** optional, switchable mid-conversation by wake phrase |
| TTS | **Kokoro** (ONNX, default) + **Chatterbox-TTS** (zero-shot voice cloning per persona) |
| Wake | "hey babel" / "hey claude" via Silero VAD + phrase match |
| Skills | LLM tool-calling: time, timers, weather (Open-Meteo), web search, **BBC live radio (11 stations via mpv)**, **BBC on-demand shows (RSS + yt-dlp)**, persona switch, **sound effects (Sony Whoosh foley)** |
| Personas | YAML-declared; voice command / LLM tag / skill intent routing; per-persona TTS backend |
| Audio I/O | Pipecat LocalAudioTransport on Jabra USB speakerphone, 16k in / 24k out, keepalive silence |
| Hosting | All on `127.0.0.1` (Ollama, Chatterbox, Whoosh, optional Stable Audio); each service auto-launched and health-polled by `run.sh` |
| Hardware | Apple Silicon Mac only |

**Architectural identity:** premium, single-user, Mac-native, LLM-first, audio-rich.

---

## 2. The landscape, in three camps

### Camp A — Intent-first, mass-market (Home Assistant Voice / Rhasspy / Wyoming)
The "by-the-book" Alexa replacement. Built for smart home control. Skills are template-based intents, not LLM tools. ~2M HA users.

### Camp B — Plugin-bus, classic FOSS lineage (OVOS, Neon, Mycroft heritage)
Modular message-bus assistants with mature skill ecosystems. LLM support exists but is bolted on, not central.

### Camp C — Framework/agent-first (Pipecat, LiveKit Agents, Willow, LocalAI, Speaches, AURA)
LLM-driven pipelines. Where voice-chatbot lives. Willow is the one hardware-first member of this camp.

---

## 3. Deep dives

### Camp A: Home Assistant Voice + Wyoming ecosystem

**Home Assistant Voice Preview Edition** — $69 ESP32-S3 hardware satellite, dual-mic, XMOS audio coproc, open hardware. Released Dec 2024; 2025.10 added dual-wake-word multilingual pipelines. ~8.9% of HA installs use Wyoming. The single most-traction hardware in this space.

**Wyoming protocol** — streaming JSON-over-stdin/TCP for STT / TTS / wake / intent. Lets you compose `wyoming-faster-whisper` + `wyoming-piper` + `openWakeWord` into a server, with $10–$30 satellites streaming only post-wake audio upstream.

**Rhasspy** — Mike Hansen's original (v2 final, v3 dev preview). The repos `rhasspy/rhasspy` and `rhasspy/piper` were **archived October 6, 2025**; active dev is now under the **Open Home Foundation (OHF-Voice)** umbrella. Wyoming components keep shipping (`wyoming-piper 2.2.2` Feb 2026, `piper-tts 1.4.2` Apr 2026).

**Piper TTS** — neural TTS, 100+ voices, 30+ languages, CPU-only, RPi-friendly. No zero-shot cloning; cloning means GPU fine-tuning. Quality is solid but distinctly below Kokoro/Chatterbox for naturalness.

**openWakeWord** — 20+ languages, train custom wake on Colab in ~75 min, 15–20 models on a single RPi3 core, <0.5 false-accept/hour. The de-facto open wake word.

**LLM in HA** — `Ollama` integration (HA 2025.6+) routes prompts through Assist API to local models, but it's a fallback after intent matching, not the primary brain. HA recommends exposing <25 entities to keep LLM context small.

**Strengths vs voice-chatbot:** real $69 hardware satellites; multilingual; smart home is first-class; openWakeWord is genuinely better-engineered than ad-hoc phrase matching; 2M-user ecosystem with stable releases.

**Gaps vs voice-chatbot:** intent-first not LLM-first (skills must be hand-authored); Piper < Kokoro/Chatterbox for voice; no built-in voice cloning, sound design, or persona switching; no native Mac story; no "play me BBC In Our Time from last Thursday" without writing a custom integration.

---

### Camp B: OVOS / Neon / Mycroft

**Mycroft AI** — company shut down Feb 2023 after patent litigation. IP dispersed; codebase lives on in OVOS/Neon.

**OpenVoiceOS (OVOS)** — the dominant fork. Active dev, NLnet NGI Zero Commons grant Oct 2025. Architecture: websocket JSON messagebus, Plugin Manager for swappable STT/TTS/wake/solver. Core 2.1.1; many repos updated through Apr 2026. Default skills: weather, timers, alarms, date/time, music. New Pre-Wake-VAD filter (Nov 2025) improves wake reliability. Wyoming bridges expose OVOS to HA. HiveMind for distributed multi-device.

**LLM in OVOS** — `ovos-solver-openai-plugin` points at any OpenAI-compatible endpoint (Ollama, llama.cpp). Personas wrap solvers + optional translation. But still **intent-first**: the LLM is a "solver" that runs when other parsers (Adapt keyword, Padatious neural) fail.

**Neon AI** — Mycroft fork that kept Mark II hardware commercial. Multi-user support, 4+ releases/month. Skill set ≈ OVOS.

**Mimic 3** — deprecated; Piper is the spiritual successor. **Precise** wake word — stagnant since 2022.

**Voice cloning in OVOS** — `ovos-tts-plugin-coqui` adds XTTS-style 3-second cloning across 17 languages. Closest thing in this camp to your Chatterbox setup.

**Strengths vs voice-chatbot:** mature plugin/skills ecosystem; messagebus enables multi-device & external agents (MCP, A2A, HiveMind); HA bridge via Wyoming; battle-tested multilingual.

**Gaps vs voice-chatbot:** intent-parsing-first means each new capability needs a skill author; LLM solver is stateless (no equivalent to your mid-conversation backend swap); Mac support is essentially Linux-VM only — no MLX, no Metal; no integrated sound design layer; typical E2E latency 3–5s vs your sub-3s.

---

### Camp C: Framework- and agent-first

**Pipecat** (your framework, Daily.co, v1.2.1 May 2026) — 40+ AI service integrations. Strength is exactly what you use: pluggable LLM/STT/TTS, real-time streaming, multimodal, tool calling. Pipecat Cloud exists but is enterprise-priced — self-hosting is free.

**LiveKit Agents** (v1.5.12 May 2026, 3,346 commits) — competing pipeline framework. Strong semantic turn detection, interruption handling, OpenAI Realtime + Gemini Live audio-in/out support (bypasses text). Production-orchestration focus; weaker on voice cloning / persona / sound design. Has a self-hosted SIP stack — useful if you ever want a phone-callable Babel.

**Willow** (HeyWillow / Tovera, v0.4.2 Feb 2026) — ESP32-S3-BOX, <500ms end-to-action latency, 25-foot far-field pickup. Ecosystem = **WIS** (inference server: Whisper base/medium/large-v2 simultaneously in <6GB VRAM, WebRTC/REST/WS) + **WAS** (web UI, OTA) + **WAC** (Willow Auto-Correct — fuzzy-matches transcriptions against learned successful commands via Typesense). This is the **only project with hardware satellites in this camp.** Skills max ~400 on-device commands; less LLM-driven than you.

**AURA** (research, 2026 arxiv:2506.23049) — speech-native agent with real-world tool calling (calendar, email, search, contacts), ReAct reasoning, 92.75% OpenBookQA, 90% human task success. Closest published research to your "LLM + tools" direction.

**Vocode** — formerly gated, now fully open-source (2026). Simpler than Pipecat/LiveKit; smaller ecosystem.

**LocalAI** (v4.2.6 May 2026) — drop-in OpenAI-compatible server. Supports Realtime API (voice-in/voice-out), tool calling, Pocket-TTS, Qwen3-TTS, Chatterbox. Useful as a backend if you ever want to abstract away Ollama+Chatterbox behind one OpenAI endpoint.

**Speaches** (formerly faster-whisper-server) — OpenAI-compatible local STT, plus Kokoro/Piper TTS. Drop-in for any client that talks Whisper API. Could replace your direct Whisper MLX usage if you ever need to expose STT to other apps.

**Leon AI** (v2.0 Developer Preview) — modular agent rebuilt around tools/memory/context. No voice cloning. Tooling design worth watching for ideas about portable skill schemas.

**SEPIA** — Java client-server, Elasticsearch brain, pure command-matching, no LLM. Mature, stable, niche.

**Chatterbox** (Resemble AI, MIT) — you already use it. Industry-grade: 5-sec zero-shot cloning, emotion control, sub-200ms, 23+ languages. State of the art for OSS cloning.

**Kokoro** — 82M-param StyleTTS2 derivative, ~300MB, CPU-fast. Cannot clone natively; **KokoClone** community fork adds zero-shot multilingual cloning if you ever need it.

**Strengths vs voice-chatbot:** Willow's $30 ESP32 satellite story is killer for whole-house coverage; WAC's fuzzy-learning auto-correct is a clever idea you don't have; LiveKit gives you SIP/phone if desired; LocalAI is a cleaner abstraction layer; AURA points the way for richer agent loops.

**Gaps vs voice-chatbot:** Willow has no LLM reasoning, no persona switching, no sound design, no voice cloning; LiveKit weaker on cloning/sound; none of them play live BBC radio out of the box; nobody integrates procedural foley like your Whoosh skill.

---

## 4. Head-to-head matrix

| Capability | voice-chatbot | HA Voice + Wyoming | OVOS / Neon | Willow | LiveKit Agents |
|---|---|---|---|---|---|
| LLM-first design | **Yes** (Ollama+Claude, hot-swap) | No (intent-first, LLM fallback) | No (solver-as-fallback) | No (command-matching) | Yes |
| Voice cloning | **Yes** (Chatterbox per persona) | No (Piper fine-tune only) | Optional (Coqui XTTS plugin) | No | Via plugins |
| Sound design / foley | **Yes** (Whoosh + Stable Audio) | No | No | No | No |
| Persona switching | **Yes** (voice cmd / LLM tag / tool) | No | Partial (solvers) | No | Custom |
| Live media (BBC, podcasts) | **Yes** (mpv + yt-dlp + RSS) | Via HA media_player | Via skills | Limited | Custom |
| Smart home depth | None | **First-class** | Strong | Decent | Custom |
| Multilingual | English only | **21+ languages** | **1127 via MMS** | Limited | Decent |
| Wake word quality | OK (phrase match + VAD) | **openWakeWord** | openWakeWord / Precise | openWakeWord | Configurable |
| Hardware satellite | No — Mac required | **$69 PE / DIY $30** | RPi / Mark II | **ESP32-S3-BOX ~$30** | No |
| Apple Silicon optimized | **Yes** (MLX) | No | No | N/A | No |
| Multi-user | No | Partial | **Yes** (Neon) | No | Yes |
| Privacy: fully local | Yes (Claude optional) | Yes (cloud optional) | **Yes** | **Yes** | Self-hostable |
| E2E latency | ~sub-3s | 1–4s | 3–5s | **<500ms (commands)** | sub-second |
| Last release as of 2026 | active | active monthly | active | v0.4.2 Feb 2026 | v1.5.12 May 2026 |

---

## 5. Where voice-chatbot is clearly ahead

1. **LLM-first design is rare.** Almost everything else is intent-first with LLM bolted on. Your "hey babel" → Gemma tool-call → spoken prose pattern is the right architecture for the post-Alexa era; the rest of the OSS world is catching up.
2. **Voice cloning per persona.** Only OVOS via Coqui XTTS comes close, and it's not first-class. Chatterbox-per-persona is genuinely differentiated.
3. **Sound design / Whoosh foley.** Nobody else does this. It's a real product idea, not just a flourish.
4. **Live BBC integration.** Live radio + on-demand show RSS is the kind of "thing Alexa actually does well" that the FOSS ecosystem largely ignores.
5. **Apple Silicon performance.** Whisper MLX is much faster on M-series than faster-whisper-on-CPU, which is what the rest of the ecosystem assumes.
6. **Backend hot-swap by wake phrase.** "hey babel" → local; "hey claude" → cloud. Nothing else does this cleanly.

---

## 6. Ideas worth borrowing

Ranked by effort vs payoff for your project specifically.

1. **openWakeWord** to replace the current phrase-matching. Train custom "hey babel" / "hey claude" / per-persona models. Low effort (Wyoming-piper-bridge or direct embed), big quality jump, multilingual-ready. — high ROI.
2. **Willow Auto-Correct concept.** Log successful (transcription → tool-call) pairs, fuzzy-match new transcriptions against them with Typesense or even SQLite FTS, and use the match as a hint to the LLM. Cheap, reduces repeat mistakes. — high ROI.
3. **ESP32-S3 satellite mode.** If you ever want Babel in another room without another Mac: write a Wyoming-protocol-speaking adapter so HA Voice PE hardware (or DIY ESP32-S3-BOX) can stream audio to your Pipecat pipeline. Lets you keep your stack but get cheap room satellites. — medium effort, opens the household to Babel.
4. **HiveMind / Wyoming bridge out.** Even if you don't switch to OVOS/HA, exposing your STT and TTS via Wyoming makes Babel composable with other people's stacks. Low effort, future-proofing.
5. **LocalAI as STT/TTS abstraction.** If you ever want to point a third-party tool (a phone, a watch, an IDE plugin) at Babel's voice, putting LocalAI in front of Kokoro+Chatterbox gives you a stable OpenAI-Realtime-compatible endpoint. — medium effort, optional.
6. **Multilingual Piper voices as fallback personas.** Cheap way to add a French/German/Spanish persona without training Chatterbox. — low effort if you ever want it.
7. **HA bridge for smart home.** You don't currently do home control. If you want to (lights, music speakers, locks), the cleanest path is `homeassistant` Python client from a new `home_skills.py`, not reinventing device protocols. — medium effort, big capability unlock.

---

## 7. Ideas to deliberately NOT borrow

- **Intent-template skill authoring** (Padatious / HA custom_sentences). You already have LLM tool-calling; this would be a downgrade.
- **Messagebus architecture** (OVOS). Adds latency and ops complexity; Pipecat's single-process pipeline is faster and simpler for one user + one mic.
- **Piper as default TTS.** Kokoro and Chatterbox are better. Piper is a viable fallback voice or for adding cheap multilingual personas, not the main road.
- **Wyoming as primary protocol.** It's good for satellites and bridges; not worth restructuring your core around. Keep Pipecat as the brain.

---

## 8. Recommended next steps

The highest-leverage moves, in order:

1. **Swap phrase-match wake → openWakeWord.** Cleanest single quality win available.
2. **Add a "command memory" inspired by WAC.** Log good transcription/tool pairs to SQLite; expose them as recent-context hints to the LLM (or as direct shortcuts when fuzzy match is high-confidence).
3. **Decide whether you want satellite hardware.** If yes, build a small Wyoming-protocol adapter that pipes a HA Voice PE or DIY ESP32-S3 into your Pipecat pipeline. This is the question that most shapes Babel's future trajectory: stay laptop-only, or become whole-house.
