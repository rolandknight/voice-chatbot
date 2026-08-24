# Open-Weight Conversational Speech Models for Mac Studio M4 Max 36GB

Research date: 2026-08-23. Web-only survey. Requirements: open weights, Moshi-class
conversational (ideally full-duplex), runs on a Mac Studio M4 Max 36 GB, extendable to
call out to bigger models and external services.

## Bottom line

Three realistic candidates, trading off **conversational naturalness vs.
brains/extensibility**:

| | Pick if you want… | Runs on the Mac via | Native tools? | Voice cloning? |
|---|---|---|---|---|
| **NVIDIA NemotronLabs VoiceChat 11B** | Only open full-duplex model with built-in tool calling; best reasoning-per-duplex | `speech-swift` (MLX INT8 ≈12 GB) | **Yes** — function channel, on-hold lines | **No** — single fixed voice |
| **PersonaPlex 7B** (+ Kyutai RL checkpoint) | Best turn-taking/naturalness, role + voice prompts | `speech-swift` / `personaplex-mlx` (8-bit ≈9.5 GB) | No — MoshiRAG-style side channel needed | **Yes** — zero-shot, 18 presets |
| **MiniCPM-o 4.5 (9B)** | Smartest brain (Qwen3-8B), 30 languages, vision | `llama.cpp-omni` (Q8 ≈12 GB, Metal) | Backbone can; server exposes no API | Yes (audio reference) |

Recommendation: **VoiceChat first, PersonaPlex as fallback**, cascade as the
extensibility baseline. If voice cloning is a hard requirement, PersonaPlex is the only
Mac-runnable full-duplex option that has it.

## The constraint

The 36 GB Mac Studio is the *binned* M4 Max (410 GB/s memory bandwidth — roughly
M2 Max class, not the 546 GB/s of 48/64 GB models).

1. Budget ~26–28 GB for models after macOS. Everything below fits, including a side LLM
   for tool routing.
2. Full-duplex models must finish a step every **80 ms** (Mimi frame) or audio glitches.
   M2 Max numbers are the best proxy; M5 Pro numbers (GPU neural accelerators) are
   optimistic. **Benchmark on the actual box before committing.**

Viable frameworks: **MLX** (Python or Swift), **llama.cpp/Metal**, Kyutai **Rust/Candle
`--features metal`**. Anything CUDA/vLLM-only is out.

## How the open full-duplex field ranks (Artificial Analysis, Aug 2026)

| Model | Big Bench Audio (reasoning) | Full-duplex conv. dynamics | Params |
|---|---|---|---|
| **Nemotron VoiceChat** | 29.2% | 77.8% | 11–12B |
| Freeze-Omni | 33.9% | 58.7% | 7B |
| **PersonaPlex** | 12.6% | **91.0%** | 7B |
| FLM-Audio | 5.3% | 62.0% | 7B |
| Moshi | 1.7% | 61.0% | 7B |

VoiceChat is "the only open weights model in the top 3 on both" axes. **All open models
are far behind proprietary** (Step-Audio R1.1: 96% on the same reasoning benchmark).
The local model should own *timing and voice*, not *knowledge*.

Full-Duplex-Bench v1 (NVIDIA-reported): PersonaPlex user-interruption latency
**0.070 s** vs Moshi 0.265 s vs Gemini Live 1.301 s; smooth turn-taking 0.170 s.
VoiceChat: smooth turn-taking 0.82 @ 448 ms, take-over rate 1.00 @ 480 ms.

## Candidate 1 — NVIDIA NemotronLabs VoiceChat 11B

Released **2026-08-03**. Fast Conformer streaming encoder → **Nemotron Nano v2
(Nemotron-H, Mamba2/Transformer hybrid, 9B)** → NVIDIA TTS decoder + codec, plus a
**separate output channel for tool-call scripts**. "The first open full-duplex model to
support tool calling." Trained on ~550k h audio. License OpenMDW-1.1 (permissive) but
NVIDIA labels the checkpoint **research-only**.

**Tool calling** — `<TOOLCALL>` JSON on the side channel; the speech channel goes idle
so JSON never leaks into audio; operator-defined **on-hold lines** play immediately to
cover API latency; results return as `<TOOL_RESPONSE>`. Full-Duplex-Bench v3: tool
selection 82.5%, argument accuracy ~43%, Pass@1 33%. BFCL-v3 average 56.1%.
Constraints: max 5 tools/session, no reliable parallel calls, user cannot interrupt
during a call.

**Voice** — **No voice cloning.** NeMo docs: "The released NemotronLabs VoiceChat
checkpoint uses a single fixed voice and does not support voice cloning." The
"reference speaker WAV" in the config is a checkpoint-conversion input, not a runtime
option. The MLX port carries an "immutable 37-frame Aria speaker prompt."

**On the Mac** — NVIDIA's reference needs 80 GB VRAM/Linux, but Soniqo `speech-swift`
(Apache-2.0) has an MLX port:

- INT8: 12.1 GB, 100% text parity. INT5: 8.6 GB, 92.5% parity. Peak physical footprint
  ~15.7 GB.
- M5 Pro 48 GB: pipeline RTF 0.92–0.94, per-frame p95 76–79 ms — under the 80 ms
  deadline with almost no margin; "thermal state and concurrent GPU work materially
  affect this margin." On binned M4 Max expect INT5 to be necessary, INT8 borderline.
- Tool calling implemented (`<SPECIAL_20>`/`<SPECIAL_21>` markers, `<TOOL_RESPONSE>`
  injection, `allow`/`confirm` execution policies) but **disabled by default**. Demo:
  MCP tool call into Apple Reminders at RTF 0.92 on M5 Pro.

**Documented failure modes** — 2-minute audio context ceiling; "degradation into
non-recoverable gibberish after several turns"; runaway self-talk; system prompts and
tool responses must be ASCII and TTS-friendly. Only one Mac port exists (Swift).

**Extension path** — the model's own tool channel. Register `ask_big_model(question)`
hitting Claude/etc. with an on-hold line. Argument accuracy is mediocre, so keep tool
schemas tiny and let the big model parse.

## Candidate 2 — PersonaPlex 7B (Moshi lineage)

NVIDIA, Jan 2026. Moshi weights + hybrid system prompt: **text role prompt forced onto
the agent text channel** (audio silent) plus a **voice sample on the agent audio
channel** for zero-shot cloning. Trained on ~2,250 h synthetic dialogues
(Qwen3-32B/GPT-OSS-120B scripts, Dia/Chatterbox TTS) + Fisher. MIT code, NVIDIA Open
Model License weights (commercial OK). English only.

**Full-duplex?** Yes, natively — inherits Moshi's simultaneous user-audio /
agent-audio / agent-text streams at 12.5 Hz. No VAD, no turn detector. Paper title:
"Voice and Role Control for Full Duplex Conversational Speech Models."

**Why compelling** — 91% conversational dynamics; 0.07 s interruption latency;
role-adherence GPT-4o score 4.48 vs Moshi 1.75. Kyutai's June 2026
`personaplex-rl-seamless` (GRPO on 4,000 h Seamless Interaction) further cuts barge-ins
and improves backchanneling; drop-in compatible; RL delta is **CC BY-NC 4.0**
(non-commercial). Base PersonaPlex stays commercial-OK.

**On the Mac** — three ports:

- `speech-swift`: 8-bit ~9.5 GB (recommended; 4-bit "significantly degrades" quality),
  ~0.94 RTF on M2 Max. Includes AEC, VAD, tool-call loop, pluggable LLM.
- `Acelogic/personaplex-mlx`: Python, web UI, 4/5/6/8-bit switchable, ~8 GB at 4-bit,
  ~65 ms/step. No AEC.
- `mu-hashmi/personaplex-mlx`: Python, local/web/offline WAV. No AEC — "use headphones."

**The catch** — NVIDIA (HF discussion #2): "We don't have tool-calling support like
that now." Developers found "text_prompt is fine-tuned for persona shaping only, not
content injection" — adopts name/role reliably but **ignores factual content in the
prompt**. NVIDIA (discussion #23): future models "packaged with finetuning flows and
toolcalling support"; custom voice "extremely difficult to get legal approval for."
Reasoning: 12.6% Big Bench Audio.

**Extension path** — the MoshiRAG pattern (see `moshi-function-calling.md`, same
architecture): watch the inner-monologue text stream for a trigger phrase or
fine-tuned token, ship transcript to a big model, inject results via the conditioner
path (`streaming_sum`) or NVIDIA's suggested hack of drip-feeding result text at 80 ms
intervals. `moshi-finetune` (LoRA) works on this lineage. Weeks, not days.

## Candidate 3 — MiniCPM-o 4.5 (OpenBMB, 9B)

SigLIP2 + Whisper-medium + **Qwen3-8B** + CosyVoice2-style speech head. Full-duplex via
"Omni-Flow" time-aligned streams (TAIL). Apache-2.0, 30+ languages.

**On the Mac** — first-party `llama.cpp-omni`, Metal auto-detected. Official
requirement for full-duplex: "Apple M4 Max with at least 24 GB RAM." Measured on
M4 Max: Q4_K_M ~8.5 GB, Q8_0 ~12 GB, F16 ~19 GB; TTFT <650 ms; decode ~12 ms/token;
Token2Wav RTF 0.47. Text + audio system prompts, voice cloning via reference audio.

**Trade-offs** — turn-taking is nowhere near PersonaPlex (duplex controller over a
Qwen brain, not a native Moshi-style model). `llama.cpp-omni` exposes HTTP/SSE with
**no OpenAI-compatible endpoint and no function-calling support**. The Qwen3-8B
backbone can do tool calls in text mode — a plumbing gap, not a model gap.

## Baseline: cascade

Kyutai's Unmute (STT + OpenAI-compatible LLM + TTS) is **not supported on Mac** and has
no tool calling. Mac-native equivalents:

- **STT:** Kyutai STT 1B via `moshi-mlx` (semantic VAD, 0.5 s delay), or **Voxtral
  Mini 4B Realtime** via MLX (Apache-2.0, 80 ms streaming, ~2.5 GB 4-bit).
- **LLM:** anything — local MLX for latency, Claude for hard questions, tools free.
- **TTS:** Kyutai Pocket TTS (CPU, 6 languages since May 2026), Kokoro, Qwen3-TTS,
  Chatterbox (already in use, cloning works).

Latency 500–800 ms, no true duplex. Full-Duplex-Bench v3: cascades hit **100%
turn-take rate but ~10 s task latency vs 4–7 s for native models** on tool tasks, and
17.6% on self-correction scenarios. Orchestrate with Pipecat or Soniqo `speech-core`
(five-state turn detector, tool-call loop, FunctionGemma 270M for structured output).

## Everything else

| Model | Verdict for this Mac |
|---|---|
| **Moshi / MoshiRAG** (Kyutai 7B) | Moshi MLX q4/q8 runs but 1.7% reasoning. MoshiRAG Rust backend supports Metal but needs 4 services. Research substrate only; PersonaPlex is a strictly better base. |
| **Step-Audio 2 mini** (8B) | Native tool calling + RAG, Apache-2.0 — CUDA-only, ~24 GB VRAM, no Mac port. |
| **Qwen3-Omni 30B-A3B** | Open, function calling. MLX only does text; Talker (speech output) not ported. Qwen3.5-Omni: API only. |
| **Covo-Audio-Chat-FD** (Tencent 7B, Mar 2026) | CC-BY-4.0, THINK/SHIFT/BREAK duplex tokens, top of 7B on audio understanding. Transformers/CUDA only. |
| **Fun-Audio-Chat 8B** (Alibaba) | Strong on speech function-calling benchmarks; Duplex variant weights not released; ~24 GB CUDA. |
| **BayLing-Duplex** (GLM-4-Voice 9B fine-tune, Jun 2026) | 100% interruption success on own eval; CUDA, ZH/EN. |
| **FLM-Audio** (7B) | Open, EN/ZH, weak reasoning (5.3%). CUDA. |
| **Freeze-Omni** (7B) | Best reasoning of older duplex models; CUDA. |
| **LFM2.5-Audio-1.5B** (Liquid) | Turn-based, not duplex; beats Moshi on VoiceBench; MLX + GGUF; LFM Open License. Tiny fallback / iPhone. |
| **Hertz-dev** (8.5B) | Base model, no instruction tuning, effectively abandoned. |
| **DuplexSLA** (May 2026) | Research: three-channel duplex with a rate-limited *action* channel for tool calls — the right architecture; weights status unclear. Watch. |

## Side note: PersonaPlex on an RTX 2060 (Turing, sm_75)

- **Official PyTorch path: no.** BF16 reference uses ~19 GB VRAM on an RTX 3090; NVIDIA
  recommends 24 GB. Turing has no native bf16 (needs fp16 cast). `--cpu-offload`
  exists (needs `accelerate`) but "will significantly degrade the real-time
  performance." Issue #10 ("8GB VRAM Support?") is unanswered.
- **moshi.cpp path: borderline yes.** [Codes4Fun/moshi.cpp](https://github.com/Codes4Fun/moshi.cpp)
  (ggml, CUDA/Vulkan/CPU backends) runs PersonaPlex from
  [q4_k GGUF](https://huggingface.co/Codes4Fun/personaplex-7b-v1-q4_k-GGUF) (~4 GB) with
  text prompt and voice cloning (`personaplex -v voice.wav -p "..."`). Benchmarked on an
  **8 GB RTX 2070 laptop: 17.8 fps** speech-to-speech vs the **12.5 fps** real-time
  floor. A desktop RTX 2060 has ~75% of that card's memory bandwidth (336 vs ~448 GB/s),
  so expect ~13 fps — real-time with almost no headroom. 6 GB VRAM is tight for
  weights + Mimi + KV cache; the 12 GB 2060 variant is comfortable.
- Quality at q4_k is degraded (the MLX port calls 4-bit "significantly degraded").

## Plan

1. **Week 1:** `brew install speech`; run PersonaPlex 8-bit and VoiceChat INT5/INT8 side
   by side on the real box. Measure p95 step time under load — nothing on the web
   answers whether VoiceChat holds 80 ms on binned-M4-Max bandwidth.
2. **If VoiceChat holds:** enable its tool channel, register one `consult(question)`
   tool hitting Claude with an on-hold line, test the "gibberish after several turns"
   failure mode aggressively (may force session resets every ~2 min). Accept fixed
   voice or re-voice output with streaming voice conversion (adds latency).
3. **If it doesn't, or cloning is required:** PersonaPlex for voice + trigger-and-inject
   side channel to a big model (MoshiRAG pattern). Knowledge lives entirely off-model.
4. **Either way**, keep the cascade running as the reliability floor. Duplex model
   handles chit-chat and timing; anything needing facts or actions escalates.

Watch for NVIDIA's promised finetuning flows for VoiceChat — that is the point where
custom tool schemas become trainable rather than worked around.

## Sources

- [Artificial Analysis — Nemotron VoiceChat Pareto](https://artificialanalysis.ai/articles/nemotron-3-voicechat-leader-speech-pareto) · [VoiceChat HF card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) · [NeMo Speech branch](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat) · [MarkTechPost](https://www.marktechpost.com/2026/08/09/nvidia-releases-nemotronlabs-voicechat-11b-an-open-full-duplex-speech-to-speech-model-with-450-ms-turn-taking-and-live-tool-calling/)
- [speech-swift](https://github.com/soniqo/speech-swift) · [VoiceChat on MLX](https://github.com/soniqo/speech-swift/blob/main/docs/models/voicechat.md) · [PersonaPlex on MLX](https://github.com/soniqo/speech-swift/blob/main/docs/models/personaplex.md) · [Soniqo voice agents](https://soniqo.audio/voice-agents)
- [PersonaPlex paper](https://arxiv.org/html/2602.06053) · [repo](https://github.com/NVIDIA/personaplex) · [HF card](https://huggingface.co/nvidia/personaplex-7b-v1) · [customisation thread](https://huggingface.co/nvidia/personaplex-7b-v1/discussions/2) · [roadmap thread](https://huggingface.co/nvidia/personaplex-7b-v1/discussions/23) · [Acelogic/personaplex-mlx](https://github.com/Acelogic/personaplex-mlx) · [mu-hashmi/personaplex-mlx](https://github.com/mu-hashmi/personaplex-mlx)
- [kyutai/personaplex-rl-seamless](https://huggingface.co/kyutai/personaplex-rl-seamless) · [kyutai/moshika-rl-seamless](https://huggingface.co/kyutai/moshika-rl-seamless) · [Interactivity alignment paper](https://arxiv.org/abs/2606.11167) · [Kyutai blog](https://kyutai.org/blog/)
- [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) · [MiniCPM-o 4.5 paper](https://arxiv.org/abs/2604.27393) · [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) · [GGUF card](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf/blob/main/README.md)
- [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) · [FDB v1/v1.5 results](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v1_v1.5) · [FDB-v3 tool-use paper](https://arxiv.org/html/2604.04847v1) · [HumDial ICASSP 2026](https://arxiv.org/html/2604.21406v2) · [WavBench](https://arxiv.org/html/2602.12135) · [Awesome-Full-Duplex-SDM](https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM)
- [Unmute](https://github.com/kyutai-labs/unmute) · [Unmute Mac issue #74](https://github.com/kyutai-labs/unmute/issues/74) · [moshi-rag](https://github.com/kyutai-labs/moshi-rag) · [moshi_mlx README](https://raw.githubusercontent.com/kyutai-labs/moshi/main/moshi_mlx/README.md) · [Kyutai STT](https://kyutai.org/stt/)
- [Covo-Audio](https://github.com/Tencent/Covo-Audio) · [Covo-Audio report](https://arxiv.org/abs/2602.09823) · [BayLing-Duplex](https://github.com/BayLing-Models/BayLing-Duplex) · [FLM-Audio](https://github.com/cofe-ai/flm-audio) · [Fun-Audio-Chat-8B](https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B) · [Step-Audio 2](https://github.com/stepfun-ai/Step-Audio2) · [Step-Audio-2-mini hardware](https://huggingface.co/stepfun-ai/Step-Audio-2-mini/discussions/13) · [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) · [Qwen3-Omni MLX 4-bit](https://huggingface.co/pherber3/Qwen3-Omni-30B-A3B-Instruct-4bit-mlx) · [LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) · [liquid-audio](https://github.com/Liquid4All/liquid-audio) · [Hertz-dev](https://huggingface.co/si-pbc/hertz-dev) · [DuplexSLA](https://arxiv.org/abs/2605.20755)
- [Voxtral Mini 4B Realtime MLX](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit) · [mlx-audio](https://github.com/Blaizzy/mlx-audio) · [qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent) · [Ultravox v0.7](https://www.ultravox.ai/blog/introducing-ultravox-v0-7-the-world-s-smartest-speech-understanding-model)
