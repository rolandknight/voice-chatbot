# Streamable, voice-cloning TTS for a Mac Studio M4 Max 36 GB

Research date: 2026-08-24. Web survey (six parallel sweeps, ~550 page fetches; the
per-cluster reports with every URL are in [`streamable-tts-mac/`](streamable-tts-mac/))
plus a review of this repo's own measurements (`poc-tts/bench-m4-max.md`,
`poc-tts-streaming/results-rtx-2060.md`) and the earlier duplex survey
(`open-duplex-models-mac.md`).

**Requirements, in priority order:** (1) zero-shot voice-clone quality from a 5–15 s
English reference clip, at least Chatterbox-Turbo class; (2) low time-to-first-audio on
a *streaming* path that actually runs on Apple Silicon; then expressive controls
(emotion / intonation presets, tags, instruct prompts), footprint, licence.

**The bar** (this repo's own numbers): Chatterbox Flash on this Mac via MLX fp16, tuned,
whole-utterance = **0.92 / 1.37 / 4.20 s** for 30 / 104 / 317-char sentences (RTF ~0.28);
chatterbox-flash exposes no streaming API and its MLX path runs S3Gen on PyTorch CPU, so
first audio ≈ first-chunk generation ≈ **0.9 s**. The RTX 2060 block-streaming hack
reached a flat 0.44 s engine TTFA (0.68 s in the browser) — that is what a Mac candidate
has to beat, at Chatterbox clone quality.

## Bottom line

Nobody has published a Mac TTFA for *any* cloning-capable model. Every number below
0.9 s on Apple Silicon belongs to a non-cloning or ~100M model. So the answer is a
ranked shortlist to benchmark on the box, not a winner:

| rank | candidate | why | clone SIM (test-en, one scale†) | Mac streaming path | licence |
|---|---|---|---|---|---|
| **1** | **Qwen3-TTS-12Hz 1.7B-Base / 0.6B-Base** (Alibaba, Jan 2026) | best independently-measured WER (1.23) of anything with a Mac streaming path; SIM ≥ Chatterbox; the **only model whose maintained MLX port documents token-chunk streaming** (`mlx-audio`, 4-token ≈ 320 ms chunks; also `speech-swift`, `audio.cpp`) | 0.717 | mlx-audio `stream=True, streaming_interval=0.32`; M2 Max RTF 0.55, 37 ms/step | Apache-2.0 |
| **2** | **Chatterbox Turbo via mlx-audio** (`chatterbox_turbo`) | the incumbent's voice, with S3Gen ported to MLX and a token-chunk `stream_generate` (≥10 tokens ≈ 0.4 s audio) — the cheapest experiment: same clips, same personas | (Chatterbox 0.5B: 0.685; Turbo not evaluated) | mlx-audio; unmeasured; known chunk-seam clicks | MIT |
| **3** | **VoxCPM 1.5 (0.8B) / VoxCPM2 (2B)** (OpenBMB, Dec 2025 / Apr 2026) | highest SIM of any AR model (VoxCPM2 0.753), native 160 ms patch streaming, voice design + style instruct | 0.714 / 0.753 | mlx-audio + audio.cpp Metal (stream flag on voxcpm2); 8-bit VoxCPM2 ≈ 0.85× real time on MLX — 1.5 is the size to test | Apache-2.0 |
| **4** | **ZONOS2** (Zyphra, Jun 2026) | 900M-active MoE, ECAPA speaker embedding, explicit emotion vectors + rate buckets; official ggml runtime with Metal (`zonos2.cpp`) *and* an mlx-audio port, both streaming | 0.645–0.668 (own eval, ReDimNet — different scale) | zonos2.cpp Metal Q8 8.5 GB; mlx-audio `zonos2` | Apache-2.0 |
| **5** | **dots.tts SOAR / MF** (RedNote, Jun–Aug 2026) | best open SIM after LongCat (0.771 en), 225 ms first chunk when cloning on CUDA, 48 kHz | 0.771 | only `mlx-speech` streams it ("bounded waveform streaming"); the two faster MLX ports are whole-utterance | Apache-2.0 |
| 6 | Higgs Audio v3 TTS 4B (Boson, Jun 2026) | richest inline control (21 emotion tokens, prosody tokens), Elo 1042 > Chatterbox 1020 | not published (v2: 0.677) | mlx-audio `higgs_audio_v3`, streams | **non-commercial** |
| 7 | Kyutai Pocket TTS 100M (Jan–May 2026) | the only sub-300 ms Apple-Silicon TTFA in the field (~200 ms first chunk, M4 Air **CPU**), clones from a clip; use as the latency floor / fallback voice | not published | native CPU, mlx-audio `pocket_tts`, CoreML (FluidAudio) | CC-BY 4.0 |

† Seed-TTS-eval test-en speaker similarity, WavLM-based SV model, mostly from the VoxCPM2
report's cross-model table (Jun 2026) and the Chatterbox-Flash paper (Aug 2026). Human
reference on this scale = 0.734; ±0.02 is noise. See "Clone-quality evidence".

**Recommendation.** Benchmark #2 first (a day: same clips, same UI, `mlx-audio` already
wraps it), then #1 as the likely winner on WER and expressive control, then #3. Keep
Pocket TTS as the "instant filler" voice if the PRD's ≤1.5 s wake→first-audio budget
needs it. Treat #4–#6 as second-round tests only if the first three miss the bar.
Kyutai TTS 1.6B, Fish S2 Pro, IndexTTS 2.5, CosyVoice 3, Orpheus, CSM, Dia2, F5-TTS,
XTTS are all out for one of: cannot clone your own clip / no Mac streaming / too slow
on MLX / licence — details in "Everything else".

## What "streamable on this Mac" actually requires

Three independent layers must all hold, and most models fail at layer 3:

1. **The architecture streams** — audio tokens/frames are emitted before the utterance
   ends (AR codec-token models, DSM, block diffusion). Flow-matching NAR models (F5,
   LuxTTS/ZipVoice, LongCat-Audio-DiT, Raon, OmniVoice, IndexTTS' S2M stage) do not: their
   "streaming" is sentence-chunking, first packet = one full generation.
2. **A runtime exposes it** — Qwen3-TTS, Chatterbox Flash and IndexTTS all stream on
   paper but their *official* repos return whole files (Qwen: "not planned"; Flash: no
   API; IndexTTS: "not planned"). Streaming lives in forks, `mlx-audio`, `vllm-omni`,
   SGLang, or `audio.cpp`.
3. **The Mac port exposes it** — and PyTorch MPS is not a viable path for the
   flow-matching vocoders (S3Gen/HiFT/BigVGAN): CosyVoice on Mac is CPU-only, IndexTTS
   hits unsupported ops and memory leaks, Chatterbox MPS measured RTF 4.55. Every fast
   Mac port re-implements the vocoder in MLX, CoreML or ggml.

Runtimes that satisfy layer 3 today, with the models that both **stream and clone**
through them:

| runtime | maintainer / cadence | streams + clones | notes |
|---|---|---|---|
| **mlx-audio** (Python, MLX) | Blaizzy; releases every 1–2 weeks (v0.5.0 2026-08-17) | Qwen3-TTS (documented 0.32 s chunks), Chatterbox Turbo (token chunks, from source), Higgs v3, ZONOS2, VoxCPM2 (undocumented whether patch-level), Pocket TTS, CSM/Marvis | `stream=True` + `streaming_interval` (default 2.0 s — lower it); OpenAI-compatible server but `stream:true` has a seam bug (#898, Aug 2026); no Apple-Silicon numbers published for anything |
| **speech-swift** (Swift, MLX + CoreML/ANE) | Soniqo; weekly (v0.0.26 2026-08-17) | Qwen3-TTS (MLX or CoreML), CosyVoice3 (MLX) | `speech-server` REST `/v1/audio/speech` (v0.0.23); M2 Max Qwen3-TTS RTF 0.55, 37 ms/step; 14 TTS engines incl. IndexTTS2/VoxCPM2/F5 (clone, no stream) |
| **audio.cpp** (C++/ggml, Metal) | 0xShug0; v0.6.1 2026-08-18 | flags stream+clone on qwen3_tts, voxcpm2, dots_tts, confucius4, omnivoice, neutts | `audiocpp_server` HTTP; GGUFs for Chatterbox, Pocket, Higgs v3, MOSS, ZONOS2…; absolute RTFs only on RTX 5090; "VoxCPM2 2.56× faster after Metal work" |
| **mlx-speech** (Python, MLX) | appautomaton; 200 commits | dots.tts (only model it streams) | also Fish S2 Pro, Step-Audio-EditX, MOSS Local — non-streaming |
| **zonos2.cpp** (ggml, Metal) | Zyphra (official) | ZONOS2 | Q8_0 8.5 GB "effectively lossless"; `zonos2-server` streams PCM |
| **Pocket TTS** native / FluidAudio CoreML | Kyutai / FluidInference | Pocket TTS | 80 ms frames; 1–30 s reference |
| kyutai `delayed-streams-modeling` `tts_mlx.py` | Kyutai; moshi-mlx unreleased for a year | Kyutai TTS 1.6B — **preset voice embeddings only** | streams frames but "word-by-word, choppy" on M2 Max (issue #170, open) |
| llama.cpp `tools/tts` | ggml-org | Qwen3-TTS, Pocket TTS clone via `--tts-speaker-file` | **writes a WAV, no streaming** |
| sherpa-onnx | k2-fsa | ZipVoice clones (sentence callback); Pocket TTS clone reported to sound wrong (#3180) | ONNX/CoreML EP |

Memory budget matters more than it looks: the PRD runs `gemma4:26b` plus Whisper on the
same 36 GB. Qwen3-TTS 1.7B bf16 (~4.5 GB) and Chatterbox Turbo fp16 (~3–4 GB) fit
comfortably; ZONOS2 Q8 (8.5 GB), Fish S2 Pro 8-bit (~6 GB), Higgs v3 bf16 (~8 GB) and
dots.tts (6–13 GB peak) compete with the LLM. Check `ollama ps` before choosing.

## Clone-quality evidence

Three sources, in order of trust. **Never compare SIM across papers that use different
speaker-verification models** — IndexTTS 2.5 reports itself at 0.855 and Gepard's table
puts Chatterbox at 0.796, both on inflated scales; the ordering is what transfers.

**1. Seed-TTS-eval, test-en, WavLM-large SV (VoxCPM2 report Table 3, Jun 2026, plus
the Chatterbox-Flash paper v3, Aug 2026, and the Raon-OpenTTS table, Jun 2026).**

| model | WER % ↓ | SIM ↑ | Mac path streams? |
|---|---|---|---|
| human reference | 2.14 | 0.734 | — |
| Seed-TTS (closed) | 2.25 | 0.762 | — |
| dots.tts SOAR 2B | 1.30 | **0.771** | mlx-speech only |
| LongCat-Audio-DiT 3.5B | 1.50 | 0.786 | no (flow-matching) |
| **VoxCPM2 2B** | 1.84 | **0.753** | audio.cpp / mlx-audio (undocumented) |
| Raon-OpenTTS-1B | 1.78 | 0.749 | no; CC-BY-NC |
| OmniVoice 0.6B | 1.60 | 0.741 | no (NAR); licence conflict |
| MOSS-TTS Local | 1.85–1.93 | 0.733 | no (Realtime variant CUDA-only) |
| VoxCPM-0.5B | 1.85 | 0.729 | mlx-audio |
| CosyVoice 3-0.5B | 2.02 | 0.718 | no (mlx-audio-plus, whole-utterance) |
| **Qwen3-TTS-12Hz-1.7B-Base** | **1.23** | 0.717 | **yes — mlx-audio / speech-swift / audio.cpp** |
| VoxCPM1.5 0.8B | 2.12 | 0.714 | mlx-audio |
| IndexTTS 2 | 2.23 | 0.706 | no |
| **Chatterbox Flash 0.5B** (α=0.5) | 2.04 | 0.704 | no (MLX whole-utterance) |
| Confucius4-TTS | 1.49 | 0.700 | unverified |
| Chatterbox 0.5B (AR, Flash's base) | 2.20 | 0.685 | mlx-audio `chatterbox` — whole-utterance |
| Higgs Audio v2 3B | 2.44 | 0.677 | mlx-audio (v3 streams; v3 SIM unpublished) |
| F5-TTS 0.3B | 1.83–2.04 | 0.670 | no |
| Voxtral TTS 4B | 2.19 | 0.663 | MLX streams but **cannot clone** (presets only); CC-BY-NC |
| CosyVoice 2 | 2.57–3.09 | 0.655 | no |
| Audio8-TTS 0.6B | 1.51 | 0.632 | ONNX CPU; unverified |
| Spark-TTS | 3.14 | 0.573 | no |
| OpenAudio S1-mini | 1.94 | 0.550 | no |
| Fish Audio S2 Pro | **0.99** | *not reported* | no (RTF 3.77 on MLX 8-bit) |
| Chatterbox Turbo, Kokoro, Orpheus, CSM, Dia, Kyutai TTS/Pocket, NeuTTS, Marvis | not evaluated on this set | | |

Gepard's own table (Aug 2026, different SV model) corroborates the ordering that matters
here: Chatterbox 0.796 < Qwen3-TTS 0.833 < VoxCPM2 0.867. A separate on-device
benchmark (Gradium, Apr 2026, WavLM cosine) puts the small cloners far below: NeuTTS Air
47.5 %, KaniTTS-2 40.7 %.

**2. Blind arenas (Elo; Artificial Analysis open-weights board, seen 2026-08-24)** —
preset-voice naturalness, not cloning: Fish S2 Pro 1125 · Step-Audio-EditX 1102 ·
Voxtral TTS 1082 · Magpie 357M 1066 · Kokoro 1060 · Maya1 1045 · Higgs v3 1042 ·
**Chatterbox 1020** · Zonos-v0.1 1000 · VibeVoice 7B 969 · XTTS-v2 920. The
*controlled-voice* arena (same 8 cloned voices for every model) is the closest thing to a
clone-quality arena: open models Voxtral 1010, Fish S2 Pro 1002, and Chatterbox ~930 in
the launch snapshot — Qwen3-TTS, VoxCPM, dots.tts are not listed.

**3. Expressiveness (EmergentTTS-Eval win-rate vs gpt-4o-mini-tts)** — Higgs TTS 3
53.7 % (own run), dots.tts 47–49 %, Fish S2 Pro 43.8 %, Qwen3-TTS 38.8–42.8 %, Orpheus
29.4 %, F5 15–17 %.

## Master comparison

Cloning-capable, open-weight, with *some* Mac path. "TTFA" = published time-to-first-audio
(hardware in brackets); "Mac" = published Apple-Silicon figure. Empty = not published.

| model | params / Mac footprint | licence | clone (ref) | native streaming; TTFA | Mac runtime; streams?; Mac numbers | biggest caveat |
|---|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz 1.7B / 0.6B Base** | 1.7B ≈ 4.5 GB bf16; 0.6B ≈ 1.5 GB | Apache-2.0 | ≥3 s + transcript | arch: 4-token packets; 97–101 ms (unnamed GPU, vLLM); official repo **no**; forks 208 ms | mlx-audio **yes** (0.32 s chunks); speech-swift **yes** (MLX/CoreML); audio.cpp; M2 Max RTF 0.55, 37 ms/step; Soniqo "120 ms first packet" (hardware unstated) | no Mac TTFA measured; Metal watchdog caps ~40 s per call; ~72 open mlx-audio issues |
| **Chatterbox Turbo** | 350M T3 + S3Gen; MLX fp16 ~3–4 GB, 8-bit ~2 GB | MIT | >5 s (10 s rec.) | official repo **no**; hosted "75 ms"; community token-chunk fork 0.47 s TTFA (RTX 4090) | mlx-audio `chatterbox_turbo` **yes** (`stream_generate`, chunk ≥10 tokens ≈ 0.4 s, S3Gen in MLX); CoreML port whole-utterance ~1.5 s / 20 tokens (M3 Pro) | unmeasured; chunk-boundary clicks (HF #18); ~5 % silent runaways (#531) — the same runaway this repo already gates |
| **Chatterbox Flash** | 0.5B + S3Gen; this repo's PoC | MIT | ~10 s | arch: block diffusion; **103–118 ms TTFP (H100)** — *not in the released package*; paper v3 2026-08-21 | official `[mlx]` extra, whole-utterance; HF card M4 RTF 0.78 / this repo M4 Max 0.28 tuned, 0.92 s first audio; S3Gen on CPU torch | no streaming API, single-commit repo, 0 issues, no exaggeration knob |
| Chatterbox Multilingual v3 (Jun 2026) | 0.5B; MLX 2.7 GB | MIT | ~10 s | no | mlx-audio `chatterbox` — `stream` ignored | whole-utterance; only relevant for non-English |
| Chatterbox Nano (Jul 2026) | 110M | MIT | 5 s | no | none (MPS only) | watch for an MLX port — a 110M Turbo could stream in <300 ms |
| **VoxCPM2** | 2B; MLX bf16 5.0 GB, 8-bit 3.2 GB, 4-bit 2.3 GB | Apache-2.0 | short clip, no transcript needed; +transcript for best SIM | **yes**, 160 ms patches; RTF 0.30 (4090); ~250 ms on L4 (user) | mlx-audio `voxcpm2` (v0.4.4), MPS; audio.cpp Metal stream flag; MLX bf16 0.48×, 8-bit 0.85×, 4-bit 0.90× real time (chip unstated) | ≤ real time on MLX → TTFA likely > bar; click/tail-leak bug #272 open |
| **VoxCPM1.5** | 0.8B; MLX 8-bit ~1 GB LM | Apache-2.0 | same | yes, 160 ms patches; RTF 0.15 (4090) | mlx-audio; numbers not published | streaming through mlx-audio undocumented |
| **ZONOS2** | 8B total / 0.9B active MoE; GGUF Q8 8.5 GB, Q4 4.9 GB; MLX bf16 ~16 GB | Apache-2.0 | short clean clip (ECAPA embedding) | yes, chunked; TTFA not published | zonos2.cpp Metal (official) + mlx-audio `zonos2` (v0.4.5), both stream; no Mac numbers | embedding clone = timbre only; 8B weights stream from memory; 44.1 kHz × 9 codebooks |
| **dots.tts SOAR / MF** | 2B; int4 2.4 GB weights, 6–13 GB peak | Apache-2.0 | ref + transcript (Swift port: transcript-free) | yes; **225 ms clone / 69 ms text-only first chunk** (hardware unstated); RTF 0.20 | mlx-speech streams; sb1992 & sammcj MLX ports whole-utterance (M5 Max int4 speedups only); audio.cpp GGUF | every Mac port is one person; heavy 48 kHz continuous-latent decode |
| Higgs Audio v3 TTS 4B | 4B; 4-bit 2 GB, bf16 8 GB | **non-commercial** | ref + transcript | yes, frame-level; 617 ms mean latency (H100, not TTFA) | mlx-audio (v0.4.4) streams; audio.cpp GGUF; no Mac numbers | licence; 4B decoder at 25 fps |
| Kyutai Pocket TTS | 100M; <1 GB | CC-BY 4.0 (gated, auto-approved) | 1–30 s clip | **yes**, 80 ms frames; **~200 ms first chunk, ~6× RT on M4 Air CPU** | native CPU is the Mac path; mlx-audio `pocket_tts`; FluidAudio CoreML | no SIM/WER published; v2.1 word-repeat regression on Apple Silicon (#221, open); no speed/emotion control |
| CosyVoice 3-0.5B | 0.5B; <3 GB | Apache-2.0 | ref (+transcript) | yes, "150 ms" (hardware unstated) | mlx-audio-plus fork, GGUF — **whole-utterance**; upstream CPU-only on Mac | no Mac streaming |
| IndexTTS 2.5 (Aug 2026) | 0.8B; MLX ~5 GB | bilibili licence (commercial by arrangement) | ≤15 s | **no** ("not planned") | index-tts-2.5-mlx: RTF 0.45 M5 Pro, ~1.4 s wall per 3 s clip | non-streaming by design |
| Fish Audio S2 Pro | 4.4B; MLX 8-bit ~6 GB, bf16 11 GB | Fish research licence (paid commercial) | 10–30 s | yes (SGLang) ~100 ms TTFA (H200) | mlx-audio / mlx-speech **non-streaming**; MLX 8-bit RTF 3.77; 4-bit "decodes to noise" | far below real time on MLX |
| Kyutai TTS 1.6B | 1.8B | CC-BY 4.0 | **cannot clone own clip** (encoder unreleased, issue #404 open since Feb 2026) | yes, text-in + frame-out; 220 ms claim | DSM `tts_mlx.py` streams but choppy (#170) | fails requirement 1 |
| Sesame CSM-1B / Marvis 250M | 1B / 0.3B | Apache-2.0 | context audio + transcript | frame-level (community) | csm-mlx / mlx-audio `sesame` stream; ports dormant since Aug–Oct 2025 | unstable cloning; abandoned |
| Orpheus 3B | 3B; Q4 2.5 GB + SNAC | Apache-2.0 | effectively fine-tune-per-voice | yes; 180–280 ms (H100/A100) | mlx-audio / llama.cpp Metal; "1.5–4× RT on M3" | zero-shot clone weak |
| NeuTTS Air 0.5B | GGUF Q8 ~0.6 GB | Apache-2.0 | 3–15 s + transcript | yes, 25-frame (0.5 s) chunks | llama.cpp **CPU** (Metal officially off); LM 111 tok/s on iMac M4 | SIM 47.5 % WavLM — well below Chatterbox |
| F5-TTS / LuxTTS (ZipVoice) | 0.3B / 0.12B | CC-BY-NC / Apache-2.0 | 5–10 s + transcript | **no** (sentence chunks; ~2 s first packet) | f5-tts-mlx (stale, ~4 s/utt M3 Max); LuxTTS-mlx fp32 with Metal bugs | non-streaming |
| VibeVoice-Realtime-0.5B / 1.5B | 0.5B / 1.5B | MIT | **Realtime cannot clone**; 1.5B clones, non-streaming | RT: ~200–300 ms | mlx-audio, speech-swift; M2 Max RTF 0.53 (RT int8) | clone and stream are different checkpoints |
| Dia2 1B/2B, VoXtream2, FlashTTS, Gepard 1.0, MOSS-TTS-Realtime, GLM-TTS, MiMo, KaniTTS-2 | — | — | yes | yes (Gepard 32 ms, VoXtream2 63 ms on 3090/5090) | **CUDA-only, no Mac path** | — |
| XTTS-v2 | 0.75B | CPML non-commercial | 6 s | yes, CPU | MPS broken (wontfix) | Elo ~900; licence |

## Capabilities: presets, tags, instruct, knobs

What each shortlisted model lets you *control*, beyond timbre from the clip. This is where
the candidates differ most and where the personas (`marvin`'s delivery, `one_one`) would
gain or lose.

| model | emotion / intonation presets | inline tags | free-text instruct | voice design from description | speed / pitch / duration | multi-speaker | languages |
|---|---|---|---|---|---|---|---|
| **Qwen3-TTS** | CustomVoice: 9 preset timbres + `instruct` string ("very happy and excited", "whispering") | none | **yes** (CustomVoice / VoiceDesign) | **yes** (VoiceDesign 1.7B) | instruct only — no numeric knob | no | 10 + zh dialects |
| **Chatterbox Turbo** | `exaggeration` 0–1, `cfg_weight`, temperature (persona-level in `personas.yaml` today) | **9 paralinguistic tags**: `[laugh] [sigh] [gasp] [groan] [chuckle] [cough] [sniff] [shush] [clear throat]` (mlx port passes them as text — verify) | no | no | no | no | EN |
| Chatterbox Flash | `num_steps`, temperature, `cfg_scale`, `time_shift_tau` — **no exaggeration** | none | no | no | no | no | EN |
| **VoxCPM2 / 1.5** | "controllable cloning": reference + style instruction (emotion, pace, expression); InstructTTSEval-EN best (84.2) | none | **yes** | **yes** (v2) | via instruct; CFG α 1.5–3.0, diffusion steps | no | 30 + 9 dialects; singing (prelim.) |
| **ZONOS2** | emotion **direction vectors** (happy / sad / angry / surprised + valence / arousal); "stable" vs "expressive" modes; quality conditioning (bandwidth, SNR) | none | no | no | **8 speaking-rate buckets** or bytes/s target | no | 42+ |
| dots.tts | none documented (separate `.edit` model for speech editing) | none | no | no | no | no | 24 |
| Higgs Audio v3 | **21 emotion tokens** (`<\|emotion:amusement\|>` …), styles (singing / whispering / shouting) | 9 SFX tokens; prosody tokens **speed, pitch, pause, `expressive_high`** | no | no | yes (tokens) | v2 documented | 100+ |
| Pocket TTS | none | none | no | no | **none** (no speed, no pause) | no | 6 |
| IndexTTS 2.5 | **8-dim emotion vector** (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm); emotion from a *second* reference clip with `emo_alpha`; emotion inferred from text | none | `emo_text` description | no | `duration_factor` 0.5–2.0; pinyin/CMU pronunciation | no | zh/en/ja/es/ar |
| CosyVoice 3 | instruct: emotions, 18+ dialects/accents, speed, volume | (v2 had `[laughter]`/`[breath]`) | yes | no | instruct; phoneme control | no | 9 |
| Fish S2 Pro | 15,000+ free-form tags | `[whisper] [excited] [angry] [laughing] [singing] [inhale] [emphasis] [pause]` …, sub-word placement | yes (NL style) | no | speed | `<\|speaker:i\|>` + multi-turn | 80+ |
| Orpheus / Maya1 | 8 preset voices / design-by-description | `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` (Maya1: 20+) | no / yes | no / **yes** | no | no | EN |
| Kyutai TTS 1.6B | Expresso/EARS emotional preset voices (NC) | none | no | no | unreliable `padding_bonus` | no | EN/FR |
| Dia2 | — | 19 non-verbal tags, `[S1]/[S2]` | no | no | speed factor | **yes** | EN (CUDA only) |

Two things fall out of this table. Chatterbox is the *only* candidate whose expressive
control is the pair of numeric knobs `personas.yaml` already uses — every alternative
would change the persona schema (instruct string, emotion vector, or tokens). And
Qwen3-TTS' `instruct` plus VoxCPM2's style instruction are the natural fit for an
LLM-driven system that could emit a per-utterance style hint alongside the text;
ZONOS2's vectors and IndexTTS' 8-dim vector are the natural fit for per-persona
presets.

## Shortlist notes

### 1. Qwen3-TTS-12Hz (1.7B-Base, 0.6B-Base; CustomVoice / VoiceDesign siblings)
Qwen3 "talker" → 12.5 Hz 16-codebook tokenizer → causal ConvNet decoder (no DiT),
24 kHz, dual-track text/audio interleaving. Official package returns whole files and
the maintainers closed streaming as "not planned" — streaming exists in `mlx-audio`
(`generate(stream=True, streaming_interval=0.32)`, tokenizer `streaming_decode`, ICL
cache for the reference), `speech-swift` (`--first-chunk-frames`/`--chunk-frames`,
MLX or CoreML fp16 2.1 GB, ECAPA speaker encoder), `audio.cpp` (Metal), and two CUDA
forks (208 ms first chunk with a 21 ms Hann crossfade). M2 Max: RTF ≈ 0.55, 37 ms per
decode step after compile warm-up — a first 4-token packet plus reference prefill and
decode should land around 0.25–0.45 s on M2 Max (estimate, not a measurement), less on
the M4 Max. Known issues: Metal watchdog forces a ~500-token (~40 s) cap per call;
`split_pattern` ignored for CustomVoice/VoiceDesign; preset voices carry a Chinese accent
in English (presets, not clones). Base clones; CustomVoice adds instruct; VoiceDesign
builds a voice from a description — three checkpoints, not one. The July 2026
"Qwen-Audio-3.0-TTS" (#1 on the arena) is hosted-only.

### 2. Chatterbox Turbo through mlx-audio
`mlx_audio/tts/models/chatterbox_turbo/` ports T3, S3Gen and the S3 tokenizer to MLX
(voice encoder via numpy) and implements `stream_generate(chunk_size = max(10,
int(streaming_interval*25)))` — token-chunk streaming, min 10 speech tokens ≈ 0.4 s of
audio, with sentence-regex text chunking. That is exactly the shape of this repo's
`engine_blockstream.py`, but on Metal end-to-end (no CPU S3Gen). Nothing published on
speed. Risks: the port drops tag handling and hallucination guards; chunk-boundary
clicks are a known Turbo-streaming artefact (HF discussion #18, unanswered) because the
single-step decoder's output is context-dependent — the same cross-fade problem this
repo solved for HiFT joins; ~5 % silent 1000-token runaways (#531) — the same runaway
`poc-tts-streaming` already trims and gates. Knobs: `exaggeration`, `cfg_weight`,
`temperature 0.8`, `top_p`, `repetition_penalty 1.2`. Also note the mlx-audio server's
`/v1/audio/speech stream:true` emits one container per chunk (issue #898) — consume the
Python generator in-process instead.

### 3. VoxCPM 1.5 / VoxCPM2
Tokenizer-free diffusion-autoregressive: text-semantic LM (MiniCPM-4 0.5B / 1B) → FSQ →
residual acoustic LM → local DiT generating continuous AudioVAE latents patch by patch
(160 ms per AR step); VoxCPM2 decodes to 48 kHz. Best SIM of any AR model (0.753 en, 0.795
with reference + continuation), best S-MOS (4.74), best InstructTTSEval-EN, voice design
and style instruct. Native `generate_streaming()` at patch granularity; RTF 0.30 on a
4090. On MLX the 2B is at or below real time (8-bit 0.85×) so its TTFA will not beat
the bar on this box; VoxCPM1.5 (0.8B, SIM 0.714 — Qwen-class) is the one to measure.
Whether `mlx-audio`'s generic `stream=True` yields patches or whole segments is not
documented; `audio.cpp` flags voxcpm2 as stream+clone with Metal. Open bug #272:
click/chirp at segment start and reference-tail leakage in one-shot cloning.

### 4. ZONOS2
Sparse MoE (16 experts, 900M active / 8B total), UTF-8 bytes + ECAPA-TDNN speaker
embedding → 9 DAC codebooks at 44.1 kHz; CFG removed, "4× throughput vs v0.1". Own eval:
WER 2.8–3.1, SIM 64.5–66.8 (ReDimNet), UTMOS 4.36–4.40, lossless at Q8. Embedding-based
cloning captures timbre, not delivery — but delivery is *explicitly* controllable
(emotion vectors, valence/arousal, rate buckets, stable/expressive), which suits fixed
personas. Two streaming Mac runtimes and zero published Mac numbers; the memory-traffic
question (8B of weights per token, 9 codebooks per 44.1 kHz frame) is the one to answer
on the binned M4 Max's 410 GB/s.

### 5. dots.tts
Fully continuous end-to-end AR (frozen 48 kHz AudioVAE → Qwen2.5-1.5B → flow-matching
head), ~2B. Highest open SIM with a streaming architecture (0.771 en; SOAR checkpoint is
the "highest speaker similarity" one; MF = MeanFlow-distilled 1–4-step head). Native
`generate_stream()`: 225 ms first chunk cloning / 69 ms text-only (hardware unstated).
Three MLX ports: `mlx-speech` streams ("bounded waveform streaming", no numbers);
`sb1992/dots-tts-mlx` and `sammcj/mlx-swift-dots-tts` are faster but whole-utterance
(the Swift port explicitly says "streaming … absent from this port"). Peak memory
6–13 GB at int4 for 30 s clips; 24 languages; no expressive controls documented.

### 6. Higgs Audio v3 TTS 4B
Qwen3-4B decoder with a fused 8-codebook head at 25 fps, 24 kHz; the richest inline
control vocabulary in the field and Elo 1042 vs Chatterbox 1020. `mlx-audio` port streams
(overlap-add), `audio.cpp` GGUF. SIM unpublished for v3 (v2 was 0.677, below Chatterbox
Flash). The licence is "Research and Non-Commercial" — fine for a household system if
you accept it, but it closes the door on anything else, and the 4B decoder must sustain
25 fused tokens/s on this box.

### 7. Kyutai Pocket TTS
~100M (FlowLM 70M + 1-step LSD sampler 10M + Mimi decoder 20M), 80 ms frames, 24 kHz,
6 languages. The only sub-300 ms Apple-Silicon TTFA anywhere in this survey — ~200 ms to
first chunk and ~6× real time on **two CPU cores** of an M4 Air (Kyutai measured no GPU
gain at batch 1 on Apple Silicon). Clones from a plain wav (gated repo; the ungated
variant has presets only). HN consensus: "voice cloning worked much better than
expected", "comparable realism to Chatterbox at half the speed w/o GPU", but Kokoro
"better by far" as a plain TTS; text-skipping on long inputs; v2.1.0 intermittently
repeats words with exported profiles on Apple Silicon (#221, open, roll back to v1).
No SIM/WER anywhere. No speed, pause or emotion control.

## Everything else, in one line each

| model | verdict for this Mac |
|---|---|
| Chatterbox Flash (released package) | Paper now shows SIM 0.704 / WER 2.0 and 103–118 ms TTFP on H100, but 0.1.0 is still the only release, whole-utterance, S3Gen on CPU. This repo's block-streaming fork is the only streaming Flash that exists. |
| Kyutai TTS 1.6B | Excellent DSM streaming design, but the voice-embedding encoder is unreleased (issue #404, Feb 2026, unanswered) — you get 228 donated voices, not your clips; MLX streaming choppy (#170). |
| Fish Audio S2 Pro | Top open arena Elo, WER 0.99, 15k tags — and 4.4B running at RTF 3.77 on MLX 8-bit with no streaming port; commercial use paid. |
| IndexTTS 2 / 2.5 | Best emotion controllability; no streaming anywhere ("not planned"); ~1.4 s per 3 s clip on M5 Pro; MPS memory leaks; restrictive licence. |
| CosyVoice 3-0.5B | Mature CUDA streaming (150 ms), SIM ≈ Qwen; Mac = community fork, whole-utterance; upstream MPS requests open since 2024. |
| Voxtral TTS (Mistral) | Elo 1082, streams on MLX — but the MLX port can't clone (presets only, issue #694 open) and it's CC-BY-NC. |
| MOSS-TTS family | Local 4B SIM 0.733; Realtime 1.7B 180 ms TTFB — CUDA serving stacks only; MLX ports whole-utterance. Nano 100M streams on CPU but no SIM published. |
| Confucius4-TTS (Aug 2026) | SIM 0.700, transcript-free cloning, `generate_stream` — CUDA 12.6; mlx-community int8 exists without a documented runtime; 54 GB package. |
| Audio8-TTS 0.6B | SIM 0.632; ONNX INT4 runs on Mac CPU; below Chatterbox. |
| Orpheus 3B | Tags and streaming, but zero-shot cloning is not trained (fine-tune per voice). |
| Sesame CSM-1B / Marvis | Frame-streaming ports exist but are dormant since 2025; cloning unstable. |
| Dia 1.6B / Dia2 | Dia2 streams and clones — CUDA 12.8 only; Dia 1.6B on MLX is whole-utterance. |
| Zonos v0.1 | Superseded by ZONOS2. |
| VibeVoice | Streaming (0.5B) and cloning (1.5B) are different checkpoints. |
| F5-TTS, LuxTTS/ZipVoice, LongCat-Audio-DiT, Raon-OpenTTS, OmniVoice | Flow-matching NAR: strong SIM (0.67–0.786), first packet = whole generation. LongCat/Raon/OmniVoice also NC or licence-conflicted. |
| NeuTTS Air / Nano, KaniTTS-2, Gepard 1.0 | Small AR cloners with real streaming; SIM 40–58 % — a class below. Gepard is CUDA-only. |
| Kokoro, Supertonic 3, Soprano, Piper, Magpie 357M, Hume TADA, Maya1 | No zero-shot cloning. Kokoro/Supertonic/Piper set the Mac latency floor (40–500 ms per sentence). Supertonic's paid Voice Builder shuts 2026-08-31. |
| XTTS-v2, OuteTTS 1B, Spark-TTS, LLaSA, GLM-TTS, MiMo-Audio, Step-Audio-EditX, MisoTTS 8B, VoXtream2, FlashTTS | Licence, SIM, dormancy, or CUDA-only — see the cluster reports. |

## What changed vs. this repo's earlier research

- `bench-m4-max.md` established that MLX Flash is fast enough to stream (RTF 0.28) but
  not plumbed for it. That is still true of the *released* Flash; the Aug-2026 paper
  confirms the architecture streams at ~100 ms on H100, which validates the
  `engine_blockstream.py` approach — but the streaming loop is still unshipped, so the
  in-house fork remains the only streaming Flash.
- `open-duplex-models-mac.md` listed Pocket TTS, Kokoro, Qwen3-TTS and Chatterbox as
  cascade TTS options. This survey ranks them: Qwen3-TTS is the only one that clones,
  streams on Mac, and beats Chatterbox on the benchmarks; Pocket TTS is the latency
  floor; Kokoro stays the non-cloning default. Kyutai TTS 1.6B is out (no cloning).
- PersonaPlex (duplex) remains the only Mac-runnable full-duplex model with cloning;
  nothing here changes that — this survey is about the TTS leg of the cascade.

## Plan: benchmark on the box

The three baseline sentences (`poc-tts/bench.py`), the `marvin.mp3` / `one-one.mp3`
reference clips, warm model, best-of-3, on the Mac Studio. Record engine TTFA (first
chunk yielded), RTF, peak memory (`mlx.core.metal.get_peak_memory()`), chunk-seam step
ratio (reuse `spike_analysis seams`), WER (whisper) and SIM (WavLM-large-SV cosine
against the reference — the Seed-TTS-eval recipe, so numbers land on the table above).

1. **Chatterbox Turbo, mlx-audio `chatterbox_turbo`, `stream=True`, `streaming_interval`
   0.4 → 0.2** — one afternoon; same voices, same UI, same expectations. If it lands
   ≤ 0.5 s TTFA with the repo's existing silence-trim/runaway gate and a cross-fade at
   chunk joins, the migration is a port of `poc-tts-streaming`'s engine, not a model
   change.
2. **Qwen3-TTS 1.7B-Base and 0.6B-Base, mlx-audio `stream=True, streaming_interval=0.32`**,
   bf16 then 8-bit; also `speech-swift` `speech-server` as a second implementation. Test
   CustomVoice's `instruct` with the persona descriptions to see whether it survives
   cloning (it is a separate checkpoint — check whether Base accepts instruct at all).
3. **VoxCPM1.5 8-bit** through mlx-audio; confirm whether `stream=True` is patch-level by
   timing the first chunk against a 10 s utterance. If it is, it is the clone-quality
   upgrade over Qwen at Qwen's size.
4. **Pocket TTS** native CPU path with an exported voice state — establishes the floor and
   whether it can serve as the "instant acknowledgement" voice while the main model
   warms.
5. Second round only if 1–3 miss: ZONOS2 via `zonos2.cpp` Q8 (memory-bandwidth question),
   dots.tts via `mlx-speech`, Higgs v3 4-bit via mlx-audio (licence decision first).

Report the results in `poc-tts/bench-m4-max-streaming.md` alongside the existing
Flash tables so the comparison is on the same sentences and the same clips.

## Sources

Every URL is in the six cluster reports under [`streamable-tts-mac/`](streamable-tts-mac/):

- [`01-chatterbox-kyutai.md`](streamable-tts-mac/01-chatterbox-kyutai.md) — Chatterbox Flash / Turbo / Multilingual v3 / Nano; Kyutai TTS 1.6B, Pocket TTS
- [`02-chinese-llm-tts.md`](streamable-tts-mac/02-chinese-llm-tts.md) — Qwen3-TTS, CosyVoice 2/3, Fish S2 Pro, IndexTTS 2/2.5, VoxCPM 0.5B/1.5/2, Spark, Step-Audio-EditX, GLM-TTS, MiMo, LLaSA
- [`03-western-expressive.md`](streamable-tts-mac/03-western-expressive.md) — Orpheus, CSM, Marvis, Dia/Dia2, Zonos/ZONOS2, Higgs v2/v3, Maya1, VibeVoice, XTTS-v2
- [`04-small-fast.md`](streamable-tts-mac/04-small-fast.md) — NeuTTS, KaniTTS-2, Gepard, Soprano, Supertonic, Kokoro, F5/LuxTTS, OuteTTS, Piper
- [`05-runtimes-benchmarks.md`](streamable-tts-mac/05-runtimes-benchmarks.md) — mlx-audio, speech-swift/core, Kyutai MLX, sherpa-onnx, llama.cpp, audio.cpp, MPS status, FluidAudio/MetalRT; Artificial Analysis / TTS Arena / Seed-TTS-eval / EmergentTTS-Eval tables
- [`06-new-2026.md`](streamable-tts-mac/06-new-2026.md) — Feb–Aug 2026 releases: dots.tts, Voxtral TTS, MOSS-TTS family, Confucius4, Audio8, VoXtream2, FlashTTS, MisoTTS, Raon, OmniVoice, Magpie, TADA; full search log
