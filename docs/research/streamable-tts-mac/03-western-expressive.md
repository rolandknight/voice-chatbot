# 03 — Western / startup expressive LLM-TTS models (researched 2026-08-24)

Bar to beat (from BRIEF): Chatterbox Flash on this M4 Max via MLX ≈ 0.9 s first audio (no streaming API), RTF ~0.28; clone quality ≈ Chatterbox. Numbers below carry their dates; "not published" means no page I saw gives the number.

Cross-cutting finding: the Mac story for this whole cluster runs through **mlx-audio (Blaizzy / Prince Canuma)**. Its `mlx_audio/tts/models/` tree (fetched 2026-08-24) contains `llama` (Orpheus), `sesame` (CSM / Marvis), `dia`, `vibevoice`, `higgs_audio`, `higgs_audio_v3`, `zonos2`, `pocket_tts`, `chatterbox`, `chatterbox_turbo` … but **no `maya`, no `zonos` (v0.1), no `xtts`**. Release cadence is weekly-to-monthly: v0.4.3 (2026-04-28, OmniVoice), v0.4.4 (2026-06-06, Higgs Audio v3), v0.4.5 (2026-07-09, "ZONOS2 TTS with streaming and batching"), v0.4.7 (2026-08-03, Chatterbox Multilingual v3), v0.5.0 (2026-08-17). Generic streaming is `--stream` on the CLI / `for result in model.generate(...)` in Python, but **mlx-audio publishes no Apple-Silicon RTF/TTFA numbers for any model** — every Mac number below comes from third-party cards or blogs.

---

### Orpheus TTS (Canopy Labs, released 2025-03-18; no v2 as of 2026-08)
- **What it is:** Llama-3.2-3B-Instruct fine-tune emitting SNAC codec tokens (7 tokens / frame, ~150 tokens per second of audio) → SNAC decoder, 24 kHz.
- **Weights/license:** https://huggingface.co/canopylabs/orpheus-3b-0.1-ft (+ `-pretrained`), Apache-2.0, commercial OK. Multilingual research preview (7 language pairs, 24 voices) 2025-04-10.
- **Size/memory:** 3B. README checklist still lists 1B/400M/150M as *planned*; only 3B (and multilingual 3B) weights exist — HF discussion #15 asks "150M how to use" with no weights. Third-party marketing pages claiming "4 sizes" are not backed by HF weights. Mac: Q4_K_M GGUF ≈ 2.49 GB on disk + ~0.5 GB SNAC (codersera, updated 2026-05-10); MLX quants `mlx-community/orpheus-3b-0.1-ft-{4bit,6bit,bf16}` (bf16 ≈ 6.5 GB est.).
- **Voice cloning:** "Zero-shot" is advertised, but the model card / README state it was *not trained on a zero-shot objective*: the pretrained model needs "one or more existing text-speech pairs in the prompt" and "the more pairs you pass, the more reliably it will generate in the correct voice". GitHub issue #6 (2025-03-19, "code example for zero-shot cloning?") has no maintainer answer. Consensus (Unsloth TTS guide, HF/Reddit): zero-shot "captures tone but misses pacing/expression", frequently drifts; the intended path is **LoRA/full fine-tune per voice** (Unsloth, mlx-tune). No SIM/WER published. Not on the Artificial Analysis / TTS Arena boards (checked 2026-08-24). → **timbre: weak/unreliable; style: not cloned; fine-tune path: strong.**
- **Streaming:** native token-level streaming (decode every 7 tokens → 1 SNAC frame). README claim "~200 ms, ~100 ms with input streaming". Measured in issue #61 (2025): **280 ms on A100, 180 ms on H100** via vLLM; one HN user: ~1 s at bf16 on RTX 4070 Super. Chunk joins are fine (contiguous codec frames).
- **Mac path:** (1) mlx-audio `llama` model (Orpheus) + SNAC in MLX; supports `--stream`; (2) llama.cpp/LM Studio Metal + `isaiahbjork/orpheus-tts-local` (SNAC on PyTorch CPU) — the codersera 2026 guide calls this "the only practical path for M-series"; (3) mlx-audio-swift includes Orpheus + SNAC. **Measured Mac numbers: only "1.5–4× real-time on M3 16 GB with Q4_K_M" (codersera, 2026-05-10); no TTFA published for any Apple chip.** A 3B decoder at ~150 tok/s of audio needs ≥150 tok/s to hold RTF 1; MLX 4-bit 3B on an M4 Max is roughly in that band, so expect RTF ~0.5–1.0 and TTFA of several hundred ms (estimate, not measured). Maintainer: mlx-audio (active, Aug 2026); Canopy repo quiet since May 2025 (Baseten partnership).
- **Controls/capabilities:** 8 preset voices (tara, leah, …); inline tags `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>`; temperature/top_p/repetition_penalty; no speed/pitch knobs; single speaker; English (+7-language preview).
- **Verdict vs the bar:** Loses on clone quality (no reliable zero-shot; Chatterbox is clearly better) and probably loses on Mac TTFA (3B AR decoder before the first SNAC frame; nothing measured < 0.9 s on Apple Silicon). Biggest caveat: zero-shot cloning is effectively a fine-tune-per-voice model.
- **Sources:**
  - Orpheus-TTS README — https://github.com/canopyai/Orpheus-TTS
  - Model card — https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
  - Issue #61 minimal latency (A100 280 ms / H100 180 ms) — https://github.com/canopyai/Orpheus-TTS/issues/61
  - Issue #6 zero-shot cloning — https://github.com/canopyai/Orpheus-TTS/issues/6
  - HF discussion #1 (vs CSM) — https://huggingface.co/canopylabs/orpheus-3b-0.1-ft/discussions/1
  - mlx-community quants — https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-4bit , https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-bf16
  - macOS guide (LM Studio + orpheus-tts-local, "1.5–4× RT on M3") — https://codersera.com/blog/install-and-run-orpheus-3b-tts-on-macos-a-complete-guide/
  - HN thread — https://news.ycombinator.com/item?id=43417894
  - mlx-audio-swift (Orpheus + SNAC) — https://github.com/Blaizzy/mlx-audio-swift
  - Unsloth TTS fine-tuning guide — https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning

### Sesame CSM-1B (Sesame AI Labs, released 2025-03-13; no CSM-2 / no update through 2026-08)
- **What it is:** Llama-style 1B backbone + ~100M audio decoder predicting Mimi RVQ codes (12.5 Hz, 32 codebooks), 24 kHz. Card lists "2B" total tensor count.
- **Weights/license:** https://huggingface.co/sesame/csm-1b , Apache-2.0, commercial OK (card forbids impersonation). MLX: https://huggingface.co/mlx-community/csm-1b (6.2 GB fp32 safetensors).
- **Size/memory:** ~1B backbone; bf16 ≈ 2–3 GB; senstella/csm-mlx supports `nn.quantize` (4/8-bit) — "nearly real-time on M2 Air but loses quality".
- **Voice cloning:** context-conditioned: pass `Segment(speaker, text, audio)` pairs (reference audio **plus exact transcript**) as conversation history; no dedicated speaker encoder. Max context 1861 audio frames (≈2.5 min). Consensus (HF discussion #7, isaiahbjork/csm-voice-cloning, Orpheus discussion #1): "results aren't the best but you can recognize the cloned voice"; quality highly variable, "sometimes exactly like the sample, other times background noise/slamming at start or nothing like the reference" (johnys.io mlx-audio write-up). No SIM/WER published; not on the arena boards. Fine-tune path: HF Trainer (native Transformers since 4.52), senstella/csm-mlx finetune (LoRA/full/DPO/KTO), davidbrowne17/csm-streaming finetune. → **timbre: recognisable but unstable; style: partially inherited from context.**
- **Streaming:** frame-level in principle (one 80 ms Mimi frame per backbone step). Reference code is whole-utterance; community: davidbrowne17/csm-streaming (CUDA, **RTF 0.28 on RTX 4090**, "first chunks in milliseconds"; no TTFA number); senstella/csm-mlx `stream_generate(accumulation_size=…)` (added 2025-08-15).
- **Mac path:** (a) senstella/csm-mlx — streaming + context cloning + quantization; last commit **2025-08-15**; TODO still lists "fix RoPE, watermarking, optimize for real-time" → effectively dormant; no published RTF/TTFA. (b) mlx-audio `sesame` model (also serves Marvis / "MisoTTS"): `ref_audio` cloning, `--stream`; no numbers. (c) akashjss/sesame-csm Gradio/OpenAI API with MLX backend (2025-03-20). **No measured Mac TTFA anywhere I could find.**
- **Controls/capabilities:** none explicit — no emotion tags, no speed/pitch knobs; style comes only from context audio; temperature/top-k; 2-speaker turn IDs; English only (weak elsewhere).
- **Verdict vs the bar:** Loses on clone quality (less reliable than Chatterbox) and unproven on Mac TTFA (streaming exists in MLX but nobody has published a number; 1B backbone + Mimi is light enough that sub-0.9 s is plausible). Biggest caveat: stale (Sesame moved on; ports last touched Aug 2025).
- **Sources:**
  - Model card — https://huggingface.co/sesame/csm-1b
  - HF discussion #7 voice cloning — https://huggingface.co/sesame/csm-1b/discussions/7
  - senstella/csm-mlx README + commits — https://github.com/senstella/csm-mlx , https://github.com/senstella/csm-mlx/commits/master
  - davidbrowne17/csm-streaming — https://github.com/davidbrowne17/csm-streaming
  - mlx-community/csm-1b — https://huggingface.co/mlx-community/csm-1b
  - akashjss/sesame-csm (MLX backend) — https://github.com/akashjss/sesame-csm
  - mlx-audio README (CSM/MisoTTS cloning) — https://github.com/Blaizzy/mlx-audio
  - johnys.io mlx-audio cloning notes — https://blog.johnys.io/local-text-to-speech-tts-and-voice-cloning-with-mlx-audio/
  - Orpheus-vs-CSM discussion — https://huggingface.co/canopylabs/orpheus-3b-0.1-ft/discussions/1

### Marvis TTS 250M (Marvis Labs / Marvis-AI — Prince Canuma & Lucas Newman; v0.1 2025-08-26, v0.2 2025-10-20)
- **What it is:** CSM-1B architecture shrunk: 250M multimodal backbone + 60M audio decoder → Kyutai Mimi RVQ codec, 24 kHz. Trained on Emilia-YODAS, ~2M steps on one GH200 (~$2k). Also a 100M v0.2.
- **Weights/license:** https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2 (MLX), `-transformers` variants, https://huggingface.co/Marvis-AI/marvis-tts-100m-v0.2 ; Apache-2.0, commercial OK.
- **Size/memory:** ~310M; "414 MB quantized", "2 GB RAM" (HF blog 2025-08-27); trivially fits.
- **Voice cloning:** yes, "just 10 s of reference audio" via `ref_audio` in mlx-audio (`sesame` loader). Quality: no SIM/WER, no arena entry; README admits hallucinated words on short sentences / new vocabulary; johnys.io found cloning "variable" across the mlx-audio CSM family. → **timbre: rough; style: no.**
- **Streaming:** native — designed for streaming on iPhone/iPad/Mac; chunk = Mimi frames as generated; `python -m mlx_audio.tts.generate --model Marvis-AI/marvis-tts-250m-v0.2 --stream`. **TTFA: not published** (blog and cards give no ms figure).
- **Mac path:** MLX-native (mlx-audio, mlx-audio-swift); streams on Mac. Activity: GitHub repo has 6 commits, last **2025-08-28**; HF v0.2 2025-10-20; nothing in 2026 → dormant.
- **Controls/capabilities:** 2 preset voices (conversational_a/b); no tags, no knobs beyond temperature; EN (v0.2 adds FR, DE).
- **Verdict vs the bar:** Almost certainly beats 0.9 s TTFA on an M4 Max (310M params, streaming path exists in MLX) but **no one has measured it**, and it loses clearly on clone quality/robustness. Biggest caveat: hallucinations and an abandoned repo.
- **Sources:**
  - GitHub — https://github.com/Marvis-Labs/marvis-tts , commits — https://github.com/Marvis-Labs/marvis-tts/commits/main
  - HF blog (numbers, training) — https://huggingface.co/blog/prince-canuma/introducing-marvis-tts
  - v0.2 card — https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2 ; 100M — https://huggingface.co/Marvis-AI/marvis-tts-100m-v0.2

### Dia 1.6B (Nari Labs, 2025-04) and Dia2 1B/2B (Nari Labs, 2025-11-19)
- **What it is:** Dia-1.6B: 1.6B encoder-decoder (SoundStorm/Parakeet-inspired) → Descript Audio Codec, 44.1 kHz, dialogue in one pass. Dia2: decoder-only 1B/2B → Kyutai Mimi (12.5 Hz), streaming — starts synthesising after the first few input tokens.
- **Weights/license:** https://huggingface.co/nari-labs/Dia-1.6B , https://huggingface.co/nari-labs/Dia2-2B (and Dia2-1B); Apache-2.0 (Mimi keeps its own license), commercial OK.
- **Size/memory:** Dia-1.6B ~6.5 GB bf16 on disk, ~7–8 GB in memory on MLX (codersera Mac guide); Dia2-2B ~4 GB bf16 est.
- **Voice cloning:** audio-prompt style: prepend reference audio + its transcript (Dia) / "prefix audio for [S1] and [S2]" (Dia2). No SIM/WER published; card warns voices differ per generation without a prompt/seed; community consensus: strong at dialogue realism, cloning is "quality varies per generation". → **timbre: moderate; style/paralinguistics: strong.**
- **Streaming:** Dia-1.6B: none (whole utterance). Dia2: native streaming at Mimi frame granularity; ~40 tok/s on A4000 (≈0.46× RT there, faster on bigger GPUs); README roadmap "Dia2 TTS Server: real streaming support" still listed as upcoming; TTFA not published.
- **Mac path:** Dia-1.6B: mlx-audio `dia` + `mlx-community/Dia-1.6B` (whole-utterance; no streaming); MPS PyTorch fails with shape errors (issue #129). **Dia2: no MLX port, no MPS support, CUDA 12.8+ required** (last commit 2025-11-29; no releases). The 2026 guides recommend running Dia2 on a cloud GPU.
- **Controls/capabilities:** `[S1]/[S2]` dialogue; non-verbal tags `(laughs) (clears throat) (sighs) (gasps) (coughs) (singing) (mumbles) (groans) (sniffs) (claps) (screams) (inhales) (exhales) (applause) (burps) (humming) (sneezes) (chuckle) (whistles)`; speed factor / cfg scale; English only.
- **Verdict vs the bar:** Dia 1.6B on MLX loses on TTFA (no streaming, ~1.6B whole-utterance) and is a coin-flip on clone quality; Dia2 has the streaming design but is **CUDA-only** — irrelevant for this Mac unless someone ports it. Biggest caveat: no Dia2 Mac path at all.
- **Sources:**
  - Dia-1.6B card — https://huggingface.co/nari-labs/Dia-1.6B ; repo — https://github.com/nari-labs/dia
  - Dia2 repo/README — https://github.com/nari-labs/dia2 ; commits — https://github.com/nari-labs/dia2/commits/main ; releases (none) — https://github.com/nari-labs/dia2/releases
  - Dia2-2B card — https://huggingface.co/nari-labs/Dia2-2B
  - Dia MPS/MLX issue #7 — https://github.com/nari-labs/dia/issues/7 ; M3 Pro failure #129 — https://github.com/nari-labs/dia/issues/129
  - Mac guide (mlx-audio v0.4.3, memory) — https://codersera.com/blog/run-nari-dia-16b-on-mac-installation-guide/

### Zonos v0.1 (Zyphra, 2025-02) → ZONOS2 (Zyphra, 2026-06-12)
- **What it is:** v0.1: 1.6B transformer or transformer+Mamba hybrid, eSpeak phonemes → DAC tokens, 44 kHz. ZONOS2: sparse MoE (MoE++, 16 experts, 28 layers, 2048 hidden), **900M active / 8B total**, NeMo-normalised UTF-8 bytes + ECAPA-TDNN speaker embedding → 9 DAC codebooks, 44.1 kHz; CFG removed; "4× throughput vs v0.1".
- **Weights/license:** v0.1 https://huggingface.co/Zyphra/Zonos-v0.1-transformer (Apache-2.0). ZONOS2 https://huggingface.co/Zyphra/ZONOS2 + https://huggingface.co/Zyphra/ZONOS2-GGUF ; Apache-2.0 (GitHub repo shows MIT); commercial OK.
- **Size/memory:** ZONOS2 GGUF: F16 15.3 GB, Q8_0 8.5 GB ("effectively lossless"), Q6_K 6.8, Q5_K 5.9, Q4_K 4.9 GB + DAC 254 MB + ECAPA 24 MB — all fit in 36 GB; MLX bf16 ≈ 16 GB.
- **Voice cloning:** ECAPA speaker embedding from a reference clip (v0.1 asks 10–30 s; ZONOS2 "short reference clip, clean speech-only"). Published (GGUF card, clean-English test set): **WER 2.79–3.07, speaker-similarity 64.5–66.8 (ReDimNet), UTMOS 4.36–4.40** across quants — same SIM ballpark as Higgs v2 (67.7 on Seed-TTS). Zyphra's ZTTS1-Eval vs competitors: numbers not on pages fetched. Arena: Zonos-v0.1 Elo 1000 (Artificial Analysis, Aug 2026; Chatterbox 1020); ZONOS2 not yet listed. Embedding-based cloning → **timbre: good; delivery/style: not cloned** (emotion/rate are explicit knobs instead).
- **Streaming:** v0.1: none (issue #41 unanswered). ZONOS2: chunked streaming in Mini-SGLang server, in zonos2.cpp (`zonos2-server`, "low-latency streaming PCM"), and in mlx-audio (chunks of ~0.5–2.0 s). **TTFA: not published** on any page (only "runs in real time on GPU").
- **Mac path:** v0.1: PyTorch CPU/MPS pure-torch build works on M1/M4 (HF discussion, Feb 2025), slow, no streaming. **ZONOS2: two real paths** — (1) `Zyphra/zonos2.cpp` (official, ggml; "CPU, CUDA, Apple Metal, Vulkan from the same GGUF files"; Metal on by default; macOS desktop app needs Xcode ≥16.3; 63 commits); (2) mlx-audio `zonos2` (v0.4.5, 2026-07-09; `ref_audio` cloning, streaming, batching). **No measured Apple-Silicon RTF/TTFA published for either** as of 2026-08-24. With 900M active params the decode should be light, but each 44.1 kHz DAC frame needs 9 codebooks and the MoE weights must stream from memory.
- **Controls/capabilities:** v0.1: 8-dim emotion vector (happiness, sadness, disgust, fear, surprise, anger, other, neutral), pitch-std, speaking-rate, fmax, quality; ZONOS2: emotion direction vectors (happy/sad/angry/surprised + valence/arousal), 8 speaking-rate buckets or bytes/s target, "stable" vs "expressive" modes, quality conditioning (bandwidth/volume/SNR), 42+ languages in 3 tiers, code-switching; no inline tags, no multi-speaker.
- **Verdict vs the bar:** ZONOS2 plausibly **matches or beats Chatterbox on timbre cloning** (SIM ~65 published) and is the only model here with an official Metal C++ runtime plus an MLX port that both stream — but nobody has published a Mac TTFA, so it must be benchmarked. Biggest caveat: 8B-total MoE memory traffic and zero Mac numbers.
- **Sources:**
  - Zonos v0.1 README — https://github.com/Zyphra/Zonos ; card — https://huggingface.co/Zyphra/Zonos-v0.1-transformer
  - Streaming issue #41 — https://github.com/Zyphra/Zonos/issues/41 ; Apple-Silicon thread — https://huggingface.co/Zyphra/Zonos-v0.1-hybrid/discussions/2
  - ZONOS2 repo — https://github.com/Zyphra/Zonos2 ; blog — https://www.zyphra.com/our-work/zonos2 ; card — https://huggingface.co/Zyphra/ZONOS2
  - ZONOS2-GGUF (quants, WER/SIM table) — https://huggingface.co/Zyphra/ZONOS2-GGUF/blob/main/README.md
  - zonos2.cpp (Metal) — https://github.com/Zyphra/zonos2.cpp
  - mlx-audio zonos2 — https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/zonos2 ; releases — https://github.com/Blaizzy/mlx-audio/releases
  - Artificial Analysis leaderboard — https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice

### Higgs Audio v2 3B (Boson AI, 2025-07) / v2.5 1B / Higgs Audio v3 TTS 4B (Boson AI + SGLang, 2026-06-04)
- **What it is:** v2: 3B Llama-backed "generation variant" with DualFFN, unified semantic+acoustic tokenizer. **v3: Qwen3-4B decoder (36 layers, 2560 hidden) with fused 8-codebook audio head, 25 fps (40 ms frames), 24 kHz, 8k context.**
- **Weights/license:** v2 https://github.com/boson-ai/higgs-audio (README_V2.md) — Apache-style open. **v3 https://huggingface.co/bosonai/higgs-audio-v3-tts-4b — "Boson Higgs Audio v3 Research and Non-Commercial License" (Creator Use Grant for attributed monetised content; production use needs a separate commercial licence).**
- **Size/memory:** v3 4B: bf16 ≈ 8 GB; `Reza2kn/Higgs-Audio-v3-TTS-4bit-MLX` 2.04 GB; mlx-audio 4/8-bit quants. v2 needed 24 GB CUDA.
- **Voice cloning:** zero-shot from reference audio + transcript (transcript improves fidelity). v2 published **Seed-TTS-eval WER 2.44 / SIM 67.70**, ESD emotion-sim 86.1, EmergentTTS-Eval win-rates 75.7 % (emotions) / 55.7 % (questions) vs gpt-4o-mini-tts. v3: Seed-TTS WER/CER 1.11 (2 langs), Higgs-Multilingual 3.61 over 111 languages; **SIM not published for v3**; Artificial Analysis Elo **1042 (Aug 2026) vs Chatterbox 1020**. Consensus: best-in-class expressive cloning among open Western models (v2 topped EmergentTTS-Eval). → **timbre: strong; delivery: strong (emotion carried from reference and via tags).**
- **Streaming:** native, frame-level (SSE chunks as the vocoder decodes); "sub-second TTFA"; **1×H100 bf16 CUDA-graph: 617 ms mean latency at concurrency 1, RTF 0.147** (LMSYS blog 2026-06-04) — note that is end-to-end latency, not TTFA, and on an H100.
- **Mac path:** mlx-audio `higgs_audio_v3` (added v0.4.4, 2026-06-06; PR #770 merged 2026-06-05): zero-shot cloning with pre-encoded reference reuse, inline tokens, batch, CLI/API, generic `--stream`. **No Apple-Silicon RTF/TTFA published**; a 4B Qwen3 decoder must emit ~25 fused tokens/s to hold RTF 1 — on an M4 Max 4-bit that should be achievable (≈40–60 tok/s for a 4B MLX model is typical), so TTFA well under 0.9 s is plausible but unmeasured. `Reza2kn` MLX quant is a bare transformer artifact (needs a custom loader).
- **Controls/capabilities:** inline tokens: 21 emotions (`<|emotion:amusement|>` …), styles (singing/whispering/shouting), 9 sound effects, prosody (speed, pitch, pause, `expressive_high`); multi-speaker dialogue (v2 documented; v3 card does not explicitly list it); 100+ languages (85 production-quality); temperature/top-k.
- **Verdict vs the bar:** Likely **beats Chatterbox on clone quality** (best published SIM/emotion numbers in this cluster) and has a real MLX streaming path — but Mac TTFA is unmeasured and the **v3 licence is non-commercial**, which may rule it out for production. Biggest caveat: licence.
- **Sources:**
  - Repo README (v3) — https://github.com/boson-ai/higgs-audio ; README_V2 (v2 benchmarks) — https://github.com/boson-ai/higgs-audio/blob/main/README_V2.md
  - v3 card (licence, tokens) — https://huggingface.co/bosonai/higgs-audio-v3-tts-4b
  - LMSYS/SGLang blog (H100 latency table) — https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/
  - mlx-audio issue #766 / PR #770 — https://github.com/Blaizzy/mlx-audio/issues/766 , https://github.com/Blaizzy/mlx-audio/pull/770
  - mlx-audio higgs_audio_v3 README — https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/higgs_audio_v3
  - Reza2kn MLX 4-bit — https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-MLX
  - Third-party spec sheet (release 2026-06-09, tables) — https://popsoda2002.github.io/higgs_tts_v3_intro.html
  - Artificial Analysis leaderboard — https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice

### Maya1 (Maya Research, 2025-11; no update through 2026-08)
- **What it is:** 3B Llama-style decoder → SNAC (~0.98 kbps, 7 tokens/frame), 24 kHz; voice *design* from a natural-language description.
- **Weights/license:** https://huggingface.co/maya-research/maya1 , Apache-2.0, commercial OK.
- **Size/memory:** 3B; card says 16 GB+ VRAM (vLLM); `nhe-ai/maya1-mlx-4Bit` 1.86 GB (mlx-lm conversion).
- **Voice cloning:** **none** — no reference-audio path; identity comes from `<description="40-yr old, low-pitch, warm">`; fine-tuning is the only way to a specific voice. Arena: Elo 1045–1051 (Artificial Analysis Aug 2026 / TTS Arena May 2026) — above Chatterbox (1006–1020) on *preset-voice* quality.
- **Streaming:** native token-level via vLLM (`vllm_streaming_inference.py`, APC), "sub-100 ms latency" target on A100/H100/4090 — TTFA not measured on any page.
- **Mac path:** no mlx-audio model; only a community mlx-lm 4-bit conversion of the LLM (text→SNAC tokens) — you would have to wire SNAC decoding and streaming yourself (Orpheus-style). No numbers.
- **Controls/capabilities:** voice-design prompt; 20+ inline emotion tags (`<laugh> <cry> <whisper> <angry> <gasp> <sigh> <giggle> <chuckle> …`); English (multi-accent); single speaker.
- **Verdict vs the bar:** Fails the primary requirement (no cloning) and has no maintained Mac runtime; interesting only if a *designed* voice is acceptable. Biggest caveat: no cloning.
- **Sources:**
  - Model card — https://huggingface.co/maya-research/maya1 ; vLLM streaming script — https://huggingface.co/maya-research/maya1/blob/main/vllm_streaming_inference.py
  - MLX 4-bit conversion — https://huggingface.co/nhe-ai/maya1-mlx-4Bit
  - MarkTechPost launch — https://www.marktechpost.com/2025/11/11/maya1-a-new-open-source-3b-voice-model-for-expressive-text-to-speech-on-a-single-gpu/
  - TTS Arena 2026 table — https://offlinetts.com/blog/tts-arena-leaderboard-2026/

### VibeVoice-1.5B / VibeVoice-Large(7B) / VibeVoice-Realtime-0.5B (Microsoft; 1.5B 2025-08, Realtime-0.5B 2025-12-03, multilingual speakers 2025-12-16; repo last updated 2026-07-23 with ASR-BitNet)
- **What it is:** Qwen2.5 LLM + 4-layer (~40 M) diffusion head producing continuous acoustic latents from a σ-VAE tokenizer at **7.5 Hz**, 24 kHz. Realtime-0.5B uses an interleaved windowed design (encode incoming text chunks while diffusing audio). The often-cited "0.3B" is actually **0.5B**.
- **Weights/license:** https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B , microsoft/VibeVoice-1.5B; MIT; commercial OK (card says research/dev "not recommended for commercial applications"). The 7B "Large" checkpoint was withdrawn by Microsoft; community mirrors remain (arena still lists "VibeVoice 7B" Elo 960–970).
- **Size/memory:** Realtime-0.5B ≈ 1 GB bf16 / 350 MB int4; 1.5B ≈ 3 GB bf16 / 1 GB int4.
- **Voice cloning:** **Realtime-0.5B: no** — ships without the acoustic encoder, only embedded voice caches ("to mitigate deepfake risks"). **1.5B: yes**, raw reference waveform + transcript, one-shot (acoustic encoder present). No SIM/WER published; Artificial Analysis Elo 970 (1.5B) / 969 (7B) vs Chatterbox 1020 → consensus: clone timbre is decent for long-form, below Chatterbox in blind ranking. → **timbre: OK; style: weak.**
- **Streaming:** Realtime-0.5B: native streaming text-in / audio-out, "~200–300 ms first audible speech (hardware dependent)", "NVIDIA T4 / Mac M4 Pro achieve real-time". 1.5B: whole-utterance in the reference code; mlx ports expose chunked output.
- **Mac path:** mlx-audio `vibevoice` (mlx-community/VibeVoice-Realtime-0.5B-{fp16,8bit}, VibeVoice-1.5B; card gated 401 for 1.5B); Soniqo speech-swift (both variants, streaming); appautomaton/mlx-speech (VibeVoice Large cloning). **Measured: aufklarer INT8 bundle on M2 Max — 1.20 s audio in 0.64 s, RTF 0.53 (bf16 RTFx 1.48, int4 2.31×); Soniqo: 1.5B INT4 on M2 Max RTFx 1.48.** Microsoft's own doc lists "Mac M4 Pro real-time". No Mac TTFA published; the design targets ~300 ms and the M4 Max is faster than the M2 Max above, so < 0.9 s is likely for the 0.5B — but that variant cannot clone.
- **Controls/capabilities:** no emotion tags/instruct; CFG scale, diffusion steps (10 default); up to 4 speakers / 90 min (1.5B); English + Chinese (1.5B), English + 9 "experimental" (0.5B); unstable on ≤3-word inputs; no code/formula handling.
- **Verdict vs the bar:** The streaming variant probably beats 0.9 s TTFA on this Mac but has **no cloning**; the cloning variant (1.5B) is slower, non-streaming in the reference path, and ranks below Chatterbox. Biggest caveat: cloning and streaming live in different checkpoints.
- **Sources:**
  - microsoft/VibeVoice README — https://github.com/microsoft/VibeVoice ; Realtime doc — https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md
  - Realtime-0.5B card — https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
  - mlx-community 8-bit — https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-8bit/blob/main/README.md ; fp16 — https://huggingface.co/mlx-community/VibeVoice-Realtime-0.5B-fp16
  - aufklarer INT8 (M2 Max RTF 0.53) — https://huggingface.co/aufklarer/VibeVoice-Realtime-0.5B-MLX-INT8
  - Soniqo guide (variants, cloning, M2 Max RTFx) — https://soniqo.audio/guides/vibevoice ; speech-swift — https://github.com/soniqo/speech-swift
  - appautomaton/mlx-speech — https://github.com/appautomaton/mlx-speech
  - Community fork — https://github.com/vibevoice-community/VibeVoice
  - Artificial Analysis leaderboard — https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice

### XTTS-v2 (Coqui → maintained by Idiap as `coqui-tts`; model 2023-11, package 0.27.5 2026-01-26)
- **What it is:** Tortoise-derived GPT-2 (~750 M) → HiFiGAN decoder with speaker/perceiver conditioning, 24 kHz.
- **Weights/license:** https://huggingface.co/coqui/XTTS-v2 — **Coqui Public Model License (CPML): non-commercial only**; code MPL-2.0 (https://github.com/idiap/coqui-ai-TTS).
- **Size/memory:** ~1.9 GB fp32 checkpoint; ~2–4 GB RAM on CPU.
- **Voice cloning:** zero-shot from ~6 s (single or multiple references), 16–17 languages. Arena: **Elo 886–920 (TTS Arena May 2026 / Artificial Analysis Aug 2026) vs Chatterbox 1006–1020** — clearly below modern models in blind ranking, though some hands-on reviewers still like its raw timbre match. Fine-tune path: Gradio fine-tune (v0.19+). → **timbre: decent; style: weak; robustness: dated.**
- **Streaming:** native `inference_stream()` chunked GPT+HiFiGAN, "< 200 ms latency" claimed (GPU, unspecified); chunk-join is a known artefact point (stream_chunk_size trade-off).
- **Mac path:** PyTorch only. **MPS does not work** — speaker-encoder conv hangs/unsupported (coqui-ai/TTS #3649, "wontfix", 2024-03; HF discussion #48 open since 2024-04); run on **CPU** ("rather quickly" on M1 Max, no numbers). No MLX/CoreML port. Idiap fork is maintained (PyPI 0.27.5, Jan 2026, Python 3.10–3.14) but adds no Mac acceleration.
- **Controls/capabilities:** no emotion tags/knobs; temperature, speed; 17 languages; single speaker per call.
- **Verdict vs the bar:** Loses on clone quality (arena ~100–130 Elo below Chatterbox), loses on Mac TTFA (CPU-only streaming; nothing measured), and the CPML licence bars commercial use. Biggest caveat: licence + no Metal path.
- **Sources:**
  - Idiap fork — https://github.com/idiap/coqui-ai-TTS ; PyPI — https://pypi.org/project/coqui-tts/
  - XTTS docs (streaming API, languages, CPML) — https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md
  - MPS bug #3649 — https://github.com/coqui-ai/TTS/issues/3649 ; HF MPS request — https://huggingface.co/coqui/XTTS-v2/discussions/48
  - TTS Arena 2026 table — https://offlinetts.com/blog/tts-arena-leaderboard-2026/ ; Artificial Analysis — https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice
  - Reviewer comparison (XTTS vs Chatterbox cloning) — https://localaimaster.com/blog/kokoro-vs-xtts-vs-chatterbox

---

## Cluster ranking against the two primary requirements (Mac M4 Max 36 GB)

| Model | Clone quality vs Chatterbox | Mac streaming path | Mac TTFA evidence | Commercial |
|---|---|---|---|---|
| **ZONOS2** | likely ≥ (SIM ~65 published) | zonos2.cpp Metal + mlx-audio, both stream | none published | Apache-2.0 |
| **Higgs Audio v3** | likely > (Elo 1042; v2 SIM 67.7) | mlx-audio v0.4.4, streams | none published | **non-commercial** |
| VibeVoice-Realtime-0.5B | no cloning | mlx-audio / speech-swift, streams | M2 Max RTF 0.53; "M4 Pro real-time" | MIT |
| VibeVoice-1.5B | < (Elo 970) | MLX, chunked | M2 Max INT4 RTFx 1.48 | MIT |
| Marvis 250M | < | MLX-native streaming | none, but tiny | Apache-2.0 |
| Sesame CSM-1B | < (unstable) | csm-mlx / mlx-audio stream | none | Apache-2.0 |
| Orpheus 3B | << (fine-tune only) | mlx-audio / llama.cpp Metal | "1.5–4× RT on M3" | Apache-2.0 |
| Dia 1.6B / Dia2 | ≈ / CUDA-only | Dia: MLX no stream; Dia2: none | none | Apache-2.0 |
| Maya1 | no cloning | none (mlx-lm LLM only) | none | Apache-2.0 |
| XTTS-v2 | < (Elo ~900) | CPU only, MPS broken | none | **CPML non-commercial** |

Recommendation for the next step: benchmark **ZONOS2 (mlx-audio `zonos2` and zonos2.cpp Metal, Q8_0)** and **Higgs Audio v3 (mlx-audio 4/8-bit)** on the M4 Max for first-chunk latency with a 5–15 s reference; they are the only two in this cluster that could beat Chatterbox on both axes, and neither has a published Apple-Silicon number.

**Out of cluster, flagged:** Fish Audio S2 Pro (highest open-weight Elo 1125, cloning, in mlx-audio v0.4.1); Step Audio EditX (Elo 1102, cloning, MLX via mlx-speech); Voxtral TTS (Mistral, Elo 1082, mlx-audio streaming via overlap-add); Arktts / Audio8-TTS (zero-shot cloning, mlx-audio v0.4.7); Chatterbox Multilingual v3 (mlx-audio v0.4.7 — check whether it adds a streaming path relevant to the existing Chatterbox stack); Kyutai Pocket TTS already has a `pocket_tts` model in mlx-audio.
