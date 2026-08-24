# Cluster 02 — LLM-based TTS from Chinese labs (Qwen3-TTS, CosyVoice, Fish/OpenAudio, IndexTTS, VoxCPM, Spark, Step/GLM/MiMo/LLaSA)

Researched 2026-08-24. Bar: Chatterbox-class clone quality and < ~0.9 s TTFA on an M4 Max 36 GB via a *streaming* Mac path.

**One cross-model reference table worth reading first** — the VoxCPM2 technical report (arXiv 2606.06928, 2026-06-05, Table 3) re-tabulates Seed-TTS-eval **test-en WER / SIM** for most models in this cluster under one SIM speaker model (values in %):
Qwen3-TTS 1.7B 1.23 / 71.7 · CosyVoice 3-0.5B 2.02 / 71.8 · CosyVoice 2 3.09 / 65.9 · IndexTTS2 2.23 / 70.6 · Fish Audio S2 0.99 / – (Fish does not report SIM) · OpenAudio-s1-mini 1.94 / 55.0 · Spark-TTS 3.14 / 57.3 · VoxCPM2 1.84 / 75.3 · VoxCPM1.5 2.12 / 71.4 · VoxCPM-0.5B 1.85 / 72.9 · MOSS-TTS 1.85 / 73.4 · OmniVoice 1.60 / 74.1 · LongCat-Audio-DiT 1.50 / 78.6 (best open SIM) · closed: Seed-TTS 2.25 / 76.2, MiniMax-Speech 1.65 / 69.2, CosyVoice3.5 1.57 / 73.8.
Caveat: each lab's own paper uses a different speaker-verification model (e.g. IndexTTS 2.5 reports itself at SS 0.855 and CosyVoice 3 at 0.811, while CosyVoice's own paper says 0.720 for the same model) — never compare SIM across papers, only within one table.

---

### Qwen3-TTS-12Hz 0.6B / 1.7B — Base / CustomVoice / VoiceDesign (Alibaba Qwen, released 2026-01-22; no newer open weights as of 2026-08-24)
- **What it is:** Qwen3 LLM "talker" (0.6B or 1.7B) predicting a 12.5 Hz, 16-codebook (1 semantic + 15 acoustic) tokenizer stream; waveform from a lightweight *causal ConvNet* decoder (no DiT), 24 kHz. A 25 Hz variant exists in the paper but only the 12 Hz models were released. "Dual-track" text/audio interleaving so audio tokens are emitted as text arrives.
- **Weights/license:** https://github.com/QwenLM/Qwen3-TTS and https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base (+ 0.6B-Base, 0.6B/1.7B-CustomVoice, 1.7B-VoiceDesign, Tokenizer-12Hz). Apache-2.0 — commercial OK. Note: the July 2026 "Qwen-Audio-3.0-TTS" (Flash/Plus, #1 on the Artificial Analysis TTS arena per press) is **hosted-only, no weights** (MarkTechPost 2026-07-20).
- **Size/memory:** 1.7B bf16 MLX bundle ~4.5 GB on disk; 0.6B ~1.5 GB (suckerfish/qwen3-tts-mlx README). mlx-audio README's 6-bit benchmark table shows 3.88–4.10 GB in use (device not stated). Easily fits 36 GB.
- **Voice cloning:** zero-shot from ≥3 s reference + transcript (ICL). Seed-TTS-eval test-en WER 1.24 (1.7B) / 1.32 (0.6B), test-zh 0.77 / 0.92 — Qwen's README/paper table (arXiv 2601.15621) reports WER only, **no SIM**. VoxCPM2 Table 3 puts Qwen3-TTS-1.7B at SIM 71.7 (en), S-MOS 4.69 (Table 12, 2nd behind VoxCPM2 4.74, IndexTTS2 4.71). Qwen's own multilingual set: SIM 0.775–0.829 across languages. Qualitative: TextToLab-style reviews say it "edged out Chatterbox" in cloning but "the gap is hard to hear"; a myByways M2 test (2026-02-02) reports preset voices carry "too strong a Chinese accent when speaking English" (presets, not clones). Fine-tune path: none official in the repo (not published).
- **Streaming:** *Architecturally* token-level (packet = 4 tokens = 320 ms of audio), paper first-packet latency 97 ms (0.6B) / 101 ms (1.7B) at concurrency 1, RTF 0.288 / 0.313, on an unspecified "single typical" GPU with vLLM-V0 + CUDA graphs. **But the official `qwen-tts` package does not stream**: maintainer on HF discussion #3 — "Audio is generated completely before being returned. Streaming will be supported after the pipeline is disaggregated"; GitHub issues #10 and #77 closed "not planned"; vLLM-Omni day-0 support is offline-only. Community CUDA forks fill the gap: rekuenkdr/Qwen3-TTS-streaming reports first-chunk 208 ms (5-frame first chunk, then 12-frame chunks, 21 ms Hann crossfade; GPU not stated), 570 ms baseline; dffdeeq/Qwen3-TTS-streaming (RTX 5090, ~6× faster than official, emit every 4 frames). Chunk-boundary clicks are acknowledged and handled with overlap-trim + crossfade.
- **Mac path:** **mlx-audio (Blaizzy)** — first-class model; all generate methods accept `stream=True` with `streaming_interval` (0.32 s ≈ 4 tokens suggested for Qwen3-TTS); tokenizer has `streaming_decode` with chunk_size/left_context; "ICL cache" (v0.4.4) and "incremental decoding throttling fixes" in recent releases; v0.5.0 released 2026-08-17. So **the Mac path does stream at chunk (≈4-token/320 ms) granularity**. Measured Mac numbers: M2 Max, 1.7B bf16 — RTF ≈ 0.55, ~37 ms per decode step after compile warm-up (soniqo/speech-swift docs, which build on the same MLX port); M3 Air 8 GB, 0.6B — RTF 0.4–0.7 at full 15 codebooks (eris-voice, custom MLX, has sentence-level streaming, no cloning); M2 Mac mini, 1.7B 8-bit — ~1000 chars/min (myByways). **No published Mac TTFA.** Rough estimate from the M2 Max step time: 4 tokens × 37 ms + prompt prefill (ref-audio ICL adds ~12.5 tok/s of reference) + ConvNet decode ≈ 0.25–0.45 s on M2 Max, likely less on M4 Max — *my estimate, not a measurement*. Other ports: AtomGradient/swift-qwen3-tts (Swift MLX, "streaming" only emits token events, audio delivered at end — not audio streaming), kapi2800 and suckerfish GUIs (wrap mlx-audio, no numbers). Known issues: Metal watchdog forces a ~500-token (~40 s) cap per call (soniqo); `split_pattern` ignored for CustomVoice/VoiceDesign (myByways); mlx-audio issue tracker ~72 open issues.
- **Controls/capabilities:** CustomVoice: 9 preset timbres + free-text `instruct` (emotion/style, e.g. "Very happy and excited"); VoiceDesign: build a voice from a description; Base: cloning. 10 languages + Chinese dialects. No inline tags. Speed/pitch: not exposed as parameters (instruct only). No multi-speaker dialogue mode.
- **Verdict vs the bar:** Clone quality ≈ Chatterbox (SIM 71.7, WER best-in-class); on Mac it is the **only model in this cluster whose maintained MLX port actually streams**, so it plausibly beats 0.9 s TTFA — but nobody has published a Mac TTFA, and the official CUDA repo still has no streaming API at all. Biggest caveat: you must measure it yourself; 1.7B at RTF ~0.55 on M2 Max leaves little headroom on a 36 GB M4 Max if you also run an LLM.
- **Sources:**
  - Qwen3-TTS GitHub README — https://github.com/QwenLM/Qwen3-TTS
  - Qwen3-TTS technical report (arXiv 2601.15621) — https://arxiv.org/html/2601.15621
  - HF discussion "Can't Stream The Model Locally" — https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/discussions/3
  - Issue #77 "Streaming support inference" (closed not planned) — https://github.com/QwenLM/Qwen3-TTS/issues/77 ; Issue #10 — https://github.com/QwenLM/Qwen3-TTS/issues/10
  - Changelog (no releases after Jan 2026) — https://qwenlm-qwen3-tts.mintlify.app/resources/changelog
  - rekuenkdr/Qwen3-TTS-streaming (208 ms first chunk, CUDA) — https://github.com/rekuenkdr/Qwen3-TTS-streaming
  - dffdeeq/Qwen3-TTS-streaming — https://github.com/dffdeeq/Qwen3-TTS-streaming
  - mlx-audio Qwen3-TTS README (stream=True, streaming_interval) — https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/README.md
  - mlx-audio streaming guide — https://blaizzy.github.io/mlx-audio/guides/streaming/
  - mlx-audio releases (v0.5.0 2026-08-17) — https://github.com/Blaizzy/mlx-audio/releases
  - Soniqo docs (M2 Max RTF 0.55, 37 ms/step) — https://soniqo.audio/guides/speak
  - myByways M2 test — https://mybyways.com/blog/qwen3-tts-with-mlx-audio-on-macos
  - eris-voice (M3 Air numbers) — https://github.com/eris-ths/eris-voice
  - swift-qwen3-tts — https://github.com/AtomGradient/swift-qwen3-tts
  - Qwen-Audio-3.0-TTS hosted-only (MarkTechPost 2026-07-20) — https://www.marktechpost.com/2026/07/20/alibabas-tongyi-lab-releases-qwen-audio-3-0-tts-a-hosted-text-to-speech-model-in-flash-and-plus-tiers-across-16-languages/
  - Qwen3-TTS vs Chatterbox review — https://texttolab.com/blog/qwen3-tts-review

---

### CosyVoice 2 (0.5B) and Fun-CosyVoice 3 (0.5B-2512) (Alibaba FunAudioLLM; CosyVoice2 Dec 2024, Fun-CosyVoice3-0.5B-2512 Dec 2025)
- **What it is:** Qwen2.5-0.5B LLM → supervised semantic tokens (S3 tokenizer, 25 Hz) → chunk-aware causal flow-matching → HiFT/vocoder, 24 kHz. CosyVoice 3 adds RL post-training and a 1.5B variant in the paper.
- **Weights/license:** https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (base + RL), https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B; Apache-2.0, commercial OK. **The 1.5B CosyVoice 3 has not been released** (roadmap lists v3.0 as 0.5B only; no 1.5B on HF as of Aug 2026).
- **Size/memory:** 0.5B LLM + flow/vocoder; MLX 8-bit / 4-bit bundles exist (mlx-community). Mac memory not published; small (< 3 GB expected).
- **Voice cloning:** zero-shot from reference audio (+ transcript for "zero-shot" mode; cross-lingual mode without transcript). CosyVoice 3 paper Table 4 (WavLM SS): 0.5B test-en WER 2.02 / SS 0.720; 1.5B 2.21 / 0.720; 1.5B-RL 1.45 / 0.695; CosyVoice 2: 2.57 / 0.652. Dec-2025 HF card (RL variant): en WER 1.68, SIM 69.5. VoxCPM2 table: CV3-0.5B 71.8, CV2 65.9. Consensus: clean timbre, prosody somewhat flat vs newer models; RL helps WER at a small SIM cost. Fine-tune: official SFT scripts in repo.
- **Streaming:** native bidirectional (text-in and audio-out) streaming, "latency as low as 150 ms" (HF card/README; hardware not stated; CosyVoice 2 paper gives the first-packet formula L = M·d_lm + M·d_fm + M·d_voc and shows streaming ≈ offline quality on test-en). Granularity: token chunks (chunk-aware flow matching with chunk-M / chunk-2M masks). GitHub issues report P99 outliers under concurrency on a 4090 (#1835). vLLM and TensorRT-LLM paths for CUDA.
- **Mac path:** No official MLX/MPS support (issue #1166 "support for apple's MLX" open, stale, unanswered since 2025-04). Community: **mlx-audio-plus** (DePasqualeOrg fork, MIT) has `cosyvoice2` and `cosyvoice3` modules; mlx-community/CosyVoice2-0.5B-4bit and Fun-CosyVoice3-0.5B-2512-{4,8}bit (converted by depasquale, ~8 months ago) support cross-lingual, zero-shot clone, instruct and voice-conversion modes. **Streaming on the MLX port is not documented** (the model cards only show `generate_audio(...)` whole-utterance calls). Also a GGUF via CrispASR (cstr/cosyvoice3-0.5b-2512-GGUF; described as streaming, Metal not mentioned). Mac RTF/TTFA: not published.
- **Controls/capabilities:** instruct mode: language, 18+ Chinese dialects/accents, emotions, speed, volume; fine-grained pinyin / CMU phoneme control; 9 languages. Inline event tags ([laughter]/[breath]) were a CosyVoice 2 feature — not re-verified for v3 in this pass. Voice conversion mode. No multi-speaker dialogue.
- **Verdict vs the bar:** Clone SIM (~0.72) is on par with Qwen3-TTS but WER is worse; CUDA streaming is mature (150 ms), but the Mac port is a community fork with whole-utterance generation and no published numbers — no evidence it beats 0.9 s TTFA. Biggest caveat: Mac streaming does not exist yet.
- **Sources:**
  - Fun-CosyVoice3-0.5B-2512 HF card — https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
  - CosyVoice GitHub README — https://github.com/FunAudioLLM/CosyVoice
  - CosyVoice 3 paper (arXiv 2505.17589) — https://arxiv.org/html/2505.17589v1
  - CosyVoice 2 paper (arXiv 2412.10117) — https://arxiv.org/html/2412.10117v1
  - MLX request issue #1166 — https://github.com/FunAudioLLM/CosyVoice/issues/1166
  - mlx-community/Fun-CosyVoice3-0.5B-2512-8bit — https://huggingface.co/mlx-community/Fun-CosyVoice3-0.5B-2512-8bit/blob/main/README.md
  - mlx-audio-plus (DePasqualeOrg) — https://github.com/DePasqualeOrg/mlx-audio-plus
  - cstr/cosyvoice3-0.5b-2512-GGUF — https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF
  - P99 latency issue #1835 — https://github.com/FunAudioLLM/CosyVoice/issues/1835

---

### Fish Audio S2 Pro (successor to Fish Speech / OpenAudio S1 & S1-mini) (Fish Audio, S2 beta 2026-03-10; tech report 2026-03-09)
- **What it is:** Dual-AR: Qwen3-4B "slow" AR over time + 400M 4-layer "fast" AR over 10 RVQ codebooks (1 semantic + 9 acoustic) of a ModifiedDAC codec at ~21.5 Hz, 44.1 kHz output; GRPO RL-aligned. ~4.4B total.
- **Weights/license:** https://huggingface.co/fishaudio/s2-pro ; https://github.com/fishaudio/fish-speech. **Fish Audio Research License** — research/non-commercial free; commercial deployment requires a paid license (business@fish.audio). Not commercial-OK out of the box. (OpenAudio S1-mini 0.5B remains under its earlier NC license; no "S2-mini" exists — only S2-Pro is published.)
- **Size/memory:** ~4.4B params. MLX bf16 bundle 11 GB (mlx-community/fish-audio-s2-pro-bf16); 8-bit ~4.5–6.7 GB; majentik cards say "≥16 GB unified, comfortable at 24 GB+". Fits 36 GB.
- **Voice cloning:** zero-shot from 10–30 s reference (README). Seed-TTS-eval test-en WER **0.99 (best of all systems)**, test-zh 0.54; Fish reports no SIM on Seed-TTS-eval. Other SIM: MiniMax-MLS English SIM 79.7 (VoxCPM2 Table 7), long-form SIM-mean 0.523 (own report), S-MOS 4.69 (VoxCPM2 Table 12). Audio Turing Test 0.515, EmergentTTS-Eval 81.88% win rate. Consensus: extremely expressive/controllable, clones delivery as well as timbre; S1-mini (0.5B) was weak on SIM (55.0). Fine-tune: official fine-tuning code released with S2.
- **Streaming:** native via SGLang-Omni engine: TTFA ~100 ms, RTF 0.195, 3000+ acoustic tok/s on a **single H200**; vLLM-Omni also supported. Token-level (per ~46 ms frame).
- **Mac path:** (a) **mlx-audio** added "Fish Audio S2 Pro TTS" in v0.4.1 (2026-03-14; module `fish_qwen3_omni`), with cloning, speaker tags, long-form chunking — README does not mention streaming or Mac numbers; (b) **appautomaton/mlx-speech** (MIT, pure MLX, int8 4.5 GB) claims "~21 tokens/s on Apple Silicon with int8" (≈ RTF 1.0 at 21.5 Hz; chip not stated), explicitly *non-streaming*; (c) groxaxo deployment notes (8-bit MLX, chip not stated): **RTF 3.77, ~5.7 semantic tok/s, 90.7 s for 24.1 s of audio, 23 s cold start**, "4-bit decodes to noise on MLX", local server is "an offline workhorse", no streaming; (d) community 8-bit/4-bit conversions (majentik, aufklarer). PyTorch MPS for S2 is not documented in the repo. No Mac TTFA published; every Mac path is whole-utterance.
- **Controls/capabilities:** 15,000+ free-form inline tags ([whisper], [excited], [angry], [laughing], [singing], [inhale], [emphasis], [pause]) with sub-word placement; natural-language style descriptions; multi-speaker via <|speaker:i|> and multi-turn context; 80+ languages (Tier 1 en/zh/ja); temperature/top_p/top_k/speed.
- **Verdict vs the bar:** Best intelligibility and by far the richest expressive control in the cluster, but a 4.4B model that runs at or well below real time on MLX with no streaming path — loses badly on Mac TTFA — and the license is non-commercial. Biggest caveat: license + size.
- **Sources:**
  - fish-speech GitHub — https://github.com/fishaudio/fish-speech
  - fishaudio/s2-pro HF card (license, H200 numbers) — https://huggingface.co/fishaudio/s2-pro
  - Fish Audio S2 technical report (arXiv 2603.08823) — https://arxiv.org/html/2603.08823v2
  - fish-speech releases (S2 beta 2026-03-10) — https://github.com/fishaudio/fish-speech/releases
  - fish.audio/s2 product page — https://fish.audio/s2/
  - mlx-community/fish-audio-s2-pro-bf16 — https://huggingface.co/mlx-community/fish-audio-s2-pro-bf16
  - mlx-community/fishaudio-s2-pro-8bit-mlx — https://huggingface.co/mlx-community/fishaudio-s2-pro-8bit-mlx
  - mlx-audio fish module — https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/fish_qwen3_omni
  - appautomaton/mlx-speech Fish S2 doc (21 tok/s int8) — https://github.com/appautomaton/mlx-speech/blob/main/docs/fish-s2-pro.md
  - groxaxo local-deploy notes (RTF 3.77, 5.7 tok/s) — https://github.com/groxaxo/fish-s2-pro-mlx-local-deploy
  - majentik 8-bit / 4-bit cards — https://huggingface.co/majentik/fishaudio-s2-pro-MLX-8bit , https://huggingface.co/majentik/fishaudio-s2-pro-MLX-4bit
  - VoxCPM2 report Table 3/7/12 (cross-model SIM) — https://arxiv.org/pdf/2606.06928

---

### IndexTTS 2 (2025-09-08) and IndexTTS 2.5 (Bilibili IndexTeam; 2.5 paper 2026-01-07, open release 2026-08-10)
- **What it is:** GPT-style AR (0.8B in 2.5) over semantic codec tokens (50 Hz in v2 → 25 Hz in 2.5) → Semantic-to-Mel (U-DiT in v2 → Zipformer in 2.5) → BigVGAN vocoder. Timbre/emotion disentangled via GRL; explicit duration control.
- **Weights/license:** https://huggingface.co/IndexTeam/IndexTTS-2.5 ; https://github.com/index-tts/index-tts. "bilibili Model Use License Agreement" — commercial use only by arrangement (indexspeech@bilibili.com). Not plainly commercial-OK.
- **Size/memory:** ~0.8B GPT + S2M + BigVGAN; official CUDA needs ~6 GB VRAM; MLX bundle ~5 GB (index-tts-2.5-mlx), IndexTTS2 fp16 MLX 2.0 GB (vanch007).
- **Voice cloning:** zero-shot from a clip up to ~15 s. IndexTTS 2.5 paper: test-en WER 1.889 / SS 0.855, test-zh 1.426 / 0.848 (its own speaker model; it lists CosyVoice 3 at 0.811 in the same table). Cross-lab table (VoxCPM2): IndexTTS2 SIM 70.6, S-MOS 4.71 (2nd). Consensus: strong timbre clone with the best emotion controllability; long text is split and concatenated with silences so prosody is not modelled across segments (HF card).
- **Streaming:** **none** on any path. README documents batch RTF only (2.5 bf16 RTF 0.2065 vs v2 0.3257 on RTX 4090; A10 total RTF 0.136); xinference streaming request #4207 "closed as not planned"; no first-packet numbers exist.
- **Mac path:** (a) **index-tts-2.5-mlx** (PyPI, yunfengwang, v0.1.1 2026-08-14): int8 GPT, pure MLX, RTF ≈ 0.45–0.47 on **M5 Pro**, "2.4× faster than official PyTorch MPS"; stage breakdown for ~3 s of audio: GPT 0.35 s + CFM 0.48 s + BigVGAN 0.56 s ≈ 1.4 s wall — whole-utterance, no streaming, deliberately no emotion tags (emotion comes from the reference); (b) vanch007/mlx-indextts2 via solar2ain/mlx-indextts: M3 Max 128 GB, 8-bit RTF 0.91–1.04, fp16 1.46–1.58 (IndexTTS2, non-streaming); (c) official PyTorch on MPS works but slower; (d) mlx-audio has an `indextts` module whose files (gpt2, bigvgan, conformer, perceiver) look like IndexTTS 1.x, not 2/2.5 — unverified.
- **Controls/capabilities:** 8-dim emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]; emotion from a separate reference clip with `emo_alpha` 0–1; emotion inferred from text (`use_emo_text`) or from an explicit `emo_text` description; `duration_factor` 0.5–2.0; pinyin / CMU / kana pronunciation control; zh/en/ja/es/ar (2.5).
- **Verdict vs the bar:** Clone/emotion quality is top-tier, but there is no streaming anywhere and the best Mac port needs ~1.4 s for a 3 s utterance on M5 Pro (TTFA = whole utterance), so it loses on TTFA; license is also restrictive. Biggest caveat: non-streaming by design.
- **Sources:**
  - index-tts GitHub README (2.5 release 2026-08-10, RTF table) — https://github.com/index-tts/index-tts
  - IndexTTS 2.5 technical report (arXiv 2601.03888) — https://arxiv.org/html/2601.03888v2
  - IndexTeam/IndexTTS-2.5 HF card — https://huggingface.co/IndexTeam/IndexTTS-2.5
  - index-tts-2.5-mlx on PyPI — https://pypi.org/project/index-tts-2.5-mlx/
  - vanch007/mlx-indextts2-standard-fp16 (M3 Max RTF) — https://huggingface.co/vanch007/mlx-indextts2-standard-fp16
  - xinference streaming request #4207 (closed not planned) — https://github.com/xorbitsai/inference/issues/4207
  - mlx-audio indextts module — https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/indextts

---

### VoxCPM-0.5B (2025-09) / VoxCPM1.5 (2025-12) / VoxCPM2 (OpenBMB; VoxCPM2 weights 2026-04, report arXiv 2606.06928 2026-06-05)
- **What it is:** Tokenizer-free diffusion-autoregressive: LocEnc → Text-Semantic LM (MiniCPM-4-0.5B in 0.5B/1.5; MiniCPM-4-1B in v2) with FSQ bottleneck → Residual Acoustic LM → Local Diffusion Transformer generating continuous AudioVAE latents patch by patch. Patch = 4 × 25 Hz frames = 6.25 Hz LM rate (160 ms per AR step) in 1.5/2. VoxCPM2 = 2B, AudioVAE V2 encodes 16 kHz, decodes 48 kHz; VoxCPM1.5 44.1 kHz; 0.5B 16 kHz.
- **Weights/license:** https://github.com/OpenBMB/VoxCPM , https://huggingface.co/openbmb/VoxCPM2 , /VoxCPM1.5 , /VoxCPM-0.5B. Apache-2.0, commercial OK.
- **Size/memory:** VoxCPM2 2B (bf16 MLX 4.96 GB; 8-bit 3.23 GB; 4-bit 2.30 GB); VoxCPM1.5 ~0.8B backbone (mlx-community 8-bit card says 0.3B/1.02 GB — LM-only count); 0.5B ~5 GB VRAM. CUDA VRAM: ~8 / 6 / 5 GB.
- **Voice cloning:** zero-shot from a short reference; v2 adds an isolated reference-audio pathway (no transcript needed) plus optional "reference + continuation" (with transcript) for best SIM. Seed-TTS-eval test-en (own report, Table 3): **VoxCPM2 WER 1.84 / SIM 75.3** (highest SIM of the AR models, 2nd open overall behind LongCat-Audio-DiT 78.6), test-zh 0.97 / 79.5; VoxCPM1.5 2.12 / 71.4; 0.5B 1.85 / 72.9. S-MOS 4.74 (best), N-MOS 4.78. Reference-only recipe: SIM 75.3; ref+continuation 79.5. Known issue #272 (open, no maintainer reply): chirp/click at segment start and tail-of-reference leakage in one-shot cloning. Fine-tune: official SFT + LoRA scripts.
- **Streaming:** native `generate_streaming()` — patch-level (each 160 ms latent patch is decoded immediately by a stateful causal VAE decoder). RTF 0.30 (PyTorch) / 0.13 (Nano-vLLM) on RTX 4090 for v2; 0.15 for 1.5; 0.17 for 0.5B. First-packet latency not published; a user in #272 reports ~250 ms "response time" on an L4. vLLM-Omni serving.
- **Mac path:** (a) official PyTorch supports **MPS** (`--device auto`); (b) **mlx-audio**: VoxCPM 0.5B and 1.5 (v0.2.7 conversions, contributed by voxmenthe, Dec 2025) and VoxCPM2 (added v0.4.4; `voxcpm2` module) — mlx-community/VoxCPM2-8bit card: **bf16 0.48× real-time, 8-bit 0.85×, 4-bit 0.90×** at 7 diffusion timesteps (numbers are speed multipliers, i.e. 8-bit is *slower than real time* — RTF ≈ 1.2; chip not stated); the aufklarer bf16 port advertises "patch-level decoding for low-latency synthesis"; whether mlx-audio's generic `stream=True` truly streams VoxCPM patches is not documented. OpenBMB's own docs say the MLX-Audio backend covers 1.0/1.5 and "VoxCPM2 not yet available" (docs lag mlx-audio). No Mac TTFA published.
- **Controls/capabilities:** voice design from natural-language description; controllable cloning (reference + style instruction: emotion, pace, expression); InstructTTSEval-EN 84.2/83.2/71.4 (best); 30 languages + 9 Chinese dialects; CFG scale (α default 2.0, 1.5–3.0 useful), inference timesteps knob; preliminary singing. No inline tags; no multi-speaker mode.
- **Verdict vs the bar:** Beats Chatterbox-class SIM on paper (75.3) and has a real patch-level streaming design, but the 2B model runs around/below real time on MLX and the port's streaming is undocumented — likely no TTFA win on Mac today; VoxCPM1.5 (0.8B, SIM 71.4) is the size to test. Biggest caveat: Mac speed of the 2B model and the open click/leakage bug.
- **Sources:**
  - VoxCPM GitHub README — https://github.com/OpenBMB/VoxCPM
  - VoxCPM2 technical report (Tables 1, 3, 4, 11, 12) — https://arxiv.org/pdf/2606.06928
  - OpenBMB MLX-Audio deployment doc — https://voxcpm.readthedocs.io/en/latest/deployment/mlx_audio.html
  - mlx-community/VoxCPM2-8bit (Mac speed table) — https://huggingface.co/mlx-community/VoxCPM2-8bit
  - mlx-community/VoxCPM1.5-8bit — https://huggingface.co/mlx-community/VoxCPM1.5-8bit
  - aufklarer/VoxCPM2-MLX-bf16 — https://huggingface.co/aufklarer/VoxCPM2-MLX-bf16
  - Issue #272 chirp/click + consistency — https://github.com/OpenBMB/VoxCPM/issues/272
  - OpenBMB MLX announcement — https://x.com/OpenBMB/status/2000913854973534666
  - mlx-audio releases (VoxCPM2 in v0.4.4) — https://github.com/Blaizzy/mlx-audio/releases

---

### Spark-TTS 0.5B (SparkAudio; paper 2025-03-04, last news 2025-03-12)
- **What it is:** Qwen2.5-0.5B LLM predicting single-stream BiCodec tokens (semantic + global speaker tokens); audio reconstructed directly from LLM codes (no separate flow/vocoder stage), 16 kHz.
- **Weights/license:** https://github.com/SparkAudio/Spark-TTS ; HF SparkAudio/Spark-TTS-0.5B. Apache-2.0, commercial OK.
- **Size/memory:** 0.5B; ~1–2 GB expected on Mac (not published).
- **Voice cloning:** zero-shot, cross-lingual/code-switch. Paper: test-en WER 1.98 / SIM 0.584; VoxCPM2 table: 3.14 / 57.3 — clearly the weakest SIM in this cluster. Consensus: intelligible, but thin timbre match.
- **Streaming:** none in the repo; Triton serving RTF 0.136 with 876 ms latency at concurrency 1 on an L20 (whole utterance).
- **Mac path:** mlx-audio and mlx-audio-plus both ship a `spark` module (bicodec, audio_tokenizer); no README, no streaming statement, no Mac numbers published. Project effectively dormant since March 2025.
- **Controls/capabilities:** coarse attribute tokens — gender, pitch level, speaking-rate level (plus fine-grained pitch/speed values) to create synthetic speakers; zh/en only.
- **Verdict vs the bar:** Loses on clone quality (SIM ~0.58) and has no streaming anywhere; only interesting for its gender/pitch/speed tokens. Biggest caveat: SIM.
- **Sources:**
  - Spark-TTS GitHub — https://github.com/SparkAudio/Spark-TTS
  - Spark-TTS paper (arXiv 2503.01710) — https://arxiv.org/pdf/2503.01710
  - mlx-audio spark module — https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/spark
  - VoxCPM2 report Table 3 — https://arxiv.org/pdf/2606.06928

---

### Step-Audio-EditX 3B (StepFun; Nov 2025, updated model 2026-01-29)
- **What it is:** 3B audio LLM over a dual-codebook tokenizer + flow-matching decoder; RL-trained for iterative *editing* (emotion, style, paralinguistics) with zero-shot TTS/cloning.
- **Weights/license:** https://github.com/stepfun-ai/Step-Audio-EditX ; Apache-2.0, commercial OK.
- **Size/memory:** 3B; 12–16 GB VRAM on CUDA; mlx-speech int8 bundle (size not published).
- **Voice cloning:** zero-shot (zh/en/Sichuanese/Cantonese); claims to beat MiniMax/Doubao on cloning + emotion in its own eval; no Seed-TTS-eval numbers published.
- **Streaming:** none (README; mlx-speech: "this family does not expose streaming in the current public API").
- **Mac path:** appautomaton/mlx-speech has an int8 MLX port (clone + edit, non-stream, "gated and manual"); no Mac numbers. Official: Linux/CUDA only.
- **Controls/capabilities:** 14+ emotions, 30+ styles, 10 paralinguistic tags (breath, laughter, sigh, throat-clear...), speed/denoise/VAD edits — applied as post-hoc edits to an existing clip.
- **Verdict vs the bar:** Non-streaming, CUDA-first; irrelevant for TTFA, potentially useful as an offline "emotion editor". Sources: https://github.com/stepfun-ai/Step-Audio-EditX ; https://arxiv.org/abs/2511.03601 ; https://github.com/appautomaton/mlx-speech/blob/main/docs/step-audio-editx.md

### GLM-TTS 1.5B (Zhipu / zai-org; open-sourced 2025-12-11, report arXiv 2512.14291)
- **What it is:** Llama-architecture 1.5B AR → speech tokens → flow matching → vocoder; GRPO multi-reward RL (SIM, CER, emotion, laughter).
- **Weights/license:** https://huggingface.co/zai-org/GLM-TTS ; https://github.com/zai-org/GLM-TTS — HF card says MIT, GitHub says Apache-2.0; either way commercial OK.
- **Size/memory:** 1.5B (report); VRAM not published.
- **Voice cloning:** zero-shot from 3–10 s. Seed-TTS-eval **zh** CER 1.03 / SIM 76.1 (RL: 0.89 / 76.4); **no English numbers published**; "primarily Chinese, English mixed text".
- **Streaming:** "supports real-time streaming audio generation" — no latency numbers, chunk sizes not published; vLLM-Omni recipe exists.
- **Mac path:** none found (no MLX/MPS mention; Ascend NPU documented).
- **Verdict vs the bar:** CUDA-only and Chinese-first; skip for English cloning on a Mac. Sources: https://huggingface.co/zai-org/GLM-TTS ; https://github.com/zai-org/GLM-TTS ; https://arxiv.org/html/2512.14291v1

### MiMo-Audio 7B (Xiaomi; 2025)
- **What it is:** 7B audio-language LLM (+1.2B tokenizer) trained on 100M+ hours; few-shot speech continuation, voice conversion, instruct-TTS. Apache-2.0.
- **Cloning/TTS:** instruct-TTS SOTA claims (InstructTTSEval-ZH 75.7/74.3/61.5 in VoxCPM2 Table 9); no Seed-TTS-eval SIM published.
- **Streaming / Mac:** no streaming numbers; requires Linux, CUDA ≥ 12, flash-attn; no Mac path. 7B + tokenizer would fit 36 GB in 4-bit but no port exists.
- **Verdict vs the bar:** research model, CUDA-only — not a candidate. Sources: https://github.com/XiaomiMiMo/MiMo-Audio ; https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct

### LLaSA 1B / 3B / 8B (HKUST Audio; Feb 2025) and LLaSA+ (Aug 2025)
- **What it is:** LLaMA 1B/3B/8B extended with XCodec2 single-codebook (65,536) tokens at 50 Hz; 250k h zh/en. LLaSA+ (arXiv 2508.06262) adds a plug-in multi-token-prediction module + verification for faster, streaming token-level generation on the frozen LLaSA.
- **Weights/license:** https://huggingface.co/HKUSTAudio/Llasa-1B (also 3B, 8B, 1B-Multilingual) — **CC-BY-NC-4.0**, no commercial use. LLaSA+ weights/code availability: not verified.
- **Voice cloning:** zero-shot via speech prompt; no Seed-TTS-eval numbers on the cards (paper reports test-en results; not re-fetched).
- **Streaming:** base LLaSA none (50 Hz single stream + XCodec2 decode would allow chunking, but nothing shipped); LLaSA+ streaming numbers are in the PDF's tables, not extractable here — not published in a form I could verify.
- **Mac path:** none found (no MLX conversion, no llama.cpp GGUF+XCodec2 pipeline surfaced; mlx-audio's generic `llama` module is not documented for LLaSA).
- **Verdict vs the bar:** NC license and no Mac/streaming path — skip. Sources: https://huggingface.co/HKUSTAudio/Llasa-1B ; https://arxiv.org/pdf/2508.06262

---

## Cluster summary vs the bar (M4 Max 36 GB, TTFA < ~0.9 s, Chatterbox-class cloning)

| Model | en SIM (VoxCPM2 table) | CUDA streaming | Mac runtime | Mac path streams? | Mac numbers |
|---|---|---|---|---|---|
| Qwen3-TTS 1.7B | 71.7 | arch yes / official repo no (forks: 208 ms) | mlx-audio (maintained, v0.5.0 2026-08-17) | **yes, chunk ≈ 320 ms** | RTF ~0.55, 37 ms/step on M2 Max; TTFA not published |
| VoxCPM2 2B | 75.3 | yes, patch 160 ms | mlx-audio + MPS | undocumented | 8-bit ≈ 0.85× real-time (chip n/a) |
| VoxCPM1.5 0.8B | 71.4 | yes | mlx-audio | undocumented | not published |
| CosyVoice3 0.5B | 71.8 | yes, 150 ms | mlx-audio-plus (fork) | no | not published |
| IndexTTS 2.5 0.8B | (v2: 70.6) | no | index-tts-2.5-mlx | no | RTF 0.45 M5 Pro, ~1.4 s per 3 s clip |
| Fish S2 Pro 4.4B | – (WER 0.99) | yes, 100 ms H200 | mlx-audio / mlx-speech | no | RTF 3.77 (8-bit, chip n/a) |
| Spark-TTS 0.5B | 57.3 | no | mlx-audio | no | not published |

Only **Qwen3-TTS via mlx-audio** offers a maintained, streaming Mac path today; it is the one to benchmark next (1.7B-Base bf16 and 0.6B-Base, `stream=True, streaming_interval=0.32`, with a 5–10 s reference). VoxCPM1.5/2 are the clone-quality upgrade to test if the mlx-audio port's `stream=True` proves to be patch-level.

**Out of cluster, flagged:** LongCat-Audio-DiT 3.5B (Meituan; SIM 78.6 = best open, in mlx-audio and mlx-speech, flow-matching so likely non-streaming); Higgs Audio v3 (mlx-audio v0.4.4 with "overlap-add mid-generation streaming"); Voxtral TTS 4B (Mistral; mlx-audio streaming overlap-add); ArkTTS/Audio8-TTS zero-shot cloning (mlx-audio v0.4.7); MOSS-TTS 8B (SIM 73.4); OmniVoice 0.8B (SIM 74.1, 646 languages, masked-NAR, in mlx-audio); Zonos2 and dots.tts (mlx-audio `zonos2`, mlx-speech "bounded waveform streaming" for dots.tts); Kyutai Pocket TTS is present in mlx-audio as `pocket_tts`.
