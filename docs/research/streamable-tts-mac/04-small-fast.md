# Cluster 04 — Small / fast on-device TTS (≤1B) where low TTFA is the selling point: which of them also clone?

Research date: 2026-08-24. Bar to beat: Chatterbox Flash on this Mac (MLX fp16) ≈ 0.9 s first audio,
whole-utterance RTF ~0.28; Chatterbox-class clone quality. All numbers below are quoted from the page
they came from, with the date of that page; anything I derived is labelled "derived".

**One-paragraph summary.** In this size class only three models both clone zero-shot *and* have a
real streaming path: **NeuTTS Air/Nano** (llama.cpp GGUF, token-level streaming with 25-frame chunks,
Mac CPU path documented, Apache-2.0 for Air), **KaniTTS-2** (cloning via WavLM speaker embedding, but
streaming exists only in community vLLM servers — CUDA-only; the Mac MLX port is for the old
non-cloning v1 model), and the brand-new **Gepard 1.0** (same lab as Kani, 32 ms TTFA — but vLLM/CUDA
only, and its own eval table puts its speaker similarity at 0.585 vs Chatterbox 0.796). The
flow-matching cloners (**F5-TTS**, **LuxTTS/ZipVoice**) have the best SIM numbers in the cluster but
are whole-chunk generators: "streaming" is sentence-chunk granularity and the first packet costs one
full generation. **Kokoro, Soprano, Supertonic, Piper** do not clone (Supertonic's paid Voice Builder
closes 2026-08-31). Nothing here is verified to beat 0.9 s TTFA on Apple Silicon *while* matching
Chatterbox's clone quality; NeuTTS Air is the only candidate where a sub-second streamed first chunk
on the M4 Max is plausible, and its published speaker similarity (47.5 % WavLM cosine, Gradium
Apr 2026) is well below the flow-matching cloners.

---

### NeuTTS Air / NeuTTS Nano / NeuTTS-2E (Neuphonic, repo last commit 2026-07-22)
- **What it is:** Qwen-0.5B-class LM backbone (Air: ~360M active / ~552M with embeddings; Nano: ~120M
  active / ~229M; 2E: ~125M active, text-input, emotion-conditioned) → NeuCodec, a 50 Hz single-codebook
  neural codec, 24 kHz output; phoneme input (Air/Nano, needs espeak-ng), text input (2E).
- **Weights/license:** https://huggingface.co/neuphonic/neutts-air — Apache-2.0 (Air; commercial OK).
  Nano and 2E: "NeuTTS Open License 1.0" (HF shows "license: other"; terms not spelled out on the
  card — treat commercial use as unverified). GGUF Q4/Q8 for all; ONNX codec decoder (fp32/int8).
  Outputs are Perth-watermarked by default.
- **Size/memory:** 748M-param HF checkpoint (Air, bf16); GGUF Q4/Q8 backbone + NeuCodec. Mac memory not
  published; derived: Air Q8 GGUF ≈ 0.6 GB + codec, well under 2 GB.
- **Voice cloning:** zero-shot, 3–15 s mono 16–44 kHz WAV, **reference transcript required** (ref text
  goes into the prompt; ref codes pre-encodable to `.pt`). Gradium's on-device benchmark (Seed-TTS
  test-en, 1 008 utts, Whisper-large-v3 WER, WavLM-large cosine SIM, April 2026): **NeuTTS Air WER
  2.18 % / SIM 47.51 %; NeuTTS Nano WER 1.71 % / SIM 40.15 %** (vs Kani-TTS2 4.97 % / 40.73 %). No
  Neuphonic-published SIM/WER/MOS. Qualitative: roundups praise "voice similarity" but note artefacts
  on noisy references; the Gradium SIM figure is the only quantitative point and it is not
  cross-comparable to the SIM-o numbers in the F5/ZipVoice papers (different pipelines). Clones timbre;
  no style/emotion control on Air/Nano; 2E has emotions but **fixed speakers only (no cloning)**.
  Fine-tune path: none documented.
- **Streaming:** native, token-level, **GGUF backbone only** (torch backend raises
  `NotImplementedError` for streaming). From `neutts/neutts.py` (fetched 2026-08-24):
  `streaming_frames_per_chunk = 25` (+5 look-forward) → first decode after 30 codec tokens = 0.5 s of
  audio; lookback 50 frames, 1-frame overlap, `_linear_overlap_add` crossfade between chunks;
  watermark applied per chunk. The example script logs TTFA/RTF but **no TTFA number is published**.
  Chunk-join quality: overlap-add with 1-frame overlap — no complaints found, but not independently
  measured.
- **Mac path:** llama-cpp-python. README explicitly recommends **Metal OFF** on M-series
  (`-DGGML_METAL=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple`, i.e. Accelerate CPU). Published Mac
  number (README, Q4, llama-bench, 500 prefill / 250 gen, **LM only, codec excluded**): iMac M4 16 GB
  CPU — **Air 111 tok/s, Nano 195 tok/s**. Derived: at 50 tokens/s of audio that is ≈2.2× realtime
  (Air) / ≈3.9× (Nano) for the LM alone; the first 30-token chunk needs ≈0.27 s of decode (Air) plus
  prefill (ref codes + text, not published) plus one ONNX codec decode — a sub-second TTFA on an
  M4 Max is plausible but **not published**. Maintainer: Neuphonic (active; 2E merged 2026-07-22).
  Known issues: watermark dependency missing under `uv sync`; unofficial "neutts.com" clones.
- **Controls/capabilities:** Air/Nano: none beyond sampling (temperature, top-k). 2E: 7 emotions
  (angry, disgusted, fearful, happy, neutral, sad, surprised) via `emotion=` arg, 4 fixed speakers,
  English only, "early alpha". Languages: Air EN; Nano EN/ES/FR/DE (+ JA/KO/ZH listed on Neuphonic
  site). No speed/pitch control, no inline tags, single speaker.
- **Verdict vs the bar:** Likely **beats the 0.9 s TTFA bar** on the M4 Max (true token streaming,
  0.5 s first chunk, CPU-only path already ≈2× realtime) but **loses on clone quality** (47.5 % WavLM
  SIM; timbre-only). Biggest caveat: no published Mac TTFA and Metal is officially discouraged — you
  are running the LM on CPU cores.
- **Sources:**
  - GitHub neuphonic/neutts (README, benchmarks, Metal advice) — https://github.com/neuphonic/neutts
  - neutts/neutts.py (streaming constants) — https://raw.githubusercontent.com/neuphonic/neutts/main/neutts/neutts.py
  - examples/README.md, basic_streaming_example.py — https://raw.githubusercontent.com/neuphonic/neutts/main/examples/README.md
  - HF neuphonic/neutts-air — https://huggingface.co/neuphonic/neutts-air
  - HF neuphonic/neutts-nano — https://huggingface.co/neuphonic/neutts-nano
  - HF neuphonic/neutts-2e — https://huggingface.co/neuphonic/neutts-2e
  - Recent commits — https://api.github.com/repos/neuphonic/neutts/commits
  - Gradium on-device benchmark 2026 — https://gradium.ai/content/on-device-tts-benchmark-2026
  - Neuphonic NeuTTS Nano page — https://www.neuphonic.com/models/neutts-nano

### KaniTTS-2 (nineninesix.ai, HF updated 2026-02-19)
- **What it is:** 400M — LiquidAI LFM2-350M causal LM (4 tokens/frame, learnable RoPE theta) → NVIDIA
  NeMo NanoCodec (FSQ, 22.05 kHz, 0.6 kbps, 12.5 fps).
- **Weights/license:** https://huggingface.co/nineninesix/kani-tts-2-en — HF card says `lfm1.0`
  (LFM Open License); GitHub package says Apache-2.0; codec under NVIDIA Open Model License. Check
  LFM1.0 revenue clause before commercial use.
- **Size/memory:** 0.4B; "3 GB VRAM" (HF card), "~4–8 GB for bf16 inference" (repo). Mac memory not
  published.
- **Voice cloning:** zero-shot via a 128-d WavLM speaker embedding (`Orange/Speaker-wavLM-tbr`),
  3–30 s reference (10–20 s recommended), **no transcript needed**; repo suggests averaging 5–10 refs.
  Gradium (Seed-TTS test-en, Apr 2026): **WER 4.97 % / SIM 40.73 %** — worst WER and lowest SIM in
  that four-model table. No nineninesix-published SIM/WER. Consensus: timbre transfer from an embedding
  (no style/prosody cloning).
- **Streaming:** **not in the official `kani-tts-2` package** (generates whole utterances). Community
  vLLM servers stream: mohammed-bahumaish/kani-tts-2-vllm — SSE/WebSocket, 25 frames per decode
  iteration (= 2.0 s audio at 12.5 fps) with 15-frame lookback crossfade, **TTFB 250 ms single-stream
  on RTX 5090** (728 ms at 12 concurrent). Official kanitts-vllm (v1 400m model, no cloning): "<300 ms"
  first chunk on RTX 5090.
- **Mac path:** effectively none for KaniTTS-2. Official `kani-mlx` supports only the v1
  `kani-tts-370m-MLX` (no cloning): 25-frame chunks, 15-frame lookback, **time-to-first-chunk 2–3 s on
  MacBook Air M2 8 GB**. KaniTTS-2 on PyTorch: CUDA or CPU ("20–60 s per 10 s of audio" on CPU,
  i.e. RTF 2–6); MPS not mentioned. Not in mlx-audio's model list (fetched 2026-08-24).
- **Controls/capabilities:** none documented beyond sampling; language-specific 400m checkpoints
  (EN, ZH, DE, AR, ES, KO, JA) in the v1 family; KaniTTS-2 EN + pretrained. Degrades beyond ~40 s.
- **Verdict vs the bar:** **Loses on both** for a Mac: no streaming Mac path (only a 2–3 s-first-chunk
  MLX port of a non-cloning model) and the weakest published clone SIM in the cluster. Caveat: the
  250 ms TTFB is a CUDA/vLLM number that does not transfer.
- **Sources:**
  - HF nineninesix/kani-tts-2-en — https://huggingface.co/nineninesix/kani-tts-2-en
  - GitHub nineninesix-ai/kani-tts-2 (README) — https://github.com/nineninesix-ai/kani-tts-2
  - GitHub nineninesix-ai/kani-mlx — https://github.com/nineninesix-ai/kani-mlx
  - GitHub nineninesix-ai/kani-tts (v1, GPU RTF table) — https://github.com/nineninesix-ai/kani-tts
  - GitHub nineninesix-ai/kanitts-vllm — https://github.com/nineninesix-ai/kanitts-vllm
  - GitHub mohammed-bahumaish/kani-tts-2-vllm — https://github.com/mohammed-bahumaish/kani-tts-2-vllm
  - HF nineninesix org listing — https://huggingface.co/nineninesix
  - Gradium on-device benchmark 2026 — https://gradium.ai/content/on-device-tts-benchmark-2026
  - MarkTechPost KaniTTS-2 (2026-02-15) — https://www.marktechpost.com/2026/02/15/meet-kani-tts-2-a-400m-param-open-source-text-to-speech-model-that-runs-in-3gb-vram-with-voice-cloning-support/

### Gepard 1.0 (nineninesix.ai, HF updated 2026-08-06) — new, in-cluster
- **What it is:** ~556M total — Qwen3.5 full-attention decoder (14 layers, hidden 1024, ~500M) that
  samples a whole NanoCodec frame (32 FSQ channels) per step, no depth transformer → NeMo NanoCodec
  22.05 kHz, 21.5 fps, 1.89 kbps. Built to run single-pass inside stock vLLM.
- **Weights/license:** https://huggingface.co/nineninesix/gepard-1.0 — Apache-2.0 (codec: NVIDIA Open
  Model License).
- **Size/memory:** 0.56B; no memory figure published; CUDA only.
- **Voice cloning:** zero-shot from "a few seconds" reference, speaker profile extracted once at
  prefill (transcript not mentioned). **Own eval table (Seed-TTS-eval, 1 088 prompts): Gepard WER 0.036
  / SIM 0.585 / UTMOS 2.64 / NISQA 4.25 vs Chatterbox WER 0.063 / SIM 0.796 / UTMOS 2.70 / NISQA 4.19;
  Qwen3-TTS SIM 0.833; VoxCPM2 SIM 0.867.** The card itself says it "trades some speaker similarity
  (SIM) and word accuracy" for streaming-first speed; two-pass CFG mode lowers similarity further.
- **Streaming:** native, frame-level; **TTFA ≈ 0.032 s, RTF ≈ 0.040 single stream on RTX 5090**,
  ≈204× aggregate under load (gepard-inference README).
- **Mac path:** none — vLLM/CUDA only; no MLX/MPS/CPU mention; Python 3.12 + NVIDIA driver required.
- **Controls/capabilities:** EN (US/UK), ES-MX, PT-BR, NL; no emotion/tag controls documented.
- **Verdict vs the bar:** Would crush the TTFA bar on CUDA (32 ms) but has **no Mac path**, and by its
  own numbers **loses clearly to Chatterbox on speaker similarity** (0.585 vs 0.796). Watch for an MLX
  port; today it is not a candidate.
- **Sources:**
  - HF nineninesix/gepard-1.0 — https://huggingface.co/nineninesix/gepard-1.0
  - GitHub nineninesix-ai/gepard-inference README — https://github.com/nineninesix-ai/gepard-inference/blob/main/README.md
  - GitHub nineninesix-ai/gepard-train — https://github.com/nineninesix-ai/gepard-train

### Soprano 1.1-80M (ekwek, released 2026-01-14)
- **What it is:** 80M decoder-only LM over audio tokens (Soprano-Encoder) → Vocos-style vocoder,
  32 kHz; trained on ~1 000 h.
- **Weights/license:** https://huggingface.co/ekwek/Soprano-1.1-80M — Apache-2.0 (commercial OK).
- **Size/memory:** 80M, "<1 GB".
- **Voice cloning:** **none** — "Soprano is currently English-only and does not support voice
  cloning" (HF card and README); roadmap item only. Soprano-Factory (Jan 2026) lets you train/fine-tune
  your own Soprano, which is the only route to a custom voice.
- **Streaming:** native "lossless streaming", `chunk_size` parameter; published **<15 ms latency on
  GPU, <250 ms on CPU** (README, hardware unspecified); RTF up to 2000× GPU / 20× CPU.
- **Mac path:** PyTorch with `device="mps"` supported (README); also listed in mlx-audio (no cloning).
  **No Mac latency numbers published.** Maintainer: ekwek1 (active through Aug 2026 per Soprano-Factory
  coverage).
- **Controls/capabilities:** temperature / top-p / repetition penalty only; English; mispronounces rare
  words, needs numbers spelled out.
- **Verdict vs the bar:** Probably beats the TTFA bar (sub-250 ms even on CPU) but **irrelevant for
  cloning** — fixed voice. Caveat: 80M/1 000 h model; quality is "for its size".
- **Sources:**
  - GitHub ekwek1/soprano — https://github.com/ekwek1/soprano
  - HF ekwek/Soprano-1.1-80M — https://huggingface.co/ekwek/Soprano-1.1-80M
  - HF ekwek/Soprano-Encoder — https://huggingface.co/ekwek/Soprano-Encoder
  - GitHub ekwek1/soprano-factory — https://github.com/ekwek1/soprano-factory
  - mlx-audio model list — https://github.com/Blaizzy/mlx-audio

### Supertonic 2 / Supertonic 3 (Supertone, v3 released 2026-04-29; README updated 2026-05-20)
- **What it is:** ~66M (v1/v2) / ~99M (v3) ONNX pipeline: text encoder → flow-matching text-to-latent
  (2–5 steps) → speech autoencoder decoder, 44.1 kHz.
- **Weights/license:** https://huggingface.co/Supertone/supertonic-3 — model OpenRAIL-M (use-based
  restrictions, commercial otherwise OK), code MIT.
- **Size/memory:** 66M / 99M; small (hundreds of MB).
- **Voice cloning:** **no zero-shot cloning in the open weights** — "fixed-voice, local TTS … does not
  include an official voice-cloning pipeline". Custom voices came only from the paid **Voice Builder**
  (≤1 min recording → v2/v3 style JSON); **new purchases closed 2026-07-23, service shuts 2026-08-31**.
  Zero-shot cloning exists only in Supertone's cloud (Play / API). WER only: EN 2.06 on Minimax-MLS-test
  (v3 card). No SIM.
- **Streaming:** none native; `supertonic-py` chunks long text and concatenates → sentence-level at best.
- **Mac path:** ONNX Runtime (CoreML EP possible), WebGPU, Swift/iOS/Flutter-macOS SDKs; sherpa-onnx
  int8 build (2026-03-06). Published Mac numbers (v1 66M card): **M4 Pro CPU RTF 0.015→0.012 (2-step),
  0.023→0.018 (5-step) for 59→266-char text; M4 Pro WebGPU RTF 0.006–0.024**; "167× realtime". Derived:
  a 59-char sentence (~3.5 s audio) finishes in ≈50–80 ms, so whole-utterance latency is already far
  under the bar. No Supertonic-3 Mac table published.
- **Controls/capabilities:** speed 0.7–2.0, quality steps 5–12, 10 inline tags (`<laugh>`, `<breath>`,
  `<sigh>` …), 31 languages (v3), preset voices only.
- **Verdict vs the bar:** Demolishes the TTFA bar (~50 ms whole sentence on M4 Pro) but **cannot clone**;
  the one custom-voice path is being switched off next week. Not a cloning candidate.
- **Sources:**
  - GitHub supertone-inc/supertonic — https://github.com/supertone-inc/supertonic
  - README.md — https://github.com/supertone-inc/supertonic/blob/main/README.md
  - HF Supertone/supertonic (v1 benchmark table) — https://huggingface.co/Supertone/supertonic
  - HF Supertone/supertonic-3 — https://huggingface.co/Supertone/supertonic-3
  - Voice Builder (shutdown notice) — https://supertonic.supertone.ai/voice-builder
  - sherpa-onnx Supertonic int8 — https://huggingface.co/csukuangfj2/sherpa-onnx-supertonic-tts-int8-2026-03-06

### Kokoro-82M (hexgrad) — latency-floor reference; no cloning
- **What it is:** 82M StyleTTS2-derived decoder (ISTFTNet), 24 kHz, misaki/espeak G2P, 256-d style
  vectors per voice.
- **Weights/license:** https://huggingface.co/hexgrad/Kokoro-82M — Apache-2.0.
- **Size/memory:** 82M; ~300–900 MB resident depending on runtime (M5 Max test: ~900 MB peak).
- **Voice cloning:** **none** — voices are trained style vectors. "Cloning hacks": (a)
  `eryawww/kokoro_hack` optimises the style vector with PSO against a Wav2Vec2 emotion encoder — for
  emotion, not timbre; (b) `Ashish-Patnaik/kokoclone` (189 stars) bolts a separate zero-shot voice
  *conversion* stage ("Kanade tokenizer") onto Kokoro-ONNX output from a 3–10 s sample — no SIM/MOS
  published, ~8.9 s chunk ceiling. Neither is style-vector *fitting* to a reference and neither has any
  published similarity figure; **not credible as a Chatterbox-class cloner**.
- **Streaming:** sentence/segment-level in every runtime (whole segment synthesised, then played).
- **Mac path (published numbers, whole utterance = first audio):**
  - RunAnywhere/MetalRT blog, **M4 Max 64 GB, 2026-03-12**: 4 words — MetalRT 178 ms / **mlx-audio
    493 ms** / sherpa-onnx 504 ms; 36 words — 604 / 706 / 2 115 ms. (MetalRT is proprietary.)
  - `mattmireles/kokoro-coreml` (June 2026): fp16 CoreML, fixed buckets 3/7/10/15/30 s; **M2 Studio:
    51 ms for a 3-s bucket, 126 ms for 10 s**; M1 mini 1 959 ms for 30 s.
  - Contra Collective, **M5 Max, 2026-07-06**, ONNX+CoreML: RTF 0.08, **~90 ms first audio**.
  Maintainers: mlx-audio (Blaizzy, active), kokoro-coreml (active June 2026).
- **Controls/capabilities:** 54 preset voices, speed, 8 languages; no emotion/instruct.
- **Verdict vs the bar:** The latency floor on this Mac is ~50–200 ms per sentence (CoreML/MetalRT) or
  ~0.5 s on mlx-audio for a short sentence — so anything cloning-capable that lands at 0.3–0.5 s TTFA is
  "within 2–3× of Kokoro". No cloning, so no quality comparison.
- **Sources:**
  - RunAnywhere MetalRT speech post — https://www.runanywhere.ai/blog/metalrt-speech-fastest-stt-tts-apple-silicon
  - HF blog runanywhere/metalrt (table) — https://huggingface.co/blog/runanywhere/metalrt-fastest-inference-apple-silicon
  - HF mattmireles/kokoro-coreml — https://huggingface.co/mattmireles/kokoro-coreml
  - Contra Collective M5 Max comparison — https://contracollective.com/blog/kokoro-vs-piper-vs-xtts-local-text-to-speech-m5-max-2026
  - GitHub eryawww/kokoro_hack — https://github.com/eryawww/kokoro_hack
  - GitHub Ashish-Patnaik/kokoclone — https://github.com/Ashish-Patnaik/kokoclone
  - GitHub Blaizzy/mlx-audio — https://github.com/Blaizzy/mlx-audio

### F5-TTS / E2-TTS (SWivid; f5-tts-mlx by lucasnewman, last commit 2025-03-19)
- **What it is:** 336M (F5) / 333M (E2) non-autoregressive flow-matching DiT on mel, ConvNeXt text
  encoder, Vocos vocoder, 24 kHz; whole-utterance generation with NFE 16–32 steps + CFG.
- **Weights/license:** https://huggingface.co/SWivid/F5-TTS — CC-BY-NC-4.0 (Emilia-trained base;
  **non-commercial**); code MIT. f5-tts-mlx: MIT code, same model weights.
- **Size/memory:** 336M; MLX 4-/8-bit quant supported; ~1–2 GB on Mac (derived).
- **Voice cloning:** zero-shot, 5–10 s ref (≤12 s; auto-clipped), **ref transcript needed** (or ASR).
  Paper (Oct 2024): **Seed-TTS test-en WER 1.83 / SIM-o 0.67 (32 NFE)**, LibriSpeech-PC WER 2.42 /
  SIM-o 0.66; E2-TTS SIM-o 0.71 test-en but WER 2.19/2.95. Cross-Lingual F5 paper (Feb 2026) still
  reports baseline SIM-o 0.668 on LibriSpeech-PC. Consensus: among the strongest timbre cloners in the
  ≤500M class; also copies the reference's delivery/pace (duration is inferred from ref speech rate);
  no explicit style control. Fine-tune path: official finetune scripts.
- **Streaming:** **text-chunk level only.** `infer_batch_process(streaming=True)` runs the full
  flow-matching pass for each text chunk, then slices the finished waveform into `chunk_size=2048`
  sample pieces (utils_infer.py, fetched 2026-08-24) — first audio = one full chunk generation.
  Issue #1225 (2025-11-21, open, "help wanted"): **first packet ≈ 2 s** with nfe 64 in a FastAPI
  streamer; issue #700 asked for true real-time streaming (closed, no implementation). socket_server.py
  = same chunk streaming. Chunk joins use 0.15 s cross-fade.
- **Mac path:** `lucasnewman/f5-tts-mlx` — whole-utterance only, no streaming; README: "~4 s" per
  sample on M3 Max (earlier ~11 s); **stale: last commit 2025-03-19 (v1 model + quant)**. PyTorch
  MPS works (`PYTORCH_ENABLE_MPS_FALLBACK=1` set by the code). Rapid-MLX ships F5-TTS as one of four
  cloning models on MLX (no latency numbers). Not in mlx-audio.
- **Controls/capabilities:** speed (duration scaling), NFE steps, CFG strength, sway sampling; EN+ZH
  base, community multilingual finetunes; no emotion tags/instruct; multi-speaker via chunking only.
- **Verdict vs the bar:** **Matches/beats Chatterbox on timbre SIM** (0.67 SIM-o class) but **loses
  badly on TTFA** — no streaming below sentence level and ~4 s per utterance on the only MLX port
  (unmaintained 17 months). Caveat: CC-BY-NC weights.
- **Sources:**
  - GitHub lucasnewman/f5-tts-mlx — https://github.com/lucasnewman/f5-tts-mlx
  - f5-tts-mlx commits — https://api.github.com/repos/lucasnewman/f5-tts-mlx/commits
  - F5-TTS paper (results tables) — https://arxiv.org/html/2410.06885v1
  - F5-TTS infer README — https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md
  - utils_infer.py (streaming mechanics) — https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/utils_infer.py
  - Issue #1225 first-packet latency — https://github.com/SWivid/F5-TTS/issues/1225
  - Issue #700 real-time streaming request — https://github.com/SWivid/F5-TTS/issues/700
  - Cross-Lingual F5-TTS (2026 SIM table) — https://arxiv.org/html/2509.14579v4
  - Rapid-MLX — https://rapidmlx.com/

### LuxTTS (Yatharth Sharma / ysharma3501, released 2026-01; README last commit 2026-06-05)
- **What it is:** ZipVoice (123M Zipformer flow-matching NAR, k2-fsa) distilled to 4 sampling steps
  with a custom 48 kHz vocoder (Vocos-family). Fits in <1 GB VRAM.
- **Weights/license:** https://huggingface.co/YatharthS/LuxTTS — Apache-2.0 (commercial OK).
- **Size/memory:** ~123M base (ZipVoice figure; LuxTTS card gives no count) + 48 kHz vocoder; <1 GB.
- **Voice cloning:** zero-shot, ≥3 s reference; ZipVoice requires a prompt transcript (LuxTTS
  README does not state otherwise). **No LuxTTS SIM/WER published.** Base-model proxy (ZipVoice paper,
  June 2025): **ZipVoice-Distill 4 NFE Seed-TTS test-en WER 1.64 / SIM-o 0.679; F5-TTS 32 NFE in the
  same table WER 1.85 / SIM-o 0.664** — so the architecture is F5-class on timbre at ~1/3 the size.
  Claim "SOTA voice cloning on par with models 10× larger" is the author's; no arena/blind data found.
  `return_smooth=True` exists to hide "metallic artifacts". Fine-tune: ZipVoice training recipes.
- **Streaming:** none (whole-utterance NAR); nothing in repo or MLX fork.
- **Mac path:** PyTorch MPS merged 2026-01-28 (PR #15); community pure-MLX port
  `jishnuvenugopal/LuxTTS-mlx` v0.1.0 (2026-02-13, synced to Jan-28 upstream) — vocoder in MLX,
  fp32 only (fp16 "almost 2×" on roadmap), **known Metal kernel errors** (fallback `--vocoder torch`).
  **No Mac RTF/latency published** anywhere; GPU claim 150× realtime, CPU ">1×". Derived from ZipVoice
  CPU RTF 1.22 (single Xeon thread, 4 NFE): a multi-core M4 Max should land well under RTF 1 but the
  whole utterance must finish before audio starts. Maintainer: solo dev; "v1.5" planned, no date.
- **Controls/capabilities:** num_steps (3–4), t_shift naturalness knob, English only (multilingual
  requested in issue #2); no emotion/tags.
- **Verdict vs the bar:** Potentially **matches Chatterbox on timbre** (ZipVoice-class SIM ~0.68) at a
  fraction of the compute, but **loses on TTFA**: non-streaming, fp32-only Mac ports with no measured
  numbers. Caveat: quality claims are unverified and the MLX port is a one-person fork with Metal bugs.
- **Sources:**
  - GitHub ysharma3501/LuxTTS — https://github.com/ysharma3501/LuxTTS
  - HF YatharthS/LuxTTS — https://huggingface.co/YatharthS/LuxTTS
  - LuxTTS commits — https://api.github.com/repos/ysharma3501/LuxTTS/commits
  - LuxTTS issues (Mac) — https://github.com/ysharma3501/LuxTTS/issues?q=mps+OR+mac+OR+apple+OR+mlx
  - PR #25 MLX link — https://github.com/ysharma3501/LuxTTS/issues/25
  - GitHub jishnuvenugopal/LuxTTS-mlx — https://github.com/jishnuvenugopal/LuxTTS-mlx
  - ZipVoice paper (tables) — https://arxiv.org/html/2506.13053
  - GitHub k2-fsa/ZipVoice — https://github.com/k2-fsa/ZipVoice
  - TheMenonLab LuxTTS write-up — https://blog.themenonlab.com/blog/luxtts-voice-cloning-150x-realtime-1gb-vram/

### OuteTTS 1.0 (OuteAI; 1B GGUF card dated 2025-03; repo effectively dormant since 2025-06)
- **What it is:** Llama-3.2-1B (or 0.6B) continued-pretrained LM → DAC codec (2 codebooks), 24 kHz;
  llama.cpp-native GGUF.
- **Weights/license:** https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B-GGUF — Llama 3.2 Community
  License + **CC-BY-NC-SA-4.0** (1B, non-commercial); OuteTTS-1.0-0.6B under Apache-2.0.
- **Size/memory:** 1B / 0.6B; GGUF Q4–Q8 ≈ 0.5–1.2 GB.
- **Voice cloning:** one-shot speaker profile from ~10 s audio (`create_speaker`), transcript via
  built-in alignment; no SIM/WER published; 23 languages. Consensus: usable timbre clone, prone to
  drift/hallucination without the mandated 64-token-window repetition penalty.
- **Streaming:** **no streaming API** in the interface (whole utterances, up to ~42 s). Batched RTF
  chart on L40S only.
- **Mac path:** llama.cpp with `-DGGML_METAL=on` (documented); mlx-audio lists OuteTTS 0.6B **without
  cloning**. No Mac latency numbers published. Last substantive commit 2025-06-21 (one file deletion
  2026-03-23).
- **Controls/capabilities:** none (sampling only); multilingual.
- **Verdict vs the bar:** Loses on TTFA (no streaming) and offers no evidence of Chatterbox-class SIM;
  dormant. Only relevance is as a llama.cpp-native cloning baseline.
- **Sources:**
  - HF OuteAI/Llama-OuteTTS-1.0-1B-GGUF — https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B-GGUF
  - GitHub edwko/OuteTTS — https://github.com/edwko/OuteTTS
  - OuteTTS commits — https://api.github.com/repos/edwko/OuteTTS/commits
  - OuteAI 1.0 release blog — https://outeai.com/blog/outetts-1-0-release

### Piper (OHF-Voice/piper1-gpl; v1.6.0 July 2026 per search summary) — reference floor only
- **What it is:** VITS-style ONNX voices (~15–60M) with espeak-ng phonemisation, 16–22 kHz.
- **Weights/license:** https://github.com/OHF-Voice/piper1-gpl — **GPL-3.0** (moved from MIT when
  rhasspy/piper went read-only Oct 2025); voices individually licensed.
- **Size/memory:** ~300 MB peak (M5 Max test).
- **Voice cloning:** none (fine-tune a voice from a dataset only).
- **Streaming:** sentence-level — `PiperVoice.synthesize()` yields an audio chunk per sentence.
- **Mac path:** ONNX Runtime (+CoreML EP): **M5 Max, 2026-07-06: RTF 0.03, ~40 ms first audio**.
  OHF is "looking for maintainers".
- **Controls/capabilities:** length_scale (speed), noise scales, multi-speaker voices; many languages.
- **Verdict vs the bar:** Latency floor (~40 ms) and nothing else; no cloning.
- **Sources:**
  - GitHub OHF-Voice/piper1-gpl — https://github.com/OHF-Voice/piper1-gpl
  - Python API doc — https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
  - Contra Collective M5 Max comparison — https://contracollective.com/blog/kokoro-vs-piper-vs-xtts-local-text-to-speech-m5-max-2026

---

## Cross-model ranking inside this cluster (clone quality × Mac TTFA)

| Model | Clones? | Best published SIM (metric) | Streams on Mac? | Best published Mac first-audio |
|---|---|---|---|---|
| NeuTTS Air (Q4/Q8 GGUF) | yes, 3–15 s + transcript | 47.5 % WavLM-cos (Gradium, Apr 2026) | **yes, token-level, CPU/Accelerate** | not published; LM 111 tok/s on iMac M4 CPU ⇒ sub-second plausible (derived) |
| LuxTTS (ZipVoice-distill) | yes, ≥3 s | proxy 0.679 SIM-o (ZipVoice-Distill 4 NFE) | no (whole utterance, MPS/MLX fp32) | not published |
| F5-TTS (f5-tts-mlx) | yes, 5–10 s + transcript | 0.67 SIM-o Seed-TTS test-en | no (text-chunk only) | ~4 s/utterance M3 Max (Mar 2025) |
| KaniTTS-2 | yes, 3–30 s embedding | 40.7 % WavLM-cos (Gradium) | no (MLX port = v1, no cloning; 2–3 s first chunk M2) | n/a |
| Gepard 1.0 | yes | 0.585 SIM (own Seed-TTS eval; Chatterbox 0.796 same table) | no Mac path (vLLM/CUDA) | n/a (32 ms on RTX 5090) |
| OuteTTS 1.0 | yes, ~10 s | not published | no streaming API (llama.cpp Metal ok) | not published |
| Kokoro / Soprano / Supertonic / Piper | no | — | sentence-level | 40–500 ms |

Bottom line for the project: **NeuTTS Air** is the only ≤1B model that plausibly beats the 0.9 s
TTFA bar on the M4 Max with a real streaming path *and* clones — but every quality datapoint puts
it below flow-matching cloners and well below Chatterbox-class similarity, and there is no published
Mac TTFA (measure it: GGUF Q8 + ONNX decoder + pre-encoded ref, Accelerate build). **LuxTTS** is the
most interesting quality/size candidate if you can live with whole-utterance generation and a
one-person MLX port. Nothing in this cluster delivers both.

**Out of cluster, flagged:**
- Gradium **Phonon** (~100M, cloning from 10 s, no transcript; Seed-TTS WER 1.48 % → 1.00 % (May 2026),
  SIM 56.4 %) — **proprietary, private beta, licensed binary for Android/iOS/browser**; not open weights,
  not a candidate. https://gradium.ai/content/gradium-phonon-on-device-tts
- `sb1992/dots-tts-mlx` — pure-MLX port of rednote dots.tts (multilingual zero-shot clone) surfaced in
  search; not researched. https://github.com/sb1992/dots-tts-mlx
- Chatterbox reference point from Gepard's table: Seed-TTS-eval **SIM 0.796 / WER 0.063** — useful as the
  bar when other clusters report SIM on the same pipeline.
