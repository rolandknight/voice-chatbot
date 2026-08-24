# 05 — Mac runtimes and cross-model leaderboards (as seen 2026-08-24)

Scope: (A) what each Mac runtime can run *today*, whether it streams and clones, and what Apple-Silicon numbers are published; (B) the cross-model quality evidence (arenas, Seed-TTS-eval, EmergentTTS-Eval) and any Apple-Silicon latency comparisons. Every number carries the URL it came from and the date it was seen (all fetched 2026-08-24 unless noted). "Not published" means the page I fetched did not contain it.

Judged against the bar from the brief: Chatterbox Flash on this M4 Max via MLX = ~0.9 s first audio (whole-utterance), RTF ~0.28, no streaming API.

---

## (A) Mac runtimes

### A1. mlx-audio (Blaizzy/mlx-audio) — Python, MLX

**Activity.** Releases every 1–2 weeks: v0.5.0 (2026-08-17), v0.4.8 (08-10), v0.4.7 (08-03), v0.4.6 (07-25), v0.4.5 (07-09), v0.4.4 (06-06) — https://github.com/Blaizzy/mlx-audio/releases. Repo shows 948 commits on main — https://github.com/Blaizzy/mlx-audio. Open-issue count not shown on the pages fetched.

**TTS model table as rendered from the README today** (https://raw.githubusercontent.com/Blaizzy/mlx-audio/main/README.md). Caveat: the fetch tool reconstructed the "Voice cloning" / "Streaming" columns from the README's prose; the README itself does not have a streaming column, and the streaming guide (below) says *all* TTS models accept `stream=True`. Treat the cloning column as "cloning mentioned in the README description", not an exhaustive test.

| Model | README description | Languages | Cloning mentioned |
|---|---|---|---|
| Kokoro | fast multilingual, 54 voice presets | EN, JA, ZH, FR, ES, IT, PT, HI | No |
| KittenTTS | compact edge TTS | EN | No |
| Qwen3-TTS | "multilingual TTS with voice cloning, emotion control, and voice design" | ZH, EN, JA, KO, + | Yes |
| Higgs Audio v3 | 4B conversational, inline controls | 100 | Yes |
| Higgs Audio v2 | 3B Llama-backed "real-time cloning" | EN, ZH, KO, DE, ES | Yes |
| OmniVoice | zero-shot multilingual, 646+ languages | 646+ | Yes |
| CSM / MisoTTS | Sesame-style conversational | EN | Yes |
| Dia | dialogue TTS | EN | No |
| OuteTTS | | EN | No |
| Spark | SparkTTS | EN, ZH | No (upstream Spark does clone; README text doesn't say) |
| Chatterbox | "expressive multilingual (v2/v3)" | 23 | Not stated in the README line (upstream clones; v0.4.6 added Chatterbox Multilingual v3) |
| Soprano | | EN | No |
| Ming Omni TTS | cloning + style control | EN, ZH | Yes |
| KugelAudio | 7B AR+diffusion European | 24 | No |
| Voxtral TTS | Mistral 4B, 20 preset voices | 9 | No |
| LongCat-AudioDiT | diffusion in waveform latent | ZH, EN | Yes |
| MeloTTS | VITS2 | EN | No |
| MOSS-TTS | 8B delay-pattern, cloning | 31 | Yes |
| MOSS-TTS-Nano | tiny cloning TTS | 20 | Yes |
| (releases) arktts, Confucius4-TTS, ZONOS2, Irodori-TTS v3/v4, Fish S2 Pro style instructions | added Jun–Aug 2026 | | arktts/Confucius4: "zero-shot voice cloning" |

Not in the current README table: **Orpheus** and **Pocket TTS** (both *are* in mlx-audio-swift, see A7). IndexTTS is also not in the Python README table today (it is in mlx-audio-swift).

**Streaming semantics** (https://blaizzy.github.io/mlx-audio/guides/streaming/):
- `stream=True` "yields `GenerationResult` objects as chunks rather than waiting for full synthesis"; each result carries `.audio`, `.segment_idx`, `.sample_rate` (https://raw.githubusercontent.com/Blaizzy/mlx-audio/main/mlx_audio/tts/generate.py).
- Granularity is time-based: `--streaming_interval` "controls how frequently audio chunks are emitted (in seconds)", **default 2.0 s**; the guide says Qwen3-TTS recommends **0.32 s ≈ 4 tokens at 12.5 Hz**. "Smaller values reduce latency but add per-chunk overhead."
- The guide claims all TTS models support it via `generate(stream=True)`, explicitly naming Kokoro and Qwen3-TTS (incl. `generate_custom_voice()` / `generate_voice_design()`). For models whose upstream implementation is whole-utterance (e.g. Chatterbox's S3Gen, Dia), whether the 'chunk' is intra-utterance or a sentence segment is **not documented** — the doc index lists per-model pages but the fetched pages did not state it. Flag: verify per model before relying on it.
- **Server:** `mlx_audio.server --host 0.0.0.0 --port 8000`, OpenAI-compatible `POST /v1/audio/speech`; with `"stream": true` the client "receives continuous WAV data" (chunked HTTP, not SSE). v0.4.7 "optimized MLX server thread ownership for model loading" and added a model-kind registry.
- Quantization: 3/4/6/8-bit, affine / mxfp4 / mxfp8 / nvfp4 (README).

**Published Apple-Silicon numbers:** none on the README, docs index, or streaming guide ("no chip-specific or millisecond-precision latency measurements"). Only third-party datapoint: MetalRT's blog (2026-03-09) says its Kokoro-82M is "2.8x faster than mlx-audio on short phrases" on an M4 Max where MetalRT takes 178 ms for a 4-word phrase — which *implies* mlx-audio Kokoro ≈ 0.5 s for the same phrase on M4 Max (derived, not published) — https://www.runanywhere.ai/blog/metalrt-speech-fastest-stt-tts-apple-silicon.

**How a Python server consumes it:** `pip install mlx-audio`, either call `model.generate(..., stream=True, streaming_interval=…)` in-process and forward chunks, or run `mlx_audio.server` and proxy `/v1/audio/speech` with `stream: true`.

**Sibling:** vllm-mlx 0.4.1 (2026-08-12) bundles TTS for Kokoro, Chatterbox, VibeVoice, VoxCPM ("11 voices, 15+ languages"), `pip install vllm-mlx[audio]`; no streaming granularity or Mac numbers on the PyPI page — https://pypi.org/project/vllm-mlx/.

### A2. soniqo/speech-swift and speech-core

**speech-swift** (Swift, MLX + CoreML/ANE; https://github.com/soniqo/speech-swift, README fetched via https://raw.githubusercontent.com/soniqo/speech-swift/main/README.md). 14 TTS engines:

| Engine | Backend | Cloning | Streaming | Languages | Published perf |
|---|---|---|---|---|---|
| Qwen3-TTS | MLX + CoreML | Yes | **Yes** | 10 | M2 Max RTF ≈ 0.55, ~37 ms/step (https://soniqo.audio/guides/speak); "120 ms first-packet latency" appears in Soniqo's docs snippet but the fetched guide page did not name hardware for it |
| CosyVoice3 | MLX | Yes | **Yes** | 9 | not published |
| VoxCPM2 | MLX bf16/int8 | Yes | No | 30 | not published |
| IndexTTS2 | MLX fp16 | Yes | No | EN/ZH | not published |
| F5-TTS | MLX fp16 | Yes | No | EN/ZH | non-commercial |
| Higgs TTS 3 | MLX bf16 | Yes | No | 100+ | non-commercial |
| Chatterbox Multilingual | MLX fp16 | Yes | No | 23 | not published |
| OmniVoice | MLX fp16/int8 | Yes | No | 600+ | not published |
| CSM | MLX int8/fp16 | Yes | No | EN | not published |
| VibeVoice 0.5B / 1.5B | MLX | No | 0.5B yes | EN/ZH | not published |
| Magpie-TTS Multilingual | MLX/CoreML | No | Yes (MLX) | 9 | not published |
| Kokoro-82M | CoreML (ANE) | No | No | 10 | 0.08 RTF on iPhone 16 Pro |
| Supertonic-3 | CoreML (ANE) | No | No | 31 | 0.15 RTF on iPhone 16 Pro |

Qwen3-TTS specifics (https://soniqo.audio/guides/speak): sizes 0.6B Base 8-bit 1.3 GB, 1.7B Base 8-bit 2.4 GB, 1.7B Base bf16 3.9 GB, 0.6B CustomVoice bf16 1.8 GB, CoreML fp16 2.1 GB; streaming chunk size set by `--first-chunk-frames` / `--chunk-frames`; default cap 500 tokens ≈ 40 s at 12.5 Hz. Speaker encoder is ECAPA-TDNN on the reference audio (https://github.com/soniqo/speech-swift/blob/main/docs/models/tts-model.md); reference-length requirement not documented there.

Python consumption: `speech-server --port 8080` exposes OpenAI-compatible REST incl. `/v1/audio/speech` (added v0.0.23, 2026-07-18) with WAV/raw-PCM output — https://github.com/soniqo/speech-swift/releases. Whether that HTTP endpoint streams chunks is not stated on the release page.

Activity: v0.0.22 (07-14: Chatterbox multilingual cloning in Swift/MLX, F5-TTS, Higgs TTS 3, IndexTTS2, OmniVoice), v0.0.23 (07-18), v0.0.24/25 (08-16: Nemotron VoiceChat, "Qwen3-TTS code predictor frame compilation", real-time on ANE), v0.0.26 (08-17). (The fetch tool printed the year as 2024 for these; the content — Higgs TTS 3, Nemotron VoiceChat 11B — makes them 2026.) Swift Forums announcement: https://forums.swift.org/t/speech-swift-on-device-speech-processing-for-apple-silicon-asr-tts-diarization-speech-to-speech/85182.

**speech-core** (C++17, ONNX Runtime + LiteRT, Apache-2.0, v0.0.11, 69 stars; https://github.com/soniqo/speech-core): TTS = VoxCPM 0.5B (16 kHz + cloning), VoxCPM2 2B (48 kHz + cloning), CosyVoice3 0.5B, Chatterbox, Supertonic 3, Kokoro 82M, Pocket TTS 100M ("streaming TTS, fixed Alba voice"). Cloning explicitly only for VoxCPM/VoxCPM2; streaming only Pocket TTS. Runs macOS but via ONNX/LiteRT (CPU / CoreML EP), no Mac numbers published. Sibling desktop app: speech-studio (Tauri + VoxCPM2 cloning) — https://github.com/soniqo/speech-studio.

### A3. Kyutai: moshi-mlx, delayed-streams-modeling, Pocket TTS

- **Kyutai TTS 1.6B** (card: https://huggingface.co/kyutai/tts-1.6b-en_fr): 1.8B params (1B backbone + 600M depth transformer, partial weight sharing); 12.5 Hz frames, 32 audio tokens/frame; audio delayed **16 steps (1.28 s)** relative to the text stream — i.e. the model needs ~1.28 s of text lead before first audio; "starts to output audio as soon as the first few words" arrive. Voices are **pre-computed voice embeddings only**; the embedding model is not released ("no voice cloning from arbitrary audio samples"). EN/FR. CC-BY 4.0. The oft-quoted 220 ms figure is on kyutai.org, which returned an empty page for me — **not confirmed on a page I saw**.
- **MLX path** (https://raw.githubusercontent.com/kyutai-labs/delayed-streams-modeling/main/README.md, https://raw.githubusercontent.com/kyutai-labs/delayed-streams-modeling/main/scripts/tts_mlx.py): `scripts/tts_mlx.py` with `--quantize 8` or `4`; it is a genuine frame-streamer: frames are queued as generated and consumed by a sounddevice callback when output is `-` (80 ms frames, "generated {frames/12.5:.2f}s"); with a file path it accumulates and writes at the end. Voices via `--voice-repo` + `--voice` (default `expresso/ex03-ex01_happy_001_channel1_334s.wav`), loaded through `tts_model.get_voice_path`. No MLX RTF/TTFA published in the README.
- **Rust/Candle**: DSM README says the Rust server "can process multiple streaming queries in parallel"; moshi repo: build with `--features metal` on macOS; "tested the MLX version on a MacBook Pro M3"; no numbers — https://github.com/kyutai-labs/moshi. `moshi_mlx` is the Moshi (S2S) inference package, not a TTS package; the TTS MLX code lives in the DSM repo scripts.
- **Pocket TTS** (https://github.com/kyutai-labs/pocket-tts): 100M params, MIT, EN/FR/DE/PT/IT/ES; "~200 ms to get the first audio chunk" on CPU; "~6x real-time on a CPU of MacBook Air M4"; uses 2 CPU cores; **accepts a reference wav** (`get_state_for_audio_prompt`), voice state exportable to safetensors; T4 GPU ~2.6× speed-up. Community ports listed: MLX, Rust (Candle), ONNX, C++, WASM. `pocket-tts-mlx` v0.2.1 (MIT; `from pocket_tts_mlx import TTSModel`; cloning "requires Hugging Face access to kyutai/pocket-tts", i.e. the gated repo) — no latency numbers published — https://github.com/jishnuvenugopal/pocket-tts-mlx. Also in mlx-audio-swift, FluidAudio, sherpa-onnx, llama.cpp, audio.cpp (see below).

### A4. sherpa-onnx, kokoro-onnx, CoreML conversions

- **sherpa-onnx TTS families** (https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html): VITS, Piper, MMS, Matcha, Kokoro, KittenTTS, ZipVoice, PocketTTS, Supertonic. The `all-models.html` URL from the brief is a 404.
- **Cloning-capable — the "probably none" is wrong:** **ZipVoice** (k2-fsa's flow-matching zero-shot TTS) takes `zipvoiceReferenceAudio` + `zipvoiceReferenceText` (PR #2487 https://github.com/k2-fsa/sherpa-onnx/pull/2487; issue #3439 https://github.com/k2-fsa/sherpa-onnx/issues/3439). **PocketTTS** also takes a reference audio, but issue #3180 (2026-02-13) reports the sherpa-onnx output "sounds distinctly different" from upstream Pocket TTS for the same reference, with no maintainer resolution visible — https://github.com/k2-fsa/sherpa-onnx/issues/3180.
- Streaming: sherpa-onnx TTS uses a generated-audio callback (per-sentence chunks for VITS/Matcha/Kokoro). No macOS TTFA numbers on the pages fetched. macOS/iOS binaries are provided.
- **kokoro-onnx** (https://github.com/thewh1teagle/kokoro-onnx): "fast performance near real-time on macOS M1", v1.0 models (`model-files-v1.1`), preset voices only; no numbers.
- **CoreML Kokoro:** FluidAudio's KokoroAne "3–11× RTFx on Apple Silicon" (https://github.com/FluidInference/FluidAudio; weights https://huggingface.co/FluidInference/kokoro-82m-coreml); mattmireles/kokoro-coreml ANE pipeline (https://github.com/mattmireles/kokoro-coreml). argmaxinc/ttskit-coreml, referenced by kakoo issue #23, is now a 404.

### A5. llama.cpp TTS on Metal

- `tools/tts` README (https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md): the tool "used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool" on top of `libmtmd`. Supported today: **Qwen3-TTS-12Hz-1.7B-Base-GGUF** (`--tts-speaker-file` wav/mp3 = zero-shot cloning; `--tts-lang`, 10 languages) and **Pocket TTS** (requires a speaker reference file; "the model produces almost no audio without it"). Output is a WAV file (`-n` frames); **no streaming output, no latency numbers, no Metal notes** on that page.
- **Orpheus-GGUF + SNAC:** the working Mac recipe is still the isaiahbjork GGUF in LM Studio/llama.cpp (Metal) with a separate Python SNAC decoder (https://codersera.com/blog/install-and-run-orpheus-3b-tts-on-macos-a-complete-guide/, https://news.ycombinator.com/item?id=43419983). No Apple-Silicon TTFA published on those pages. Orpheus is tagged "Linux-only" in tts-bench (June 2026) — https://github.com/5uck1ess/tts-bench.
- **NeuTTS / NeuCodec:** not in llama.cpp's tts tool; it is in **audio.cpp** (0xShug0, pure-C++ ggml engine; v0.6 2026-08-13 added NeuTTS, dots.tts, MiniMax-H3; v0.6.1 2026-08-18) — https://github.com/0xShug0/audio.cpp. audio.cpp TTS families: chatterbox, confucius4_tts, dots_tts, dramabox, fish_audio, higgs_audio_tts, index_tts2, irodori_tts, magpie_tts, miotts, moss_tts_local, moss_tts_nano, neutts, omnivoice, personaplex, pocket_tts, qwen3_tts, supertonic, vibevoice, voxcpm2 (+ community f5_tts, glm_tts, inflect_v2, outetts, vietneu_tts). "Stream" flag on: confucius4_tts, dots_tts, neutts, omnivoice, personaplex, qwen3_tts, supertonic, voxcpm2. Cloning flag on: chatterbox, confucius4_tts, dots_tts, dramabox, fish_audio, higgs_audio_tts, index_tts2, irodori_tts, miotts, moss_tts_*, omnivoice, pocket_tts, qwen3_tts, vibevoice, voxcpm2. Metal builds exist; only relative Apple-Silicon number: "VoxCPM2 end-to-end runs up to 2.56x faster" after Metal work. Absolute RTFs are RTX 5090 only (pocket_tts 31.09×, moss_tts_nano 9.91×, supertonic 187.62×). Interfaces: `audiocpp_cli`, `audiocpp_server` (C++ HTTP), WebUI.

### A6. PyTorch MPS viability for flow-matching vocoders (S3Gen / HiFT / BigVGAN)

Issue-tracker evidence, 2025–2026:
- **Chatterbox** (https://github.com/resemble-ai/chatterbox/issues?q=is%3Aissue+mps): #357 "doesn't work on Apple silicon due to the lack of CUDA" (open, 2025-11-16); #336 dependency-install script for Apple Silicon (open, 2025-10-24); #218 "Critical Memory Leak (Apple Silicon)" (open, 2025-08-18); #275 "No apple support for multilingual?" (closed 2025-09-10). README lists `"mps"` as a device but publishes no Mac timings (https://github.com/resemble-ai/chatterbox).
- **IndexTTS 2 (BigVGAN vocoder)** (https://github.com/index-tts/index-tts/issues?q=is%3Aissue+mps): #351 "NotImplementedError: Output channels > 65536 not supported at the MPS device" (closed 2025-09-15); #414 M3 Ultra 256 GB memory leak (open 2025-09-20); #360 M4 Max 128 GB memory overflow on long text (open 2025-09-14); #308 Mac memory anomaly; #410 slow/weird output. Users fall back to CPU and lose speed.
- **CosyVoice (HiFT vocoder)** (https://github.com/FunAudioLLM/CosyVoice/issues?q=is%3Aissue+mps): #1767 "Mac GPU support" closed 2026-02-18; #1011 "Mac Mini M4 CPU at 100%, GPU unused" (open, 2025-02-22); #672, #614, #134 MPS requests open since 2024 — i.e. upstream CosyVoice is effectively CPU-only on Mac.
- **PyTorch MPS generic (2025–2026):** bf16 unsupported on some MPS configs (#141864 https://github.com/pytorch/pytorch/issues/141864); "MPS built but not available on macOS 26, PyTorch 2.9.1/2.10 nightly" (#167679 https://github.com/pytorch/pytorch/issues/167679); corrupted MPS output on macOS 27 beta (#187280 https://github.com/pytorch/pytorch/issues/187280). Secondary: fp16 matmul only ~1.1× fp32 on M4 Pro, so fp16 on MPS buys memory not speed (https://id2thomas.medium.com/apple-silicon-experiment-4-pytorch-mps-backend-e6a9c1687cbc).
- Practical conclusion: the projects that actually run these vocoders fast on Mac re-implemented them in MLX (speech-swift CosyVoice3/IndexTTS2/Chatterbox, mlx-audio Chatterbox), in Swift/CoreML, or in ggml (audio.cpp) — none rely on MPS. This matches the brief's observation that chatterbox-flash's MLX path still runs S3Gen on PyTorch CPU.

### A7. Swift/CoreML-native TTS with cloning

- **speech-swift** — see A2: the only Swift/CoreML-native path today with *cloning + streaming* is its Qwen3-TTS engine (MLX or CoreML fp16 2.1 GB); CosyVoice3 clones+streams on MLX. Kokoro/Supertonic on ANE are preset-voice only.
- **FluidAudio** (https://github.com/FluidInference/FluidAudio): CoreML/ANE. TTS backends: **PocketTTS** — streaming, cloning from 1–30 s samples, EN/ES/DE/IT/PT/FR; **KokoroAne** — 82M, batch only, "3–11× RTFx on Apple Silicon", EN + Mandarin. No TTFA numbers published. Version string on the page 0.12.4.
- **mlx-audio-swift** (https://github.com/Blaizzy/mlx-audio-swift): 11 TTS models — Qwen3-TTS, OmniVoice, **Fish Audio S2 Pro**, Soprano, VyvoTTS, **Orpheus**, MOSS-TTS, **IndexTTS**, Marvis TTS, **Pocket TTS**, Irodori TTS; "streaming audio generation for real-time TTS" via async event iteration; MIT; 322 commits, 13 open issues, 6 open PRs; no perf numbers.
- **MetalRT** (RunAnywhere, proprietary; blog 2026-03-09 https://www.runanywhere.ai/blog/metalrt-speech-fastest-stt-tts-apple-silicon): Kokoro-82M on **M4 Max**: 178 ms (4 words), 230 ms (10), 381 ms (18), 604 ms (36) whole-utterance; "2.8x faster than mlx-audio on short phrases". Kokoro only; no cloning, no streaming mentioned.
- Apple ships no cloning TTS; WhisperKit-style Argmax `ttskit-coreml` is gone (404).

---

## (B) Leaderboards and benchmarks (open-weight rows only)

### B1. Artificial Analysis — Speech Arena (seen 2026-08-24)

**Open-weights leaderboard** (https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights; 15 open-weight rows out of 82 total; no "last updated" date on the page):

| # | Model (creator) | Elo |
|---|---|---|
| 1 | Fish Audio S2 Pro (Fish Audio) | 1,125 |
| 2 | Step Audio EditX – Mar 2026 (StepFun) | 1,102 |
| 3 | Voxtral TTS (Mistral) | 1,082 |
| 4 | Magpie-Multilingual 357M – Feb 2026 (NVIDIA) | 1,066 |
| 5 | Kokoro 82M v1.0 | 1,060 |
| 6 | Maya1 (Maya Research) | 1,045 |
| 7 | Higgs Audio V3 TTS (Boson AI) | 1,042 |
| 8 | Chatterbox (Resemble AI) | 1,020 |
| 9 | Magpie-Multilingual 357M (NVIDIA) | 1,004 |
| 10 | Zonos-v0.1 (Zyphra) | 1,000 |
| 11 | VibeVoice 7B (Microsoft) | 969 |
| 12 | OpenVoice v2 | 954 |
| 13 | XTTS v2 (Coqui) | 920 |
| 14 | StyleTTS 2 | 892 |
| 15 | MetaVoice v1 | 844 |

Not on the AA open-weights board at all: Qwen3-TTS, Orpheus, Dia, CosyVoice, IndexTTS, VoxCPM, dots.tts, Pocket TTS, Kyutai TTS. (Qwen-Audio-3.0-TTS-Plus sits #2 overall at 1,238 but as an API model — https://www.siliconflow.com/articles/benchmark/text-to-speech-models, secondary.)

**Controlled Voice Arena** (same 8 cloned voices for every model — the closest arena proxy for *clone* quality; https://artificialanalysis.ai/text-to-speech/leaderboard/controlled-voice, seen 2026-08-24): #1 Cartesia Sonic 3.6 1,119; #2 Sonic 3.5 1,096; #3 ElevenLabs v3 1,064; #4 StepAudio 2.5 TTS 1,052; #5 Inworld Realtime TTS-2 1,048. Open weights: **#12 Voxtral TTS 1,010** (2,394 votes), **#15 Fish Audio S2 Pro 1,002** (3,532 votes), #37 XTTS v2 821, #38 OpenVoice v2 799. An earlier snapshot reported in the launch coverage had Fish S2 Pro 1,034 / Voxtral 1,024 / **Chatterbox 930** (https://x.com/WesRoth/status/2075248621797384271) — Chatterbox was not in the rows my fetch rendered today. Note Voxtral TTS is preset-voice on Mac ("Mac (MLX, preset-voice only)" per tts-bench), so its controlled-voice rank is not usable for cloning on this machine.

### B2. TTS Arena V2 (Hugging Face, TTS-AGI)

The space (https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2) and its leaderboard route (https://tts-agi-tts-arena-v2.hf.space/leaderboard) are JS-rendered; neither fetch returned rows. The only transcription I could get is secondary (offlinetts.com, snapshot "as of May 2026", https://offlinetts.com/blog/tts-arena-leaderboard-2026/): Inworld Realtime TTS 1.5 Max 1,209.6; Gemini 3.1 Flash TTS 1,205.8; "every model in the top 10 is closed-source"; **Fish Audio S2 Pro #11 at 1,128.7** (top open weight); **Kokoro 82M v1.0 #32 at 1,056.2**. Those Elo values track the AA board closely, so the blog may be blending sources — treat as unverified. No open-model rows for Chatterbox, Higgs, Qwen3-TTS were quoted.

### B3. Seed-TTS-eval, English test set (test-en) — zero-shot cloning: WER (%) ↓ / SIM ↑

SIM values are only comparable *within* a row's source (different papers use different speaker encoders; IndexTTS2's "SS" column uses its own encoder and is inflated relative to the WavLM-based SIM used elsewhere). Dates = paper version seen.

| Model | WER | SIM | Source (who measured) |
|---|---|---|---|
| Human reference | 2.14 | 0.734 | Raon-OpenTTS Table 1 (Jun 2026) https://arxiv.org/html/2605.20830 ; CosyVoice 3 paper https://arxiv.org/html/2505.17589 |
| Seed-TTS (closed, reference point) | 2.25 | 0.762 | dots.tts Table 2 (2026-08-10) https://arxiv.org/html/2606.07080 ; VoxCPM Table 3 https://arxiv.org/html/2509.24650v1 |
| **dots.tts (SOAR)** 2B | **1.30** | **0.771** | dots.tts Table 2 ; Qwen-Audio-3.0-TTS Table 3 https://arxiv.org/html/2607.23938v1 |
| dots.tts (Pretrain) | 1.34 | 0.768 | dots.tts Table 2 |
| **VoxCPM 2** | 1.84 | **0.753** | dots.tts Table 2 ; Qwen-Audio-3.0-TTS Table 3 (0.753) |
| Raon-OpenTTS-1B (open data) | 1.78 | 0.749 | Raon-OpenTTS Table 1 |
| **Qwen3-TTS-12Hz-1.7B-Base** | **1.24** (own report) / 1.23 / 1.46 | 0.717 / 0.715 | Qwen3-TTS report Table 5 (WER only) https://arxiv.org/html/2601.15621v1 ; dots.tts Table 2 ; Raon Table 1 |
| Qwen3-TTS-12Hz-0.6B-Base | 1.32 | not published | Qwen3-TTS report Table 5 |
| **CosyVoice 3-1.5B** | 2.21 / 2.22 | 0.720 | CosyVoice 3 paper Table (test-en) ; dots.tts Table 2 ; Raon Table 1 |
| CosyVoice 3-0.5B (open) | 2.02 / 2.50 | 0.718 / 0.698 | CosyVoice 3 paper ; Raon Table 1 |
| **VoxCPM (0.5B)** | 1.85 / 1.98 | 0.729 / 0.730 | VoxCPM Table 3 (own) ; Raon Table 1 |
| **IndexTTS 2** (1.5B) | 2.23 / 2.18 (1.521 own) | 0.706 / 0.709 (0.860 own "SS") | dots.tts Table 2 ; Raon Table 1 ; IndexTTS2 paper Table 1 https://arxiv.org/html/2506.21619 |
| MaskGCT | 2.62 / 2.57 | 0.717 / 0.713 | VoxCPM Table 3 ; Raon Table 1 |
| **F5-TTS** (0.3B) | 2.00 / 1.83 / 2.04 | 0.670 / 0.647 / 0.671 | dots.tts ; CosyVoice 3 paper ; Raon Table 1 |
| Higgs Audio v2 (3B) | 2.44 | 0.677 | VoxCPM Table 3 |
| Higgs TTS 3 (4B, non-commercial) | 1.11 (own eval; Fish S2 Pro 1.31, Qwen3-TTS-1.7B 1.30 in same run) | not published | model card https://huggingface.co/bosonai/higgs-audio-v3-tts-4b |
| **Fish Audio S2 / S2 Pro** | **0.99** (own; test-zh 0.54) | not published in the Seed-TTS table | Fish S2 tech report Table 1 https://arxiv.org/html/2603.08823v1 |
| Voxtral TTS (4B, preset voices) | 2.19 | 0.663 | Raon Table 1 |
| CosyVoice 2 (0.5B) | 2.57–3.09 | 0.652–0.659 | CosyVoice 3 paper ; VoxCPM Table 3 ; Raon Table 1 |
| FireRedTTS 2 | 1.95 | 0.665 | dots.tts Table 2 |
| Spark TTS | 1.98 / 3.14 | 0.584 / 0.573 | CosyVoice 3 paper ; VoxCPM Table 3 |
| Llasa 8B | 3.63 | 0.581 | Raon Table 1 |
| Qwen-Audio-3.0-TTS (open-weight status not stated on the paper page) | 1.54 | 0.762 | Qwen-Audio-3.0-TTS Table 3 (Jul 2026) |
| MiniMax-Speech (closed) | 1.65 / 1.90 | 0.738 | Qwen3-TTS Table 5 ; Qwen-Audio-3.0 Table 3 |
| Chatterbox (any variant) | not published | not published | README/HF cards show only a Podonos preference image, no Seed-TTS-eval — https://github.com/resemble-ai/chatterbox , https://huggingface.co/ResembleAI/chatterbox-turbo |
| Kokoro, Orpheus, Dia, Kyutai TTS 1.6B, Pocket TTS | not evaluated on Seed-TTS-eval by any source seen | — | Kyutai card reports no WER/SIM (https://huggingface.co/kyutai/tts-1.6b-en_fr) |

Reading: on independent (third-party) measurements, the open-weight SIM ordering on test-en is dots.tts (0.771) > VoxCPM 2 (0.753) > Raon-1B (0.749) > VoxCPM 1 / CosyVoice 3 / Qwen3-TTS-1.7B / IndexTTS 2 (0.71–0.73, within noise of each other) > MaskGCT > F5-TTS / Higgs v2 (0.67) > Voxtral / CosyVoice 2 (0.65–0.66). Qwen3-TTS-1.7B has the best independently-measured WER (1.23) of anything with a Mac streaming path; Fish S2 claims lower (0.99) but self-reported and without SIM. Chatterbox has **no** Seed-TTS-eval numbers anywhere, so its clone quality relative to these can only be judged by arena (AA open-weights Elo 1,020, controlled-voice ~930 in the launch snapshot) and listening.

### B4. EmergentTTS-Eval and other 2026 leaderboards

- **EmergentTTS-Eval public leaderboard** (Boson; judge gemini-2.5-pro, thinking budget 256; baseline gpt-4o-mini-tts = 50 % win rate; https://github.com/boson-ai/EmergentTTS-Eval-public): open-source rows — Orpheus TTS 29.44 % (WER 17.71), F5-TTS 17.11 % (16.47), Tortoise 16.36 % (28.62), Bark 9.02 %, VITS-VCTK 8.47 %. Closed top: Gemini-2.5-Flash-Preview-TTS 75.57 %. Paper: https://arxiv.org/abs/2505.23009 (NeurIPS 2025).
- **dots.tts report Table 5** (2026-08-10, https://arxiv.org/html/2606.07080): dots.tts Pretrain 49.2 % (WER 10.86), SOAR 47.6 % (10.45), MF4 47.9 %, **Qwen3-TTS 42.8 % (WER 17.32)**, F5-TTS 15.3 %.
- **Higgs TTS 3 card** (own run, https://huggingface.co/bosonai/higgs-audio-v3-tts-4b): Higgs TTS 3 53.65 %, Fish Audio S2 Pro 43.80 %, Qwen3-TTS-1.7B 38.84 % overall; paralinguistics 68.57 / 53.75 / 44.29.
- Fish's own claim of "81.88 % win rate on EmergentTTS-Eval" for S2 Pro (https://fish.audio/blog/fish-audio-open-sources-s2/) is inconsistent with the two third-party runs above (43.8 %) — different judge/baseline; treat as marketing.
- No 2026 "Open TTS leaderboard" from Hume or ElevenLabs with open models was found; the MarkTechPost/Pinggy/SiliconFlow roundups only re-quote Artificial Analysis.

### B5. Apple-Silicon TTS latency comparisons (2026)

No source compares several *cloning* models' TTFA on M-series. What exists:
- **tts-bench** (5uck1ess, June 2026, https://github.com/5uck1ess/tts-bench): Mac rig = Apple M4 16 GB, "CPU + MPS" rows. Fastest on Mac: Piper, warm TTFA 208 ms, 32× RTFx. Marked Linux-only (no Mac row): Fish S2-Pro, MetaVoice, Step-Audio-EditX, Higgs Audio v3, dots.tts, Zonos2, Orpheus, CosyVoice 3. Voxtral: "Mac (MLX, preset-voice only)". Per-model Mac rows for Chatterbox/Qwen3-TTS/Kokoro are on its live site, not in the README I fetched.
- **MetalRT blog** (2026-03-09): Kokoro only, M4 Max 178–604 ms whole-utterance (see A7).
- **Soniqo**: Qwen3-TTS M2 Max RTF ≈ 0.55, 37 ms/step; Kokoro CoreML 0.08 RTF and Supertonic 0.15 RTF on iPhone 16 Pro (A2).
- **Pocket TTS**: ~200 ms first chunk on CPU, ~6× RT on MacBook Air M4 CPU (A3).
- **kakoo issue #23** (2026-04-13, https://github.com/ragaeeb/kakoo/issues/23) is a *request* to benchmark mlx-audio / MetalRT / llama.cpp / sherpa-onnx on Apple Silicon; it contains no results.

---

## Bottom line for the brief's question

1. **Runtimes that stream *and* clone on a Mac today:** speech-swift (Qwen3-TTS on MLX/CoreML; CosyVoice3 on MLX), mlx-audio (Qwen3-TTS with `streaming_interval≈0.32 s`; other models only via its generic chunker with undocumented intra-utterance behaviour), audio.cpp Metal (qwen3_tts, voxcpm2, dots_tts, omnivoice, neutts flagged "stream"+clone), FluidAudio/Pocket TTS (CoreML, 1–30 s reference), Kyutai DSM tts_mlx.py (frame-streams but preset voice embeddings only, and audio lags text by 1.28 s by design), sherpa-onnx ZipVoice (clone, sentence-chunk callback). llama.cpp's tts tool clones (Qwen3-TTS, Pocket TTS) but writes a WAV — no streaming.
2. **Nobody publishes a Mac TTFA for a cloning model.** The only Apple-Silicon numbers are Kokoro (178 ms M4 Max, MetalRT), Piper (208 ms M4, tts-bench), Pocket TTS (~200 ms first chunk on CPU), and Qwen3-TTS RTF 0.55 on M2 Max (Soniqo) plus a "120 ms first packet" figure without hardware. Whether Qwen3-TTS-1.7B via mlx-audio/speech-swift beats 0.9 s on the M4 Max has to be measured locally; the 0.32 s streaming interval and 37 ms/step suggest first audio well under 0.9 s is plausible but unproven.
3. **Clone-quality evidence:** the independently-measured Seed-TTS-eval leaders among open weights are dots.tts, VoxCPM 2, then a cluster of Qwen3-TTS-1.7B / CosyVoice 3 / IndexTTS 2 / VoxCPM 1 (SIM 0.71–0.73, all above F5-TTS 0.67 and CosyVoice 2 0.65). Chatterbox has no published SIM/WER and sits at Elo 1,020 (AA open-weights) / ~930 (controlled-voice launch snapshot), below Fish S2 Pro (1,125 / 1,002) and Higgs v3 (1,042). Of the SIM leaders, only Qwen3-TTS and VoxCPM 2 have a Mac path that also streams (VoxCPM 2 only via audio.cpp; speech-swift's VoxCPM2 is non-streaming).
4. **PyTorch MPS is not a viable path** for S3Gen/HiFT/BigVGAN-class vocoders per the issue trackers (CPU fallback, >65536-channel op unsupported, memory leaks); every fast Mac port re-implements them in MLX, CoreML or ggml.

**Out of cluster, flagged:** dots.tts 2B (Aug 2026; best open SIM 0.771; in audio.cpp with stream+clone, "Linux-only" per tts-bench — check its Metal status); VoxCPM 2 (SIM 0.753; audio.cpp Metal stream+clone, speech-swift MLX clone-only); Raon-OpenTTS-1B (open data, CC-BY-4.0, SIM 0.749, no Mac port seen); Qwen-Audio-3.0-TTS (Jul 2026 paper, SIM 0.762, open-weight status unconfirmed); Fish Audio S2 Pro (top open Elo, in mlx-audio-swift and mlx-audio, but tts-bench marks it Linux-only — Mac streaming status unverified).
