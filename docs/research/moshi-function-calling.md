# Extending Moshi for Function Calling

Research date: 2026-08-23. Web-only survey of how to add tool/function calling to
[kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) and its derivatives.

## Bottom line

Moshi has no native function calling, and the maintainers never answered the one
issue that asked for it ([#84](https://github.com/kyutai-labs/moshi/issues/84), open
since Sept 2024). But Kyutai shipped the blueprint in April 2026: **MoshiRAG** is
function calling with exactly one function. Forking it is the cheapest path.

## Why the obvious approach doesn't work

Moshi is an RQ-Transformer running **9 parallel streams at 12.5 Hz**: 1 text stream +
8 Mimi audio codebooks, each with its own delay, driven by a 7B temporal transformer
and a small depth transformer per frame.

The text stream is the "inner monologue" — **time-aligned to the speech Moshi is
about to say**, not a free scratchpad. Consequences:

- Emitting a ~60-token JSON tool call costs **60 frames = 4.8 s of wall clock**, and it
  gets spoken unless the model is trained not to.
- There is no system prompt. The [FAQ](https://github.com/kyutai-labs/moshi/blob/main/FAQ.md):
  changing voice or personality "would require fine tuning, which is not currently
  supported."
- Text vocab is a fixed 32k SentencePiece; `existing_text_padding_id=3`,
  `existing_text_end_padding_id=0`.

The design question is therefore not "how do I make Moshi emit JSON" but **"how do I
get a decision out of Moshi cheaply, and a result back in without stalling it."**

## Route A — Cascade it away (Unmute)

Kyutai's production answer is [Unmute](https://github.com/kyutai-labs/unmute): their
STT (with semantic VAD) + **any OpenAI-compatible LLM** + their 1.6B streaming TTS. Tool
calling comes free from the LLM. Cost: ~400–750 ms conversational latency instead of
Moshi's ~200 ms, and no true full-duplex behaviour.

Caveats: Unmute is **explicitly not supported on Mac** (issue #74, open) and has no tool
calling itself ("we would appreciate a contribution").

## Route B — Trigger token + external resolver (the MoshiRAG pattern) ← recommended

[Paper](https://arxiv.org/abs/2604.12928) · [code](https://github.com/kyutai-labs/moshi-rag) ·
[weights](https://huggingface.co/kyutai/moshika-rag-pytorch-bf16) (CC-BY-4.0, PyTorch + Candle).

Mechanism end to end:

1. **A single silent control token.** `<ret>` (`rag_token_id`, default **4**) is added to
   the text stream. It produces *no audio* — a pure control signal in the text channel.
2. **Detection in the serving loop.** The Python `Channel`/`InferenceJob` watches
   `text_token` in `StepOutput`; the Rust backend has a `RagManager` on a worker
   thread. Generation does **not** block.
3. **Context assembly.** A `TurnManager` keeps `user_text_buffer` (ASR) and
   `model_text_buffer` (Moshi's inner monologue), VAD-windowed; on turn switch they
   flush into `conversation_context`, which is cleaned and shipped to the backend LLM.
4. **Moshi keeps talking.** It is trained to emit "pre-RAG content" — a coarse answer
   plus filler ("let me check that for you…"). This exploits the **keyword delay** (~3 s
   between response onset and the informative word) to hide a **<2 s** retrieval
   budget. Measured time-to-first-audio-token: **0.0 s**.
5. **Result injection.** Returned text → **ARC-Encoder** (frozen, 4× sequence
   compression) → one trainable linear projection → **summed into the temporal
   transformer's input embeddings**, streaming, for `l` frames starting at delay `d`.
   That is the `streaming_sum` fuser — not cross-attention, not prompt prepending.
   Sequence length is preserved, so it works mid-utterance.
6. **Packaging.** The encoder runs as its own FastAPI service (`server_conditioner.py`):
   `GET /spec`, `POST /embed {"text": ...}` → safetensors blob.

Turning this into function calling:

| Concern | Approach |
|---|---|
| Which tool? | One generic `<tool>` token; let a small text LLM pick the tool from the transcript. Per-tool tokens scale badly and need per-tool training data. |
| Arguments? | **Never from Moshi.** The transcript (user ASR + inner monologue) is the argument source; a text LLM emits the structured call. Moshi decides *whether*, the LLM decides *what*. |
| Tool result? | Identical path to retrieved references — text → conditioner → streaming sum. |
| Latency? | The tool must return inside the ~2 s filler window, or fallback behaviour must be trained. |

The paper names this gap as a limitation: it "cannot handle multiple tools or
structured arguments simultaneously."

What it cost Kyutai: full fine-tuning (no LoRA), backbone trainable, encoder frozen,
0.2 dropout on reference docs. Training data: **~1.9M synthetic conversations** from
three Gemma-3-27B instances role-playing user / assistant / reference-provider,
seeded from NQ (307k), HotpotQA (90k), TriviaQA (76k) topics, 1.5–2.5 min each.

Mac note: the moshi-rag Rust backend supports `--features metal`; running it needs
four services (ARC encoder, retrieval LLM, Moshi server, STT).

## Route C — Native in-stream tool calling (research-grade)

Three under-documented hooks already in `LMModel.__init__`:

- **`extra_heads_num_heads` / `extra_heads_dim`** — auxiliary per-frame prediction
  heads. Kyutai STT uses these for semantic-VAD pause prediction at 0.5/1/2/3 s. A
  "call tool *k* now" head costs **zero text-stream bandwidth** and doesn't perturb the
  inner-monologue distribution. Cleanest place for a trigger.
- **`text_card_out`** — output text vocab can differ from input vocab, so control
  tokens can be *output-only* without touching the tokenizer or input embedding.
- **`demux_second_text_stream`** — a second text stream multiplexed into one vocab
  (used for TTS lookahead). Plausible carrier for a *silent action stream* holding JSON
  args, decoupled from spoken text. Speculative — no released checkpoint does this.

Plus the conditioner stack (`ConditionProvider`, `ConditionFuser` with
`sum`/`prepend`/`cross`/`streaming_sum`, `LUTConditioner`, `TensorCondition` =
`[B,T,D]` + mask) as the injection surface for tool results *and* a tool-manifest
pseudo-system-prompt.

MoshiVis is the precedent for the heavier variant: gated cross-attention adapters
(~206M params) over a **frozen** 7B backbone, gates zeroing out to exactly recover
base Moshi.

## Training practicalities

[moshi-finetune](https://github.com/kyutai-labs/moshi-finetune): LoRA (rank ≤128,
scaling 2.0, lr 2e-6), stereo WAV (**left = Moshi, right = user**) plus per-file JSON
transcript with timestamps, catalogued in a `.jsonl` of `{"path": ..., "duration": ...}`;
`annotate.py` generates transcripts. 8×H100 → ~10.7k tok/s/GPU at 23.7 GB. Serve with
`--lora-weight=.../lora.safetensors`.

Caveats:

- The repo trains **existing** streams. A new token, head, or conditioner means
  patching `lm.py` and the trainer.
- Data pipeline is the hard part: generate text dialogues with tool calls → TTS each
  side into a separate channel → force-align → insert the control token **at the
  correct frame**. The trigger must fire *before* the model commits to speaking.
- Tune trigger precision/recall deliberately: over-triggering destroys latency,
  under-triggering destroys factuality.

## Route D — Skip Moshi

If "native tool calling from speech" is the requirement rather than "Moshi
specifically": [Step-Audio 2](https://github.com/stepfun-ai/Step-Audio2) ships
JSON-style function calling from the audio stream with tool-calling accuracy on par
with text LLMs; Qwen3-Omni supports OpenAI-format function calling natively;
**NVIDIA NemotronLabs VoiceChat 11B** (Aug 2026) is the first open *full-duplex* model
with a native tool channel. See `open-duplex-models-mac.md`.

## Recommended ladder

1. **Days** — prototype tool logic on a cascade. Validates the tool surface without
   touching model internals.
2. **Weeks** — fork `moshi-rag`, swap the retrieval backend for a tool router. Inherits
   the trigger token, conditioner microservice, turn manager, Rust serving path, and
   released weights. Generic trigger + external argument synthesis.
3. **Months** — only if 2 is insufficient: `extra_heads` for the trigger, conditioners
   for results, MoshiVis-style gated adapters to protect the backbone.

## Sources

- [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) · [moshi/README.md](https://github.com/kyutai-labs/moshi/blob/main/moshi/README.md) · [FAQ](https://github.com/kyutai-labs/moshi/blob/main/FAQ.md) · [Issue #84](https://github.com/kyutai-labs/moshi/issues/84)
- [Moshi paper](https://kyutai.org/Moshi.pdf) · [DeepWiki: LMModel/LMGen](https://deepwiki.com/kyutai-labs/moshi/4.1-core-language-model-(lmmodel-and-lmgen))
- [MoshiRAG paper](https://arxiv.org/abs/2604.12928) · [repo](https://github.com/kyutai-labs/moshi-rag) · [DeepWiki RAG pipeline](https://deepwiki.com/kyutai-labs/moshi-rag) · [getting started](https://deepwiki.com/kyutai-labs/moshi-rag/1.1-getting-started)
- [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) · [nu-dialogue/moshi-finetune](https://github.com/nu-dialogue/moshi-finetune)
- [MoshiVis](https://github.com/kyutai-labs/moshivis) · [Vision-Speech Models paper](https://arxiv.org/html/2503.15633)
- [Unmute](https://github.com/kyutai-labs/unmute) · [Kyutai STT](https://kyutai.org/stt/) · [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling/)
- [Step-Audio 2 report](https://arxiv.org/abs/2507.16632) · [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)
