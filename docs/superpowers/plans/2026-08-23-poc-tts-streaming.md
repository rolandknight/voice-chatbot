# poc-tts-streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `poc-tts-streaming/` — a copy of `poc-tts/` on port 8006 that streams Chatterbox Flash audio to the browser sentence-by-sentence over WebRTC, speaking the OpenAI Realtime API (GA vocabulary, WebRTC transport only), with a browser test page that measures time-to-first-audio.

**Architecture:** The engine gains a `synthesize_stream()` generator (sentence-chunk pipelining; torch runs on one worker thread). A pure-Python `RealtimeSession` state machine turns Realtime client events into the server-event sequence and pushes PCM chunks into an audio sink. An aiortc layer (`realtime/webrtc.py`) binds one session per peer: `oai-events` data channel in, `PcmQueueTrack` (20 ms frames, silence on underrun) out. `server.py` exposes `/v1/realtime/client_secrets`, `/v1/realtime/calls`, and a chunked-PCM `/v1/audio/speech` that is the integration seam for the Rust PoC. The session module never imports aiortc or torch, so every protocol test is GPU-free and the module doubles as the spec for a future Rust port.

**Tech Stack:** Python 3.10 (mise-pinned, scoped to `poc-tts-streaming/`), `chatterbox-flash==0.1.0`, torch ≥ 2.6, FastAPI + uvicorn, aiortc (+ PyAV), pydantic 2, pytest, httpx. Browser: vanilla JS, `RTCPeerConnection`, WebAudio `AnalyserNode`.

**Spec:** `docs/superpowers/specs/2026-08-23-poc-tts-streaming-design.md`

## Global Constraints

- Python pinned to **3.10** via `poc-tts-streaming/mise.toml`; the rest of the repo stays hermit-managed. No root `mise.toml`.
- `chatterbox-flash` pinned **exactly** `0.1.0`. aiortc floor `>=1.9,<2`; record the resolved version in the README after first setup.
- **Never modify `poc-tts/`** — it stays on :8005 for side-by-side A/B. **Never modify `vendor/chatterbox-tts-server/`** or its venv.
- Server binds **port 8006**, host `127.0.0.1`.
- Engine output is **24 kHz mono float32**; the track emits **exactly 480-sample (20 ms) frames** — larger frames trip aiortc's same-timestamp-per-packet bug (`docs/web-rtc.md`).
- Event names are the **GA** Realtime names: `response.output_audio_transcript.delta`, `conversation.item.added`, `conversation.item.done`, `output_audio_buffer.started/stopped/cleared`, session `type: "realtime"`, `output_modalities`, `audio.output.voice`. They live only in `poc_tts_streaming/realtime/events.py`.
- **No `response.output_audio.delta` events.** Audio travels on the media track only.
- Chatterbox-specific parameters live under `session.x_chatterbox` (and `response.x_chatterbox` for per-response overrides). Standard fields are never overloaded.
- `session.audio.output.voice` is a reference-clip filename validated against the voice search paths; unknown → `error` event, never a fallback.
- One synthesis at a time per engine: a single-thread `SynthWorker`; further responses queue in arrival order.
- Dtype/backend resolution, OOM reporting, and voice discovery are copied verbatim from `poc-tts` with their tests. Do not "improve" them here.
- Work on branch `poc-tts-streaming`. Commit after every task with the `feat(poc-tts-streaming): …` / `test(poc-tts-streaming): …` / `docs(poc-tts-streaming): …` prefixes used by the previous PoC.
- All `make` and `pytest` commands below run **from `poc-tts-streaming/`** unless a path says otherwise. `make test` runs `.venv/bin/python -m pytest tests -v`.

---

### Task 1: Copy poc-tts, rename the package, toolchain on :8006

**Files:**
- Create: `poc-tts-streaming/` (copy of `poc-tts/`, see step 1 for exclusions)
- Modify: `poc-tts-streaming/config.yaml`, `Makefile`, `requirements.txt`, `setup.sh`, `README.md`, `.gitignore`
- Modify: `Makefile` (repo root, after the `poc-tts-test` target at lines 307-308)

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `poc_tts_streaming` package whose copied tests pass; `make run` serves the copied GUI on :8006; root `make poc-tts-streaming*` targets.

- [ ] **Step 1: Copy the directory without build artefacts**

From the repo root:

```bash
git checkout -b poc-tts-streaming
rsync -a --exclude .venv --exclude .pytest_cache --exclude '__pycache__' \
  --exclude reports --exclude .env --exclude 'bench-*.md' \
  --exclude mac-gpu-build-plan.md poc-tts/ poc-tts-streaming/
mv poc-tts-streaming/poc_tts poc-tts-streaming/poc_tts_streaming
grep -rl "poc_tts" poc-tts-streaming --include='*.py' --include='Makefile' --include='*.md' \
  | xargs sed -i 's/poc_tts\b/poc_tts_streaming/g'
```

Then check nothing slipped through:

```bash
grep -rn "poc_tts\b" poc-tts-streaming | grep -v poc_tts_streaming   # expect no output
```

- [ ] **Step 2: Port, title, requirements**

`poc-tts-streaming/config.yaml`: change `port: 8005` → `port: 8006`, and add at the end:

```yaml
realtime:
  model: chatterbox-flash
  default_voice: one-one.mp3
  client_secret_ttl_s: 600
```

Also set the tuned sweep result as the default generation block (the spec makes the tuned config the default here):

```yaml
engine:
  device: auto
  dtype: auto
  backend: auto
  drf_block_size: 32

generation:
  temperature: 0.6
  exaggeration: 0.5
  cfg_scale: 1.0
  num_steps: 4
  n_cfm_timesteps: 1
  chunk_size: 120
  split_text: true
  split_on_clauses: true
```

`poc-tts-streaming/requirements.txt`:

```
chatterbox-flash==0.1.0
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pyyaml==6.0.2
soundfile==0.12.1
aiortc>=1.9,<2
python-multipart==0.0.20
pytest==8.3.4
httpx==0.28.1
```

`poc-tts-streaming/poc_tts_streaming/server.py`: change the FastAPI title to `"poc-tts-streaming: Chatterbox Flash over Realtime/WebRTC"` and the `port` default in `main()` to `8006`.

- [ ] **Step 3: setup.sh — prove aiortc and torch coexist**

Append to `poc-tts-streaming/setup.sh` before the final `echo`:

```bash
# aiortc pulls PyAV with its own ffmpeg/libopus. Prove it imports beside
# torch in this venv; a wheel mismatch here would otherwise surface as a
# confusing failure on the first /calls.
./.venv/bin/python - <<'PY'
import av, aiortc, torch
print(f"aiortc {aiortc.__version__}, av {av.__version__}, torch {torch.__version__}")
PY
```

- [ ] **Step 4: Makefile targets**

In `poc-tts-streaming/Makefile` change the `run` help text to `:8006` and add a `bench-stream` target (the script arrives in Task 14):

```make
bench-stream: setup  ## TTFA / total / audio_s per baseline sentence -> reports/stream_runs.jsonl
	@$(LOAD_ENV) $(PY) -m poc_tts_streaming.bench_stream
```

Add `bench-stream` to `.PHONY`. In the repo-root `Makefile`, after the `poc-tts-test` target (line 307-308), add:

```make
poc-tts-streaming-setup:  ## poc-tts-streaming: mise python 3.10, venv, deps, aiortc probe (idempotent)
	@$(MAKE) -C poc-tts-streaming setup

poc-tts-streaming:    ## poc-tts-streaming: Flash streamed over Realtime/WebRTC on :8006
	@$(MAKE) -C poc-tts-streaming run

poc-tts-streaming-test:  ## poc-tts-streaming: GPU-free unit + loopback tests
	@$(MAKE) -C poc-tts-streaming test

poc-tts-streaming-bench:  ## poc-tts-streaming: streaming TTFA bench -> poc-tts-streaming/reports/stream_runs.jsonl
	@$(MAKE) -C poc-tts-streaming bench-stream
```

and append the four names to the root `.PHONY` line that lists `poc-tts-test`.

- [ ] **Step 5: README stub**

Replace `poc-tts-streaming/README.md` with:

```markdown
# poc-tts-streaming — Chatterbox Flash over the OpenAI Realtime API (WebRTC)

Copy of `poc-tts/` that streams audio sentence-by-sentence over WebRTC on
:8006, speaking the OpenAI Realtime API (`POST /v1/realtime/calls`,
`oai-events` data channel). poc-tts keeps :8005 so both run side by side.

    make              # install anything missing, then serve on :8006
    make test         # GPU-free unit + loopback WebRTC tests
    make bench-stream # TTFA per baseline sentence -> reports/stream_runs.jsonl
    make clean

Design: `docs/superpowers/specs/2026-08-23-poc-tts-streaming-design.md`
```

- [ ] **Step 6: Run setup and the copied tests**

Run: `make test`
Expected: setup completes (prints the aiortc/av/torch line), and every copied test passes under the new package name.

- [ ] **Step 7: Commit**

```bash
git add poc-tts-streaming Makefile
git commit -m "feat(poc-tts-streaming): copy poc-tts as the streaming PoC on :8006"
```

---

### Task 2: Clause splitting in `chunk_text`

**Files:**
- Modify: `poc-tts-streaming/poc_tts_streaming/engine_flash.py` (the `chunk_text` block)
- Test: `poc-tts-streaming/tests/test_chunking.py`

**Interfaces:**
- Produces: `chunk_text(text: str, chunk_size: int, split_on_clauses: bool = True) -> list[str]`. With `split_on_clauses=False` behaviour is byte-identical to poc-tts.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunking.py`:

```python
def test_overlong_sentence_splits_on_clauses_by_default():
    text = ("The door opened slowly, the corridor was dark, the air was cold; "
            "nobody had been here for years, and the dust proved it.")
    chunks = chunk_text(text, chunk_size=60)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_clause_splitting_can_be_disabled():
    text = ("The door opened slowly, the corridor was dark, the air was cold; "
            "nobody had been here for years, and the dust proved it.")
    assert chunk_text(text, chunk_size=60, split_on_clauses=False) == [text]


def test_short_sentences_are_never_clause_split():
    assert chunk_text("Yes, sir.", chunk_size=120) == ["Yes, sir."]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_chunking.py -v`
Expected: the two new tests FAIL (`TypeError: unexpected keyword` / wrong chunk count).

- [ ] **Step 3: Implement**

Replace the `chunk_text` block in `engine_flash.py` with:

```python
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _pack(units: list[str], chunk_size: int) -> list[str]:
    """Pack whole units together up to chunk_size; a unit longer than
    chunk_size is emitted on its own rather than cut mid-word."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= chunk_size:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int, split_on_clauses: bool = True) -> list[str]:
    """Split text into chunks of roughly chunk_size characters.

    Sentences are the unit: each generate() call is an independent draw with
    its own prosody and trailing silence, so anything smaller than a clause
    sounds like a list being read. Whole sentences are packed up to
    chunk_size. A sentence longer than chunk_size is split on clause
    punctuation (, ; :) when split_on_clauses is set -- the cheapest way to
    bring time-to-first-audio down on long sentences -- and otherwise
    emitted whole.
    """
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        if split_on_clauses and len(sentence) > chunk_size:
            units.extend(c.strip() for c in _CLAUSE_END.split(sentence) if c.strip())
        else:
            units.append(sentence)
    return _pack(units, chunk_size)
```

- [ ] **Step 4: Run the whole suite**

Run: `make test`
Expected: PASS, including the pre-existing `test_sentence_longer_than_chunk_size_is_not_dropped` (its sentence has no clause punctuation, so it is still emitted whole).

- [ ] **Step 5: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/engine_flash.py poc-tts-streaming/tests/test_chunking.py
git commit -m "feat(poc-tts-streaming): split over-long sentences on clauses"
```

---

### Task 3: `FlashEngine.synthesize_stream`

**Files:**
- Modify: `poc-tts-streaming/poc_tts_streaming/engine_flash.py` (`FlashEngine.synthesize`)
- Test: `poc-tts-streaming/tests/test_engine.py`

**Interfaces:**
- Produces:
  ```python
  def synthesize_stream(self, text: str, voice: str, *,
      temperature: float | None = None, exaggeration: float | None = None,
      cfg_scale: float | None = None, num_steps: int | None = None,
      n_cfm_timesteps: int | None = None, chunk_size: int = 120,
      split_text: bool = True, split_on_clauses: bool = True,
      cancel: threading.Event | None = None,
  ) -> Iterator[tuple[str, np.ndarray]]
  ```
  Yields `(chunk_text, mono_float32_pcm)` per chunk, in order. Raises `FileNotFoundError` / `ValueError` **before the first yield** (voice missing / text empty), `OutOfMemoryError` from inside the loop. Returns early, silently, once `cancel.is_set()` is observed between chunks.
- `synthesize(...)` keeps its signature and return type and is now implemented over `synthesize_stream`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine.py`:

```python
import threading


def _loaded_engine(tmp_path, samples_per_chunk=1000):
    (tmp_path / "a.wav").write_bytes(b"x")
    eng = _engine(tmp_path)
    fake_model = MagicMock()
    fake_model.sr = 24000
    fake_model.generate.return_value = torch.zeros(1, samples_per_chunk)
    with patch("poc_tts_streaming.engine_flash.ChatterboxFlashTTS") as cls:
        cls.from_pretrained.return_value = fake_model
        eng.load()
    return eng, fake_model


def test_synthesize_stream_yields_one_chunk_per_sentence_in_order(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    text = "First sentence here. Second sentence here. Third sentence here."
    out = list(eng.synthesize_stream(text, "a.wav", chunk_size=25))
    assert [t for t, _ in out] == [
        "First sentence here.", "Second sentence here.", "Third sentence here."]
    assert all(pcm.dtype == np.float32 and pcm.shape == (1000,) for _, pcm in out)
    assert model.generate.call_count == 3


def test_synthesize_stream_is_lazy(tmp_path):
    """The first chunk must come back before the second is generated --
    that is the whole point of streaming."""
    eng, model = _loaded_engine(tmp_path)
    gen = eng.synthesize_stream("One. Two.", "a.wav", chunk_size=4)
    next(gen)
    assert model.generate.call_count == 1


def test_synthesize_stream_stops_after_current_chunk_on_cancel(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    cancel = threading.Event()
    gen = eng.synthesize_stream("One. Two. Three.", "a.wav", chunk_size=4, cancel=cancel)
    next(gen)
    cancel.set()
    assert list(gen) == []
    assert model.generate.call_count == 1


def test_synthesize_stream_missing_voice_raises_before_first_yield(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    with pytest.raises(FileNotFoundError):
        next(eng.synthesize_stream("Hello.", "missing.wav"))
    model.generate.assert_not_called()


def test_synthesize_stream_forwards_split_on_clauses(tmp_path):
    eng, model = _loaded_engine(tmp_path)
    text = "alpha beta, gamma delta, epsilon zeta, eta theta."
    whole = list(eng.synthesize_stream(text, "a.wav", chunk_size=20, split_on_clauses=False))
    split = list(eng.synthesize_stream(text, "a.wav", chunk_size=20, split_on_clauses=True))
    assert len(whole) == 1 and len(split) > 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine.py -v -k stream`
Expected: FAIL with `AttributeError: 'FlashEngine' object has no attribute 'synthesize_stream'`.

- [ ] **Step 3: Implement**

Add `import threading` and `from typing import Iterator` at the top of `engine_flash.py`. Replace `FlashEngine.synthesize` with:

```python
    def synthesize_stream(
        self,
        text: str,
        voice: str,
        *,
        temperature: float | None = None,
        exaggeration: float | None = None,
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        n_cfm_timesteps: int | None = None,
        chunk_size: int = 120,
        split_text: bool = True,
        split_on_clauses: bool = True,
        cancel: threading.Event | None = None,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (chunk_text, mono float32 pcm) per sentence chunk, in order.

        Validation (voice, text) happens before the first yield so callers
        can fail fast. Cancellation is checked between chunks: a chunk
        already inside generate() finishes (~1 s tuned) and is discarded by
        the caller. generate() itself cannot be interrupted.
        """
        if not self.loaded:
            raise RuntimeError("model is not loaded -- call load() first")
        gen = self._generation_cfg
        prompt = str(resolve_voice_path(voice, self._voice_paths))
        if split_text:
            chunks = chunk_text(text, chunk_size, split_on_clauses=split_on_clauses)
        else:
            chunks = [t for t in [text.strip()] if t]
        if not chunks:
            raise ValueError("text is empty")

        for chunk in chunks:
            if cancel is not None and cancel.is_set():
                return
            try:
                wav = self._model.generate(
                    chunk,
                    audio_prompt_path=prompt,
                    temperature=temperature if temperature is not None else gen["temperature"],
                    exaggeration=exaggeration if exaggeration is not None else gen["exaggeration"],
                    cfg_scale=cfg_scale if cfg_scale is not None else gen["cfg_scale"],
                    num_steps=num_steps if num_steps is not None else gen["num_steps"],
                    n_cfm_timesteps=(
                        n_cfm_timesteps if n_cfm_timesteps is not None
                        else gen["n_cfm_timesteps"]
                    ),
                    backend=self.backend,
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise OutOfMemoryError(
                    f"ran out of VRAM during generation. {_vram_report()}"
                ) from exc
            yield chunk, wav.detach().float().cpu().numpy().reshape(-1)

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        temperature: float | None = None,
        exaggeration: float | None = None,
        cfg_scale: float | None = None,
        num_steps: int | None = None,
        n_cfm_timesteps: int | None = None,
        chunk_size: int = 120,
        split_text: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Whole-utterance synthesis: synthesize_stream concatenated."""
        pieces = [
            pcm for _, pcm in self.synthesize_stream(
                text, voice,
                temperature=temperature, exaggeration=exaggeration,
                cfg_scale=cfg_scale, num_steps=num_steps,
                n_cfm_timesteps=n_cfm_timesteps, chunk_size=chunk_size,
                split_text=split_text,
            )
        ]
        return np.concatenate(pieces), self.sr
```

- [ ] **Step 4: Run the suite**

Run: `make test`
Expected: PASS. The pre-existing `synthesize` tests (`concatenates_chunks`, `rejects_blank_text`, OOM translation) still pass through the new path.

- [ ] **Step 5: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/engine_flash.py poc-tts-streaming/tests/test_engine.py
git commit -m "feat(poc-tts-streaming): synthesize_stream yields audio per sentence chunk"
```

---

### Task 4: `audio.py` — int16 conversion and 20 ms framing

**Files:**
- Create: `poc-tts-streaming/poc_tts_streaming/audio.py`
- Test: `poc-tts-streaming/tests/test_audio.py`

**Interfaces:**
- Produces: `SAMPLE_RATE = 24000`, `FRAME_SAMPLES = 480`, `to_int16(pcm: np.ndarray) -> np.ndarray[int16]`, `silence_frame() -> np.ndarray[int16] (480,)`, and
  ```python
  class FrameSlicer:
      def push(self, pcm_int16: np.ndarray) -> list[np.ndarray]   # full 480-sample frames
      def flush(self) -> list[np.ndarray]                        # zero-padded tail, at most one frame
      def clear(self) -> None
  ```

- [ ] **Step 1: Write the failing tests**

`tests/test_audio.py`:

```python
import numpy as np

from poc_tts_streaming.audio import FRAME_SAMPLES, FrameSlicer, silence_frame, to_int16


def test_to_int16_clips_and_scales():
    out = to_int16(np.array([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float32))
    assert out.dtype == np.int16
    assert out.tolist() == [-32767, -32767, 0, 16383, 32767]


def test_slicer_emits_full_frames_and_carries_the_remainder():
    s = FrameSlicer()
    frames = s.push(np.arange(1001, dtype=np.int16))
    assert [len(f) for f in frames] == [480, 480]
    assert frames[0][0] == 0 and frames[1][0] == 480
    tail = s.flush()
    assert len(tail) == 1 and len(tail[0]) == FRAME_SAMPLES
    assert tail[0][:41].tolist() == list(range(960, 1001))
    assert not tail[0][41:].any()


def test_slicer_joins_across_pushes():
    s = FrameSlicer()
    assert s.push(np.ones(300, dtype=np.int16)) == []
    frames = s.push(np.ones(300, dtype=np.int16))
    assert len(frames) == 1 and frames[0].all()


def test_flush_on_empty_slicer_emits_nothing():
    assert FrameSlicer().flush() == []


def test_clear_drops_the_carry():
    s = FrameSlicer()
    s.push(np.ones(300, dtype=np.int16))
    s.clear()
    assert s.flush() == []


def test_silence_frame_shape():
    f = silence_frame()
    assert f.dtype == np.int16 and f.shape == (FRAME_SAMPLES,) and not f.any()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: poc_tts_streaming.audio`.

- [ ] **Step 3: Implement**

`poc_tts_streaming/audio.py`:

```python
"""PCM helpers shared by the WebRTC track and the chunked-PCM endpoint.

Everything here is numpy-only: no aiortc, no torch.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def to_int16(pcm: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] -> int16, clipped. Same scaling as poc-tts's WAV path."""
    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def silence_frame() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


class FrameSlicer:
    """Re-frame arbitrary-length int16 PCM into exact FRAME_SAMPLES frames.

    aiortc stamps every RTP packet cut from one AudioFrame with the same
    timestamp, so frames larger than 20 ms lose all but one packet
    (docs/web-rtc.md). Exact 480-sample frames are therefore a hard rule,
    and this is the one place that rule is enforced.
    """

    def __init__(self) -> None:
        self._carry = np.zeros(0, dtype=np.int16)

    def push(self, pcm_int16: np.ndarray) -> list[np.ndarray]:
        buf = np.concatenate([self._carry, np.asarray(pcm_int16, dtype=np.int16)])
        n_full = len(buf) // FRAME_SAMPLES
        frames = [buf[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n_full)]
        self._carry = buf[n_full * FRAME_SAMPLES:]
        return frames

    def flush(self) -> list[np.ndarray]:
        if len(self._carry) == 0:
            return []
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        frame[:len(self._carry)] = self._carry
        self._carry = np.zeros(0, dtype=np.int16)
        return [frame]

    def clear(self) -> None:
        self._carry = np.zeros(0, dtype=np.int16)
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_audio.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/audio.py poc-tts-streaming/tests/test_audio.py
git commit -m "feat(poc-tts-streaming): int16 framing helpers for 20 ms WebRTC frames"
```

---

### Task 5: `track.py` — `PcmQueueTrack`

**Files:**
- Create: `poc-tts-streaming/poc_tts_streaming/track.py`
- Test: `poc-tts-streaming/tests/test_track.py`

**Interfaces:**
- Consumes: `audio.FrameSlicer`, `audio.to_int16`, `audio.silence_frame`, `audio.SAMPLE_RATE`.
- Produces:
  ```python
  class PcmQueueTrack(aiortc.MediaStreamTrack):   # kind = "audio"
      def __init__(self, *, paced: bool = True)
      def push(self, pcm_float32: np.ndarray) -> None   # event-loop thread only
      def flush(self) -> None                            # pad the partial frame
      def clear(self) -> None                            # drop everything queued
      async def drained(self) -> None                    # resolves when the queue is empty
      async def recv(self) -> av.AudioFrame              # 480 samples, s16 mono 24 kHz
      @property
      def queued_frames(self) -> int
  ```
  This is the `AudioSink` the session talks to (Task 7 declares the Protocol).

- [ ] **Step 1: Write the failing tests**

`tests/test_track.py`:

```python
import asyncio

import numpy as np

from poc_tts_streaming.track import PcmQueueTrack


def run(coro):
    return asyncio.run(coro)


async def _frames(track, n):
    return [await track.recv() for _ in range(n)]


def test_recv_returns_20ms_s16_mono_frames_with_advancing_pts():
    track = PcmQueueTrack(paced=False)
    track.push(np.full(1001, 0.5, dtype=np.float32))
    track.flush()
    frames = run(_frames(track, 4))
    assert [f.samples for f in frames] == [480, 480, 480, 480]
    assert [f.pts for f in frames] == [0, 480, 960, 1440]
    assert all(f.sample_rate == 24000 and f.format.name == "s16" and f.layout.name == "mono"
               for f in frames)
    first = np.frombuffer(bytes(frames[0].planes[0]), dtype=np.int16)
    assert first[0] == 16383
    fourth = np.frombuffer(bytes(frames[3].planes[0]), dtype=np.int16)
    assert not fourth.any(), "underrun must produce silence, never a stall"


def test_clear_drops_queued_audio():
    track = PcmQueueTrack(paced=False)
    track.push(np.ones(4800, dtype=np.float32))
    assert track.queued_frames == 10
    track.clear()
    assert track.queued_frames == 0
    frame = run(track.recv())
    assert not np.frombuffer(bytes(frame.planes[0]), dtype=np.int16).any()


def test_drained_resolves_when_queue_empties():
    async def main():
        track = PcmQueueTrack(paced=False)
        track.push(np.ones(960, dtype=np.float32))
        waiter = asyncio.ensure_future(track.drained())
        await asyncio.sleep(0)
        assert not waiter.done()
        await track.recv()
        await track.recv()
        await asyncio.wait_for(waiter, 1)
    run(main())


def test_drained_on_empty_track_returns_immediately():
    async def main():
        track = PcmQueueTrack(paced=False)
        await asyncio.wait_for(track.drained(), 1)
    run(main())
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_track.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`poc_tts_streaming/track.py`:

```python
"""Outbound WebRTC audio track fed from a queue of PCM chunks.

Owns pacing and framing; knows nothing about Realtime events or the engine.
"""

from __future__ import annotations

import asyncio
import fractions
import time
from collections import deque

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame

from poc_tts_streaming.audio import FRAME_SAMPLES, SAMPLE_RATE, FrameSlicer, silence_frame, to_int16

_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class PcmQueueTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, *, paced: bool = True) -> None:
        super().__init__()
        self._paced = paced
        self._slicer = FrameSlicer()
        self._queue: deque[np.ndarray] = deque()
        self._pts = 0
        self._start: float | None = None
        self._waiters: list[asyncio.Future] = []

    # ---- producer side (event loop thread) --------------------------------

    def push(self, pcm_float32: np.ndarray) -> None:
        self._queue.extend(self._slicer.push(to_int16(pcm_float32)))

    def flush(self) -> None:
        self._queue.extend(self._slicer.flush())

    def clear(self) -> None:
        self._queue.clear()
        self._slicer.clear()
        self._resolve_waiters()

    @property
    def queued_frames(self) -> int:
        return len(self._queue)

    async def drained(self) -> None:
        if not self._queue:
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        await fut

    def _resolve_waiters(self) -> None:
        for fut in self._waiters:
            if not fut.done():
                fut.set_result(None)
        self._waiters.clear()

    # ---- consumer side (aiortc RTP sender) ---------------------------------

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise Exception("track ended")
        if self._paced:
            if self._start is None:
                self._start = time.monotonic()
            wait = self._start + self._pts / SAMPLE_RATE - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

        if self._queue:
            samples = self._queue.popleft()
            if not self._queue:
                self._resolve_waiters()
        else:
            samples = silence_frame()

        frame = AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = _TIME_BASE
        self._pts += FRAME_SAMPLES
        return frame
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_track.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/track.py poc-tts-streaming/tests/test_track.py
git commit -m "feat(poc-tts-streaming): PcmQueueTrack with 20 ms pacing and silence on underrun"
```

---

### Task 6: `realtime/ids.py` and `realtime/events.py`

**Files:**
- Create: `poc-tts-streaming/poc_tts_streaming/realtime/__init__.py` (empty)
- Create: `poc-tts-streaming/poc_tts_streaming/realtime/ids.py`
- Create: `poc-tts-streaming/poc_tts_streaming/realtime/events.py`
- Test: `poc-tts-streaming/tests/test_realtime_events.py`

**Interfaces:**
- Produces:
  - `ids.new_id(prefix: str) -> str` (`"sess_…"`, 16 hex chars), `ids.now() -> int` (unix seconds).
  - `events.E` — string constants for every server event type used.
  - `events.EventError(code, message, *, param=None, event_id=None, error_type="invalid_request_error")` — exception carrying an OpenAI-shaped error.
  - `events.parse_client_event(raw: str) -> ClientEvent` — raises `EventError` for bad JSON, unknown type, unsupported type, or schema failure.
  - Client models: `SessionUpdate(session: dict)`, `ConversationItemCreate(item: dict, previous_item_id: str|None)`, `ConversationItemDelete(item_id: str)`, `ResponseCreate(response: dict)`, `ResponseCancel(response_id: str|None)`, `OutputAudioBufferClear()`. All have `type` and `event_id: str | None`.
  - `events.server_event(type_: str, **fields) -> dict` (adds `event_id`), `events.error_event(err: EventError) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_realtime_events.py`:

```python
import json

import pytest

from poc_tts_streaming.realtime.events import (
    E, EventError, ResponseCreate, SessionUpdate, error_event, parse_client_event, server_event,
)
from poc_tts_streaming.realtime.ids import new_id


def test_new_id_has_prefix_and_is_unique():
    a, b = new_id("sess"), new_id("sess")
    assert a.startswith("sess_") and a != b and len(a) == len("sess_") + 16


def test_parse_session_update():
    ev = parse_client_event(json.dumps({
        "type": "session.update", "event_id": "evt_1",
        "session": {"audio": {"output": {"voice": "marvin.wav"}}}}))
    assert isinstance(ev, SessionUpdate)
    assert ev.event_id == "evt_1"
    assert ev.session["audio"]["output"]["voice"] == "marvin.wav"


def test_parse_response_create_defaults_to_empty_response():
    ev = parse_client_event('{"type": "response.create"}')
    assert isinstance(ev, ResponseCreate) and ev.response == {}


def test_bad_json_is_an_event_error():
    with pytest.raises(EventError) as exc:
        parse_client_event("{not json")
    assert exc.value.code == "invalid_json"


def test_unknown_type_is_an_event_error_with_the_event_id_echoed():
    with pytest.raises(EventError) as exc:
        parse_client_event('{"type": "nope.nothing", "event_id": "evt_9"}')
    assert exc.value.code == "unknown_event" and exc.value.event_id == "evt_9"


@pytest.mark.parametrize("t", [
    "input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.clear",
    "conversation.item.truncate", "conversation.item.retrieve",
])
def test_known_but_unsupported_types_say_so(t):
    with pytest.raises(EventError) as exc:
        parse_client_event(json.dumps({"type": t}))
    assert exc.value.code == "unsupported_event"
    assert "not supported" in exc.value.message


def test_schema_failure_names_the_param():
    with pytest.raises(EventError) as exc:
        parse_client_event('{"type": "conversation.item.delete"}')
    assert exc.value.code == "missing_required_parameter"
    assert exc.value.param == "item_id"


def test_server_event_adds_an_event_id():
    ev = server_event(E.SESSION_CREATED, session={"id": "sess_x"})
    assert ev["type"] == "session.created"
    assert ev["event_id"].startswith("event_")
    assert ev["session"] == {"id": "sess_x"}


def test_error_event_shape():
    ev = error_event(EventError("invalid_value", "bad voice", param="session.audio.output.voice",
                                event_id="evt_3"))
    assert ev["type"] == "error"
    assert ev["error"] == {
        "type": "invalid_request_error", "code": "invalid_value", "message": "bad voice",
        "param": "session.audio.output.voice", "event_id": "evt_3",
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_realtime_events.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement ids.py**

`poc_tts_streaming/realtime/ids.py`:

```python
"""Identifier and clock helpers shaped like the OpenAI Realtime API's."""

from __future__ import annotations

import secrets
import time


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def now() -> int:
    return int(time.time())
```

- [ ] **Step 4: Implement events.py**

`poc_tts_streaming/realtime/events.py`:

```python
"""Realtime API event vocabulary and client-event validation.

The single place GA event names live. Verified against the API reference on
2026-08-23. No aiortc, no torch.
"""

from __future__ import annotations

import json
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from poc_tts_streaming.realtime.ids import new_id


class E:
    """Server event types."""
    ERROR = "error"
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    CONVERSATION_CREATED = "conversation.created"
    ITEM_ADDED = "conversation.item.added"
    ITEM_DONE = "conversation.item.done"
    ITEM_DELETED = "conversation.item.deleted"
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"
    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"
    AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
    AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
    AUDIO_DONE = "response.output_audio.done"
    OUTPUT_AUDIO_BUFFER_STARTED = "output_audio_buffer.started"
    OUTPUT_AUDIO_BUFFER_STOPPED = "output_audio_buffer.stopped"
    OUTPUT_AUDIO_BUFFER_CLEARED = "output_audio_buffer.cleared"


class EventError(Exception):
    """An error to report as an `error` event (or an HTTP error body)."""

    def __init__(self, code: str, message: str, *, param: str | None = None,
                 event_id: str | None = None, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.code, self.message, self.param = code, message, param
        self.event_id, self.error_type = event_id, error_type

    def as_dict(self) -> dict:
        return {"type": self.error_type, "code": self.code, "message": self.message,
                "param": self.param, "event_id": self.event_id}


# ---- client events ---------------------------------------------------------

class _Base(BaseModel):
    event_id: Optional[str] = None


class SessionUpdate(_Base):
    type: Literal["session.update"]
    session: dict = Field(default_factory=dict)


class ConversationItemCreate(_Base):
    type: Literal["conversation.item.create"]
    item: dict
    previous_item_id: Optional[str] = None


class ConversationItemDelete(_Base):
    type: Literal["conversation.item.delete"]
    item_id: str


class ResponseCreate(_Base):
    type: Literal["response.create"]
    response: dict = Field(default_factory=dict)


class ResponseCancel(_Base):
    type: Literal["response.cancel"]
    response_id: Optional[str] = None


class OutputAudioBufferClear(_Base):
    type: Literal["output_audio_buffer.clear"]


ClientEvent = Union[
    SessionUpdate, ConversationItemCreate, ConversationItemDelete,
    ResponseCreate, ResponseCancel, OutputAudioBufferClear,
]
_ADAPTER = TypeAdapter(ClientEvent)

# Real Realtime client events this TTS server deliberately does not implement.
UNSUPPORTED = frozenset({
    "input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.clear",
    "conversation.item.truncate", "conversation.item.retrieve",
})


def parse_client_event(raw: str) -> ClientEvent:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EventError("invalid_json", f"invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise EventError("invalid_json", "event must be a JSON object")
    event_id = data.get("event_id") if isinstance(data.get("event_id"), str) else None
    type_ = data.get("type")
    if type_ in UNSUPPORTED:
        raise EventError("unsupported_event",
                         f"'{type_}' is not supported by this server (text-to-speech only)",
                         param="type", event_id=event_id)
    try:
        return _ADAPTER.validate_python(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = [str(p) for p in first.get("loc", ()) if p not in ("tagged-union",)]
        # A type that matches no model shows up as a union tag failure on "type".
        if not isinstance(type_, str) or first.get("type") in ("union_tag_invalid", "union_tag_not_found"):
            raise EventError("unknown_event", f"unknown event type {type_!r}",
                             param="type", event_id=event_id) from exc
        param = loc[-1] if loc else None
        code = "missing_required_parameter" if first.get("type") == "missing" else "invalid_value"
        raise EventError(code, first.get("msg", "invalid event"), param=param,
                         event_id=event_id) from exc


# ---- server events ---------------------------------------------------------

def server_event(type_: str, **fields) -> dict:
    return {"type": type_, "event_id": new_id("event"), **fields}


def error_event(err: EventError) -> dict:
    return server_event(E.ERROR, error=err.as_dict())
```

- [ ] **Step 5: Run**

Run: `.venv/bin/python -m pytest tests/test_realtime_events.py -v`
Expected: PASS. If `test_unknown_type_is_an_event_error…` fails on pydantic's error `type` string, print `exc.errors()[0]["type"]` once and add that string to the tuple in `parse_client_event` — pydantic 2.10 reports `union_tag_invalid` for a tag value that matches no member.

- [ ] **Step 6: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/realtime poc-tts-streaming/tests/test_realtime_events.py
git commit -m "feat(poc-tts-streaming): Realtime event vocabulary and client-event validation"
```

---

### Task 7: `realtime/session.py` — the protocol state machine

**Files:**
- Create: `poc-tts-streaming/poc_tts_streaming/realtime/session.py`
- Test: `poc-tts-streaming/tests/test_realtime_session.py`

**Interfaces:**
- Consumes: Task 6 models and helpers.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ChatterboxKnobs:
      temperature: float; exaggeration: float; cfg_scale: float
      num_steps: int; n_cfm_timesteps: int; chunk_size: int
      split_text: bool; split_on_clauses: bool
      @classmethod def from_config(cls, generation_cfg: dict) -> "ChatterboxKnobs"
      def merged(self, patch: dict, *, param_prefix: str) -> "ChatterboxKnobs"   # raises EventError
      def as_engine_kwargs(self) -> dict
      def as_dict(self) -> dict

  Synthesizer = Callable[[str, str, ChatterboxKnobs, threading.Event], Iterator[tuple[str, np.ndarray]]]
  # (text, voice, knobs, cancel) -> chunks; runs on the worker thread

  class AudioSink(Protocol):
      def push(self, pcm: np.ndarray) -> None
      def flush(self) -> None
      def clear(self) -> None
      async def drained(self) -> None

  class SynthWorker:                      # one per engine; ThreadPoolExecutor(max_workers=1)
      def submit(self, fn) -> concurrent.futures.Future
      def shutdown(self) -> None

  class RealtimeSession:
      def __init__(self, *, send: Callable[[dict], None], synthesizer: Synthesizer,
                   sink: AudioSink, worker: SynthWorker, voices: Callable[[], list[str]],
                   voice: str, knobs: ChatterboxKnobs, model: str = "chatterbox-flash",
                   session_patch: dict | None = None)
      id: str
      async def open(self) -> None         # session.created, conversation.created
      async def handle(self, raw: str) -> None
      async def close(self) -> None
      def session_object(self) -> dict
      def apply_session_patch(self, patch: dict) -> None   # raises EventError
  ```

- [ ] **Step 1: Write the failing tests**

`tests/test_realtime_session.py`:

```python
import asyncio
import json
import threading

import numpy as np
import pytest

from poc_tts_streaming.realtime.events import EventError
from poc_tts_streaming.realtime.session import ChatterboxKnobs, RealtimeSession, SynthWorker

KNOBS = ChatterboxKnobs.from_config({
    "temperature": 0.6, "exaggeration": 0.5, "cfg_scale": 1.0,
    "num_steps": 4, "n_cfm_timesteps": 1, "chunk_size": 120,
    "split_text": True, "split_on_clauses": True,
})


class FakeSink:
    def __init__(self):
        self.pushed, self.flushed, self.cleared = [], 0, 0
    def push(self, pcm): self.pushed.append(pcm)
    def flush(self): self.flushed += 1
    def clear(self): self.cleared += 1
    async def drained(self): return None


class FakeSynth:
    """Splits on '. ' and yields 100 samples per sentence; records calls."""
    def __init__(self, gate: threading.Event | None = None):
        self.calls, self.gate = [], gate
    def __call__(self, text, voice, knobs, cancel):
        self.calls.append((text, voice, knobs))
        for i, sentence in enumerate(s for s in text.split(". ") if s):
            if self.gate is not None and i > 0:
                self.gate.wait(2)
            if cancel.is_set():
                return
            yield sentence, np.full(100, 0.1, dtype=np.float32)


def make_session(synth=None, sink=None, voice="one-one.mp3"):
    sent = []
    worker = SynthWorker()
    session = RealtimeSession(
        send=sent.append, synthesizer=synth or FakeSynth(), sink=sink or FakeSink(),
        worker=worker, voices=lambda: ["one-one.mp3", "marvin.wav"], voice=voice, knobs=KNOBS,
    )
    return session, sent, worker


def types(sent):
    return [e["type"] for e in sent]


async def until(sent, type_, timeout=5):
    for _ in range(int(timeout * 100)):
        if any(e["type"] == type_ for e in sent):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never saw {type_}; got {types(sent)}")


def run(coro):
    return asyncio.run(coro)


def test_open_sends_session_then_conversation_created():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        assert types(sent) == ["session.created", "conversation.created"]
        s = sent[0]["session"]
        assert s["type"] == "realtime" and s["object"] == "realtime.session"
        assert s["id"] == session.id and s["model"] == "chatterbox-flash"
        assert s["output_modalities"] == ["audio"]
        assert s["audio"]["output"]["voice"] == "one-one.mp3"
        assert s["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert s["x_chatterbox"]["num_steps"] == 4
        assert sent[1]["conversation"]["object"] == "realtime.conversation"
    run(main())


def test_session_update_changes_voice_and_knobs():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "session.update", "session": {
            "audio": {"output": {"voice": "marvin.wav"}},
            "x_chatterbox": {"num_steps": 8, "split_on_clauses": False}}}))
        assert types(sent)[-1] == "session.updated"
        s = sent[-1]["session"]
        assert s["audio"]["output"]["voice"] == "marvin.wav"
        assert s["x_chatterbox"]["num_steps"] == 8 and s["x_chatterbox"]["split_on_clauses"] is False
        assert s["x_chatterbox"]["temperature"] == 0.6, "untouched knobs keep their value"
    run(main())


def test_unknown_voice_is_an_error_and_session_is_unchanged():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "session.update", "event_id": "evt_7",
                                         "session": {"audio": {"output": {"voice": "ghost.wav"}}}}))
        err = sent[-1]
        assert err["type"] == "error"
        assert err["error"]["code"] == "invalid_value"
        assert err["error"]["param"] == "session.audio.output.voice"
        assert err["error"]["event_id"] == "evt_7"
        assert session.session_object()["audio"]["output"]["voice"] == "one-one.mp3"
    run(main())


def test_out_of_range_knob_is_an_error():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "session.update",
                                         "session": {"x_chatterbox": {"num_steps": 99}}}))
        assert sent[-1]["type"] == "error"
        assert sent[-1]["error"]["param"] == "session.x_chatterbox.num_steps"
    run(main())


def test_item_create_emits_added_and_done():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Hello there."}]}}))
        assert types(sent)[-2:] == ["conversation.item.added", "conversation.item.done"]
        item = sent[-1]["item"]
        assert item["id"].startswith("item_") and item["object"] == "realtime.item"
        assert item["role"] == "user" and item["status"] == "completed"
        assert item["content"] == [{"type": "input_text", "text": "Hello there."}]
        assert sent[-1]["previous_item_id"] is None
    run(main())


def test_assistant_items_are_rejected():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "x"}]}}))
        assert sent[-1]["type"] == "error" and sent[-1]["error"]["param"] == "item.role"
    run(main())


RESPONSE_SEQUENCE = [
    "response.created",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_audio_transcript.delta",   # "Hello there"
    "output_audio_buffer.started",
    "response.output_audio_transcript.delta",   # "General Kenobi."
    "response.output_audio_transcript.done",
    "response.output_audio.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.done",
    "output_audio_buffer.stopped",
]


def test_full_response_sequence_for_two_sentences():
    async def main():
        synth, sink = FakeSynth(), FakeSink()
        session, sent, _ = make_session(synth, sink)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Hello there. General Kenobi."}]}}))
        n = len(sent)
        await session.handle(json.dumps({"type": "response.create", "event_id": "evt_r"}))
        await until(sent, "output_audio_buffer.stopped")
        assert types(sent)[n:] == RESPONSE_SEQUENCE
        deltas = [e["delta"] for e in sent if e["type"] == "response.output_audio_transcript.delta"]
        assert deltas == ["Hello there ", "General Kenobi. "]
        done = [e for e in sent if e["type"] == "response.done"][0]["response"]
        assert done["status"] == "completed" and done["id"].startswith("resp_")
        assert done["output"][0]["role"] == "assistant"
        assert done["output"][0]["content"][0] == {"type": "audio", "transcript": "Hello there General Kenobi. "}
        assert done["usage"]["output_tokens"] == 0
        assert len(sink.pushed) == 2 and sink.flushed == 1
        assert synth.calls[0][:2] == ("Hello there. General Kenobi.", "one-one.mp3")
        created = [e for e in sent if e["type"] == "response.created"][0]
        assert created["response"]["status"] == "in_progress"
        started = [e for e in sent if e["type"] == "output_audio_buffer.started"][0]
        assert started["response_id"] == done["id"]
    run(main())


def test_response_create_with_inline_input_speaks_that_text_only():
    async def main():
        synth = FakeSynth()
        session, sent, _ = make_session(synth)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "ignored."}]}}))
        await session.handle(json.dumps({"type": "response.create", "response": {"input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Spoken."}]}]}}))
        await until(sent, "response.done")
        assert synth.calls[0][0] == "Spoken."
    run(main())


def test_response_x_chatterbox_and_voice_override_for_one_response():
    async def main():
        synth = FakeSynth()
        session, sent, _ = make_session(synth)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi."}]}}))
        await session.handle(json.dumps({"type": "response.create", "response": {
            "audio": {"output": {"voice": "marvin.wav"}}, "x_chatterbox": {"num_steps": 2}}}))
        await until(sent, "response.done")
        text, voice, knobs = synth.calls[0]
        assert voice == "marvin.wav" and knobs.num_steps == 2
        assert session.session_object()["audio"]["output"]["voice"] == "one-one.mp3"
    run(main())


def test_nothing_to_speak_is_an_error_and_no_response():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "response.create", "event_id": "evt_e"}))
        assert sent[-1]["type"] == "error"
        assert sent[-1]["error"]["event_id"] == "evt_e"
        assert "response.created" not in types(sent)
    run(main())


def test_second_response_while_active_is_an_error():
    async def main():
        gate = threading.Event()
        session, sent, _ = make_session(FakeSynth(gate))
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "One. Two."}]}}))
        await session.handle(json.dumps({"type": "response.create"}))
        await until(sent, "output_audio_buffer.started")
        await session.handle(json.dumps({"type": "response.create"}))
        assert sent[-1]["type"] == "error"
        assert sent[-1]["error"]["code"] == "conversation_already_has_active_response"
        gate.set()
        await until(sent, "response.done")
    run(main())


def test_cancel_closes_the_response_as_cancelled_and_discards_later_chunks():
    async def main():
        gate = threading.Event()
        sink = FakeSink()
        session, sent, _ = make_session(FakeSynth(gate), sink)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "One. Two."}]}}))
        await session.handle(json.dumps({"type": "response.create"}))
        await until(sent, "output_audio_buffer.started")
        await session.handle(json.dumps({"type": "response.cancel"}))
        await until(sent, "response.done")
        done = [e for e in sent if e["type"] == "response.done"][0]["response"]
        assert done["status"] == "cancelled"
        gate.set()
        await asyncio.sleep(0.1)
        assert len(sink.pushed) == 1, "the chunk finished after cancel must be discarded"
        assert types(sent).count("response.output_audio_transcript.delta") == 1
    run(main())


def test_output_audio_buffer_clear_clears_the_sink_and_reports():
    async def main():
        sink = FakeSink()
        session, sent, _ = make_session(sink=sink)
        await session.open()
        await session.handle(json.dumps({"type": "output_audio_buffer.clear"}))
        assert sink.cleared == 1 and sent[-1]["type"] == "output_audio_buffer.cleared"
    run(main())


def test_synthesizer_failure_marks_the_response_failed():
    async def main():
        def boom(text, voice, knobs, cancel):
            raise RuntimeError("ran out of VRAM during generation. VRAM 0.1 GB free")
            yield  # pragma: no cover
        session, sent, _ = make_session(boom)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi."}]}}))
        await session.handle(json.dumps({"type": "response.create"}))
        await until(sent, "response.done")
        done = [e for e in sent if e["type"] == "response.done"][0]["response"]
        assert done["status"] == "failed"
        assert "VRAM" in done["status_details"]["error"]["message"]
    run(main())


def test_unsupported_and_unknown_events_produce_error_events():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle('{"type": "input_audio_buffer.append", "audio": "AAAA"}')
        assert sent[-1]["error"]["code"] == "unsupported_event"
        await session.handle('{bad')
        assert sent[-1]["error"]["code"] == "invalid_json"
    run(main())


def test_knobs_merge_validates_ranges():
    with pytest.raises(EventError) as exc:
        KNOBS.merged({"temperature": 5}, param_prefix="session.x_chatterbox")
    assert exc.value.param == "session.x_chatterbox.temperature"
    assert KNOBS.merged({"chunk_size": 200}, param_prefix="x").chunk_size == 200
    assert KNOBS.as_engine_kwargs()["cfg_scale"] == 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_realtime_session.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`poc_tts_streaming/realtime/session.py`:

```python
"""OpenAI Realtime protocol state machine for a text-to-speech server.

Pure Python: no aiortc, no torch. Audio leaves through an AudioSink; events
leave through a `send` callable. This is the module a Rust port of the
protocol would mirror.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np

from poc_tts_streaming.realtime.events import (
    E, ConversationItemCreate, ConversationItemDelete, EventError, OutputAudioBufferClear,
    ResponseCancel, ResponseCreate, SessionUpdate, error_event, parse_client_event, server_event,
)
from poc_tts_streaming.realtime.ids import new_id

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


# ---- knobs -------------------------------------------------------------------

_RANGES: dict[str, tuple[type, float, float]] = {
    "temperature": (float, 0.0, 2.0),
    "exaggeration": (float, 0.0, 2.0),
    "cfg_scale": (float, 0.0, 5.0),
    "num_steps": (int, 1, 32),
    "n_cfm_timesteps": (int, 1, 8),
    "chunk_size": (int, 50, 500),
}
_BOOLS = ("split_text", "split_on_clauses")


@dataclass(frozen=True)
class ChatterboxKnobs:
    temperature: float
    exaggeration: float
    cfg_scale: float
    num_steps: int
    n_cfm_timesteps: int
    chunk_size: int
    split_text: bool
    split_on_clauses: bool

    @classmethod
    def from_config(cls, generation_cfg: dict) -> "ChatterboxKnobs":
        g = generation_cfg
        return cls(
            temperature=float(g.get("temperature", 0.6)),
            exaggeration=float(g.get("exaggeration", 0.5)),
            cfg_scale=float(g.get("cfg_scale", 1.0)),
            num_steps=int(g.get("num_steps", 10)),
            n_cfm_timesteps=int(g.get("n_cfm_timesteps", 2)),
            chunk_size=int(g.get("chunk_size", 120)),
            split_text=bool(g.get("split_text", True)),
            split_on_clauses=bool(g.get("split_on_clauses", True)),
        )

    def merged(self, patch: dict, *, param_prefix: str) -> "ChatterboxKnobs":
        if not isinstance(patch, dict):
            raise EventError("invalid_value", "x_chatterbox must be an object", param=param_prefix)
        values = dataclasses.asdict(self)
        for key, raw in patch.items():
            param = f"{param_prefix}.{key}"
            if key in _BOOLS:
                if not isinstance(raw, bool):
                    raise EventError("invalid_value", f"{key} must be a boolean", param=param)
                values[key] = raw
            elif key in _RANGES:
                typ, lo, hi = _RANGES[key]
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise EventError("invalid_value", f"{key} must be a number", param=param)
                if not lo <= raw <= hi:
                    raise EventError("invalid_value", f"{key} must be between {lo} and {hi}", param=param)
                values[key] = typ(raw)
            else:
                raise EventError("unknown_parameter", f"unknown x_chatterbox parameter {key!r}", param=param)
        return ChatterboxKnobs(**values)

    def as_engine_kwargs(self) -> dict:
        return dataclasses.asdict(self)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---- collaborators ---------------------------------------------------------

Synthesizer = Callable[[str, str, ChatterboxKnobs, threading.Event], Iterator[tuple[str, np.ndarray]]]


class AudioSink(Protocol):
    def push(self, pcm: np.ndarray) -> None: ...
    def flush(self) -> None: ...
    def clear(self) -> None: ...
    async def drained(self) -> None: ...


class SynthWorker:
    """One synthesis at a time per engine: a single-thread executor."""

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="synth")

    def submit(self, fn: Callable[[], None]) -> concurrent.futures.Future:
        return self._pool.submit(fn)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# ---- session -----------------------------------------------------------------

_DONE = object()


@dataclass
class _Response:
    id: str
    item_id: str
    cancel: threading.Event
    closed: bool = False
    status: str = "in_progress"
    transcript: str = ""
    started: bool = False
    error: dict | None = None
    metadata: dict | None = None


class RealtimeSession:
    def __init__(
        self, *, send: Callable[[dict], None], synthesizer: Synthesizer, sink: AudioSink,
        worker: SynthWorker, voices: Callable[[], list[str]], voice: str, knobs: ChatterboxKnobs,
        model: str = "chatterbox-flash", session_patch: dict | None = None,
    ) -> None:
        self.id = new_id("sess")
        self.conversation_id = new_id("conv")
        self._send, self._synthesizer, self._sink, self._worker = send, synthesizer, sink, worker
        self._voices, self._model = voices, model
        self._voice, self._knobs = voice, knobs
        self._instructions = ""
        self._items: list[dict] = []
        self._unspoken: list[str] = []
        self._active: _Response | None = None
        self._playout_token = 0
        if session_patch:
            self.apply_session_patch(session_patch)

    # ---- session object ---------------------------------------------------

    def session_object(self) -> dict:
        fmt = {"type": "audio/pcm", "rate": SAMPLE_RATE}
        return {
            "type": "realtime", "id": self.id, "object": "realtime.session", "model": self._model,
            "output_modalities": ["audio"], "instructions": self._instructions,
            "audio": {"input": {"format": fmt, "turn_detection": None},
                      "output": {"format": fmt, "voice": self._voice, "speed": 1.0}},
            "x_chatterbox": self._knobs.as_dict(),
        }

    def _check_voice(self, voice, param: str) -> str:
        if not isinstance(voice, str) or voice not in self._voices():
            raise EventError("invalid_value", f"unknown voice {voice!r}; use a reference clip filename",
                             param=param)
        return voice

    def apply_session_patch(self, patch: dict) -> None:
        """Validate the whole patch first, then apply -- an error leaves the
        session exactly as it was."""
        if not isinstance(patch, dict):
            raise EventError("invalid_value", "session must be an object", param="session")
        voice, knobs, instructions = self._voice, self._knobs, self._instructions
        out = patch.get("audio", {}).get("output", {}) if isinstance(patch.get("audio"), dict) else {}
        if "voice" in out:
            voice = self._check_voice(out["voice"], "session.audio.output.voice")
        if "x_chatterbox" in patch:
            knobs = self._knobs.merged(patch["x_chatterbox"], param_prefix="session.x_chatterbox")
        if "instructions" in patch:
            if not isinstance(patch["instructions"], str):
                raise EventError("invalid_value", "instructions must be a string", param="session.instructions")
            instructions = patch["instructions"]
        if "output_modalities" in patch and patch["output_modalities"] != ["audio"]:
            raise EventError("invalid_value", "this server only produces audio",
                             param="session.output_modalities")
        self._voice, self._knobs, self._instructions = voice, knobs, instructions

    # ---- lifecycle ----------------------------------------------------------

    async def open(self) -> None:
        self._send(server_event(E.SESSION_CREATED, session=self.session_object()))
        self._send(server_event(E.CONVERSATION_CREATED,
                                conversation={"id": self.conversation_id, "object": "realtime.conversation"}))

    async def close(self) -> None:
        if self._active is not None and not self._active.closed:
            self._active.cancel.set()
            self._active.closed = True
            self._active = None

    async def handle(self, raw: str) -> None:
        try:
            event = parse_client_event(raw)
        except EventError as err:
            self._send(error_event(err))
            return
        try:
            if isinstance(event, SessionUpdate):
                self.apply_session_patch(event.session)
                self._send(server_event(E.SESSION_UPDATED, session=self.session_object()))
            elif isinstance(event, ConversationItemCreate):
                self._on_item_create(event)
            elif isinstance(event, ConversationItemDelete):
                self._on_item_delete(event)
            elif isinstance(event, ResponseCreate):
                await self._on_response_create(event)
            elif isinstance(event, ResponseCancel):
                await self._on_response_cancel()
            elif isinstance(event, OutputAudioBufferClear):
                self._playout_token += 1
                self._sink.clear()
                self._send(server_event(E.OUTPUT_AUDIO_BUFFER_CLEARED,
                                        response_id=self._active.id if self._active else None))
        except EventError as err:
            err.event_id = err.event_id or event.event_id
            self._send(error_event(err))

    # ---- conversation items -------------------------------------------------

    @staticmethod
    def _user_text(item: dict, param: str) -> str:
        if item.get("type") != "message":
            raise EventError("invalid_value", "only message items are supported", param=f"{param}.type")
        if item.get("role") != "user":
            raise EventError("invalid_value", "only user messages can be spoken", param=f"{param}.role")
        content = item.get("content")
        if not isinstance(content, list) or not content:
            raise EventError("missing_required_parameter", "content is required", param=f"{param}.content")
        texts = []
        for i, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "input_text" or not isinstance(part.get("text"), str):
                raise EventError("invalid_value", "content parts must be input_text",
                                 param=f"{param}.content[{i}].type")
            texts.append(part["text"])
        return " ".join(texts)

    def _on_item_create(self, event: ConversationItemCreate) -> None:
        text = self._user_text(event.item, "item")
        item = {"id": event.item.get("id") or new_id("item"), "object": "realtime.item", "type": "message",
                "status": "completed", "role": "user", "content": [{"type": "input_text", "text": text}]}
        previous = self._items[-1]["id"] if self._items else None
        self._items.append(item)
        self._unspoken.append(item["id"])
        self._send(server_event(E.ITEM_ADDED, item=item, previous_item_id=previous))
        self._send(server_event(E.ITEM_DONE, item=item, previous_item_id=previous))

    def _on_item_delete(self, event: ConversationItemDelete) -> None:
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != event.item_id]
        if len(self._items) == before:
            raise EventError("item_not_found", f"no item {event.item_id!r}", param="item_id")
        self._unspoken = [i for i in self._unspoken if i != event.item_id]
        self._send(server_event(E.ITEM_DELETED, item_id=event.item_id))

    # ---- responses -----------------------------------------------------------

    def _response_object(self, resp: _Response, output: list[dict]) -> dict:
        return {"id": resp.id, "object": "realtime.response", "status": resp.status,
                "status_details": ({"type": resp.status, "error": resp.error} if resp.error else None),
                "output": output, "conversation_id": self.conversation_id,
                "output_modalities": ["audio"], "metadata": resp.metadata,
                "usage": ({"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
                          if resp.status != "in_progress" else None)}

    def _assistant_item(self, resp: _Response, done: bool) -> dict:
        return {"id": resp.item_id, "object": "realtime.item", "type": "message",
                "status": "completed" if done else "in_progress", "role": "assistant",
                "content": [{"type": "audio", "transcript": resp.transcript}] if done else []}

    async def _on_response_create(self, event: ResponseCreate) -> None:
        if self._active is not None and not self._active.closed:
            raise EventError("conversation_already_has_active_response",
                             "a response is already in progress; cancel it first")
        spec = event.response if isinstance(event.response, dict) else {}
        if "input" in spec:
            if not isinstance(spec["input"], list):
                raise EventError("invalid_value", "response.input must be a list", param="response.input")
            text = " ".join(self._user_text(i, f"response.input[{n}]") for n, i in enumerate(spec["input"]))
            spoken_ids: list[str] = []
        else:
            by_id = {i["id"]: i for i in self._items}
            text = " ".join(by_id[i]["content"][0]["text"] for i in self._unspoken if i in by_id)
            spoken_ids = list(self._unspoken)
        if not text.strip():
            raise EventError("invalid_request_error", "nothing to speak: add a user message item first",
                             param="response.input")
        voice = self._voice
        out = spec.get("audio", {}).get("output", {}) if isinstance(spec.get("audio"), dict) else {}
        if "voice" in out:
            voice = self._check_voice(out["voice"], "response.audio.output.voice")
        knobs = self._knobs
        if "x_chatterbox" in spec:
            knobs = knobs.merged(spec["x_chatterbox"], param_prefix="response.x_chatterbox")

        resp = _Response(id=new_id("resp"), item_id=new_id("item"), cancel=threading.Event(),
                         metadata=spec.get("metadata"))
        self._active = resp
        self._unspoken = [i for i in self._unspoken if i not in spoken_ids]
        self._send(server_event(E.RESPONSE_CREATED, response=self._response_object(resp, [])))
        self._send(server_event(E.OUTPUT_ITEM_ADDED, response_id=resp.id, output_index=0,
                                item=self._assistant_item(resp, done=False)))
        self._send(server_event(E.CONTENT_PART_ADDED, response_id=resp.id, item_id=resp.item_id,
                                output_index=0, content_index=0, part={"type": "audio", "transcript": ""}))
        asyncio.ensure_future(self._run_response(resp, text, voice, knobs))

    async def _run_response(self, resp: _Response, text: str, voice: str, knobs: ChatterboxKnobs) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                for chunk in self._synthesizer(text, voice, knobs, resp.cancel):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:  # noqa: BLE001 -- reported as response.failed
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        self._worker.submit(produce)
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if resp.closed:
                continue  # cancelled: discard whatever the worker still produces
            if isinstance(item, Exception):
                logger.error("response %s failed: %s", resp.id, item)
                resp.error = {"type": "server_error", "code": "synthesis_failed", "message": str(item)}
                await self._finish(resp, "failed")
                continue
            chunk_text, pcm = item
            delta = chunk_text + " "
            resp.transcript += delta
            self._send(server_event(E.AUDIO_TRANSCRIPT_DELTA, response_id=resp.id, item_id=resp.item_id,
                                    output_index=0, content_index=0, delta=delta))
            self._sink.push(pcm)
            if not resp.started:
                resp.started = True
                self._send(server_event(E.OUTPUT_AUDIO_BUFFER_STARTED, response_id=resp.id))
        if not resp.closed:
            await self._finish(resp, "completed")

    async def _finish(self, resp: _Response, status: str) -> None:
        if resp.closed:
            return
        resp.closed, resp.status = True, status
        self._sink.flush()
        common = dict(response_id=resp.id, item_id=resp.item_id, output_index=0, content_index=0)
        self._send(server_event(E.AUDIO_TRANSCRIPT_DONE, transcript=resp.transcript, **common))
        self._send(server_event(E.AUDIO_DONE, **common))
        self._send(server_event(E.CONTENT_PART_DONE, part={"type": "audio", "transcript": resp.transcript},
                                **common))
        item = self._assistant_item(resp, done=True)
        self._items.append(item)
        self._send(server_event(E.OUTPUT_ITEM_DONE, response_id=resp.id, output_index=0, item=item))
        self._send(server_event(E.RESPONSE_DONE, response=self._response_object(resp, [item])))
        if self._active is resp:
            self._active = None
        if resp.started:
            self._playout_token += 1
            asyncio.ensure_future(self._after_playout(resp.id, self._playout_token))

    async def _after_playout(self, response_id: str, token: int) -> None:
        await self._sink.drained()
        if token == self._playout_token:
            self._send(server_event(E.OUTPUT_AUDIO_BUFFER_STOPPED, response_id=response_id))

    async def _on_response_cancel(self) -> None:
        resp = self._active
        if resp is None or resp.closed:
            raise EventError("response_cancel_not_active", "no active response to cancel")
        resp.cancel.set()
        await self._finish(resp, "cancelled")
```

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_realtime_session.py -v`
Expected: PASS. If `test_full_response_sequence…` shows `output_audio_buffer.started` before the first transcript delta, keep the order as implemented (delta first, then `started`) — the test encodes the spec's order.

- [ ] **Step 5: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/realtime/session.py poc-tts-streaming/tests/test_realtime_session.py
git commit -m "feat(poc-tts-streaming): Realtime session state machine for text-to-speech responses"
```

---

### Task 8: `server.py` — drop `/tts`, add client secrets and the OpenAI error shape

**Files:**
- Modify: `poc-tts-streaming/poc_tts_streaming/server.py`
- Delete: `poc-tts-streaming/poc_tts_streaming/models.py` and `tests/test_server_tts.py` (the WAV `/tts` endpoint is gone; whole-utterance output returns as `/v1/audio/speech` `wav` in Task 11)
- Test: `poc-tts-streaming/tests/test_server_realtime_http.py`

**Interfaces:**
- Consumes: `ChatterboxKnobs`, `SynthWorker` (Task 7), `discover_voices` (copied).
- Produces:
  ```python
  class ClientSecretStore:
      def __init__(self, ttl_s: int = 600, clock: Callable[[], int] = ids.now)
      def issue(self, session_patch: dict | None) -> dict   # {"value","expires_at","session"}
      def verify(self, token: str | None) -> bool

  def openai_error(status: int, message: str, *, type_="invalid_request_error", code=None, param=None) -> JSONResponse
  def bearer_token(request: Request) -> str | None
  def create_app(engine, config: dict, voice_paths: list[Path], *, worker: SynthWorker | None = None) -> FastAPI
  def engine_synthesizer(engine) -> Synthesizer
  ```
  `app.state.secrets` (the store), `app.state.worker`, `app.state.knobs`, `app.state.realtime` (the `realtime:` config block).

- [ ] **Step 1: Write the failing tests**

`tests/test_server_realtime_http.py`:

```python
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poc_tts_streaming.server import ClientSecretStore, create_app


@pytest.fixture
def engine():
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    return eng


@pytest.fixture
def client(engine, tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    config = {"server": {"port": 8006},
              "generation": {"num_steps": 4, "n_cfm_timesteps": 1},
              "realtime": {"model": "chatterbox-flash", "default_voice": "one-one.mp3",
                           "client_secret_ttl_s": 600}}
    return TestClient(create_app(engine, config, voice_paths=[voices]))


def test_client_secret_shape(client):
    r = client.post("/v1/realtime/client_secrets", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["value"].startswith("ek_")
    assert isinstance(body["expires_at"], int)
    assert body["session"]["type"] == "realtime"
    assert body["session"]["audio"]["output"]["voice"] == "one-one.mp3"
    assert body["session"]["x_chatterbox"]["num_steps"] == 4


def test_client_secret_applies_the_session_patch(client):
    r = client.post("/v1/realtime/client_secrets",
                    json={"session": {"x_chatterbox": {"num_steps": 2}}})
    assert r.json()["session"]["x_chatterbox"]["num_steps"] == 2


def test_client_secret_rejects_a_bad_patch_with_the_openai_error_shape(client):
    r = client.post("/v1/realtime/client_secrets",
                    json={"session": {"audio": {"output": {"voice": "ghost.wav"}}}})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_value"
    assert err["param"] == "session.audio.output.voice"


def test_store_expires_tokens():
    t = [1000]
    store = ClientSecretStore(ttl_s=10, clock=lambda: t[0])
    tok = store.issue(None)["value"]
    assert store.verify(tok)
    t[0] = 1011
    assert not store.verify(tok)
    assert not store.verify("ek_nope") and not store.verify(None)


def test_tts_endpoint_is_gone(client):
    assert client.post("/tts", json={"text": "x"}).status_code == 404


def test_initial_data_still_serves_the_ui_shape(client):
    body = client.get("/api/ui/initial-data").json()
    for key in ("config", "reference_files", "predefined_voices", "presets", "initial_gen_result", "model_info"):
        assert key in body
    assert body["config"]["realtime"]["model"] == "chatterbox-flash"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_realtime_http.py -v`
Expected: FAIL (`ImportError: ClientSecretStore`).

- [ ] **Step 3: Implement**

Edit `poc_tts_streaming/server.py`:

1. Remove `_wav_bytes`, the `FlashTTSRequest` import, and the whole `@app.post("/tts")` handler. Delete `poc_tts_streaming/models.py` and `tests/test_server_tts.py` (`git rm`).
2. Add imports and the two helpers near the top:

```python
import secrets
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from poc_tts_streaming.realtime import ids
from poc_tts_streaming.realtime.events import EventError
from poc_tts_streaming.realtime.session import ChatterboxKnobs, RealtimeSession, SynthWorker


def openai_error(status: int, message: str, *, type_: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None) -> JSONResponse:
    """The error body shape api.openai.com returns, so client code paths match."""
    return JSONResponse({"error": {"type": type_, "code": code, "message": message, "param": param}},
                        status_code=status)


def bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


class ClientSecretStore:
    """In-memory ephemeral keys. Cosmetic on localhost, but it keeps the
    browser's code path identical to the one it would use against OpenAI."""

    def __init__(self, ttl_s: int = 600, clock: Callable[[], int] = ids.now) -> None:
        self._ttl, self._clock = ttl_s, clock
        self._tokens: dict[str, int] = {}

    def issue(self, session_patch: dict | None, *, session_factory=None) -> dict:
        value = f"ek_{secrets.token_urlsafe(24)}"
        expires_at = self._clock() + self._ttl
        self._tokens[value] = expires_at
        session = session_factory(session_patch).session_object() if session_factory else {}
        return {"value": value, "expires_at": expires_at, "session": session}

    def verify(self, token: str | None) -> bool:
        if not token or token not in self._tokens:
            return False
        if self._tokens[token] < self._clock():
            del self._tokens[token]
            return False
        return True


def engine_synthesizer(engine):
    """Adapt FlashEngine.synthesize_stream to the session's Synthesizer type."""
    def synthesize(text, voice, knobs: ChatterboxKnobs, cancel):
        return engine.synthesize_stream(text, voice, cancel=cancel, **knobs.as_engine_kwargs())
    return synthesize
```

3. Change the `create_app` signature and body head to:

```python
def create_app(engine, config: dict, voice_paths: list[Path], *, worker: SynthWorker | None = None) -> FastAPI:
    app = FastAPI(title="poc-tts-streaming: Chatterbox Flash over Realtime/WebRTC", version="0.1.0")
    realtime_cfg = {"model": "chatterbox-flash", "default_voice": "one-one.mp3",
                    "client_secret_ttl_s": 600, **config.get("realtime", {})}
    app.state.knobs = ChatterboxKnobs.from_config(config.get("generation", {}))
    app.state.worker = worker or SynthWorker()
    app.state.secrets = ClientSecretStore(ttl_s=int(realtime_cfg["client_secret_ttl_s"]))
    app.state.realtime = realtime_cfg

    def build_session(send, sink, session_patch: dict | None = None) -> RealtimeSession:
        return RealtimeSession(
            send=send, synthesizer=engine_synthesizer(engine), sink=sink, worker=app.state.worker,
            voices=lambda: discover_voices(voice_paths), voice=realtime_cfg["default_voice"],
            knobs=app.state.knobs, model=realtime_cfg["model"], session_patch=session_patch,
        )
    app.state.build_session = build_session
```

and keep everything else in `create_app` (UI routes, initial-data, save/reset settings, restart) as it was. Add the route:

```python
    @app.post("/v1/realtime/client_secrets")
    async def client_secrets(request: Request):
        body = await request.json() if int(request.headers.get("content-length", "0") or 0) else {}
        patch = body.get("session") if isinstance(body, dict) else None
        try:
            return app.state.secrets.issue(
                patch, session_factory=lambda p: build_session(lambda _e: None, _NullSink(), p))
        except EventError as err:
            return openai_error(400, err.message, code=err.code, param=err.param)
```

with a tiny null sink beside the helpers:

```python
class _NullSink:
    def push(self, pcm): ...
    def flush(self): ...
    def clear(self): ...
    async def drained(self): ...
```

4. `_ui_shaped_config` already spreads the raw config through, so `config.realtime` reaches the UI — no change needed there.

- [ ] **Step 4: Run the whole suite**

Run: `make test`
Expected: PASS. `tests/test_server_info.py` still passes (it only touches info/UI routes).

- [ ] **Step 5: Commit**

```bash
git add -A poc-tts-streaming
git commit -m "feat(poc-tts-streaming): client secrets, OpenAI error shape, drop the WAV /tts route"
```

---

### Task 9: `realtime/webrtc.py` and `POST /v1/realtime/calls`

**Files:**
- Create: `poc-tts-streaming/poc_tts_streaming/realtime/webrtc.py`
- Modify: `poc-tts-streaming/poc_tts_streaming/server.py` (routes + shutdown)
- Test: `poc-tts-streaming/tests/test_server_calls.py`

**Interfaces:**
- Consumes: `PcmQueueTrack` (Task 5), `RealtimeSession` (Task 7), `app.state.build_session`, `app.state.secrets`.
- Produces:
  ```python
  class CallRegistry:
      async def create(self, offer_sdp: str, build_session, *, session_patch=None) -> tuple[str, str]  # (call_id, answer_sdp)
      async def hangup(self, call_id: str) -> bool
      async def close_all(self) -> None
      def __len__(self) -> int
  ```
  Routes: `POST /v1/realtime/calls` → `201 application/sdp` + `Location`; `DELETE /v1/realtime/calls/{call_id}` → `200` / `404`. `app.state.calls` is the registry.

- [ ] **Step 1: Write the failing tests (HTTP contract only — the media path is Task 10)**

`tests/test_server_calls.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from poc_tts_streaming.server import create_app


@pytest.fixture
def app(tmp_path):
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    return create_app(eng, {"generation": {}, "realtime": {"default_voice": "one-one.mp3"}},
                      voice_paths=[voices])


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def fake_calls(app):
    app.state.calls = MagicMock()
    app.state.calls.create = AsyncMock(return_value=("call_abc", "v=0\r\nanswer"))
    app.state.calls.hangup = AsyncMock(return_value=True)
    return app.state.calls


def _token(client):
    return client.post("/v1/realtime/client_secrets", json={}).json()["value"]


def test_calls_requires_a_valid_bearer(client, fake_calls):
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp"})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_request_error"
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp", "authorization": "Bearer ek_bogus"})
    assert r.status_code == 401


def test_calls_accepts_application_sdp(client, fake_calls):
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp", "authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 201
    assert r.headers["content-type"].startswith("application/sdp")
    assert r.headers["location"] == "/v1/realtime/calls/call_abc"
    assert r.text == "v=0\r\nanswer"
    args, kwargs = fake_calls.create.call_args
    assert args[0] == "v=0\r\noffer" and kwargs["session_patch"] is None


def test_calls_accepts_multipart_with_session(client, fake_calls):
    r = client.post("/v1/realtime/calls",
                    files={"sdp": (None, "v=0\r\noffer"),
                           "session": (None, '{"x_chatterbox": {"num_steps": 2}}')},
                    headers={"authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 201
    _, kwargs = fake_calls.create.call_args
    assert kwargs["session_patch"] == {"x_chatterbox": {"num_steps": 2}}


def test_calls_rejects_other_content_types(client, fake_calls):
    r = client.post("/v1/realtime/calls", json={"sdp": "x"},
                    headers={"authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 415


def test_calls_503_when_model_not_loaded(app, client, fake_calls):
    app.state.engine.loaded = False
    r = client.post("/v1/realtime/calls", content="v=0\r\noffer",
                    headers={"content-type": "application/sdp", "authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 503


def test_hangup(client, fake_calls):
    assert client.delete("/v1/realtime/calls/call_abc").status_code == 200
    fake_calls.hangup.return_value = False
    assert client.delete("/v1/realtime/calls/call_zzz").status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_server_calls.py -v`
Expected: FAIL with 404s / `AttributeError: engine`.

- [ ] **Step 3: Implement webrtc.py**

`poc_tts_streaming/realtime/webrtc.py`:

```python
"""aiortc glue: one RTCPeerConnection + RealtimeSession + PcmQueueTrack per call.

Knows about aiortc and the session; knows nothing about FastAPI or torch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

from poc_tts_streaming.realtime.ids import new_id
from poc_tts_streaming.realtime.session import RealtimeSession
from poc_tts_streaming.track import PcmQueueTrack

logger = logging.getLogger(__name__)

EVENTS_CHANNEL = "oai-events"


@dataclass
class Call:
    id: str
    pc: RTCPeerConnection
    track: PcmQueueTrack
    session: RealtimeSession | None = None
    tasks: list[asyncio.Task] = field(default_factory=list)


class CallRegistry:
    def __init__(self) -> None:
        self._calls: dict[str, Call] = {}

    def __len__(self) -> int:
        return len(self._calls)

    async def create(self, offer_sdp: str, build_session, *, session_patch: dict | None = None) -> tuple[str, str]:
        call = Call(id=new_id("call"), pc=RTCPeerConnection(), track=PcmQueueTrack())
        self._calls[call.id] = call
        pc = call.pc
        pc.addTrack(call.track)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            if channel.label != EVENTS_CHANNEL:
                logger.info("[%s] ignoring data channel %r", call.id, channel.label)
                return

            def send(event: dict) -> None:
                if channel.readyState == "open":
                    channel.send(json.dumps(event))

            call.session = build_session(send, call.track, session_patch)

            @channel.on("message")
            def on_message(message) -> None:
                if isinstance(message, str):
                    call.tasks.append(asyncio.ensure_future(call.session.handle(message)))

            call.tasks.append(asyncio.ensure_future(call.session.open()))

        @pc.on("track")
        def on_track(track) -> None:
            # A client that offers a mic gets it accepted and drained; this
            # server never listens. Draining keeps aiortc's receiver quiet.
            async def drain() -> None:
                try:
                    while True:
                        await track.recv()
                except MediaStreamError:
                    return
            call.tasks.append(asyncio.ensure_future(drain()))

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            logger.info("[%s] connection state -> %s", call.id, pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.hangup(call.id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        await pc.setLocalDescription(await pc.createAnswer())
        return call.id, pc.localDescription.sdp

    async def hangup(self, call_id: str) -> bool:
        call = self._calls.pop(call_id, None)
        if call is None:
            return False
        if call.session is not None:
            await call.session.close()
        call.track.stop()
        for task in call.tasks:
            task.cancel()
        await call.pc.close()
        return True

    async def close_all(self) -> None:
        await asyncio.gather(*(self.hangup(cid) for cid in list(self._calls)), return_exceptions=True)
```

- [ ] **Step 4: Wire the routes**

In `server.py` add `from poc_tts_streaming.realtime.webrtc import CallRegistry` and, inside `create_app` after `app.state.build_session = build_session`:

```python
    app.state.engine = engine
    app.state.calls = CallRegistry()

    @app.post("/v1/realtime/calls")
    async def realtime_calls(request: Request):
        if not app.state.secrets.verify(bearer_token(request)):
            return openai_error(401, "missing or expired ephemeral key; POST /v1/realtime/client_secrets first",
                                code="invalid_api_key")
        ctype = request.headers.get("content-type", "")
        session_patch = None
        if ctype.startswith("application/sdp"):
            offer = (await request.body()).decode()
        elif ctype.startswith("multipart/form-data"):
            form = await request.form()
            offer = form.get("sdp")
            if isinstance(form.get("session"), str):
                try:
                    session_patch = json.loads(form["session"])
                except json.JSONDecodeError:
                    return openai_error(400, "session must be JSON", param="session")
            if not isinstance(offer, str):
                return openai_error(400, "multipart body needs an sdp field", param="sdp")
        else:
            return openai_error(415, "send the SDP offer as application/sdp", code="unsupported_media_type")
        if not app.state.engine.loaded:
            return openai_error(503, "Flash model is not loaded", type_="server_error", code="model_not_loaded")
        try:
            call_id, answer = await app.state.calls.create(offer, build_session, session_patch=session_patch)
        except EventError as err:
            return openai_error(400, err.message, code=err.code, param=err.param)
        return Response(content=answer, status_code=201, media_type="application/sdp",
                        headers={"Location": f"/v1/realtime/calls/{call_id}"})

    @app.delete("/v1/realtime/calls/{call_id}")
    async def realtime_hangup(call_id: str):
        if not await app.state.calls.hangup(call_id):
            return openai_error(404, f"no call {call_id!r}", code="not_found", param="call_id")
        return {"ok": True}

    @app.on_event("shutdown")
    async def on_shutdown():
        await app.state.calls.close_all()
        app.state.worker.shutdown()
```

Add `import json` to `server.py` if missing.

- [ ] **Step 5: Run**

Run: `make test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/realtime/webrtc.py poc-tts-streaming/poc_tts_streaming/server.py poc-tts-streaming/tests/test_server_calls.py
git commit -m "feat(poc-tts-streaming): /v1/realtime/calls SDP exchange with per-call sessions"
```

---

### Task 10: Loopback end-to-end test (aiortc client in-process, no GPU)

**Files:**
- Test: `poc-tts-streaming/tests/test_realtime_loopback.py`

**Interfaces:**
- Consumes: the full stack from Tasks 5-9 through `create_app`, with a fake synthesizer injected by monkeypatching `engine.synthesize_stream`.

- [ ] **Step 1: Write the test**

`tests/test_realtime_loopback.py`:

```python
"""Real WebRTC over loopback: an aiortc client peer in the test process
drives the app through httpx's ASGI transport and asserts the Realtime
event sequence plus audio on the remote track. No GPU: the engine's
synthesize_stream is replaced with a generator of tone chunks."""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription

from poc_tts_streaming.server import create_app


def fake_stream(text, voice, *, cancel=None, **knobs):
    for sentence in (s.strip() for s in text.split(".") if s.strip()):
        t = np.arange(24000 // 2) / 24000.0
        yield sentence + ".", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def make_app(tmp_path):
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    eng.synthesize_stream = fake_stream
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    return create_app(eng, {"generation": {}, "realtime": {"default_voice": "one-one.mp3"}},
                      voice_paths=[voices])


async def _scenario(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        token = (await http.post("/v1/realtime/client_secrets", json={})).json()["value"]
        pc = RTCPeerConnection()
        events: list[dict] = []
        got: asyncio.Queue = asyncio.Queue()
        frames: list = []

        channel = pc.createDataChannel("oai-events")
        channel.on("message", lambda m: (events.append(json.loads(m)), got.put_nowait(events[-1])))
        pc.addTransceiver("audio", direction="recvonly")

        @pc.on("track")
        def on_track(track):
            async def pull():
                while len(frames) < 60:
                    frames.append(await track.recv())
            asyncio.ensure_future(pull())

        await pc.setLocalDescription(await pc.createOffer())
        r = await http.post("/v1/realtime/calls", content=pc.localDescription.sdp,
                            headers={"content-type": "application/sdp", "authorization": f"Bearer {token}"})
        assert r.status_code == 201, r.text
        await pc.setRemoteDescription(RTCSessionDescription(sdp=r.text, type="answer"))

        async def wait_for(type_, timeout=10):
            while True:
                ev = await asyncio.wait_for(got.get(), timeout)
                if ev["type"] == type_:
                    return ev

        await wait_for("conversation.created")
        channel.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Hello there. General Kenobi."}]}}))
        await wait_for("conversation.item.done")
        channel.send(json.dumps({"type": "response.create"}))
        await wait_for("output_audio_buffer.stopped")

        types = [e["type"] for e in events]
        assert types[:2] == ["session.created", "conversation.created"]
        i = types.index("response.created")
        assert types[i:] == [
            "response.created", "response.output_item.added", "response.content_part.added",
            "response.output_audio_transcript.delta", "output_audio_buffer.started",
            "response.output_audio_transcript.delta", "response.output_audio_transcript.done",
            "response.output_audio.done", "response.content_part.done", "response.output_item.done",
            "response.done", "output_audio_buffer.stopped",
        ]
        # audio really crossed the (loopback) wire
        for _ in range(200):
            if len(frames) >= 60:
                break
            await asyncio.sleep(0.05)
        assert len(frames) >= 60
        loud = [f for f in frames if np.abs(f.to_ndarray()).max() > 1000]
        assert loud, "expected non-silent decoded audio on the remote track"

        location = r.headers["location"]
        assert (await http.delete(location)).status_code == 200
        await pc.close()


@pytest.mark.timeout(60)
def test_realtime_loopback_end_to_end(tmp_path):
    asyncio.run(_scenario(make_app(tmp_path)))
```

If `pytest.mark.timeout` is unknown (pytest-timeout is not installed), drop the decorator — the `wait_for` timeouts already bound the test.

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_realtime_loopback.py -v -s`
Expected: PASS in a few seconds. Known things to check if it does not:
  - `session.created` never arrives → the `datachannel` handler must call `session.open()` (Task 9 step 3). aiortc fires `datachannel` once the channel is open.
  - No frames → the `recvonly` transceiver on the client must exist **before** `createOffer`, and `pc.addTrack(call.track)` on the server must happen before `setRemoteDescription` so aiortc pairs it with the offered m-line.
  - Decoded frames all silent → check `PcmQueueTrack.recv` is producing non-silence (`queued_frames` > 0 after `push`) and that `to_ndarray()` is read as int16.

- [ ] **Step 3: Commit**

```bash
git add poc-tts-streaming/tests/test_realtime_loopback.py
git commit -m "test(poc-tts-streaming): loopback WebRTC end-to-end Realtime sequence"
```

---

### Task 11: `POST /v1/audio/speech` — chunked PCM (the sidecar seam) and WAV

**Files:**
- Modify: `poc-tts-streaming/poc_tts_streaming/server.py`
- Test: `poc-tts-streaming/tests/test_audio_speech.py`

**Interfaces:**
- Consumes: `engine_synthesizer`, `app.state.worker`, `app.state.knobs`, `to_int16`.
- Produces: `POST /v1/audio/speech` with JSON `{"input": str, "voice": str, "response_format": "pcm"|"wav", "x_chatterbox"?: {...}}` → `200 audio/pcm` chunked s16le 24 kHz (streamed per sentence) or `200 audio/wav` whole file. Errors use `openai_error`.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio_speech.py`:

```python
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from poc_tts_streaming.server import create_app


class GatedStream:
    """Second chunk waits on a gate so the test can prove the first chunk
    was flushed to the client before the second was generated."""
    def __init__(self):
        self.gate = threading.Event()
        self.second_started = threading.Event()
    def __call__(self, text, voice, *, cancel=None, **knobs):
        yield "One.", np.full(2400, 0.25, dtype=np.float32)
        self.second_started.set()
        self.gate.wait(5)
        yield "Two.", np.full(2400, 0.25, dtype=np.float32)


@pytest.fixture
def setup(tmp_path):
    eng = MagicMock()
    eng.loaded = True
    eng.model_info.return_value = {"loaded": True, "type": "flash"}
    eng.synthesize_stream = GatedStream()
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "one-one.mp3").write_bytes(b"x")
    app = create_app(eng, {"generation": {}, "realtime": {"default_voice": "one-one.mp3"}}, voice_paths=[voices])
    return app, eng, TestClient(app)


def test_pcm_is_streamed_chunk_by_chunk(setup):
    app, eng, client = setup
    stream = eng.synthesize_stream
    with client.stream("POST", "/v1/audio/speech",
                       json={"input": "One. Two.", "voice": "one-one.mp3", "response_format": "pcm"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/pcm")
        it = r.iter_bytes(4800)
        first = next(it)
        assert len(first) == 4800, "first sentence arrives on its own"
        assert stream.second_started.wait(5)
        stream.gate.set()
        rest = b"".join(it)
        assert len(rest) == 4800
    assert np.frombuffer(first, dtype=np.int16)[0] == 8191


def test_wav_returns_a_whole_file(setup):
    app, eng, client = setup
    eng.synthesize_stream.gate.set()
    r = client.post("/v1/audio/speech", json={"input": "One. Two.", "voice": "one-one.mp3",
                                              "response_format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF" and len(r.content) == 44 + 2 * 4800


def test_unknown_voice_and_missing_input_use_the_openai_error_shape(setup):
    _, _, client = setup
    r = client.post("/v1/audio/speech", json={"input": "x", "voice": "ghost.wav", "response_format": "pcm"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "voice"
    r = client.post("/v1/audio/speech", json={"voice": "one-one.mp3"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "input"


def test_x_chatterbox_overrides_reach_the_engine(setup):
    app, eng, client = setup
    calls = []
    def spy(text, voice, *, cancel=None, **knobs):
        calls.append(knobs)
        yield "x.", np.zeros(480, dtype=np.float32)
    eng.synthesize_stream = spy
    client.post("/v1/audio/speech", json={"input": "x.", "voice": "one-one.mp3",
                                          "response_format": "pcm", "x_chatterbox": {"num_steps": 2}})
    assert calls[0]["num_steps"] == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_audio_speech.py -v`
Expected: FAIL with 404.

- [ ] **Step 3: Implement**

In `server.py` add imports `import io, wave, asyncio, threading`, `from fastapi.responses import StreamingResponse`, `from poc_tts_streaming.audio import to_int16, SAMPLE_RATE`, `from pydantic import BaseModel, Field`, and:

```python
class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    voice: str
    response_format: Literal["pcm", "wav"] = "pcm"
    model: str | None = None
    x_chatterbox: dict = Field(default_factory=dict)


def _wav_bytes(pcm_int16: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_int16.tobytes())
    return buffer.getvalue()
```

and inside `create_app`:

```python
    async def _pcm_chunks(text: str, voice: str, knobs: ChatterboxKnobs):
        """Run the synthesizer on the worker; yield int16 bytes per chunk as they land."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()
        done = object()

        def produce():
            try:
                for _, pcm in engine_synthesizer(engine)(text, voice, knobs, cancel):
                    loop.call_soon_threadsafe(queue.put_nowait, to_int16(pcm).tobytes())
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        app.state.worker.submit(produce)
        try:
            while True:
                item = await queue.get()
                if item is done:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancel.set()

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request):
        try:
            body = SpeechRequest.model_validate(await request.json())
        except Exception as exc:  # pydantic ValidationError or bad JSON
            err = getattr(exc, "errors", lambda: [{"loc": ("body",), "msg": str(exc)}])()[0]
            return openai_error(400, err.get("msg", "invalid request"),
                                code="invalid_value", param=str(err.get("loc", ("body",))[-1]))
        if body.voice not in discover_voices(voice_paths):
            return openai_error(400, f"unknown voice {body.voice!r}", code="invalid_value", param="voice")
        if not engine.loaded:
            return openai_error(503, "Flash model is not loaded", type_="server_error", code="model_not_loaded")
        try:
            knobs = app.state.knobs.merged(body.x_chatterbox, param_prefix="x_chatterbox")
        except EventError as err:
            return openai_error(400, err.message, code=err.code, param=err.param)
        chunks = _pcm_chunks(body.input, body.voice, knobs)
        if body.response_format == "wav":
            try:
                data = b"".join([c async for c in chunks])
            except FileNotFoundError as exc:
                return openai_error(404, str(exc), code="not_found", param="voice")
            except Exception as exc:  # noqa: BLE001
                return openai_error(500, str(exc), type_="server_error", code="synthesis_failed")
            return Response(_wav_bytes(np.frombuffer(data, dtype=np.int16), SAMPLE_RATE), media_type="audio/wav")
        return StreamingResponse(chunks, media_type="audio/pcm",
                                 headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Channels": "1"})
```

Add `from typing import Literal` and `import numpy as np` if not present.

- [ ] **Step 4: Run**

Run: `make test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/server.py poc-tts-streaming/tests/test_audio_speech.py
git commit -m "feat(poc-tts-streaming): chunked-PCM /v1/audio/speech as the engine sidecar seam"
```

---

### Task 12: Browser client — `ui/realtime-client.js`

**Files:**
- Create: `poc-tts-streaming/ui/realtime-client.js`
- Modify: `poc-tts-streaming/poc_tts_streaming/server.py` (serve it with `no-store` like `script.js`)
- Modify: `poc-tts-streaming/ui/index.html` (script tag)

**Interfaces:**
- Produces a global `RealtimeTtsClient`:
  ```js
  const c = new RealtimeTtsClient({ baseUrl: "", session: {...} });
  await c.connect();                       // client secret -> offer -> /calls -> answer; resolves on conversation.created
  c.remoteStream                          // MediaStream (set on ontrack)
  await c.updateSession(patch);           // session.update; resolves on session.updated or rejects on error
  const respId = await c.speak(text);     // item.create + response.create; resolves on response.created
  c.cancel(); c.clear();                  // response.cancel / output_audio_buffer.clear
  c.on(type, fn) / c.off(type, fn)        // server events by type; "*" for all; "client-event" for outbound
  c.disconnect();                         // DELETE /calls/{id} + pc.close()
  c.state                                 // { pc, ice, dc }
  ```
  `baseUrl: "https://api.openai.com"` plus `apiKey` in the constructor makes it hit OpenAI instead (manual conformance check; not automated).

- [ ] **Step 1: Write the client**

`ui/realtime-client.js`:

```javascript
// Minimal OpenAI-Realtime-over-WebRTC client for text-to-speech.
// The only thing that should differ between this server and api.openai.com
// is `baseUrl` (and, there, a real API key): the flow below is the one
// OpenAI documents -- client secret, SDP offer to /v1/realtime/calls,
// "oai-events" data channel, audio on the media track.
class RealtimeTtsClient {
  constructor({ baseUrl = "", apiKey = null, session = {}, model = "chatterbox-flash", iceServers = [] } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.sessionPatch = session;
    this.model = model;
    this.iceServers = iceServers;
    this.pc = null; this.dc = null; this.callLocation = null;
    this.remoteStream = null;
    this.state = { pc: "new", ice: "new", dc: "closed" };
    this._handlers = new Map();
    this._pending = new Map();   // client event_id -> {resolve, reject, okType}
    this._seq = 0;
  }

  on(type, fn) { (this._handlers.get(type) || this._handlers.set(type, []).get(type)).push(fn); return this; }
  off(type, fn) { const h = this._handlers.get(type) || []; this._handlers.set(type, h.filter(x => x !== fn)); }
  _emit(type, ev) { for (const fn of [...(this._handlers.get(type) || []), ...(this._handlers.get("*") || [])]) fn(ev); }

  async _clientSecret() {
    if (this.apiKey) return this.apiKey;   // against OpenAI, mint the ephemeral key server-side; here a raw key is fine for a manual check
    const r = await fetch(`${this.baseUrl}/v1/realtime/client_secrets`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ session: { type: "realtime", model: this.model, ...this.sessionPatch } }),
    });
    if (!r.ok) throw new Error(`client_secrets ${r.status}: ${await r.text()}`);
    return (await r.json()).value;
  }

  async connect() {
    const key = await this._clientSecret();
    const pc = new RTCPeerConnection({ iceServers: this.iceServers });
    this.pc = pc;
    pc.addEventListener("connectionstatechange", () => { this.state.pc = pc.connectionState; this._emit("state", this.state); });
    pc.addEventListener("iceconnectionstatechange", () => { this.state.ice = pc.iceConnectionState; this._emit("state", this.state); });
    pc.addEventListener("track", (e) => { this.remoteStream = e.streams[0]; this._emit("track", e.streams[0]); });
    pc.addTransceiver("audio", { direction: "recvonly" });

    const dc = pc.createDataChannel("oai-events");
    this.dc = dc;
    const ready = new Promise((resolve, reject) => {
      this._onceType("conversation.created", resolve);
      dc.addEventListener("close", () => reject(new Error("data channel closed")), { once: true });
    });
    dc.addEventListener("open", () => { this.state.dc = "open"; this._emit("state", this.state); });
    dc.addEventListener("close", () => { this.state.dc = "closed"; this._emit("state", this.state); });
    dc.addEventListener("message", (e) => this._onServerEvent(JSON.parse(e.data)));

    await pc.setLocalDescription(await pc.createOffer());
    await this._waitIce(pc);
    const r = await fetch(`${this.baseUrl}/v1/realtime/calls?model=${encodeURIComponent(this.model)}`, {
      method: "POST", headers: { "content-type": "application/sdp", authorization: `Bearer ${key}` },
      body: pc.localDescription.sdp,
    });
    if (!r.ok) throw new Error(`calls ${r.status}: ${await r.text()}`);
    this.callLocation = r.headers.get("location");
    await pc.setRemoteDescription({ type: "answer", sdp: await r.text() });
    await ready;
  }

  _waitIce(pc) {
    if (pc.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve) => {
      const check = () => { if (pc.iceGatheringState === "complete") { pc.removeEventListener("icegatheringstatechange", check); resolve(); } };
      pc.addEventListener("icegatheringstatechange", check);
      setTimeout(resolve, 1500);
    });
  }

  _onceType(type, fn) { const h = (ev) => { this.off(type, h); fn(ev); }; this.on(type, h); }

  _onServerEvent(ev) {
    this._emit(ev.type, ev);
    if (ev.type === "error" && ev.error?.event_id && this._pending.has(ev.error.event_id)) {
      this._pending.get(ev.error.event_id).reject(new Error(`${ev.error.code}: ${ev.error.message}`));
      this._pending.delete(ev.error.event_id);
    }
    for (const [id, p] of this._pending) {
      if (ev.type === p.okType) { p.resolve(ev); this._pending.delete(id); }
    }
  }

  send(event, okType = null) {
    if (!this.dc || this.dc.readyState !== "open") throw new Error("not connected");
    const event_id = `evt_${++this._seq}`;
    const full = { event_id, ...event };
    this.dc.send(JSON.stringify(full));
    this._emit("client-event", full);
    if (!okType) return Promise.resolve(null);
    return new Promise((resolve, reject) => this._pending.set(event_id, { resolve, reject, okType }));
  }

  updateSession(patch) { return this.send({ type: "session.update", session: patch }, "session.updated"); }

  async speak(text, responsePatch = {}) {
    await this.send({ type: "conversation.item.create", item: {
      type: "message", role: "user", content: [{ type: "input_text", text }] } }, "conversation.item.done");
    const created = await this.send({ type: "response.create", response: responsePatch }, "response.created");
    return created.response.id;
  }

  cancel() { return this.send({ type: "response.cancel" }); }
  clear() { return this.send({ type: "output_audio_buffer.clear" }); }

  async disconnect() {
    try { if (this.callLocation) await fetch(`${this.baseUrl}${this.callLocation}`, { method: "DELETE" }); } catch {}
    try { this.dc?.close(); } catch {}
    try { this.pc?.close(); } catch {}
    this.pc = null; this.dc = null; this.remoteStream = null;
    this.state = { pc: "new", ice: "new", dc: "closed" };
    this._emit("state", this.state);
  }
}
window.RealtimeTtsClient = RealtimeTtsClient;
```

- [ ] **Step 2: Serve it**

In `server.py` next to the `script_js` route:

```python
    @app.get("/realtime-client.js", include_in_schema=False)
    async def realtime_client_js():
        return FileResponse(UI_DIR / "realtime-client.js", media_type="application/javascript", headers=_NO_STORE)
```

In `ui/index.html`, before the `script.js` tag: `<script src="realtime-client.js?v=poc1"></script>`. Remove the `vendor/wavesurfer.min.js` script tag and the `/vendor/wavesurfer.min.js` route in `server.py` (WaveSurfer is not used in this PoC).

- [ ] **Step 3: Add a route test**

Append to `tests/test_server_info.py`:

```python
def test_realtime_client_js_is_served_uncached(client):
    r = client.get("/realtime-client.js")
    assert r.status_code == 200
    assert "RealtimeTtsClient" in r.text
    assert r.headers["cache-control"] == "no-store, must-revalidate"
```

(use that file's existing `client` fixture.)

Run: `make test` → PASS.

- [ ] **Step 4: Commit**

```bash
git add poc-tts-streaming/ui/realtime-client.js poc-tts-streaming/ui/index.html poc-tts-streaming/poc_tts_streaming/server.py poc-tts-streaming/tests/test_server_info.py
git commit -m "feat(poc-tts-streaming): browser RealtimeTtsClient (client secret, /calls, oai-events)"
```

---

### Task 13: Browser test interface — stream panel, events pane, metrics

**Files:**
- Modify: `poc-tts-streaming/ui/index.html` (audio player container at line ~460; remove the Output Format select at lines ~330-335)
- Modify: `poc-tts-streaming/ui/script.js` (`submitTTSRequest` at ~1099, `initializeWaveSurfer` at ~958, generate-button handler at ~1147, loading-overlay cancel at ~1232)
- Modify: `poc-tts-streaming/ui/styles.css` (append)

**Interfaces:**
- Consumes: `RealtimeTtsClient` (Task 12), `config.realtime` from `/api/ui/initial-data`.
- Produces: a working page where **Generate** streams and **Stop** cancels; metrics rendered in `#stream-metrics`; raw events in `#events-log`.

- [ ] **Step 1: Replace the player container in index.html**

Replace `<div id="audio-player-container" class="mt-8"></div>` (line ~460) with:

```html
<section id="stream-panel" class="card mt-8">
    <h2 class="card__title">Stream</h2>
    <div class="flex-row items-center gap-2 mb-4">
        <span id="pill-pc" class="pill">pc: new</span>
        <span id="pill-ice" class="pill">ice: new</span>
        <span id="pill-dc" class="pill">dc: closed</span>
        <button type="button" id="stop-btn" class="btn secondary" disabled>Stop</button>
        <button type="button" id="disconnect-btn" class="btn secondary" disabled>Disconnect</button>
    </div>
    <audio id="remote-audio" autoplay controls></audio>
    <div id="level-meter"><div id="level-bar"></div></div>
    <dl id="stream-metrics" class="metrics">
        <dt>TTFA (browser)</dt><dd id="m-ttfa">–</dd>
        <dt>TTFA (server)</dt><dd id="m-ttfa-server">–</dd>
        <dt>Total</dt><dd id="m-total">–</dd>
        <dt>Audio</dt><dd id="m-audio">–</dd>
        <dt>Chunks</dt><dd id="m-chunks">–</dd>
    </dl>
    <p class="text-sm">Playback needs a user gesture after a reload: click Generate, not a script.</p>
    <h3 class="card__subtitle mt-4">oai-events</h3>
    <pre id="events-log" class="events-log"></pre>
</section>
```

Delete the `Output Format` form group (the `<select id="output-format">` and its label/wrapper) and the `speed_factor`/`seed`/`language` controls' form groups if they are still rendered (they are hidden by `model_info` today; removing them avoids dead `session.update` fields).

- [ ] **Step 2: Styles**

Append to `ui/styles.css`:

```css
.pill { padding: 2px 8px; border-radius: 999px; font-size: 12px; background: var(--muted, #e5e7eb); }
.pill.ok { background: #bbf7d0; } .pill.err { background: #fecaca; }
#level-meter { height: 8px; background: var(--muted, #e5e7eb); border-radius: 4px; margin: 8px 0; overflow: hidden; }
#level-bar { height: 100%; width: 0; background: #22c55e; transition: width 60ms linear; }
.metrics { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; font-variant-numeric: tabular-nums; }
.metrics dt { opacity: .7; }
.events-log { max-height: 320px; overflow: auto; font-size: 11px; white-space: pre-wrap; }
.events-log .out { color: #2563eb; } .events-log .in { color: inherit; } .events-log .err { color: #dc2626; }
```

- [ ] **Step 3: Rewrite the generation path in script.js**

Delete `initializeWaveSurfer` (lines ~958-1073) and the `wavesurfer` variable. Replace `submitTTSRequest` with the block below, and add the helpers after it. Keep `getTTSFormData` but stop reading `outputFormatSelect`, `speedFactorSlider`, `seedInput`, `languageSelect` (delete those lines and the matching `getElementById`s at the top of the file).

```javascript
    // --- Realtime streaming ---
    let rt = null;                      // RealtimeTtsClient
    let analyser = null, meterRaf = null;
    let metrics = null;                 // per-response timing
    const $ = (id) => document.getElementById(id);

    function logEvent(kind, ev) {
        const log = $('events-log');
        const line = document.createElement('div');
        line.className = kind;
        const ts = new Date().toISOString().slice(11, 23);
        line.textContent = `${ts} ${kind === 'out' ? '→' : '←'} ${JSON.stringify(ev)}`;
        log.appendChild(line);
        log.scrollTop = log.scrollHeight;
    }

    function setPill(id, label, cls) { const el = $(id); el.textContent = label; el.className = 'pill' + (cls ? ' ' + cls : ''); }

    function sessionPatchFromControls() {
        const data = getTTSFormData();
        const voice = currentVoiceMode === 'predefined' ? data.predefined_voice_id : data.reference_audio_filename;
        return {
            audio: { output: { voice } },
            x_chatterbox: {
                temperature: data.temperature, exaggeration: data.exaggeration, cfg_scale: data.cfg_weight,
                num_steps: data.num_steps, n_cfm_timesteps: data.n_cfm_timesteps,
                chunk_size: data.chunk_size, split_text: data.split_text, split_on_clauses: true,
            },
        };
    }

    async function ensureConnected() {
        if (rt && rt.state.dc === 'open') return rt;
        rt = new RealtimeTtsClient({ baseUrl: API_BASE_URL, session: sessionPatchFromControls(),
                                     iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
        rt.on('*', (ev) => logEvent(ev.type === 'error' ? 'err' : 'in', ev));
        rt.on('client-event', (ev) => logEvent('out', ev));
        rt.on('state', (s) => {
            setPill('pill-pc', `pc: ${s.pc}`, s.pc === 'connected' ? 'ok' : s.pc === 'failed' ? 'err' : '');
            setPill('pill-ice', `ice: ${s.ice}`, /connected|completed/.test(s.ice) ? 'ok' : s.ice === 'failed' ? 'err' : '');
            setPill('pill-dc', `dc: ${s.dc}`, s.dc === 'open' ? 'ok' : '');
            $('disconnect-btn').disabled = s.dc !== 'open';
        });
        rt.on('track', (stream) => { $('remote-audio').srcObject = stream; startMeter(stream); });
        rt.on('response.created', (ev) => { metrics = { id: ev.response.id, created: performance.now(), chunks: 0, firstAudio: null, serverStarted: null }; renderMetrics(); });
        rt.on('response.output_audio_transcript.delta', () => { if (metrics) { metrics.chunks++; renderMetrics(); } });
        rt.on('output_audio_buffer.started', () => { if (metrics) { metrics.serverStarted = performance.now(); renderMetrics(); } });
        rt.on('response.done', (ev) => { if (metrics) { metrics.done = performance.now(); metrics.status = ev.response.status; renderMetrics(); } $('stop-btn').disabled = true; isGenerating = false; hideLoadingOverlay(); });
        rt.on('output_audio_buffer.stopped', () => { if (metrics) { metrics.stopped = performance.now(); renderMetrics(); } });
        rt.on('error', (ev) => showNotification(`${ev.error.code}: ${ev.error.message}`, 'error'));
        await rt.connect();
        return rt;
    }

    function startMeter(stream) {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const src = ctx.createMediaStreamSource(stream);
        analyser = ctx.createAnalyser(); analyser.fftSize = 512;
        src.connect(analyser);
        const buf = new Float32Array(analyser.fftSize);
        const tick = () => {
            analyser.getFloatTimeDomainData(buf);
            let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v));
            $('level-bar').style.width = `${Math.min(100, peak * 300)}%`;
            if (metrics && metrics.firstAudio === null && peak > 0.01) { metrics.firstAudio = performance.now(); renderMetrics(); }
            meterRaf = requestAnimationFrame(tick);
        };
        if (meterRaf) cancelAnimationFrame(meterRaf);
        tick();
    }

    function renderMetrics() {
        if (!metrics) return;
        const s = (a, b) => (a != null && b != null) ? `${((b - a) / 1000).toFixed(3)} s` : '–';
        $('m-ttfa').textContent = s(metrics.created, metrics.firstAudio);
        $('m-ttfa-server').textContent = s(metrics.created, metrics.serverStarted);
        $('m-total').textContent = s(metrics.created, metrics.done) + (metrics.status ? ` (${metrics.status})` : '');
        $('m-audio').textContent = s(metrics.serverStarted, metrics.stopped);
        $('m-chunks').textContent = String(metrics.chunks);
    }

    async function submitTTSRequest() {
        isGenerating = true;
        showLoadingOverlay();
        try {
            const client = await ensureConnected();
            await client.updateSession(sessionPatchFromControls());
            $('stop-btn').disabled = false;
            await client.speak(textArea.value);
        } catch (error) {
            console.error('Realtime error:', error);
            showNotification(error.message || 'Streaming failed.', 'error');
            isGenerating = false;
            hideLoadingOverlay();
        }
    }

    $('stop-btn').addEventListener('click', async () => {
        if (!rt) return;
        try { await rt.cancel(); await rt.clear(); } catch (e) { showNotification(e.message, 'error'); }
    });
    $('disconnect-btn').addEventListener('click', async () => { if (rt) { await rt.disconnect(); rt = null; } });
```

Also change the loading-overlay cancel handler (line ~1232) so cancelling the overlay also sends `rt.cancel()`; and make the overlay non-blocking during streaming: call `hideLoadingOverlay()` in the `output_audio_buffer.started` handler (audio is playing; the overlay should not cover the meter). Keep `isGenerating` true until `response.done`.

- [ ] **Step 4: Manual verification in the browser**

Run: `make` (from `poc-tts-streaming/`), open `http://127.0.0.1:8006`.

- Type the three baseline sentences (they are in `poc_tts_streaming/bench.py`), pick `one-one.mp3`, click **Generate**. Expect: pills go `pc: connected`, `dc: open`; the events pane shows `session.created` → … → `response.created`; audio starts within ~1 s; the level meter moves; `TTFA (browser)` and `TTFA (server)` fill in; `Chunks` counts up per sentence; `response.done (completed)` then `output_audio_buffer.stopped`.
- Click **Generate** on a long paragraph, then **Stop** mid-way. Expect: audio stops within one frame, `response.done (cancelled)`, `output_audio_buffer.cleared`.
- Change `num_steps` and Generate again. Expect a `session.update`/`session.updated` pair in the pane and a different generation time.
- Pick a voice, delete the file from `voices/`, Generate. Expect an `error` event with `param: session.audio.output.voice` and a notification.
- `chrome://webrtc-internals`: one outbound audio stream from the server, packets increasing, no ICE failures.
- Record the browser TTFA for the three baseline sentences (best of 3) in `poc-tts-streaming/results-rtx-2060.md` (Task 14 creates the file).

- [ ] **Step 5: Commit**

```bash
git add poc-tts-streaming/ui
git commit -m "feat(poc-tts-streaming): stream panel with TTFA metrics, events pane, Stop"
```

---

### Task 14: `bench_stream.py`, results doc, README

**Files:**
- Create: `poc-tts-streaming/poc_tts_streaming/bench_stream.py`
- Create: `poc-tts-streaming/results-rtx-2060.md`
- Modify: `poc-tts-streaming/README.md`
- Test: `poc-tts-streaming/tests/test_bench_stream.py`

**Interfaces:**
- Consumes: `FlashEngine.synthesize_stream`, the baseline sentences from the copied `bench.py` (`SENTENCES` or equivalent name — check `bench.py` and import that constant).
- Produces: `measure(engine, text, voice, knobs: dict) -> dict` with keys `ttfa_s, gen_s, audio_s, n_chunks, first_chunk_chars, chunks: [{chars, gen_s, audio_s}]`; `main()` appends one JSON line per sentence to `reports/stream_runs.jsonl` with `host, gpu, dtype, backend, drf_block_size, vram_peak_mb` alongside.

- [ ] **Step 1: Write the failing test**

`tests/test_bench_stream.py`:

```python
import numpy as np

from poc_tts_streaming.bench_stream import measure


class FakeEngine:
    sr = 24000
    def synthesize_stream(self, text, voice, **kw):
        for s in ("One.", "Two."):
            yield s, np.zeros(12000, dtype=np.float32)


def test_measure_reports_ttfa_total_audio_and_chunks():
    row = measure(FakeEngine(), "One. Two.", "v.wav", {"num_steps": 4})
    assert row["n_chunks"] == 2
    assert row["first_chunk_chars"] == 4
    assert row["audio_s"] == 1.0
    assert 0 <= row["ttfa_s"] <= row["gen_s"]
    assert [c["audio_s"] for c in row["chunks"]] == [0.5, 0.5]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bench_stream.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`poc_tts_streaming/bench_stream.py`:

```python
"""Time-to-first-audio bench: the three baseline sentences through
synthesize_stream() with the config.yaml generation defaults.

    make bench-stream   ->   reports/stream_runs.jsonl (one line per sentence)
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from poc_tts_streaming.bench import SENTENCES  # the same three sentences every baseline uses
from poc_tts_streaming.config import load_config, voice_paths

REPORT = Path(__file__).resolve().parent.parent / "reports" / "stream_runs.jsonl"


def measure(engine, text: str, voice: str, knobs: dict) -> dict:
    t0 = time.perf_counter()
    chunks = []
    last = t0
    ttfa = None
    for chunk_text, pcm in engine.synthesize_stream(text, voice, **knobs):
        now = time.perf_counter()
        if ttfa is None:
            ttfa = now - t0
        chunks.append({"chars": len(chunk_text), "gen_s": round(now - last, 4),
                       "audio_s": round(len(pcm) / engine.sr, 4)})
        last = now
    total = time.perf_counter() - t0
    return {
        "ttfa_s": round(ttfa or total, 4), "gen_s": round(total, 4),
        "audio_s": round(sum(c["audio_s"] for c in chunks), 4),
        "n_chunks": len(chunks), "first_chunk_chars": chunks[0]["chars"] if chunks else 0,
        "chunks": chunks,
    }


def main() -> None:
    import torch
    from poc_tts_streaming.engine_flash import FlashEngine

    config = load_config()
    paths = voice_paths(config)
    engine = FlashEngine(engine_cfg=config.get("engine", {}), generation_cfg=config.get("generation", {}),
                         voice_paths=paths)
    engine.load()
    gen = config.get("generation", {})
    knobs = {k: gen[k] for k in ("chunk_size", "split_text", "split_on_clauses") if k in gen}
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    REPORT.parent.mkdir(exist_ok=True)
    # warm-up: the first generate() pays CUDA-graph / allocator costs
    list(engine.synthesize_stream("Warm up.", voice, **knobs))
    with open(REPORT, "a", encoding="utf-8") as out:
        for label, text in SENTENCES:   # list of (name, text) tuples in bench.py:39
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            best = min((measure(engine, text, voice, knobs) for _ in range(2)), key=lambda r: r["ttfa_s"])
            row = {"ts": int(time.time()), "host": platform.node(), "sentence": label, "chars": len(text),
                   "dtype": str(engine.dtype).replace("torch.", ""), "backend": engine.backend,
                   "drf_block_size": engine.drf_block_size, "generation": gen,
                   "vram_peak_mb": (round(torch.cuda.max_memory_reserved() / 2**20)
                                    if torch.cuda.is_available() else None), **best}
            out.write(json.dumps(row) + "\n")
            print(f"{label:>7}: ttfa {best['ttfa_s']:.3f}s  gen {best['gen_s']:.2f}s  "
                  f"audio {best['audio_s']:.2f}s  chunks {best['n_chunks']}")


if __name__ == "__main__":
    main()
```

`SENTENCES` in the copied `bench.py` (line 39) is a list of `(name, text)` tuples — the loop above matches it. The test only touches `measure`.

- [ ] **Step 4: Run the test, then the real bench**

Run: `.venv/bin/python -m pytest tests/test_bench_stream.py -v` → PASS.
Run: `make bench-stream` (needs the GPU and the downloaded model) → three lines printed and appended to `reports/stream_runs.jsonl`.

- [ ] **Step 5: Results doc and README**

Create `poc-tts-streaming/results-rtx-2060.md` with the measured numbers (fill from the bench output and the browser panel — do not leave placeholders; if a number was not measured, say "not measured" and why):

```markdown
# poc-tts-streaming on an RTX 2060 — time-to-first-audio

Measured <date> on `<host>`, generation config `<from config.yaml>`,
resolved dtype `<…>` backend `<…>`.

| sentence | chars | chunks | TTFA engine (s) | TTFA browser (s) | total gen (s) | audio (s) | poc-tts whole-utterance (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 30 | … | … | … | … | … | 0.59 |
| medium | 104 | … | … | … | … | … | 1.03 |
| long | 317 | … | … | … | … | … | 3.38 |

Browser TTFA = `response.create` sent → first non-silent sample at the
AnalyserNode; includes Opus encode, jitter buffer and decode. Engine TTFA is
from `reports/stream_runs.jsonl`. poc-tts column: `poc-tts/bench-rtx-2060.md`
tuned row (whole utterance must finish before any audio plays).

Gaps between chunks: <observed / not observed>, by ear and by the level meter.
```

Extend `README.md` with a "Realtime API surface" section listing the four routes and the data-channel name, the manual OpenAI swap check (`new RealtimeTtsClient({baseUrl: "https://api.openai.com", apiKey, model: "gpt-realtime"})` from the DevTools console — `x_chatterbox` will be rejected there, everything else should flow), and the resolved aiortc version from `setup.sh`'s output.

- [ ] **Step 6: Commit**

```bash
git add poc-tts-streaming/poc_tts_streaming/bench_stream.py poc-tts-streaming/tests/test_bench_stream.py poc-tts-streaming/results-rtx-2060.md poc-tts-streaming/README.md poc-tts-streaming/reports/stream_runs.jsonl
git commit -m "docs(poc-tts-streaming): TTFA bench and RTX 2060 results"
```

---

### Task 15 (optional): MediaRecorder capture for offline A/B

**Files:**
- Modify: `poc-tts-streaming/ui/script.js`, `ui/index.html`

**Interfaces:**
- Consumes: `rt.remoteStream`, the `response.created` / `output_audio_buffer.stopped` events.
- Produces: a **Download last utterance** link (`webm/opus`) under the stream panel.

- [ ] **Step 1: Add the control**

In `index.html` stream panel, after `#stream-metrics`: `<a id="download-last" class="btn secondary" download="utterance.webm" hidden>Download last utterance</a>`.

- [ ] **Step 2: Record per response**

In `ensureConnected()` (Task 13) add:

```javascript
        let recorder = null, recorded = [];
        rt.on('response.created', () => {
            if (!rt.remoteStream) return;
            recorded = [];
            recorder = new MediaRecorder(rt.remoteStream, { mimeType: 'audio/webm;codecs=opus' });
            recorder.ondataavailable = (e) => { if (e.data.size) recorded.push(e.data); };
            recorder.onstop = () => {
                const url = URL.createObjectURL(new Blob(recorded, { type: 'audio/webm' }));
                const a = $('download-last'); a.href = url; a.hidden = false;
            };
            recorder.start(250);
        });
        rt.on('output_audio_buffer.stopped', () => { if (recorder && recorder.state === 'recording') recorder.stop(); });
        rt.on('output_audio_buffer.cleared', () => { if (recorder && recorder.state === 'recording') recorder.stop(); });
```

- [ ] **Step 3: Verify and commit**

Generate, wait for `output_audio_buffer.stopped`, click the link, play the file. Commit:

```bash
git add poc-tts-streaming/ui
git commit -m "feat(poc-tts-streaming): download the last streamed utterance"
```

---

### Task 16 (optional spike, gated): intra-sentence streaming

**Gate:** run only if Task 14's browser TTFA for the *medium* sentence is above ~0.5 s with the tuned config — otherwise sentence pipelining already meets the PRD's fast-tier budget and this spike is not worth its risk. This is a **spike**: its output is a finding in `results-rtx-2060.md`, and any code stays behind a config flag defaulting to off.

**Files:**
- Create: `poc-tts-streaming/poc_tts_streaming/engine_blockstream.py` (spike code, flag-gated)
- Modify: `poc-tts-streaming/config.yaml` (`engine.block_streaming: false`)
- Modify: `poc-tts-streaming/results-rtx-2060.md` (findings)

**What to try, concretely:**

1. **Hook the block loop.** Copy `ChatterboxFlashT3.generate` (`chatterbox_flash/model.py:525` onward, the torch-SDPA path only — the FlashInfer/CUDA-graph path is irrelevant on sm_75) into `engine_blockstream.py` as `generate_blocks(t3, *, on_block: Callable[[torch.Tensor], None], **same_kwargs)`, and call `on_block(speech_tokens[:, :filled])` each time a block of `drf_block_size` tokens is finalised. Verify first, with a mocked `on_block`, that the final token tensor is identical to the unpatched `generate` for a fixed seed (`torch.manual_seed(0)` around both calls).
2. **Vocode windows.** Following the CosyVoice/`S3GenStreamer` pattern the base package's docstring points to (`chatterbox/models/s3gen/s3gen.py:278`): on each callback, run `s3gen.flow_inference(tokens_so_far, ref_dict=conds.gen, finalize=False, n_cfm_timesteps=…)` and `s3gen.hift_inference(mels_for_new_window, cache_source=prev_source)` where `hift_inference` returns `(wav, source)` and the returned `source` is the next call's `cache_source`. Emit only the samples that correspond to newly finalised tokens (2 mel frames per token; `mel2wav` upsamples 480 samples per mel frame at 24 kHz — confirm the ratio from one full-utterance run: `len(wav) / n_tokens`). On the last block call with `finalize=True`.
3. **Measure and listen.** Add a `--block-stream` flag to `bench_stream.py` that swaps the synthesizer; record `ttfa_s` per sentence beside the sentence-pipelined numbers. Listen for seams at window boundaries (clicks, pitch resets) with headphones; the level meter will show discontinuities as spikes.
4. **Go / no-go.** Go = TTFA drops by ≥ 40 % on the medium sentence **and** no audible seams on all three sentences across 3 runs. Otherwise no-go: keep the code flag-gated off, write up what broke (most likely candidates: the `trim_fade` applied per window, mel/sample ratio drift at window edges, or `finalize=False` quality on the meanflow vocoder), and stop.

- [ ] **Step 1:** Implement `generate_blocks` and the identity test against the unpatched path (`tests/test_blockstream.py`, mocked `on_block`, real T3 requires the GPU → mark `@pytest.mark.skipif(not torch.cuda.is_available())`).
- [ ] **Step 2:** Implement windowed vocoding behind `engine.block_streaming`.
- [ ] **Step 3:** Bench + listening notes into `results-rtx-2060.md`; go/no-go recorded.
- [ ] **Step 4:** Commit: `git commit -m "spike(poc-tts-streaming): intra-sentence block streaming -- <go|no-go>"`.

---

## Plan self-review

**Spec coverage.** Sentence pipelining → Tasks 3, 7; clause splitting → Task 2; engine boundary + sidecar seam → Tasks 3, 11; Realtime HTTP surface (`client_secrets`, `calls`, `DELETE`) → Tasks 8, 9; data-channel protocol and event order → Tasks 6, 7, 10; media track rules (480-sample frames, silence on underrun, inbound drain) → Tasks 4, 5, 9; browser page (client, pills, meter, TTFA, events pane, Stop, session.update on control change) → Tasks 12, 13; bench + results → Task 14; MediaRecorder → Task 15; level-2 spike → Task 16; error handling (OOM → `failed`, 503 when unloaded, disconnect cleanup, bad JSON) → Tasks 7, 9. The FlowCat integration section is deliberately follow-on work in `poc/flowcat` and has no task here.

**Interfaces checked.** `synthesize_stream(text, voice, *, …, cancel)` is called with keyword knobs from `engine_synthesizer` (Task 8) and from `bench_stream.measure` (Task 14); `ChatterboxKnobs.as_engine_kwargs()` yields exactly the keyword names `synthesize_stream` accepts (`temperature, exaggeration, cfg_scale, num_steps, n_cfm_timesteps, chunk_size, split_text, split_on_clauses`). `AudioSink` methods (`push/flush/clear/drained`) match `PcmQueueTrack` and the test `FakeSink`. `build_session(send, sink, session_patch)` is the signature used by `CallRegistry.create` and by the client-secrets route.

**Known judgement calls (from the spec's [assumption] markers) that the requester should confirm before Task 1:** copy-not-import; bearer check kept on `/calls`; tuned config as default; `x_chatterbox` namespacing; user-item text is what gets spoken (no `instructions` shortcut).
