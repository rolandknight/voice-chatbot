"""Task 16 spike: block streaming must not change what T3 generates.

Everything but the config guard needs the real model on a GPU, so those tests
are skipped off-CUDA. That keeps `make test` GPU-free while still giving the
spike a hard identity check on the machine that matters.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


def _bench_sentences() -> list[str]:
    """The three sentences every baseline in this PoC is measured on."""
    from poc_tts_streaming.bench import SENTENCES

    return [text for _label, text in SENTENCES]


def test_block_streaming_defaults_on_for_cuda_torch():
    """block_streaming defaults true in config.yaml, but the resolved engine
    only takes the spike path on device=='cuda' with backend=='torch' -- the
    only combination engine_blockstream.BlockStreamEngine can run (it hooks a
    copied torch-SDPA T3 loop). Every other resolved device/backend falls
    back to sentence streaming regardless of the flag."""
    from poc_tts_streaming.config import load_config
    from poc_tts_streaming.engine_flash import block_streaming_effective

    assert load_config()["engine"]["block_streaming"] is True

    assert block_streaming_effective(True, "cuda", "torch") is True
    assert block_streaming_effective(True, "cpu", "torch") is False
    assert block_streaming_effective(True, "cuda", "mlx") is False
    assert block_streaming_effective(False, "cuda", "torch") is False


@pytest.fixture(scope="module")
def loaded():
    """A warm FlashEngine plus the pieces generate_blocks needs."""
    from poc_tts_streaming.config import load_config, voice_paths
    from poc_tts_streaming.engine_flash import FlashEngine, resolve_voice_path

    config = load_config()
    paths = voice_paths(config)
    engine = FlashEngine(
        engine_cfg=config.get("engine", {}),
        generation_cfg=config.get("generation", {}),
        voice_paths=paths,
    )
    engine.load()
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    model = engine._model
    # Warm up: the first generate() builds the SDPA engine and the cached
    # unconditional block prior. Both are lazy and would otherwise land inside
    # whichever call happens to run first, which is not what we are comparing.
    model.generate(
        "Warm up.",
        audio_prompt_path=str(resolve_voice_path(voice, paths)),
        num_steps=4, n_cfm_timesteps=1,
        backend=engine.backend,
    )
    return engine, model, config


def _t3_kwargs(engine, model, text: str, config: dict) -> dict:
    from chatterbox_flash.tts import _speech_len_for_text_tokens

    gen = config.get("generation", {})
    text_tokens = model._encode_text(text, normalize_text=True)
    n_text = int(text_tokens.size(1))
    return dict(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        text_token_lens=torch.tensor([n_text], device=model.device),
        total_speech_len=_speech_len_for_text_tokens(n_text),
        num_steps=int(gen.get("num_steps", 4)),
        temperature=float(gen.get("temperature", 0.6)),
        cfg_scale=float(gen.get("cfg_scale", 1.0)),
        backend=engine.backend,
    )


def _prepare_conds(model, config):
    """Point the model at the bench voice and return its path."""
    from poc_tts_streaming.config import voice_paths
    from poc_tts_streaming.engine_flash import resolve_voice_path

    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    prompt = str(resolve_voice_path(voice, voice_paths(config)))
    model.prepare_conditionals(
        prompt, exaggeration=float(config["generation"]["exaggeration"]),
    )
    return voice


@gpu
def test_generate_blocks_matches_generate(loaded):
    """Same seed in, same speech tokens out -- hooking must be side-effect free.

    Driven twice, because the two runs pin different things:

    1. a **no-op** callback, which pins the copied decode loop itself against
       drift in the installed ``ChatterboxFlashT3.generate``; and
    2. the **real vocoding** callback the engine ships, which pins the RNG
       fence. Vocoding draws noise -- ``flow_matching.py:216``'s
       ``randn_like`` (unconditional even with ``noised_mels`` supplied) and
       HiFTGAN's per-call NSF source noise -- and it happens *inside* the
       decode loop. Until ``_MelWindow`` fenced those draws behind
       ``_fork_rng`` (and its own noise pre-draw with them), every window
       advanced the same global stream the loop's later blocks sample from, so
       the path that actually ships sampled a different utterance from
       ``t3.generate``'s off the same seed. Case (1) could not see that: a
       callback that does nothing draws nothing.

    The ``_MelWindow`` is built *after* the seed, exactly as ``_stream_chunk``
    builds it before ``generate_blocks``, so an unfenced pre-draw shifts the
    stream here too.
    """
    from poc_tts_streaming.engine_blockstream import _MelWindow, generate_blocks

    engine, model, config = loaded
    text = "I checked the calendar for tomorrow and you have three meetings."
    _prepare_conds(model, config)
    kwargs = _t3_kwargs(engine, model, text, config)

    torch.manual_seed(0)
    reference = model.t3.generate(**kwargs)

    seen: list[torch.Tensor] = []
    torch.manual_seed(0)
    hooked = generate_blocks(model.t3, on_block=seen.append, **kwargs)

    assert hooked.shape == reference.shape, (
        f"shape drift: hooked {tuple(hooked.shape)} vs "
        f"reference {tuple(reference.shape)}"
    )
    assert torch.equal(hooked, reference), (
        f"{int((hooked != reference).sum())} of {reference.numel()} tokens differ"
    )

    # The callback must see strictly growing prefixes of the final tensor, and
    # the last one must be the final tensor itself.
    assert seen, "on_block was never called"
    lengths = [int(t.size(1)) for t in seen]
    assert lengths == sorted(set(lengths)), f"prefixes not strictly growing: {lengths}"
    assert lengths[-1] == int(hooked.size(1))
    for prefix in seen:
        n = int(prefix.size(1))
        assert torch.equal(prefix, hooked[:, :n]), "a callback prefix diverged"

    # ---- and again with the callback the engine actually runs.
    vocoded: list[int] = []
    torch.manual_seed(0)
    window = _MelWindow(
        model.s3gen, model.conds.gen,
        int(config["generation"]["n_cfm_timesteps"]),
        int(kwargs["total_speech_len"]),
    )

    def vocode(tokens: torch.Tensor) -> None:
        trimmed = model._trim_to_eos(tokens[0])
        vocoded.append(window.push(trimmed.to(model.device), finalize=False).size)

    real = generate_blocks(model.t3, on_block=vocode, **kwargs)

    assert sum(1 for n in vocoded if n) >= 2, (
        f"the vocoding callback produced no audio ({vocoded}) -- this run has "
        "to actually draw noise or it pins nothing"
    )
    assert real.shape == reference.shape, (
        f"shape drift under the vocoding callback: {tuple(real.shape)} vs "
        f"reference {tuple(reference.shape)} -- vocoding moved the RNG stream"
    )
    assert torch.equal(real, reference), (
        f"{int((real != reference).sum())} of {reference.numel()} tokens differ "
        "under the vocoding callback -- the _fork_rng fence around _MelWindow "
        "is not holding, so the shipping path is a different draw from "
        "t3.generate's"
    )


@gpu
def test_block_count_matches_drf_block_size(loaded):
    """One callback per drf_block_size tokens (the last block may be short)."""
    from poc_tts_streaming.engine_blockstream import generate_blocks

    engine, model, config = loaded
    kwargs = _t3_kwargs(engine, model, "Sure, the kitchen light is on.", config)

    seen: list[torch.Tensor] = []
    torch.manual_seed(1)
    tokens = generate_blocks(model.t3, on_block=seen.append, **kwargs)

    bs = engine.drf_block_size
    assert bs == model.t3.drf_block_size
    # Every callback but the last lands exactly on a block boundary.
    for prefix in seen[:-1]:
        assert int(prefix.size(1)) % bs == 0, (
            f"prefix length {int(prefix.size(1))} is not a multiple of {bs}"
        )
    assert int(tokens.size(1)) > 0


@gpu
def test_samples_per_token_ratio(loaded):
    """The 960-samples-per-token constant, checked against a real utterance."""
    from poc_tts_streaming import engine_blockstream as bs

    engine, model, config = loaded
    from poc_tts_streaming.engine_flash import resolve_voice_path
    from poc_tts_streaming.config import voice_paths

    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    prompt = str(resolve_voice_path(voice, voice_paths(config)))
    model.prepare_conditionals(prompt, exaggeration=0.5)
    kwargs = _t3_kwargs(engine, model, "Sure, the kitchen light is on.", config)

    torch.manual_seed(2)
    tokens = model.t3.generate(**kwargs)
    trimmed = model._trim_to_eos(tokens[0])
    wav, _ = model.s3gen.inference(
        speech_tokens=trimmed.to(model.device),
        ref_dict=model.conds.gen,
        n_cfm_timesteps=int(config["generation"]["n_cfm_timesteps"]),
    )
    ratio = wav.numel() / int(trimmed.numel())
    assert ratio == pytest.approx(bs.SAMPLES_PER_TOKEN, rel=1e-6), (
        f"len(wav)/n_tokens is {ratio}, not {bs.SAMPLES_PER_TOKEN}"
    )


@gpu
def test_blockstream_engine_end_to_end(loaded):
    """Windows must tile the utterance exactly: no gap, no overlap, no repeat.

    The whole-utterance length is 960 x n_tokens (test_samples_per_token_ratio),
    so a concatenation that is not a multiple of 960 means the windowed vocoder
    lost or duplicated audio somewhere.
    """
    from poc_tts_streaming import engine_blockstream as bs
    from poc_tts_streaming.config import load_config, voice_paths

    _engine, _model, config = loaded
    engine = bs.BlockStreamEngine(
        # trim_silence off: this test is about the *vocoder* tiling the token
        # stream exactly, and the silence gate deliberately drops edge samples
        # on the way out, which would make the 960-multiple arithmetic below
        # meaningless. The gate's own arithmetic is
        # test_streamed_samples_match_trimmed_tokens.
        engine_cfg={**config.get("engine", {}), "trim_silence": False},
        generation_cfg=config.get("generation", {}),
        voice_paths=voice_paths(config),
    )
    engine._model = _model          # reuse the fixture's weights, don't reload
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    text = ("I checked the calendar for tomorrow and you have three meetings, "
            "the first one starting at nine fifteen.")

    windows = list(engine.synthesize_stream(text, voice, split_text=False))
    assert len(windows) >= 2, "block streaming produced a single window"
    for label, pcm in windows:
        assert pcm.size > 0, f"empty window {label}"
        assert np.isfinite(pcm).all(), f"non-finite samples in {label}"

    total = sum(len(pcm) for _, pcm in windows)
    assert total % bs.SAMPLES_PER_TOKEN == 0, (
        f"{total} samples is not a whole number of tokens "
        f"({total / bs.SAMPLES_PER_TOKEN:.3f})"
    )


@gpu
def test_block_and_sentence_engines_draw_the_same_tokens(loaded, monkeypatch, capsys):
    """One sentence, one seed, both engines -> the same speech tokens.

    ``test_generate_blocks_matches_generate`` pins the copied loop against
    ``t3.generate``. This pins the two *engines* end to end, which is the claim
    results-rtx-2060.md makes to users: switching block streaming on is not
    supposed to change what gets said, only when you hear it.

    Scope: one chunk (``split_text=False`` on a sentence), seeded immediately
    before each render. That is the fence's reach and no further -- across a
    multi-chunk utterance from a single seed the two engines still diverge from
    chunk two on, because the *sentence* path's vocoder consumes global RNG
    between chunks and the fenced block path's no longer does. See the Prosody
    section of results-rtx-2060.md.

    The tokens being identical, the two renders differ only in vocoder noise
    (a fixed pre-draw sliced per window here, one fresh draw per utterance
    there, plus per-call NSF source noise on both). That difference is
    *reported*, not asserted -- it is the windowed-vocoder divergence the spike
    already measured, not something this test is in a position to bound. Run
    with ``-s`` to see it.
    """
    from poc_tts_streaming import engine_blockstream as bs
    from poc_tts_streaming.config import voice_paths
    from poc_tts_streaming.engine_flash import FlashEngine

    _engine, model, config = loaded
    # trim_silence off on both: the gate cuts edge samples, and a per-sample
    # comparison needs the raw draws. Token identity is unaffected either way.
    engine_cfg = {**config.get("engine", {}), "trim_silence": False}
    paths = voice_paths(config)
    block = bs.BlockStreamEngine(
        engine_cfg=engine_cfg,
        generation_cfg=config.get("generation", {}),
        voice_paths=paths,
    )
    sentence = FlashEngine(
        engine_cfg=engine_cfg,
        generation_cfg=config.get("generation", {}),
        voice_paths=paths,
    )
    block._model = sentence._model = model   # one set of weights, two engines
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    text = "I checked the calendar for tomorrow and you have three meetings."

    original = model._trim_to_eos
    captured: list[torch.Tensor] = []

    def spy(speech_tokens):
        out = original(speech_tokens)
        captured.append(out.clone())
        return out

    monkeypatch.setattr(model, "_trim_to_eos", spy)

    seed = 11
    torch.manual_seed(seed)
    block_pcm = np.concatenate(
        [pcm for _, pcm in block.synthesize_stream(text, voice, split_text=False)]
    )
    block_tokens = captured[-1]          # the tail push's trimmed tokens
    n_block_calls = len(captured)

    captured.clear()
    torch.manual_seed(seed)
    sentence_pcm = np.concatenate(
        [pcm for _, pcm in sentence.synthesize_stream(text, voice, split_text=False)]
    )
    assert len(captured) == 1, (
        f"the sentence path trimmed {len(captured)} times for one chunk; the "
        "spy is not reading what it thinks it is"
    )
    sentence_tokens = captured[0]

    assert n_block_calls > 1, (
        "the block path only trimmed once -- it did not stream, so this "
        "comparison is vacuous"
    )
    assert block_tokens.shape == sentence_tokens.shape, (
        f"block drew {tuple(block_tokens.shape)} tokens, sentence "
        f"{tuple(sentence_tokens.shape)} -- the two paths diverged from seed "
        f"{seed}"
    )
    assert torch.equal(block_tokens, sentence_tokens), (
        f"{int((block_tokens != sentence_tokens).sum())} of "
        f"{sentence_tokens.numel()} tokens differ between the engines at seed "
        f"{seed} -- something outside the vocoder fence is consuming RNG"
    )

    # Reported, not asserted: same tokens, different vocoder noise. The control
    # is the *same* tokens through s3gen.inference twice, which is the floor
    # any two renders sit at -- the CFM redraws a full-length `z` every call,
    # so each render is a fresh sample of the same distribution and the two
    # waveforms are near-uncorrelated sample-wise however identical they sound.
    # That is why the raw rel-RMS is uninformative on its own and the
    # phase-blind log-mel number next to it is the one to read.
    assert block_pcm.shape == sentence_pcm.shape, (
        f"identical tokens but {block_pcm.size} vs {sentence_pcm.size} samples"
    )
    n_cfm = int(config["generation"]["n_cfm_timesteps"])
    one_shot = [
        model.s3gen.inference(
            speech_tokens=sentence_tokens.to(model.device),
            ref_dict=model.conds.gen, n_cfm_timesteps=n_cfm,
        )[0].detach().float().cpu().numpy().reshape(-1)
        for _ in range(2)
    ]

    def rel(x, y):
        return float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y ** 2)))

    def logmel_rel(x, y):
        try:
            import librosa
        except ImportError:  # pragma: no cover - librosa ships with the model
            return float("nan")
        lm = lambda v: np.log(librosa.feature.melspectrogram(  # noqa: E731
            y=v.astype(np.float32), sr=block.sr,
            n_fft=1024, hop_length=256, n_mels=80,
        ) + 1e-8)
        a, b = lm(x), lm(y)
        k = min(a.shape[1], b.shape[1])
        return float(
            np.sqrt(np.mean((a[:, :k] - b[:, :k]) ** 2))
            / np.sqrt(np.mean(b[:, :k] ** 2))
        )

    with capsys.disabled():
        print(
            f"\n[same {sentence_tokens.numel()} tokens] block vs sentence audio: "
            f"peak |d| {float(np.abs(block_pcm - sentence_pcm).max()):.4f}, "
            f"rel RMS {rel(block_pcm, sentence_pcm):.2%}, "
            f"log-mel {logmel_rel(block_pcm, sentence_pcm):.2%}  |  CONTROL "
            f"(one-shot vs one-shot, same tokens) rel RMS "
            f"{rel(one_shot[0], one_shot[1]):.2%}, "
            f"log-mel {logmel_rel(one_shot[0], one_shot[1]):.2%}"
        )


@gpu
def test_generate_blocks_stops_at_the_eos_block(loaded):
    """No block is emitted that starts after the one carrying the stop token.

    The 2026-08-24 hallucination investigation asked whether the streamed path
    keeps vocoding past EOS and speaks out the rest of the (generous) token
    budget. It does not -- ``generate_blocks`` truncates the EOS block at the
    stop token (exclusive) and returns immediately, exactly as upstream's
    ``ChatterboxFlashT3.generate`` does. This pins that down so a future edit
    to the copied loop cannot quietly reintroduce it.
    """
    import math

    from poc_tts_streaming.engine_blockstream import generate_blocks

    engine, model, config = loaded
    stop_tok = model.t3.hp.stop_speech_token
    bs = model.t3.drf_block_size

    for seed, text in enumerate(_bench_sentences()):
        kwargs = _t3_kwargs(engine, model, text, config)
        budget = int(kwargs["total_speech_len"])
        seen: list[torch.Tensor] = []
        torch.manual_seed(seed)
        tokens = generate_blocks(model.t3, on_block=seen.append, **kwargs)
        n = int(tokens.size(1))

        assert not bool((tokens == stop_tok).any()), (
            f"{text!r}: the returned tensor still carries the stop token -- "
            "truncation must be exclusive"
        )
        assert seen, f"{text!r}: on_block was never called"
        assert int(seen[-1].size(1)) == n, (
            f"{text!r}: last callback saw {int(seen[-1].size(1))} tokens but "
            f"generate_blocks returned {n}"
        )
        # One callback per block up to and including the block EOS landed in.
        # Without EOS the loop legitimately runs the whole budget.
        expected = math.ceil(budget / bs) if n >= budget else n // bs + 1
        assert len(seen) == expected, (
            f"{text!r}: {len(seen)} blocks emitted for {n} tokens "
            f"(budget {budget}, block {bs}); expected {expected} -- a block "
            "starting after the EOS block was vocoded"
        )


@gpu
def test_streamed_samples_match_trimmed_tokens(loaded, monkeypatch):
    """Two exact properties, measured on one render:

    1. the audio the vocoder produces is exactly 960 x the tokens that survive
       ``_trim_to_eos`` -- the EOS contract this test has always pinned. It
       would still pass if the stream vocoded the post-EOS budget, which is
       what test_blockstream_engine_end_to_end cannot tell you; and
    2. what the engine *emits* is exactly ``trim_edge_silence`` applied to that
       vocoded audio -- the silence gate removes edge silence and nothing else.

    Property (1) used to be asserted on the emitted samples. Those are no
    longer the same number: the gate drops leading/trailing silence on the way
    out. Rather than relax it to ``<=`` with a hand-waved bound, it is moved
    onto the un-gated vocoder output (still exact, same meaning) and (2) pins
    the entire difference -- a strictly stronger pair than before. The un-gated
    stream is captured from the same render by spying on ``_MelWindow.push``,
    so nothing depends on two renders drawing the same tokens.

    Driven one chunk per call (``split_text=False`` over a pre-chunked text) so
    the last ``_trim_to_eos`` call is unambiguously that chunk's tail push.
    """
    from poc_tts_streaming import engine_blockstream as bs
    from poc_tts_streaming.audio import trim_edge_silence
    from poc_tts_streaming.config import voice_paths
    from poc_tts_streaming.engine_flash import chunk_text

    _engine, model, config = loaded
    gen = config.get("generation", {})
    engine = bs.BlockStreamEngine(
        engine_cfg=config.get("engine", {}),
        generation_cfg=gen,
        voice_paths=voice_paths(config),
    )
    engine._model = model            # reuse the fixture's weights, don't reload
    assert engine.trim_silence, "the shipping config must leave the gate on"
    voice = config.get("bench", {}).get("voice", "one-one.mp3")

    original = model._trim_to_eos
    trimmed_lengths: list[int] = []

    def spy(speech_tokens):
        out = original(speech_tokens)
        trimmed_lengths.append(int(out.numel()))
        return out

    monkeypatch.setattr(model, "_trim_to_eos", spy)

    vocoded: list[np.ndarray] = []
    window_push = bs._MelWindow.push

    def push_spy(self, tokens, *, finalize):
        pcm = window_push(self, tokens, finalize=finalize)
        vocoded.append(pcm)
        return pcm

    monkeypatch.setattr(bs._MelWindow, "push", push_spy)

    for seed, text in enumerate(_bench_sentences()):
        for chunk in chunk_text(text, int(gen.get("chunk_size", 120))):
            trimmed_lengths.clear()
            vocoded.clear()
            torch.manual_seed(seed)
            windows = list(engine.synthesize_stream(chunk, voice, split_text=False))
            n_tokens = trimmed_lengths[-1]
            raw = np.concatenate(vocoded)
            assert len(raw) == n_tokens * bs.SAMPLES_PER_TOKEN, (
                f"{chunk!r}: vocoded {len(raw)} samples for {n_tokens} trimmed "
                f"tokens; expected {n_tokens * bs.SAMPLES_PER_TOKEN} "
                f"({(len(raw) - n_tokens * bs.SAMPLES_PER_TOKEN) / bs.SAMPLES_PER_TOKEN:+.2f} tokens)"
            )
            emitted = np.concatenate([pcm for _, pcm in windows])
            # The engine's own knobs, not the module defaults: config.yaml
            # happens to repeat them today, and a test that hardcodes -45/120
            # would go quietly vacuous the moment it stops.
            whole = trim_edge_silence(
                raw, engine.sr,
                threshold_db=engine.trim_threshold_db,
                keep_ms=engine.trim_keep_ms,
            )
            assert np.array_equal(emitted, whole), (
                f"{chunk!r}: the gate emitted {len(emitted)} samples; a whole-chunk "
                f"trim of the same audio at threshold_db={engine.trim_threshold_db} "
                f"keep_ms={engine.trim_keep_ms} gives {len(whole)} (vocoded "
                f"{len(raw)}) -- the gate must remove edge silence and nothing else"
            )


@gpu
def test_stream_chunk_encodes_speakable_text_but_labels_the_original(loaded, monkeypatch):
    """A clause fragment off chunk_text's clause split keeps its trailing
    ``, ; :`` -- a weak EOS signal for a model trained on sentence-final
    punctuation. ``_stream_chunk`` must encode ``speakable(chunk)`` for the
    model while the emitted label stays the original chunk text, mirroring
    the sentence engine's contract (test_engine.py's
    test_synthesize_stream_normalises_clause_fragments_for_the_model).
    """
    from poc_tts_streaming import engine_blockstream as bs
    from poc_tts_streaming.config import voice_paths
    from poc_tts_streaming.engine_flash import speakable

    _engine, model, config = loaded
    engine = bs.BlockStreamEngine(
        engine_cfg=config.get("engine", {}),
        generation_cfg=config.get("generation", {}),
        voice_paths=voice_paths(config),
    )
    engine._model = model
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    chunk = "it was the age of wisdom,"
    assert speakable(chunk) == "it was the age of wisdom."

    seen: list[str] = []
    original_encode = model._encode_text

    def spy(text, **kwargs):
        seen.append(text)
        return original_encode(text, **kwargs)

    monkeypatch.setattr(model, "_encode_text", spy)

    torch.manual_seed(5)
    windows = list(engine.synthesize_stream(chunk, voice, split_text=False))

    assert seen == ["it was the age of wisdom."], (
        f"model saw {seen!r}, not the speakable() fragment"
    )
    labels = [label for label, _ in windows if label]
    assert labels == [chunk], f"emitted label(s) {labels!r} != original chunk {chunk!r}"


@gpu
def test_the_chunk_label_rides_the_first_piece_emitted(loaded, monkeypatch):
    """A chunk whose first window is silence still delivers its text.

    The gate holds a silent opening window back, so the chunk's label cannot be
    attached at vocode time -- it has to travel with the first PCM that
    actually reaches the caller. ``realtime.session._run_response`` skips
    empty-text deltas and ``bench_stream``'s ``first_chunk_chars`` reads the
    first yield, so a label stranded on a swallowed window is a sentence
    missing from the transcript.
    """
    from poc_tts_streaming import engine_blockstream as bs
    from poc_tts_streaming.audio import TRIM_THRESHOLD_DB
    from poc_tts_streaming.config import voice_paths

    _engine, model, config = loaded
    engine = bs.BlockStreamEngine(
        engine_cfg=config.get("engine", {}),
        generation_cfg=config.get("generation", {}),
        voice_paths=voice_paths(config),
    )
    engine._model = model
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    text = "Sure, the kitchen light is on."

    calls = {"n": 0}
    window_push = bs._MelWindow.push

    def push_spy(self, tokens, *, finalize):
        pcm = window_push(self, tokens, finalize=finalize)
        calls["n"] += 1
        if calls["n"] == 1 and pcm.size:
            # A full second of digital silence where the first window's audio
            # would be: the state of the real _MelWindow is untouched, only
            # what the gate sees changes.
            return np.zeros(engine.sr, dtype=np.float32)
        return pcm

    monkeypatch.setattr(bs._MelWindow, "push", push_spy)

    torch.manual_seed(3)
    windows = list(engine.synthesize_stream(text, voice, split_text=False))

    assert windows, "the chunk emitted nothing at all"
    labels = [label for label, _ in windows]
    assert labels[0] == text, f"the chunk label was lost: {labels[:3]}"
    assert all(label == "" for label in labels[1:]), (
        f"the chunk text was repeated across windows: {labels}"
    )

    keep = int(engine.sr * engine.trim_keep_ms / 1000)
    loud = np.flatnonzero(np.abs(windows[0][1]) > 10 ** (TRIM_THRESHOLD_DB / 20))
    assert loud.size, "the first emitted piece is silent -- the opener leaked"
    assert int(loud[0]) <= keep, (
        f"{int(loud[0])} silent samples before the first speech; the 24000-sample "
        f"silent opener must be cut to at most {keep}"
    )
