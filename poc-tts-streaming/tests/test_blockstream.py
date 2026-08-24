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


def test_block_streaming_defaults_off():
    """The spike must not be reachable from a stock checkout."""
    from poc_tts_streaming.config import load_config

    assert load_config()["engine"]["block_streaming"] is False


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


@gpu
def test_generate_blocks_matches_generate(loaded):
    """Same seed in, same speech tokens out -- hooking must be side-effect free."""
    from poc_tts_streaming.engine_blockstream import generate_blocks

    engine, model, config = loaded
    text = "I checked the calendar for tomorrow and you have three meetings."
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
        engine_cfg=config.get("engine", {}),
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
