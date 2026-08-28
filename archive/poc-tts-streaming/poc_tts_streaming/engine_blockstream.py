"""SPIKE (Task 16): intra-sentence block streaming.

Sentence-level streaming (``engine_flash.FlashEngine.synthesize_stream``) can
only emit audio once a whole sentence has been generated *and* vocoded. This
module asks whether we can go finer: T3 fills its speech-token buffer a block
of ``drf_block_size`` tokens at a time, so in principle each finished block
could be vocoded and played while T3 is still working on the next one.

Two pieces are needed:

1. :func:`generate_blocks` -- a copy of ``ChatterboxFlashT3.generate``'s
   torch-SDPA path with an ``on_block`` callback fired every time a block is
   finalised. It is a *copy*, not a patch: the upstream method returns only
   the final tensor and there is no hook to attach to. Everything that
   consumes RNG is left byte-for-byte identical so the hooked loop reproduces
   the unhooked one for a fixed seed -- which also obliges the *callback* not
   to draw, hence :func:`_fork_rng` around the vocoder
   (``tests/test_blockstream.py``).
2. :class:`BlockStreamEngine` -- windowed vocoding of those partial token
   prefixes through S3Gen, following the CosyVoice2 streaming pattern that
   ``chatterbox/models/s3gen/s3gen.py:278`` points at.

**This is spike code.** It is gated by ``engine.block_streaming`` in
config.yaml, which defaults to ``true`` but is only effective when the
resolved engine is device=='cuda' with backend=='torch' (see
``engine_flash.block_streaming_effective``) -- server.py imports this module
only inside that branch, so it stays off the import graph everywhere else,
including the server tests. See ``results-rtx-2060.md`` for the go/no-go and
the evidence.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Callable, Iterator, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Private helpers of the installed package. Imported, never modified -- this
# module is a copy of one method's body, so it needs that method's neighbours.
from chatterbox_flash.cfg_guidance import apply_zero_text_cfg_from_logits
from chatterbox_flash.engines import build_engine
from chatterbox_flash.model import (
    _cond_emb_zero_all,
    _omnivoice_unmask_schedule,
    _pmi_count_early_step_unmask,
    _zero_text_content_keep_pad,
)
from chatterbox_flash.tts import _speech_len_for_text_tokens
from chatterbox.models.s3gen.s3gen import S3Token2Mel as _S3Token2Mel

from poc_tts_streaming.engine_flash import (
    FlashEngine,
    OutOfMemoryError,
    _vram_report,
    chunk_text,
    resolve_voice_path,
    speakable,
)

logger = logging.getLogger(__name__)


# --- Step 1: the hooked block loop -------------------------------------------

@torch.inference_mode()
def generate_blocks(
    t3,
    *,
    on_block: Callable[[Tensor], None],
    t3_cond,
    text_tokens: Tensor,
    text_token_lens: Tensor | None = None,
    total_speech_len: int = 0,
    num_steps: int = 10,
    temperature: float = 0.6,
    temperature_sampling: Literal["multinomial", "gumbel"] = "multinomial",
    time_shift_tau: float = 0.1,
    omnivoice_schedule_t_shift: float = 0.5,
    cfg_scale: float = 1.0,
    position_temperature: float = 5.0,
    pmi_uncond_prior_precompute: bool = True,
    batch_size: int = 1,
    page_size: int = 16,
    flashinfer_reserve_max_seq: int | None = None,
    use_cuda_graph: bool = True,
    backend: Literal["auto", "flashinfer", "torch"] = "auto",
) -> Tensor:
    """``ChatterboxFlashT3.generate`` with a per-block callback.

    Restricted to the single-sample, same-text case (``text_tokens`` a tensor,
    ``batch_size == 1``) -- the cross-text batch path in upstream is dead code
    for this PoC and copying it would only add divergence risk. The torch-SDPA
    engine is the only one reachable on sm_75 (see ``engine_flash``'s
    ``_flashinfer_available`` docstring), but nothing here assumes it: the
    CUDA-graph calls are kept because they are no-ops on the SDPA engine and
    removing them would change behaviour on Ampere+.

    ``on_block`` is called once per finalised block with the committed token
    prefix ``xt[0:1, :block_end]`` (a clone, safe to keep). The callback runs
    *inside* the decode loop, so it must not consume torch RNG or the tokens
    will diverge from ``generate``'s. The shipping callback vocodes, and the
    vocoder draws noise (``flow_matching.py``'s ``randn_like``, HiFTGAN's NSF
    source), so it fences every draw behind :func:`_fork_rng` -- see
    :class:`_MelWindow`. ``tests/test_blockstream.py::
    test_generate_blocks_matches_generate`` pins the invariant down with both
    a no-op callback (the loop copy) and the real vocoding one (the fence).
    """
    if isinstance(text_tokens, list):
        raise ValueError("generate_blocks handles a single sample, not a batch")
    if batch_size != 1:
        raise ValueError("generate_blocks handles batch_size=1 only")

    device = t3.device
    BS = t3.drf_block_size
    K = num_steps
    MASK = t3.mask_token_id
    dtype = next(t3.parameters()).dtype
    stop_tok = t3.hp.stop_speech_token

    B = 1
    N = total_speech_len if isinstance(total_speech_len, int) else total_speech_len[0]
    num_blocks = math.ceil(N / BS) if N > 0 else 0

    if N <= 0:
        return torch.empty((B, 0), device=device, dtype=torch.long)

    B_usr = B
    _ztb = cfg_scale > 0
    B_fwd = (2 * B_usr) if _ztb else B_usr

    cond_emb_single = t3.prepare_conditioning(t3_cond)

    prefix_emb = t3._build_prefix_emb(cond_emb_single, text_tokens, device, dtype)
    lp = prefix_emb.size(1)
    prefix_emb_cond = prefix_emb
    prefix_emb_null = None
    if _ztb:
        cond_for_null = _cond_emb_zero_all(cond_emb_single)
        text_emb = t3.text_emb(text_tokens)
        if t3.hp.input_pos_emb == "learned":
            text_emb = text_emb + t3.text_pos_emb(text_tokens)
        text_emb_zero = _zero_text_content_keep_pad(
            text_emb, text_tokens, text_token_lens,
        )
        sos = torch.full(
            (1, 1), t3.hp.start_speech_token, device=device, dtype=torch.long,
        )
        sos_emb = t3._embed_speech_tokens(sos)
        prefix_emb_null = torch.cat(
            [cond_for_null, text_emb_zero, sos_emb], dim=1,
        ).to(dtype)

    # ---- speech buffer (mask everywhere; we fill it block by block) ----
    xt = torch.full((B_fwd, N), MASK, device=device, dtype=torch.long)

    SOS = int(t3.hp.start_speech_token)
    prior_override_v: Tensor | None = None
    if pmi_uncond_prior_precompute:
        prior_override_v = t3._compute_uncond_block_prior(BS, SOS, MASK, dtype, device)

    engine_max_seq = max(lp + N + 64, flashinfer_reserve_max_seq or 0)
    cached = t3._cached_engine
    if cached is not None and cached.can_reuse(
        engine_max_seq, dtype, batch_size=B_fwd, page_size=page_size,
    ):
        engine = cached
        engine.reset()
    else:
        if (
            use_cuda_graph
            and cached is not None
            and getattr(cached, "_max_seq_len", 0) < engine_max_seq
            and getattr(cached, "_batch_size", -1) == B_fwd
            and getattr(cached, "_dtype", None) == dtype
        ):
            engine_max_seq = max(
                ((engine_max_seq + 511) // 512) * 512,
                int(getattr(cached, "_max_seq_len", 0) * 1.5),
            )
        engine = build_engine(
            t3, engine_max_seq, dtype,
            backend=backend, batch_size=B_fwd, page_size=page_size,
        )

    if _ztb:
        assert prefix_emb_null is not None
        pfx_for_engine = [
            *[prefix_emb_cond[bi : bi + 1] for bi in range(B_usr)],
            *[prefix_emb_null[bi : bi + 1] for bi in range(B_usr)],
        ]
        shift_ctx = engine.prefix_forward(pfx_for_engine)
    else:
        shift_ctx = engine.prefix_forward(prefix_emb_cond)

    if use_cuda_graph:
        engine.capture_cuda_graph(BS, speech_head=t3.speech_head)
    t3._cached_engine = engine

    block_marginal_priors: list[Tensor | None] = [None] * B_usr

    for b_idx in range(num_blocks):
        bs_ = b_idx * BS
        be_ = min(bs_ + BS, N)
        bl = be_ - bs_

        _omnivoice_sched = _omnivoice_unmask_schedule(
            n_total_mask=bl,
            num_steps=K,
            t_shift=float(omnivoice_schedule_t_shift),
        )

        for k in range(K):
            cache_start = lp + bs_
            sp_blk = t3._embed_speech_tokens(xt[:, bs_:be_]).to(dtype)

            use_graph = engine.has_cuda_graph and bl == BS
            if use_graph:
                if k == 0:
                    engine.set_shift_ctx(shift_ctx)
                blk_hidden, graph_logits = engine.block_forward_graph(sp_blk, cache_start)
                if graph_logits is not None:
                    blk_logits = graph_logits
                else:
                    shift_hidden = torch.cat([shift_ctx, blk_hidden[:, : bl - 1]], dim=1)
                    blk_logits = t3.speech_head(shift_hidden)
            else:
                blk_hidden = engine.block_forward(sp_blk, cache_start)
                shift_hidden = torch.cat([shift_ctx, blk_hidden[:, : bl - 1]], dim=1)
                blk_logits = t3.speech_head(shift_hidden)

            pmi_cfg_probs_c: Tensor | None = None
            pmi_cfg_probs_u: Tensor | None = None
            if _ztb:
                logits_cond = blk_logits[:B_usr]
                logits_uncond = blk_logits[B_usr : 2 * B_usr]
                pmi_cfg_probs_c = F.softmax(logits_cond, dim=-1)
                pmi_cfg_probs_u = F.softmax(logits_uncond, dim=-1)
                blk_logits = apply_zero_text_cfg_from_logits(
                    logits_cond, logits_uncond, cfg_scale,
                )

            step_break = False
            for bi in range(B_usr):
                pmi_c_bi = None if pmi_cfg_probs_c is None else pmi_cfg_probs_c[bi]
                pmi_u_bi = None if pmi_cfg_probs_u is None else pmi_cfg_probs_u[bi]
                r = _pmi_count_early_step_unmask(
                    blk_logits[bi],
                    xt[bi, bs_:be_],
                    bl=bl,
                    k=k, K=K,
                    MASK=MASK, device=device,
                    temperature=temperature,
                    temperature_sampling=temperature_sampling,
                    omnivoice_unmask_schedule_k=_omnivoice_sched[k],
                    time_shift_tau=time_shift_tau,
                    prior_override_v=prior_override_v,
                    block_marginal_prior=block_marginal_priors[bi],
                    pmi_cfg_probs_c_bl=pmi_c_bi,
                    pmi_cfg_probs_u_bl=pmi_u_bi,
                    cfg_scale_for_pmi=cfg_scale,
                    position_temperature=position_temperature,
                )
                xt[bi, bs_:be_] = r["xt_step"]
                block_marginal_priors[bi] = r["block_marginal_prior"]
                if _ztb:
                    xt[B_usr + bi, bs_:be_] = r["xt_step"]
                row_hit_eos = bool((xt[bi, bs_:be_] == stop_tok).any().item())
                if r["n_mask"] == 0 or row_hit_eos:
                    step_break = True
            if step_break:
                break

        # Mid-block finalize: write the committed (possibly partial) tokens of
        # this block to the KV cache so the next block sees clean context.
        if b_idx < num_blocks - 1:
            fin_emb = t3._embed_speech_tokens(xt[:, bs_:be_]).to(dtype)
            fin_hidden = engine.block_forward(fin_emb, lp + bs_)
            engine.advance_cache(bl)
            shift_ctx = fin_hidden[:, bl - 1 : bl, :].clone()

        # --- the hook. Placed after the block is committed to the KV cache so
        # the prefix handed out is exactly what the next block will condition
        # on. EOS truncation is applied here too, so the last callback carries
        # the same tensor generate() would have returned.
        eos_pos = (xt[0, bs_:be_] == stop_tok).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            out = xt[:B_usr, : bs_ + eos_pos[0].item()]
            on_block(out.clone())
            return out
        on_block(xt[:B_usr, :be_].clone())

    return xt[:B_usr]


# --- Step 2: windowed vocoding -----------------------------------------------

# S3 speech tokens are 25 Hz; S3Gen's flow emits token_mel_ratio=2 mel frames
# per token and HiFTGenerator upsamples 480 samples per mel frame at 24 kHz.
# 2 * 480 = 960 samples per token -- asserted against a real run in
# tests/test_blockstream.py rather than trusted blind.
MEL_PER_TOKEN = 2
SAMPLES_PER_MEL = 480
SAMPLES_PER_TOKEN = MEL_PER_TOKEN * SAMPLES_PER_MEL

# CosyVoice2's streaming vocoder (cosyvoice/cli/model.py, mel_cache_len=8 at
# 24 kHz) keeps a short mel/source/speech overlap across window boundaries.
# Three caches, all of them load-bearing:
#   * mel    -- the last OVERLAP_MEL frames are re-vocoded with the new window
#               so HiFTGAN sees left context;
#   * source -- seeds the NSF sine generator, which otherwise restarts its
#               phase accumulator (theta = cumsum(f0)) at every call;
#   * speech -- the previous window's rendering of the overlap region, which is
#               cross-faded with the new one. Seeding the source only fixes the
#               phase for the cached samples; the fresh cumsum picks up an
#               unrelated phase where the cache ends, so without the cross-fade
#               there is a step discontinuity exactly at each join. Measured:
#               |step| at the joins was 9-45x the local median |diff| before
#               this was added (see results-rtx-2060.md).
OVERLAP_MEL = 8
OVERLAP_SAMPLES = OVERLAP_MEL * SAMPLES_PER_MEL

# The flow encoder looks ahead pre_lookahead_len=3 tokens, so the last 3 tokens
# of a non-final window (6 mel frames) are generated without their real right
# context. Upstream's `finalize=False` exists to withhold them -- but it is
# broken in chatterbox-tts 0.1.7: flow.py:171 truncates `h` and then multiplies
# it by a `mask` built from the UNtruncated `h_lengths`, so the call dies with
#     RuntimeError: The size of tensor a (558) must match the size of tensor b
#     (564) at non-singleton dimension 2
# So we pass finalize=True and drop the trailing frames from the OUTPUT instead.
#
# That is NOT the same computation, and the difference is in our favour.
# Upstream truncates `h` *before* encoder_proj and the CFM decoder, so the
# lookahead positions never enter `mu` and cannot condition the frames that are
# kept. We leave them in, let the CFM denoise the full length, and discard
# output frames -- so our retained frames see more right context, not less.
# Measured (`spike_analysis lookahead`): ours vs an upstream-equivalent
# truncation differs by 0.77 % rel RMS against a 0.64 % floor (the CFM redraws
# prompt-region noise every call), and the per-frame profile localises it --
# the last ~5 frames differ by 3-7x the control while earlier frames sit at the
# control level. Right context reaching the window's tail is exactly what one
# would expect to change, and it is a small effect.
LOOKAHEAD_MEL = 6

# A non-final window must survive the lookahead truncation AND leave a full
# OVERLAP_MEL tail to hand to the next one: 2n - LOOKAHEAD_MEL >= OVERLAP_MEL.
MIN_WINDOW_TOKENS = (OVERLAP_MEL + LOOKAHEAD_MEL) // MEL_PER_TOKEN


def _fork_rng(device: torch.device | str):
    """Fence a vocoder call off from the global torch RNG.

    Vocoding draws noise, and on this path it happens *inside* the T3 decode
    loop -- so without a fence every window shifts the RNG stream the loop's
    later blocks sample from, and a block-streamed utterance is a different
    draw from the sentence path's off the same seed. Three unavoidable draws
    (all in the installed package, none of which takes a ``generator``):

    * ``chatterbox/models/s3gen/flow_matching.py:216`` -- ``z =
      torch.randn_like(mu)``, executed unconditionally and *then* overwritten
      in the non-prompt region when ``noised_mels`` is supplied, so passing
      fixed noise does not stop the draw;
    * ``chatterbox/models/s3gen/hifigan.py:226,282`` -- the NSF source's
      per-call ``torch.randn_like`` (plus a ``Uniform.sample`` for the sine
      phase at :212).

    ``torch.randn_like`` has no ``generator`` argument, so routing the vocoder
    at a private ``torch.Generator`` would mean forking the package. Saving and
    restoring the global state around the call is the practical fence:
    ``fork_rng`` snapshots the CPU generator and the given CUDA device's, and
    puts both back on exit. Inside the fence the vocoder still gets
    deterministic noise for a fixed seed (it sees whatever state the loop is
    at); outside it, the loop's stream is exactly the one ``t3.generate``
    would have walked.

    One fence per window (and one for the noise pre-draw), not one per
    utterance: the T3 draws must stay *outside* it. Cost is a 5056-byte CPU
    state copy plus a 16-byte CUDA one each way -- 31.8 us measured on the
    RTX 2060, 0.024 % of that box's 130 ms mean window, 0.22 ms over a 5.5 s
    utterance.

    Not thread-safe, and cannot be: the generators it saves and restores are
    process-global. One utterance at a time is the engine's shape today (one
    CUDA stream, one worker thread per ``_stream_chunk``); two concurrent
    renders would interleave their fences and corrupt each other's streams --
    but they would already be racing the same generators without it.
    """
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] \
        if dev.type == "cuda" else []
    return torch.random.fork_rng(devices=devices, enabled=True)


class _Cancelled(Exception):
    """Unwinds the T3 loop from inside on_block when the caller cancels."""


class _MelWindow:
    """Accumulates flow output and hands out only newly finalised samples.

    Each ``push`` re-runs the flow over the whole token prefix -- the meanflow
    CFM in this build has no incremental cache (``CausalConditionalCFM`` dropped
    ``flow_cache``) -- and emits only the mel frames that are new. That would
    normally make each window a fresh stochastic draw, so the frames already
    emitted would not be the ones the new draw continues. The fix is to fix the
    noise: one ``randn`` for the longest possible utterance, sliced per call, so
    every window is a prefix of the *same* CFM trajectory. That requires
    bypassing ``flow_inference`` (which redraws unconditionally) and calling
    ``S3Token2Mel.forward`` with an explicit ``noised_mels``.

    Every draw this class makes -- the pre-draw below and whatever the vocoder
    itself consumes per window -- happens inside :func:`_fork_rng`, because
    ``push`` is called from inside the T3 decode loop and must leave that
    loop's RNG stream untouched.
    """

    def __init__(self, s3gen, ref_dict, n_cfm_timesteps: int, max_tokens: int):
        self._s3gen = s3gen
        self._ref = ref_dict
        self._n_cfm = n_cfm_timesteps
        self._vocoded_mel = 0
        self._cache_mel: Tensor | None = None
        self._cache_source: Tensor | None = None
        self._cache_speech: Tensor | None = None
        self._first = True
        with _fork_rng(s3gen.device):
            self._noise = torch.randn(
                1, 80, max(max_tokens, 1) * MEL_PER_TOKEN,
                dtype=s3gen.dtype, device=s3gen.device,
            )
        # np.hamming(2 * source_cache_len) in CosyVoice; the two halves are the
        # fade-in for the new rendering and the fade-out for the cached one.
        self._xfade = torch.hamming_window(
            2 * OVERLAP_SAMPLES, periodic=False,
            dtype=torch.float32, device=s3gen.device,
        )

    def push(self, tokens: Tensor, *, finalize: bool) -> np.ndarray:
        """Vocode the token prefix and return the new float32 samples.

        The whole vocode runs inside one :func:`_fork_rng`, so a window costs
        the T3 loop that called it nothing in RNG state. One fence for both
        draws rather than one each: the CFM's ``randn_like`` and HiFTGAN's
        source noise then see the same relative stream they always did, so the
        only thing the fence changes about this window's *audio* is where its
        noise starts.
        """
        n = int(tokens.numel())
        if n == 0 or (not finalize and n < MIN_WINDOW_TOKENS):
            return np.zeros(0, dtype=np.float32)
        with _fork_rng(self._s3gen.device):
            return self._vocode(tokens, n, finalize=finalize)

    def _vocode(self, tokens: Tensor, n: int, *, finalize: bool) -> np.ndarray:
        """``push``'s body. Called only from inside the RNG fence."""
        mels = _S3Token2Mel.forward(
            self._s3gen,
            tokens.unsqueeze(0) if tokens.dim() == 1 else tokens,
            ref_wav=None,
            ref_sr=None,
            ref_dict=self._ref,
            n_cfm_timesteps=self._n_cfm,
            finalize=True,
            noised_mels=self._noise[:, :, : n * MEL_PER_TOKEN],
        ).to(dtype=self._s3gen.dtype)
        total_mel = mels.size(2) if finalize else mels.size(2) - LOOKAHEAD_MEL
        if total_mel <= self._vocoded_mel:
            return np.zeros(0, dtype=np.float32)

        new_mel = mels[:, :, self._vocoded_mel : total_mel]
        if self._cache_mel is not None:
            mel_in = torch.cat([self._cache_mel, new_mel], dim=2)
        else:
            mel_in = new_mel

        wav, source = self._s3gen.hift_inference(
            mel_in, cache_source=self._cache_source,
        )
        wav = wav.float()

        if self._cache_speech is not None:
            # Cross-fade this window's rendering of the overlap region with the
            # previous window's rendering of the same region.
            head = wav[:, :OVERLAP_SAMPLES] * self._xfade[:OVERLAP_SAMPLES]
            head = head + self._cache_speech * self._xfade[OVERLAP_SAMPLES:]
            wav = torch.cat([head, wav[:, OVERLAP_SAMPLES:]], dim=1)
        elif self._first:
            # Same spillover guard s3gen.inference() applies to a whole
            # utterance -- only the very first window gets it.
            fade = self._s3gen.trim_fade.float()
            wav = torch.cat(
                [wav[:, : len(fade)] * fade, wav[:, len(fade) :]], dim=1,
            )
            self._first = False

        if finalize:
            out = wav
            self._vocoded_mel = total_mel
            self._cache_mel = self._cache_source = self._cache_speech = None
        else:
            # Hold the tail back: the next window re-renders it with real right
            # context and cross-fades over the seam.
            out = wav[:, :-OVERLAP_SAMPLES]
            self._cache_speech = wav[:, -OVERLAP_SAMPLES:].clone()
            self._cache_mel = mel_in[:, :, -OVERLAP_MEL:].clone()
            self._cache_source = source[:, :, -OVERLAP_SAMPLES:].clone()
            self._vocoded_mel = total_mel
        if out.numel() == 0:
            return np.zeros(0, dtype=np.float32)
        return out.detach().cpu().numpy().reshape(-1)


class BlockStreamEngine(FlashEngine):
    """FlashEngine whose ``synthesize_stream`` emits sub-sentence windows.

    Yields ``(text, pcm)`` like the parent -- ordered, non-overlapping mono
    float32 at ``self.sr`` -- but a chunk now arrives as several windows
    instead of one, each covering only the tokens T3 finalised in that block.

    **Edge silence is gated, not trimmed.** The parent can trim a finished
    chunk because it holds the whole thing; here a runaway silent tail is
    spread over many windows that have already been handed to the caller by
    the time the chunk ends. So one ``audio.TrailingSilenceGate`` per chunk
    sits on the emission path: it releases each window's speech immediately,
    buffers silence until it knows whether more speech follows, and at chunk
    end keeps only ``trim_keep_ms`` of what is left. It runs on already
    cross-faded PCM and never alters a sample, so window continuity is
    untouched -- and what it emits for a chunk is byte-identical to
    ``trim_edge_silence`` over that chunk's concatenation.

    **The text field is not per-window.** The parent yields the chunk's text
    with the chunk's audio, and every consumer treats that string as new
    transcript: ``realtime.session._run_response`` appends it to the response
    transcript and sends it as an ``output_audio_transcript.delta``. Repeating
    the chunk text on each of its windows would repeat the sentence four to six
    times in the transcript. So the contract here is: **the first piece a
    chunk actually emits carries that chunk's text, every later piece of the
    same chunk carries ``""``** -- "first emitted", not "first vocoded",
    because the silence gate above may swallow an opening window whole.
    Consumers must treat an empty string as "same text, more audio" -- see the
    ``if chunk_text:`` guard in ``_run_response``. Total
    yielded text over an utterance is therefore identical between the two
    engines, which is what keeps ``bench_stream``'s ``chars`` honest.
    """

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
        if not self.loaded:
            raise RuntimeError("model is not loaded -- call load() first")
        gen = self._generation_cfg
        model = self._model
        prompt = str(resolve_voice_path(voice, self._voice_paths))
        if split_text:
            chunks = chunk_text(text, chunk_size, split_on_clauses=split_on_clauses)
        else:
            chunks = [t for t in [text.strip()] if t]
        if not chunks:
            raise ValueError("text is empty")

        pick = lambda given, key: given if given is not None else gen[key]  # noqa: E731
        exagg = pick(exaggeration, "exaggeration")
        n_cfm = int(pick(n_cfm_timesteps, "n_cfm_timesteps"))

        for chunk in chunks:
            if cancel is not None and cancel.is_set():
                return
            yield from self._stream_chunk(
                chunk, prompt,
                exaggeration=exagg, n_cfm=n_cfm,
                num_steps=int(pick(num_steps, "num_steps")),
                temperature=float(pick(temperature, "temperature")),
                cfg_scale=float(pick(cfg_scale, "cfg_scale")),
                cancel=cancel,
            )

    def _stream_chunk(
        self,
        chunk: str,
        prompt: str,
        *,
        exaggeration: float,
        n_cfm: int,
        num_steps: int,
        temperature: float,
        cfg_scale: float,
        cancel: threading.Event | None,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Generate one text chunk, vocoding each T3 block as it lands.

        The vocoding happens *inside* ``on_block``, not after ``generate_blocks``
        returns -- that is the whole point of the spike, and it is why this runs
        on a worker thread: a callback cannot ``yield`` into a generator, so the
        producer pushes finished windows onto a queue and the generator drains
        it. Everything stays on one CUDA stream, so T3 and S3Gen serialise on
        the GPU anyway; the thread only bridges the callback/generator gap.

        The model is handed ``speakable(chunk)``, not ``chunk`` itself --
        mirroring ``FlashEngine.synthesize_stream`` -- so a clause fragment's
        trailing ``, ; :`` does not reach T3 as a weak EOS signal. ``chunk``
        (the original) is still what ``put`` below labels emitted audio with.
        """
        model = self._model
        model.prepare_conditionals(prompt, exaggeration=exaggeration)
        text_tokens = model._encode_text(speakable(chunk), normalize_text=True)
        n_text = int(text_tokens.size(1))
        max_tokens = _speech_len_for_text_tokens(n_text)
        window = _MelWindow(model.s3gen, model.conds.gen, n_cfm, max_tokens)

        gate = self._silence_gate()

        out: "queue.Queue[tuple[str, np.ndarray] | None | BaseException]" = queue.Queue()
        state = {"emitted": 0}

        def put(pcm: np.ndarray) -> None:
            """Queue one piece, labelling only the first this chunk emits.

            The label rides the first PCM that actually reaches the caller, not
            the first window vocoded -- the gate may hold an opening window back
            (a chunk that starts with silence) and the chunk's text has to
            travel with whatever comes out first, or the transcript loses a
            sentence. Later pieces carry "" ("same text, more audio").
            """
            if not pcm.size:
                return
            first = state["emitted"] == 0
            state["emitted"] += 1
            out.put((chunk if first else "", pcm))

        def emit(pcm: np.ndarray) -> None:
            """Send a vocoded window through this chunk's silence gate."""
            if gate is None:
                put(pcm)
                return
            for piece in gate.push(pcm):
                put(piece)

        def on_block(tokens: Tensor) -> None:
            if cancel is not None and cancel.is_set():
                raise _Cancelled()
            trimmed = model._trim_to_eos(tokens[0])
            emit(window.push(trimmed.to(self.device), finalize=False))

        def produce() -> None:
            try:
                final = generate_blocks(
                    model.t3,
                    on_block=on_block,
                    t3_cond=model.conds.t3,
                    text_tokens=text_tokens,
                    text_token_lens=torch.tensor([n_text], device=self.device),
                    total_speech_len=max_tokens,
                    num_steps=num_steps,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    backend=self.backend,
                )
                # Tail: re-run the flow with finalize=True so the last
                # pre_lookahead_len tokens (6 mel frames) that every
                # finalize=False call withholds actually get vocoded.
                trimmed = model._trim_to_eos(final[0])
                if trimmed.numel():
                    emit(window.push(trimmed.to(self.device), finalize=True))
                # Chunk over: the gate now knows its buffered silence is
                # trailing, not a pause, and gives back at most keep_ms of it.
                if gate is not None:
                    for piece in gate.flush():
                        put(piece)
            except _Cancelled:
                pass
            except torch.cuda.OutOfMemoryError as exc:
                out.put(OutOfMemoryError(
                    f"ran out of VRAM during generation. {_vram_report()}"
                ))
            except BaseException as exc:  # noqa: BLE001 -- re-raised on the consumer
                out.put(exc)
            finally:
                out.put(None)

        worker = threading.Thread(target=produce, name="blockstream", daemon=True)
        worker.start()
        try:
            while True:
                item = out.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            worker.join(timeout=60)
