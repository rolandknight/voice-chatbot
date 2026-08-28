"""Task 16 spike: the measurements behind the go/no-go in results-rtx-2060.md.

Committed so the call is reproducible, not because it is production code.
Everything here needs a GPU.

    python -m poc_tts_streaming.spike_analysis seams      # TTFA + seams, benched config
    python -m poc_tts_streaming.spike_analysis paired     # same tokens, windowed vs one-shot
    python -m poc_tts_streaming.spike_analysis lookahead  # what our finalize workaround changes

`seams` is the load-bearing one: it renders **the configuration the bench
actually runs** -- config.yaml's knobs, so text is chunked exactly as the
shipping path chunks it -- three times per sentence, and reports block joins
(new, introduced by block streaming) separately from chunk joins (which the
sentence-level path already has). `paired` isolates the vocoder by driving both
window-wise and one-shot vocoding from an identical token sequence and an
identical CFM noise draw; that one necessarily renders a single chunk, because
its whole point is that the two sides see the same tokens.

Seam metrics, since nobody here can listen:

  step ratio  |x[j] - x[j-1]| divided by the median |diff(x)| in the
              surrounding +-50 ms. A click is a step discontinuity. The local
              denominator matters: speech is loud in places, so a percentile
              against the whole utterance over-reports.
  jump        |rms_after - rms_before| over 10 ms either side, reported both
              raw and as a percentile of the same statistic at every other
              offset in that utterance -- the raw >20% threshold fires on
              ~40% of all offsets in ordinary speech, so it needs calibrating.
  glitch scan the step ratio evaluated everywhere, to catch an artefact that
              is not at a join at all.
  tiling      total samples must equal 960 x n_tokens (`paired` only, where
              the token count is known): catches a repeat or a truncation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from poc_tts_streaming.bench import SENTENCES
from poc_tts_streaming.config import load_config, voice_paths

SR = 24000
RMS_WIN = 240        # 10 ms either side of a join
LOCAL_HALF = 1200    # +-50 ms local control
REPORTS = Path(__file__).resolve().parent.parent / "reports"
WAVS = REPORTS / "spike-wavs"


# --- metrics -----------------------------------------------------------------

def _rms_jumps(x: np.ndarray) -> np.ndarray:
    """|rms_after - rms_before| / mean, at every offset. Vectorised."""
    e = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    idx = np.arange(RMS_WIN, len(x) - RMS_WIN)
    before = np.sqrt((e[idx] - e[idx - RMS_WIN]) / RMS_WIN)
    after = np.sqrt((e[idx + RMS_WIN] - e[idx]) / RMS_WIN)
    return np.abs(after - before) / np.maximum((after + before) / 2, 1e-9)


def _jump_at(x: np.ndarray, j: int) -> tuple[float, float, float]:
    e = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    b = float(np.sqrt((e[j] - e[j - RMS_WIN]) / RMS_WIN))
    a = float(np.sqrt((e[j + RMS_WIN] - e[j]) / RMS_WIN))
    return b, a, abs(a - b) / max((a + b) / 2, 1e-9)


def step_ratio(x: np.ndarray, j: int) -> float:
    """|step| at j over the median |diff| within +-50 ms of j."""
    d = np.abs(np.diff(x))
    lo, hi = max(0, j - LOCAL_HALF), min(len(d), j + LOCAL_HALF)
    return float(abs(x[j] - x[j - 1]) / max(float(np.median(d[lo:hi])), 1e-12))


def glitch_scan(x: np.ndarray, k: int = 5) -> tuple[np.ndarray, list[int]]:
    """Step ratio everywhere (local MEAN |diff|, so it vectorises), top-k."""
    d = np.abs(np.diff(x)).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(d)])
    i = np.arange(len(d))
    lo, hi = np.clip(i - LOCAL_HALF, 0, len(d)), np.clip(i + LOCAL_HALF, 0, len(d))
    scale = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    r = d / np.maximum(scale, 1e-10)
    return r, [int(v) for v in np.argsort(r)[-k:][::-1]]


def analyse(pcm: np.ndarray, joins: list[tuple[int, str]]) -> dict:
    """Per-join metrics plus a control drawn from the same utterance."""
    jumps_all = _rms_jumps(pcm)
    rows = []
    for j, kind in joins:
        if j < RMS_WIN or j > len(pcm) - RMS_WIN:
            continue
        b, a, jump = _jump_at(pcm, j)
        rows.append({
            "at_s": round(j / SR, 3), "kind": kind,
            "rms_before": round(b, 6), "rms_after": round(a, 6),
            "jump_pct": round(100 * jump, 1),
            "jump_pctile": round(100 * float((jumps_all < jump).mean()), 2),
            "step_ratio": round(step_ratio(pcm, j), 2),
        })
    ctrl_idx = np.linspace(LOCAL_HALF, len(pcm) - LOCAL_HALF, 40).astype(int)
    control = [step_ratio(pcm, int(i)) for i in ctrl_idx if int(i) > 0]
    r, top = glitch_scan(pcm)
    nearest = [
        (min(abs(i - j) for j, _ in joins) / SR if joins else -1.0) for i in top
    ]
    return {
        "joins": rows,
        "control_step_ratio": {
            "median": round(float(np.median(control)), 2),
            "p95": round(float(np.percentile(control, 95)), 2),
            "max": round(float(np.max(control)), 2),
        },
        "jump_over_20pct_everywhere": round(100 * float((jumps_all > 0.20).mean()), 1),
        "glitch": {
            "p99_99": round(float(np.percentile(r, 99.99)), 1),
            "max": round(float(r.max()), 1),
            "top": [{"at_s": round(i / SR, 3), "r": round(float(r[i]), 1),
                     "dist_to_join_s": round(d, 3)} for i, d in zip(top, nearest)],
        },
    }


# --- render helpers ----------------------------------------------------------

def _render(stream) -> tuple[np.ndarray, list[tuple[int, str]], float, int]:
    """Drain a synthesize_stream, timing TTFA and recording typed joins.

    A window carrying text starts a new chunk; a window carrying "" continues
    the current one (BlockStreamEngine's contract). So the join before a
    labelled window is a chunk join -- which the sentence-level path has too --
    and the join before an unlabelled window is a block join, which is new.
    """
    t0 = time.perf_counter()
    ttfa = None
    parts, joins, off = [], [], 0
    for i, (label, pcm) in enumerate(stream):
        if ttfa is None:
            ttfa = time.perf_counter() - t0
        if i:
            joins.append((off, "chunk" if label else "block"))
        parts.append(pcm)
        off += len(pcm)
    gen = time.perf_counter() - t0
    return np.concatenate(parts), joins, ttfa or gen, len(parts)


def _engines(config):
    """One set of weights, both code paths.

    BlockStreamEngine subclasses FlashEngine, so the parent's unbound
    synthesize_stream gives the sentence-level behaviour off the same loaded
    model -- no second 2 GB of VRAM, and no doubt about the two paths having
    been measured against different weights.
    """
    from poc_tts_streaming.engine_blockstream import BlockStreamEngine
    from poc_tts_streaming.engine_flash import FlashEngine

    engine = BlockStreamEngine(
        engine_cfg=config.get("engine", {}),
        generation_cfg=config.get("generation", {}),
        voice_paths=voice_paths(config),
    )
    engine.load()
    return engine, {
        "blockstream": engine.synthesize_stream,
        "sentence": lambda *a, **k: FlashEngine.synthesize_stream(engine, *a, **k),
    }


# --- subcommand: seams -------------------------------------------------------

def cmd_seams(args) -> None:
    config = load_config()
    gen = config["generation"]
    knobs = {k: gen[k] for k in ("chunk_size", "split_text", "split_on_clauses")}
    voice = config.get("bench", {}).get("voice", "one-one.mp3")
    WAVS.mkdir(parents=True, exist_ok=True)
    engine, paths = _engines(config)
    list(paths["blockstream"]("Warm up.", voice, **knobs))
    list(paths["sentence"]("Warm up.", voice, **knobs))

    out: dict[str, dict] = {}
    print(f"config: chunk_size={knobs['chunk_size']} split_text={knobs['split_text']} "
          f"split_on_clauses={knobs['split_on_clauses']} -- the benched configuration\n")
    for mode, fn in paths.items():
        for label, text in SENTENCES:
            renders = []
            for r in range(args.runs):
                torch.cuda.synchronize()
                pcm, joins, ttfa, nwin = _render(fn(text, voice, **knobs))
                torch.cuda.synchronize()
                res = analyse(pcm, joins)
                res.update(ttfa_s=round(ttfa, 4), n_windows=nwin,
                           audio_s=round(len(pcm) / SR, 3),
                           n_block_joins=sum(k == "block" for _, k in joins),
                           n_chunk_joins=sum(k == "chunk" for _, k in joins))
                renders.append(res)
                if r == 0:
                    import soundfile as sf
                    sf.write(WAVS / f"{label}-{mode}.wav", pcm, SR)
            out[f"{mode}/{label}"] = renders
            print(f"[{mode:11s}] {label:>7}: ttfa {[x['ttfa_s'] for x in renders]}  "
                  f"windows {[x['n_windows'] for x in renders]}  "
                  f"block joins {[x['n_block_joins'] for x in renders]}  "
                  f"chunk joins {[x['n_chunk_joins'] for x in renders]}", flush=True)

    print("\n=== TTFA (best of runs) ===")
    print(f"{'sentence':>8} {'sentence-level':>15} {'block-stream':>13} {'change':>9}")
    for label, _ in SENTENCES:
        s = min(x["ttfa_s"] for x in out[f"sentence/{label}"])
        b = min(x["ttfa_s"] for x in out[f"blockstream/{label}"])
        print(f"{label:>8} {s:15.3f} {b:13.3f} {100*(b-s)/s:8.1f}%")

    print("\n=== step ratio at joins, by join type, pooled over all runs ===")
    print(f"{'render':>22} {'join type':>10} {'n':>4} {'median':>7} {'max':>7}"
          f" | {'control median':>14} {'p95':>6} {'max':>6}")
    for key, renders in out.items():
        ctrl = [r["control_step_ratio"] for r in renders]
        for kind in ("block", "chunk"):
            vals = [j["step_ratio"] for r in renders for j in r["joins"] if j["kind"] == kind]
            if not vals:
                continue
            print(f"{key:>22} {kind:>10} {len(vals):4d} {np.median(vals):7.2f} {max(vals):7.2f}"
                  f" | {np.median([c['median'] for c in ctrl]):14.2f}"
                  f" {np.median([c['p95'] for c in ctrl]):6.2f}"
                  f" {max(c['max'] for c in ctrl):6.2f}")

    print("\n=== rms jump at joins, as a percentile of the same statistic everywhere ===")
    print("(raw >20% is not a usable threshold: see 'of all offsets' -- that is how")
    print(" often an ARBITRARY offset in the same audio already exceeds 20%.)")
    print(f"{'render':>22} {'join type':>10} {'n':>4} {'jump% med':>10} {'pctile med':>11}"
          f" {'pctile max':>11} {'of all offsets':>15}")
    for key, renders in out.items():
        everywhere = np.median([r["jump_over_20pct_everywhere"] for r in renders])
        for kind in ("block", "chunk"):
            js = [j for r in renders for j in r["joins"] if j["kind"] == kind]
            if not js:
                continue
            print(f"{key:>22} {kind:>10} {len(js):4d} "
                  f"{np.median([j['jump_pct'] for j in js]):9.1f}% "
                  f"{np.median([j['jump_pctile'] for j in js]):10.1f} "
                  f"{max(j['jump_pctile'] for j in js):10.1f} "
                  f"{everywhere:14.1f}%")

    print("\n=== glitch scan: worst step ratio anywhere, and how far it is from a join ===")
    for key, renders in out.items():
        r0 = renders[0]["glitch"]
        tops = ", ".join(f"{t['at_s']:.3f}s r={t['r']:.1f} (join {t['dist_to_join_s']:+.3f}s)"
                         for t in r0["top"][:3])
        print(f"{key:>22}  p99.99 {r0['p99_99']:5.1f}  max {r0['max']:5.1f}  | {tops}")

    dest = REPORTS / "spike_seams.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}; wavs in {WAVS}")
    del engine


# --- subcommand: paired ------------------------------------------------------

def cmd_paired(args) -> None:
    """Same tokens, same CFM noise: windowed vocoding vs a single-shot vocode."""
    import soundfile as sf
    from chatterbox_flash.tts import _speech_len_for_text_tokens
    from poc_tts_streaming.engine_blockstream import (
        MEL_PER_TOKEN, SAMPLES_PER_TOKEN, _MelWindow, _S3Token2Mel, generate_blocks,
    )
    from poc_tts_streaming.engine_flash import FlashEngine, resolve_voice_path

    config = load_config()
    paths = voice_paths(config)
    gen = config["generation"]
    n_cfm = int(gen["n_cfm_timesteps"])
    engine = FlashEngine(engine_cfg=config.get("engine", {}),
                         generation_cfg=gen, voice_paths=paths)
    engine.load()
    m = engine._model
    prompt = str(resolve_voice_path(config.get("bench", {}).get("voice", "one-one.mp3"), paths))
    m.generate("Warm up.", audio_prompt_path=prompt, num_steps=4,
               n_cfm_timesteps=n_cfm, backend=engine.backend)
    WAVS.mkdir(parents=True, exist_ok=True)

    print("NOTE: this subcommand renders ONE chunk per sentence on purpose -- the")
    print("point is that both sides see an identical token sequence. Chunk joins")
    print("are measured by `seams`, not here.\n")
    for label, text in SENTENCES:
        m.prepare_conditionals(prompt, exaggeration=float(gen["exaggeration"]))
        tt = m._encode_text(text, normalize_text=True)
        n_text = int(tt.size(1))
        max_tok = _speech_len_for_text_tokens(n_text)

        torch.manual_seed(args.seed)
        prefixes: list[torch.Tensor] = []
        final = generate_blocks(
            m.t3, on_block=prefixes.append, t3_cond=m.conds.t3, text_tokens=tt,
            text_token_lens=torch.tensor([n_text], device=m.device),
            total_speech_len=max_tok, num_steps=int(gen["num_steps"]),
            temperature=float(gen["temperature"]), cfg_scale=float(gen["cfg_scale"]),
            backend=engine.backend,
        )
        tok = m._trim_to_eos(final[0])
        n_tok = int(tok.numel())

        torch.manual_seed(args.seed + 1)
        win = _MelWindow(m.s3gen, m.conds.gen, n_cfm, max_tok)
        noise = win._noise
        parts, joins, off = [], [], 0
        for p in prefixes:
            pcm = win.push(m._trim_to_eos(p[0]).to(m.device), finalize=False)
            if pcm.size:
                parts.append(pcm)
                off += len(pcm)
                joins.append(off)
        tail = win.push(tok.to(m.device), finalize=True)
        if tail.size:
            parts.append(tail)
        joins = joins[: len(parts) - 1]
        streamed = np.concatenate(parts)

        mels = _S3Token2Mel.forward(
            m.s3gen, tok.unsqueeze(0), ref_wav=None, ref_sr=None,
            ref_dict=m.conds.gen, n_cfm_timesteps=n_cfm, finalize=True,
            noised_mels=noise[:, :, : n_tok * MEL_PER_TOKEN],
        ).to(dtype=m.s3gen.dtype)

        def one_shot():
            wav, _ = m.s3gen.hift_inference(mels, cache_source=None)
            wav = wav.clone().float()
            fade = m.s3gen.trim_fade.float()
            wav[:, : len(fade)] *= fade
            return wav.detach().cpu().numpy().reshape(-1)

        single, control = one_shot(), one_shot()

        print(f"=== {label} ===")
        print(f"  tiling: {n_tok} tokens -> expected {n_tok*SAMPLES_PER_TOKEN} samples; "
              f"streamed {len(streamed)}; one-shot {len(single)}  "
              f"{'OK' if len(streamed)==len(single)==n_tok*SAMPLES_PER_TOKEN else 'MISMATCH'}")
        n = min(len(streamed), len(single))
        a, b, c = streamed[:n], single[:n], control[:n]
        rel = lambda x, y: float(np.sqrt(((x - y) ** 2).mean()) / np.sqrt((y**2).mean()))
        print(f"  waveform: streamed vs one-shot rel RMS {rel(a,b):6.2%} corr "
              f"{float(np.corrcoef(a,b)[0,1]):.4f}  |  CONTROL one-shot vs one-shot "
              f"{rel(c,b):6.2%} corr {float(np.corrcoef(c,b)[0,1]):.6f}")
        try:
            import librosa
            lm = lambda x: np.log(librosa.feature.melspectrogram(
                y=x.astype(np.float32), sr=SR, n_fft=1024, hop_length=256, n_mels=80) + 1e-8)
            ma, mb, mc = lm(a), lm(b), lm(c)
            k = min(ma.shape[1], mb.shape[1], mc.shape[1])
            mrel = lambda x, y: float(np.sqrt(((x[:, :k]-y[:, :k])**2).mean())
                                      / np.sqrt((y[:, :k]**2).mean()))
            print(f"  log-mel (phase-blind, the meaningful one): streamed vs one-shot "
                  f"{mrel(ma,mb):6.2%}  |  CONTROL {mrel(mc,mb):6.2%}")
        except ImportError:
            print("  log-mel: librosa unavailable, skipped")
        res = analyse(a, [(j, "block") for j in joins])
        sr_join = [j["step_ratio"] for j in res["joins"]]
        if sr_join:
            print(f"  step ratio: joins n={len(sr_join)} median {np.median(sr_join):.2f} "
                  f"max {max(sr_join):.2f}  |  control median "
                  f"{res['control_step_ratio']['median']:.2f} p95 "
                  f"{res['control_step_ratio']['p95']:.2f} max "
                  f"{res['control_step_ratio']['max']:.2f}")
        g = res["glitch"]
        print(f"  glitch scan streamed: p99.99 {g['p99_99']:.1f} max {g['max']:.1f} | " +
              ", ".join(f"{t['at_s']:.3f}s r={t['r']:.1f} (join {t['dist_to_join_s']:+.3f}s)"
                        for t in g["top"][:3]))
        gs = analyse(b, [(j, "block") for j in joins])["glitch"]
        print(f"  glitch scan one-shot: p99.99 {gs['p99_99']:.1f} max {gs['max']:.1f} | " +
              ", ".join(f"{t['at_s']:.3f}s r={t['r']:.1f}" for t in gs["top"][:3]))
        sf.write(WAVS / f"{label}-paired-streamed.wav", streamed, SR)
        sf.write(WAVS / f"{label}-paired-singleshot.wav", single, SR)
        print()


# --- subcommand: lookahead ---------------------------------------------------

def cmd_lookahead(args) -> None:
    """What our finalize=True workaround changes versus upstream's intent.

    Upstream's broken finalize=False drops the lookahead mel positions from
    `mu` BEFORE encoder_proj and the CFM, so they never condition the retained
    frames. We keep them in the CFM and discard output frames instead, which
    gives the retained frames MORE right context. This measures how much,
    against the control of simply rerunning the truncated version (the CFM
    redraws the prompt-region noise on every call, so that is the floor).
    """
    import torch.nn.functional as F
    from chatterbox.models.s3gen.utils.mask import make_pad_mask
    from chatterbox_flash.tts import _speech_len_for_text_tokens
    from poc_tts_streaming.engine_flash import FlashEngine, resolve_voice_path

    config = load_config()
    paths = voice_paths(config)
    gen = config["generation"]
    engine = FlashEngine(engine_cfg=config.get("engine", {}),
                         generation_cfg=gen, voice_paths=paths)
    engine.load()
    m = engine._model
    prompt = str(resolve_voice_path(config.get("bench", {}).get("voice", "one-one.mp3"), paths))
    m.generate("Warm up.", audio_prompt_path=prompt, num_steps=4,
               n_cfm_timesteps=int(gen["n_cfm_timesteps"]), backend=engine.backend)

    label, text = SENTENCES[1]
    m.prepare_conditionals(prompt, exaggeration=float(gen["exaggeration"]))
    tt = m._encode_text(text, normalize_text=True)
    torch.manual_seed(args.seed)
    tok = m._trim_to_eos(m.t3.generate(
        t3_cond=m.conds.t3, text_tokens=tt,
        text_token_lens=torch.tensor([tt.size(1)], device=m.device),
        total_speech_len=_speech_len_for_text_tokens(int(tt.size(1))),
        num_steps=int(gen["num_steps"]), temperature=float(gen["temperature"]),
        cfg_scale=float(gen["cfg_scale"]), backend=engine.backend)[0])

    with torch.inference_mode():
        flow = m.s3gen.flow
        ref = m.conds.gen
        token = torch.atleast_2d(tok)
        token_len = torch.LongTensor([token.size(-1)]).to(m.device)
        emb = flow.spk_embed_affine_layer(
            F.normalize(torch.atleast_2d(ref["embedding"]), dim=1))
        tk = torch.concat([ref["prompt_token"], token], dim=1)
        tl = ref["prompt_token_len"] + token_len
        msk = (~make_pad_mask(tl)).unsqueeze(-1).to(emb)
        h, _ = flow.encoder(flow.input_embedding(tk.long()) * msk, tl)
        mel_len1 = ref["prompt_feat"].shape[1]
        look = flow.pre_lookahead_len * flow.token_mel_ratio
        noise = torch.randn(1, 80, h.shape[1], dtype=m.s3gen.dtype, device=m.device)

        def run(h_used):
            mel_len2 = h_used.shape[1] - mel_len1
            mu = flow.encoder_proj(h_used).transpose(1, 2).contiguous()
            conds = torch.zeros([1, mel_len1 + mel_len2, flow.output_size],
                                device=mu.device).to(mu.dtype)
            conds[:, :mel_len1] = ref["prompt_feat"]
            mk = (~make_pad_mask(torch.LongTensor([mel_len1 + mel_len2]).to(m.device))
                  ).unsqueeze(1).to(mu)
            feat, _ = flow.decoder(mu=mu, mask=mk, spks=emb, cond=conds.transpose(1, 2),
                                  n_timesteps=int(gen["n_cfm_timesteps"]),
                                  noised_mels=noise[:, :, :mel_len2], meanflow=True)
            return feat[:, :, mel_len1:].float()

        ours, upstream, ctrl = run(h), run(h[:, :-look]), run(h[:, :-look])

    k = upstream.size(2)
    a, b, c = ours[:, :, :k], upstream, ctrl
    den = b.pow(2).mean().sqrt().item()
    rel = lambda x: (x - b).pow(2).mean().sqrt().item() / den
    per = lambda x: ((x - b).pow(2).mean(dim=(0, 1)).sqrt() / den)
    print(f"sentence: {label!r}, {int(tok.numel())} tokens")
    print(f"mel frames: ours {ours.size(2)}, upstream-equivalent {k} (lookahead {look})")
    print(f"  ours vs upstream-truncated : rel RMS {rel(a):.4%}  corr "
          f"{float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0,1]):.6f}")
    print(f"  CONTROL truncated, rerun   : rel RMS {rel(c):.4%}  corr "
          f"{float(torch.corrcoef(torch.stack([c.flatten(), b.flatten()]))[0,1]):.6f}")
    print("  (the CFM redraws prompt-region noise per call, so the control is the floor)")
    print("  per-frame rel diff, last 8 frames -- this is where right context shows:")
    print(f"    ours    : {[f'{v:.2e}' for v in per(a)[-8:].tolist()]}")
    print(f"    control : {[f'{v:.2e}' for v in per(c)[-8:].tolist()]}")
    print(f"  first 5 frames  ours: {[f'{v:.2e}' for v in per(a)[:5].tolist()]}")
    print(f"  first 5 frames  ctrl: {[f'{v:.2e}' for v in per(c)[:5].tolist()]}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seams", help="TTFA + seam analysis on the benched (chunked) config")
    s.add_argument("--runs", type=int, default=3)
    s.set_defaults(func=cmd_seams)
    p = sub.add_parser("paired", help="windowed vs one-shot vocoding of identical tokens")
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_paired)
    l = sub.add_parser("lookahead", help="what the finalize=True workaround changes")
    l.add_argument("--seed", type=int, default=7)
    l.set_defaults(func=cmd_lookahead)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
