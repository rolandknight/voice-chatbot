from pathlib import Path

from poc_qwen.bench import SENTENCES, markdown_table, run_matrix, summarize
from poc_qwen.spike_stream import measure


def test_sentences_match_poc_tts():
    src = Path(__file__).resolve().parents[2] / "poc-tts" / "poc_tts" / "bench.py"
    if not src.exists():
        return
    text = src.read_text()
    for _, sentence in SENTENCES:
        assert sentence[:40] in text


def test_run_matrix_marks_cold_and_summary_excludes_it(engine, tmp_path):
    ref = tmp_path / "v.wav"
    import soundfile as sf, numpy as np

    sf.write(ref, np.zeros(24000, dtype=np.float32), 24000)
    rows = run_matrix(engine, ref, "t", ["0.6B"], repeats=2)
    assert len(rows) == 6 and sum(r["cold"] for r in rows) == 3
    summary = summarize(rows)
    assert summary[("0.6B", "short")]["n"] == 1
    table = markdown_table(summary, ["0.6B"])
    assert "| short |" in table and "—" not in table


def test_measure():
    m = measure([(0.2, 2400), (0.5, 7680), (0.8, 7680)], 24000)
    assert m["ttfa_s"] == 0.2 and m["chunks"] == 3 and m["gap_s_median"] == 0.3
    assert measure([], 24000)["ttfa_s"] is None
