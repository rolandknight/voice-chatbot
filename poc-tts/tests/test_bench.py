import json
from unittest.mock import MagicMock, patch

from poc_tts.bench import SENTENCES, main, record_result, sweep_configs


def test_sweep_is_the_cartesian_product():
    grid = {"num_steps": [4, 10], "n_cfm_timesteps": [1, 2]}
    configs = sweep_configs(grid)
    assert len(configs) == 4
    assert {"num_steps": 4, "n_cfm_timesteps": 1} in configs


def test_sweep_of_single_valued_axes_is_one_config():
    assert len(sweep_configs({"num_steps": [10]})) == 1


def test_sentences_match_the_recorded_baselines():
    """These three are what the Turbo CUDA, Flash CUDA, and Turbo CPU rows
    were measured on. Changing them invalidates the comparison."""
    assert [name for name, _ in SENTENCES] == ["short", "medium", "long"]
    assert len(dict(SENTENCES)["long"]) > 300


def test_record_result_appends_one_json_line(tmp_path):
    path = tmp_path / "runs.jsonl"
    record_result(path, {"rtf": 0.58, "backend": "torch"})
    record_result(path, {"rtf": 0.72, "backend": "torch"})
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1])["rtf"] == 0.72


def test_record_result_creates_parent_directory(tmp_path):
    path = tmp_path / "reports" / "runs.jsonl"
    record_result(path, {"rtf": 0.5})
    assert path.exists()


def test_main_writes_36_rows_correctly_attributed_to_their_block_size(tmp_path, monkeypatch):
    """Guards the restructure: main() loads the engine once per drf_block_size
    and reuses it across the num_steps x n_cfm_timesteps x sentence grid, so a
    row must carry the drf_block_size of the engine that actually produced it
    -- not a stale or mismatched value from a different outer-loop iteration.

    Each fake engine tags its own `device` with the block size it was built
    with, so a row is only consistent if row["device"] matches
    f"cpu-block{row['config']['drf_block_size']}" -- i.e. the block size
    recorded in the row's config is the block size of the engine that
    actually ran it, not just a label that happens to be nearby.
    No real model weights or CUDA allocations are touched: FlashEngine,
    load_config, and voice_paths are all mocked.
    """
    reports_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr("poc_tts.bench.REPORTS", reports_path)

    def _make_fake_engine(*, engine_cfg, generation_cfg, voice_paths):
        block_size = engine_cfg["drf_block_size"]
        engine = MagicMock()
        engine.device = f"cpu-block{block_size}"
        engine.dtype = "float16"
        engine.backend = "torch"
        engine.synthesize.return_value = ([0.0] * 2400, 24000)
        return engine

    fake_config = {
        "engine": {"device": "cpu", "dtype": "auto", "backend": "auto"},
        "generation": {
            "temperature": 0.6, "exaggeration": 0.5, "cfg_scale": 1.0,
            "num_steps": 10, "n_cfm_timesteps": 2,
        },
        "bench": {"voice": "test.wav"},
    }

    with patch("poc_tts.config.load_config", return_value=fake_config), \
         patch("poc_tts.config.voice_paths", return_value=[tmp_path]), \
         patch("poc_tts.engine_flash.FlashEngine", side_effect=_make_fake_engine):
        main()

    rows = [json.loads(line) for line in reports_path.read_text().strip().split("\n")]
    assert len(rows) == 36

    counts = {16: 0, 32: 0}
    for row in rows:
        cfg = row["config"]
        block_size = cfg["drf_block_size"]
        assert block_size in counts
        counts[block_size] += 1
        assert "num_steps" in cfg
        assert "n_cfm_timesteps" in cfg
        # The engine that produced this row must be the one built for this
        # block size -- catches a row silently recording the wrong block size.
        assert row["device"] == f"cpu-block{block_size}"
    assert counts == {16: 18, 32: 18}
