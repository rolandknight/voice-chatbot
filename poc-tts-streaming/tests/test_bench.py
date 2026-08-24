import json
import os
from unittest.mock import MagicMock, patch

from poc_tts_streaming.bench import (
    SENTENCES,
    load_time_configs,
    main,
    record_result,
    sweep_configs,
)


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
    monkeypatch.setattr("poc_tts_streaming.bench.REPORTS", reports_path)

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

    with patch("poc_tts_streaming.config.load_config", return_value=fake_config), \
         patch("poc_tts_streaming.config.voice_paths", return_value=[tmp_path]), \
         patch("poc_tts_streaming.engine_flash.FlashEngine", side_effect=_make_fake_engine):
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


def test_load_time_configs_defaults_to_the_config_engine_section():
    """An unset environment must reproduce the pre-existing sweep exactly:
    the CUDA box runs `make bench` with no POC_TTS_BENCH_* vars set, and its
    rows have to stay comparable with the ones already in runs.jsonl."""
    config = {"engine": {"backend": "auto", "dtype": "auto"}}
    configs = load_time_configs(config, env={})
    assert configs == [
        {"backend": "auto", "dtype": "auto", "quantize_bits": None, "drf_block_size": 16},
        {"backend": "auto", "dtype": "auto", "quantize_bits": None, "drf_block_size": 32},
    ]


def test_load_time_configs_sweeps_backend_dtype_and_quantization():
    config = {"engine": {"backend": "auto", "dtype": "auto"}}
    configs = load_time_configs(
        config,
        env={
            "POC_TTS_BENCH_BACKENDS": "mlx",
            "POC_TTS_BENCH_DTYPE": "float16",
            "POC_TTS_BENCH_QUANT_BITS": ",4",
        },
    )
    assert len(configs) == 4
    assert {c["backend"] for c in configs} == {"mlx"}
    assert {c["dtype"] for c in configs} == {"float16"}
    # The empty entry in ",4" is the unquantized baseline -- losing it would
    # leave the 4-bit numbers with nothing to be compared against.
    assert [c["quantize_bits"] for c in configs] == [None, None, 4, 4]


def test_load_time_configs_keeps_quantization_an_outer_loop_axis():
    """Quantization happens when chatterbox_flash builds its MLX engine, so
    every distinct bit width needs its own model load -- it cannot be varied
    per request the way num_steps can."""
    configs = load_time_configs(
        {"engine": {"backend": "mlx", "dtype": "float16"}},
        env={"POC_TTS_BENCH_QUANT_BITS": ",4,8"},
    )
    assert len(configs) == 6
    assert len({(c["quantize_bits"], c["drf_block_size"]) for c in configs}) == 6


def test_main_records_quantization_against_the_engine_that_ran_it(tmp_path, monkeypatch):
    """Same guard as the block-size test, for the quantization axis: a 4-bit
    row must come from an engine that was built with the env var set, not
    from a neighbouring unquantized load."""
    reports_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr("poc_tts_streaming.bench.REPORTS", reports_path)
    monkeypatch.setenv("POC_TTS_BENCH_BACKENDS", "mlx")
    monkeypatch.setenv("POC_TTS_BENCH_DTYPE", "float16")
    monkeypatch.setenv("POC_TTS_BENCH_QUANT_BITS", ",4")
    monkeypatch.delenv("CHATTERBOX_FLASH_MLX_QUANT_BITS", raising=False)

    def _make_fake_engine(*, engine_cfg, generation_cfg, voice_paths):
        engine = MagicMock()
        # Tag the device with what the environment said at construction time,
        # which is when the real MLX engine reads it.
        seen = os.environ.get("CHATTERBOX_FLASH_MLX_QUANT_BITS", "none")
        engine.device = f"cpu-quant{seen}"
        engine.dtype = engine_cfg["dtype"]
        engine.backend = engine_cfg["backend"]
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

    with patch("poc_tts_streaming.config.load_config", return_value=fake_config), \
         patch("poc_tts_streaming.config.voice_paths", return_value=[tmp_path]), \
         patch("poc_tts_streaming.engine_flash.FlashEngine", side_effect=_make_fake_engine):
        main()

    rows = [json.loads(line) for line in reports_path.read_text().strip().split("\n")]
    assert len(rows) == 72
    assert {row["backend"] for row in rows} == {"mlx"}
    assert {row["dtype"] for row in rows} == {"float16"}

    quantized = [r for r in rows if "quantize_bits" in r["config"]]
    unquantized = [r for r in rows if "quantize_bits" not in r["config"]]
    assert len(quantized) == len(unquantized) == 36
    for row in quantized:
        assert row["config"]["quantize_bits"] == 4
        assert row["device"] == "cpu-quant4"
    for row in unquantized:
        assert row["device"] == "cpu-quantnone"


def test_main_leaves_no_quantization_env_var_behind(tmp_path, monkeypatch):
    """The sweep mutates a process-global env var. Leaking a 4-bit setting
    into whatever runs next in the same shell would silently quantize it."""
    monkeypatch.setattr("poc_tts_streaming.bench.REPORTS", tmp_path / "runs.jsonl")
    monkeypatch.setenv("POC_TTS_BENCH_QUANT_BITS", "4")
    monkeypatch.delenv("CHATTERBOX_FLASH_MLX_QUANT_BITS", raising=False)

    def _make_fake_engine(*, engine_cfg, generation_cfg, voice_paths):
        engine = MagicMock()
        engine.device, engine.dtype, engine.backend = "cpu", "float16", "mlx"
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

    with patch("poc_tts_streaming.config.load_config", return_value=fake_config), \
         patch("poc_tts_streaming.config.voice_paths", return_value=[tmp_path]), \
         patch("poc_tts_streaming.engine_flash.FlashEngine", side_effect=_make_fake_engine):
        main()

    assert "CHATTERBOX_FLASH_MLX_QUANT_BITS" not in os.environ
