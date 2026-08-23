import json

from poc_tts.bench import SENTENCES, record_result, sweep_configs


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
