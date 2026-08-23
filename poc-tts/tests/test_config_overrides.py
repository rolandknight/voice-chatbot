"""Machine-specific engine overrides.

config.yaml is shared between the CUDA box and the Mac. The Mac needs
`backend: mlx` + `dtype: float16` and CUDA must never see them, so the
selection has to come from the environment rather than the committed file.
"""

from poc_tts.config import apply_engine_overrides


BASE = {"engine": {"device": "auto", "dtype": "auto", "backend": "auto", "drf_block_size": 16}}


def test_unset_environment_changes_nothing():
    """The CUDA box sets none of these; its behaviour must be untouched."""
    assert apply_engine_overrides(BASE, env={}) == BASE


def test_empty_values_are_ignored():
    """An exported-but-empty var in a .env file must not be read as a request
    for the empty-string backend, which would fail resolution."""
    env = {"POC_TTS_ENGINE_BACKEND": "", "POC_TTS_ENGINE_DTYPE": "   "}
    assert apply_engine_overrides(BASE, env=env) == BASE


def test_overrides_select_the_mac_metal_path():
    env = {"POC_TTS_ENGINE_BACKEND": "mlx", "POC_TTS_ENGINE_DTYPE": "float16"}
    merged = apply_engine_overrides(BASE, env=env)
    assert merged["engine"]["backend"] == "mlx"
    assert merged["engine"]["dtype"] == "float16"


def test_overrides_leave_unmentioned_engine_keys_alone():
    """drf_block_size has no override var and must survive the merge."""
    merged = apply_engine_overrides(BASE, env={"POC_TTS_ENGINE_BACKEND": "mlx"})
    assert merged["engine"]["drf_block_size"] == 16
    assert merged["engine"]["device"] == "auto"


def test_overrides_do_not_mutate_the_input():
    env = {"POC_TTS_ENGINE_BACKEND": "mlx"}
    original = {"engine": dict(BASE["engine"])}
    apply_engine_overrides(original, env=env)
    assert original["engine"]["backend"] == "auto"


def test_overrides_work_when_config_has_no_engine_section():
    merged = apply_engine_overrides({}, env={"POC_TTS_ENGINE_DEVICE": "cpu"})
    assert merged["engine"] == {"device": "cpu"}


def test_values_are_passed_through_for_the_resolvers_to_validate():
    """apply_engine_overrides is not a second validation site -- resolve_*
    in engine_flash.py owns that, and duplicating the allowed-value list here
    would be a second place to forget to update."""
    merged = apply_engine_overrides(BASE, env={"POC_TTS_ENGINE_BACKEND": "nonsense"})
    assert merged["engine"]["backend"] == "nonsense"
