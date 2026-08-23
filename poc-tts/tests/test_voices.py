import pytest

from poc_tts.engine_flash import discover_voices, resolve_voice_path


def test_discovers_wavs_sorted(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    (d / "zeta.wav").write_bytes(b"x")
    (d / "alpha.wav").write_bytes(b"x")
    assert discover_voices([d]) == ["alpha.wav", "zeta.wav"]


def test_ignores_non_wav_files(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    (d / "a.wav").write_bytes(b"x")
    (d / "notes.md").write_bytes(b"x")
    (d / "b.mp3").write_bytes(b"x")
    assert discover_voices([d]) == ["a.wav"]


def test_missing_directory_is_skipped_not_an_error(tmp_path):
    """The vendor clone is gitignored and may be absent entirely."""
    present = tmp_path / "voices"
    present.mkdir()
    (present / "a.wav").write_bytes(b"x")
    absent = tmp_path / "does-not-exist"
    assert discover_voices([present, absent]) == ["a.wav"]


def test_duplicate_names_across_paths_are_deduplicated(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "marvin.wav").write_bytes(b"x")
    (second / "marvin.wav").write_bytes(b"x")
    assert discover_voices([first, second]) == ["marvin.wav"]


def test_resolve_returns_first_match_in_path_order(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "marvin.wav").write_bytes(b"x")
    (second / "marvin.wav").write_bytes(b"x")
    assert resolve_voice_path("marvin.wav", [first, second]).parent == first


def test_resolve_missing_voice_names_the_paths_searched(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="voices"):
        resolve_voice_path("nope.wav", [d])
