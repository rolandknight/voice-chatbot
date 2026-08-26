from qwen_tts.text import chunk_text, split_sentences


def test_split_basic():
    assert split_sentences("Hello there. How are you? Fine!") == ["Hello there.", "How are you?", "Fine!"]


def test_decimals_and_abbreviations_survive():
    assert split_sentences("Meet Dr. Smith at 9.15 today. Ok.") == ["Meet Dr. Smith at 9.15 today.", "Ok."]


def test_chunk_groups_up_to_max_chars():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_text(text, max_chars=120)
    assert len(chunks) >= 3
    assert all(len(c) <= 120 for c in chunks)
    assert " ".join(chunks) == text


def test_single_long_sentence_is_one_chunk():
    long = "word " * 100
    assert chunk_text(long.strip(), max_chars=50) == [long.strip()]


def test_empty():
    assert chunk_text("   ") == []
