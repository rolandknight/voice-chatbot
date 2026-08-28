from poc_tts.engine_flash import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello there.", chunk_size=120) == ["Hello there."]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", chunk_size=120) == []


def test_splits_on_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_text(text, chunk_size=25)
    assert len(chunks) == 3
    assert chunks[0] == "First sentence here."
    assert all(not c.startswith(" ") for c in chunks)


def test_sentences_are_packed_up_to_chunk_size():
    text = "One. Two. Three. Four."
    chunks = chunk_text(text, chunk_size=120)
    assert chunks == ["One. Two. Three. Four."]


def test_sentence_longer_than_chunk_size_is_not_dropped():
    long_sentence = "word " * 60 + "end."
    chunks = chunk_text(long_sentence, chunk_size=50)
    assert len(chunks) == 1
    assert chunks[0] == long_sentence


def test_no_text_is_lost():
    text = "Alpha beta. Gamma delta. Epsilon zeta."
    chunks = chunk_text(text, chunk_size=15)
    assert " ".join(chunks).split() == text.split()
