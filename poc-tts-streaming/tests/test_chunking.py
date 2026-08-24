from poc_tts_streaming.engine_flash import chunk_text, speakable


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


def test_overlong_sentence_splits_on_clauses_by_default():
    text = ("The door opened slowly, the corridor was dark, the air was cold; "
            "nobody had been here for years, and the dust proved it.")
    chunks = chunk_text(text, chunk_size=60)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_clause_splitting_can_be_disabled():
    text = ("The door opened slowly, the corridor was dark, the air was cold; "
            "nobody had been here for years, and the dust proved it.")
    assert chunk_text(text, chunk_size=60, split_on_clauses=False) == [text]


def test_short_sentences_are_never_clause_split():
    assert chunk_text("Yes, sir.", chunk_size=120) == ["Yes, sir."]


# --- speakable ----------------------------------------------------------


def test_speakable_leaves_period_unchanged():
    assert speakable("Hello.") == "Hello."


def test_speakable_leaves_terminal_punctuation_stacks_unchanged():
    assert speakable("Really?!") == "Really?!"


def test_speakable_leaves_terminal_punctuation_inside_quotes_unchanged():
    assert speakable('He said "go."') == 'He said "go."'


def test_speakable_leaves_terminal_punctuation_inside_parens_unchanged():
    assert speakable("(done.)") == "(done.)"


def test_speakable_treats_ellipsis_as_terminal():
    assert speakable("Wait...") == "Wait..."


def test_speakable_replaces_trailing_comma_with_period():
    assert speakable("it was the age of wisdom,") == "it was the age of wisdom."


def test_speakable_replaces_trailing_semicolon_with_period():
    assert speakable("the air was cold;") == "the air was cold."


def test_speakable_replaces_trailing_colon_with_period():
    assert speakable("here is the list:") == "here is the list."


def test_speakable_keeps_closing_quote_after_replaced_clause_mark():
    assert speakable('wisdom,"') == 'wisdom."'


def test_speakable_appends_period_when_no_punctuation():
    assert speakable("it was the age of wisdom") == "it was the age of wisdom."


def test_speakable_empty_string_unchanged():
    assert speakable("") == ""


def test_speakable_whitespace_only_unchanged():
    assert speakable("   ") == "   "
