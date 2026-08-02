from utils.preprocess import clean_text, split_sentences


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Hello, Umar !!! Welcome to AI detection") == "hello umar welcome to ai detection"


def test_clean_text_collapses_whitespace():
    assert clean_text("too    many\n\nspaces") == "too many spaces"


def test_split_sentences_splits_on_terminal_punctuation():
    sentences = split_sentences("Machine learning is powerful. It learns from data! Does it reason?")
    assert sentences == [
        "Machine learning is powerful.",
        "It learns from data!",
        "Does it reason?",
    ]


def test_split_sentences_handles_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
