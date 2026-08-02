from utils.explanation import explain_ai_signals

UNIFORM_TEXT = (
    "Artificial intelligence is transforming industries worldwide. "
    "Machine learning algorithms are improving business outcomes globally. "
    "Digital transformation initiatives are reshaping corporate strategies universally."
)

VARIED_HUMAN_TEXT = (
    "omg I can't believe it's already Friday?? this week flew by. "
    "Anyway I finally fixed that bug that's been driving me crazy for like three days. "
    "Coffee helped. Lots of it."
)


def test_uniform_text_produces_more_explanation_signals_than_varied_text():
    uniform_reasons = explain_ai_signals(UNIFORM_TEXT)
    human_reasons = explain_ai_signals(VARIED_HUMAN_TEXT)
    assert len(uniform_reasons) >= len(human_reasons)


def test_repetitive_text_flags_repetitive_phrasing():
    repetitive = "Machine learning is amazing. Machine learning is amazing indeed. The weather is nice."
    reasons = explain_ai_signals(repetitive)
    assert any("repetitive phrasing" in reason for reason in reasons)


def test_explanations_are_human_readable_strings():
    reasons = explain_ai_signals(UNIFORM_TEXT)
    assert all(isinstance(reason, str) and reason for reason in reasons)
