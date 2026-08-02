import pickle

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from models import ai_text_detector
from models.ai_detectors import sklearn_backend
from models.ai_text_detector import clear_model_cache, detect_ai_text

UNIFORM_TEXT = (
    "Artificial intelligence is transforming industries worldwide. "
    "Machine learning algorithms are improving business outcomes globally. "
    "Digital transformation initiatives are reshaping corporate strategies universally. "
    "Cloud computing solutions are optimizing organizational efficiency broadly."
)

VARIED_HUMAN_TEXT = (
    "omg I can't believe it's already Friday?? this week flew by. "
    "Anyway I finally fixed that bug that's been driving me crazy for like three days. "
    "Coffee helped. Lots of it. "
    "Gonna go take a nap now honestly, I earned it."
)


@pytest.fixture(autouse=True)
def _isolated_model_path(tmp_path, monkeypatch):
    """Every test gets its own MODEL_PATH so it doesn't depend on whether a
    real trained model happens to exist on disk. sklearn_backend.MODEL_PATH
    is what SklearnDetector actually reads (as a default constructor
    argument), so that's what must be patched -- patching the re-exported
    name on ai_text_detector alone would silently leave the real on-disk
    path in effect. The lru_cache is cleared before and after so tests can't
    leak state into each other.
    """
    fake_path = tmp_path / "ai_detector.pkl"
    monkeypatch.setattr(sklearn_backend, "MODEL_PATH", fake_path)
    monkeypatch.setattr(ai_text_detector, "MODEL_PATH", fake_path)
    clear_model_cache()
    yield
    clear_model_cache()


def test_heuristic_scores_uniform_low_diversity_text_higher_than_varied_text():
    # No trained model at MODEL_PATH, so this exercises the heuristic path.
    uniform_score, _ = detect_ai_text(UNIFORM_TEXT)
    human_score, _ = detect_ai_text(VARIED_HUMAN_TEXT)
    assert uniform_score > human_score


def test_heuristic_score_and_label_are_consistent():
    score, label = detect_ai_text(VARIED_HUMAN_TEXT)
    assert 0.0 <= score <= 100.0
    assert label in {"AI", "Human"}
    assert (label == "AI") == (score >= 50)


def test_trained_model_is_used_when_present(tmp_path, monkeypatch):
    # Train a tiny real classifier and confirm detect_ai_text_detailed picks
    # it up (backend == "sklearn") instead of falling back to heuristic --
    # this is exactly the auto-upgrade behavior a config regression could
    # silently break, so assert on `backend`, not just score/label shape.
    texts = [UNIFORM_TEXT, "Corporate synergy drives scalable digital growth initiatives."] * 3 + \
            [VARIED_HUMAN_TEXT, "lol my cat just knocked my coffee off the desk again"] * 3
    labels = ["ai"] * 6 + ["human"] * 6

    vectorizer = TfidfVectorizer()
    estimator = LogisticRegression(max_iter=1000).fit(vectorizer.fit_transform(texts), labels)

    model_path = tmp_path / "ai_detector.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "estimator": estimator}, handle)

    monkeypatch.setattr(sklearn_backend, "MODEL_PATH", model_path)
    clear_model_cache()

    result = ai_text_detector.detect_ai_text_detailed(UNIFORM_TEXT)
    assert result.backend == "sklearn"
    assert 0.0 <= result.ai_probability <= 1.0
    assert result.label in {"AI", "Human"}


def test_auto_mode_falls_back_to_heuristic_when_no_model_exists():
    result = ai_text_detector.detect_ai_text_detailed(VARIED_HUMAN_TEXT)
    assert result.backend == "heuristic"
