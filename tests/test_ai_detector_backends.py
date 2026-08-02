import pickle

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from models.ai_detectors.heuristic import HeuristicDetector
from models.ai_detectors.sklearn_backend import SklearnDetector

UNIFORM_TEXT = (
    "Artificial intelligence is transforming industries worldwide. "
    "Machine learning algorithms are improving business outcomes globally."
)
VARIED_HUMAN_TEXT = "omg I can't believe it's Friday already, this week flew by so fast honestly."


def test_heuristic_detector_result_shape():
    result = HeuristicDetector().predict(VARIED_HUMAN_TEXT)
    assert abs(result.human_probability + result.ai_probability - 1.0) < 1e-9
    assert 0.0 <= result.confidence <= 1.0
    assert result.label in {"AI", "Human"}
    assert result.backend == "heuristic"
    assert isinstance(result.explanation, list)


def test_sklearn_detector_raises_clearly_when_no_model_trained(tmp_path):
    detector = SklearnDetector(model_path=tmp_path / "missing.pkl")
    assert detector.is_available() is False
    with pytest.raises(RuntimeError):
        detector.predict("some text")


def test_sklearn_detector_predicts_when_trained(tmp_path):
    texts = ["Artificial intelligence is transforming industries."] * 4 + ["lol my dog is so silly today"] * 4
    labels = ["ai"] * 4 + ["human"] * 4
    vectorizer = TfidfVectorizer()
    estimator = LogisticRegression(max_iter=1000).fit(vectorizer.fit_transform(texts), labels)

    model_path = tmp_path / "ai_detector.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "estimator": estimator}, handle)

    detector = SklearnDetector(model_path=model_path)
    assert detector.is_available() is True
    result = detector.predict(UNIFORM_TEXT)
    assert result.backend == "sklearn"
    assert abs(result.human_probability + result.ai_probability - 1.0) < 1e-9
