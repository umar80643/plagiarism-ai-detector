from models.plagiarism_model import detect_plagiarism

REFERENCE = {
    "source_a.txt": "Machine learning is a branch of artificial intelligence that learns patterns from data.",
    "source_b.txt": "The stock market closed higher today after a strong earnings season.",
}


def test_copied_text_scores_high_against_matching_source():
    copied = "Machine learning is a branch of artificial intelligence that learns patterns from data."
    result = detect_plagiarism(copied, corpus_dir="does/not/exist", extra_documents=REFERENCE)
    assert result["score"] > 90.0
    assert result["matches"][0]["source"] == "source_a.txt"


def test_original_text_scores_low_against_unrelated_corpus():
    original = "My dog learned a new trick this weekend and I could not be prouder."
    result = detect_plagiarism(original, corpus_dir="does/not/exist", extra_documents=REFERENCE)
    assert result["score"] < 30.0


def test_empty_corpus_returns_zero_score_not_a_crash(tmp_path, monkeypatch):
    # extra_documents=None means detect_plagiarism would otherwise use any
    # cached global index; point it at an empty tmp path so this test is
    # independent of whatever's been built on this machine.
    monkeypatch.setattr("models.plagiarism_model.INDEX_PATH", tmp_path / "no_index.pkl")
    result = detect_plagiarism("Some text.", corpus_dir="does/not/exist")
    assert result == {"score": 0.0, "matches": [], "sentence_matches": [
        {"sentence": "Some text.", "source": None, "matched_sentence": None, "similarity": 0.0}
    ]}


def test_sentence_matches_identify_the_source_document():
    text = "Machine learning is a branch of artificial intelligence that learns patterns from data. This part is original."
    result = detect_plagiarism(text, corpus_dir="does/not/exist", extra_documents=REFERENCE)
    matched = [m for m in result["sentence_matches"] if m["similarity"] > 70]
    assert matched
    assert matched[0]["source"] == "source_a.txt"


def test_embedding_similarity_catches_paraphrase_not_just_exact_words():
    """The whole point of the LSA upgrade: a reworded copy should still score
    meaningfully, not just near-zero the way raw TF-IDF word-overlap would.
    """
    paraphrased = "Machine learning belongs to the field of AI and identifies patterns within data."
    result = detect_plagiarism(paraphrased, corpus_dir="does/not/exist", extra_documents=REFERENCE)
    assert result["matches"][0]["source"] == "source_a.txt"
    assert result["score"] > 20.0
