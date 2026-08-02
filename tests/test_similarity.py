from utils.similarity import (
    calculate_cosine_similarity,
    sentence_level_corpus_matches,
    sentence_similarity_analysis,
)


def test_identical_texts_score_near_100():
    text = "Artificial intelligence is transforming the world."
    assert calculate_cosine_similarity(text, text) > 99.0


def test_unrelated_texts_score_low():
    score = calculate_cosine_similarity(
        "Artificial intelligence is transforming the world.",
        "My cat knocked a plant off the windowsill this morning.",
    )
    assert score < 30.0


def test_paraphrased_texts_score_higher_than_unrelated():
    original = "Machine learning is a branch of artificial intelligence."
    paraphrase = "Machine learning is a subfield of artificial intelligence."
    unrelated = "The weather in Kanpur was cloudy all afternoon."
    assert calculate_cosine_similarity(original, paraphrase) > calculate_cosine_similarity(original, unrelated)


def test_sentence_similarity_analysis_finds_internal_repetition():
    text = "Machine learning is amazing. Deep learning is amazing too. The weather is nice today."
    results = sentence_similarity_analysis(text)
    assert len(results) == 3
    # The two "amazing" sentences should be each other's best match.
    top = max(results, key=lambda item: item["similarity"])
    assert top["similarity"] > 0


def test_sentence_similarity_analysis_handles_single_sentence():
    results = sentence_similarity_analysis("Just one sentence here.")
    assert results == [{"sentence": "Just one sentence here.", "most_similar_to": None, "similarity": 0.0}]


def test_sentence_level_corpus_matches_finds_external_match():
    reference = {"source.txt": "The sky is blue and the grass is green."}
    results = sentence_level_corpus_matches("The sky is blue and the grass is green today.", reference)
    assert results[0]["source"] == "source.txt"
    assert results[0]["similarity"] > 50.0


def test_sentence_level_corpus_matches_handles_empty_corpus():
    results = sentence_level_corpus_matches("Some sentence.", {})
    assert results == [{"sentence": "Some sentence.", "source": None, "matched_sentence": None, "similarity": 0.0}]
