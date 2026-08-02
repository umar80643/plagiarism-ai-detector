from models.reranker import LexicalOverlapReranker


def test_rerank_puts_highest_overlap_candidate_first():
    reranker = LexicalOverlapReranker()
    query = "machine learning artificial intelligence"
    candidates = [
        "the weather today is sunny and warm",
        "machine learning is a branch of artificial intelligence",
        "cooking pasta requires boiling water",
    ]
    ranked = reranker.rerank(query, candidates)
    assert ranked[0][0] == 1  # the ML/AI candidate ranks first
    assert ranked[0][1] > ranked[-1][1]


def test_rerank_handles_empty_candidates():
    assert LexicalOverlapReranker().rerank("query", []) == []


def test_rerank_returns_original_indices_not_reordered_candidates():
    reranker = LexicalOverlapReranker()
    ranked = reranker.rerank("apple banana", ["banana cherry", "apple banana cherry date"])
    indices = {index for index, _score in ranked}
    assert indices == {0, 1}
