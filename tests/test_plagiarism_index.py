from embeddings.tfidf_lsa import TfidfLsaEmbedder
from models.plagiarism_index import PlagiarismIndex, build_index
from models.reranker import LexicalOverlapReranker

CORPUS = {
    "source_a.txt": "Machine learning is a branch of artificial intelligence that learns patterns from data.",
    "source_b.txt": "The stock market closed higher today after a strong earnings season.",
}


def test_build_index_scores_matching_document_highest():
    index = build_index(CORPUS, embedder=TfidfLsaEmbedder())
    matches = index.score_document("Machine learning is a branch of artificial intelligence.")
    assert matches[0]["source"] == "source_a.txt"
    assert matches[0]["similarity"] > matches[1]["similarity"]


def test_add_documents_makes_new_content_searchable_without_rebuilding():
    index = build_index(CORPUS, embedder=TfidfLsaEmbedder())
    original_document_store_id = id(index.document_store)

    index.add_documents({"source_c.txt": "The Eiffel Tower is located in Paris, France."})

    assert "source_c.txt" in index.document_names
    assert id(index.document_store) == original_document_store_id  # same store, added to in place
    matches = index.score_document("The Eiffel Tower is a famous landmark in Paris.")
    assert matches[0]["source"] == "source_c.txt"


def test_reranker_is_used_when_configured():
    index = build_index(CORPUS, embedder=TfidfLsaEmbedder(), reranker=LexicalOverlapReranker(), rerank_top_k=5)
    results = index.score_sentences("Machine learning is a branch of artificial intelligence.")
    assert results[0]["source"] == "source_a.txt"
    assert results[0]["similarity"] > 0


def test_score_sentences_without_reranker_still_works():
    index = build_index(CORPUS, embedder=TfidfLsaEmbedder())  # reranker=None
    results = index.score_sentences("The stock market closed higher today.")
    assert results[0]["source"] == "source_b.txt"


def test_save_and_load_roundtrip_preserves_search_behavior(tmp_path):
    index = build_index(CORPUS, embedder=TfidfLsaEmbedder())
    path = tmp_path / "index.pkl"
    index.save(path)

    loaded = PlagiarismIndex.load(path)
    matches = loaded.score_document("Machine learning is a branch of artificial intelligence.")
    assert matches[0]["source"] == "source_a.txt"

    # Loaded index should still support incremental add, continuing ids
    # rather than colliding with what was saved.
    loaded.add_documents({"source_c.txt": "Completely unrelated content about gardening."})
    assert len(loaded.document_names) == 3


def test_empty_corpus_returns_empty_results_not_a_crash():
    index = build_index({}, embedder=TfidfLsaEmbedder())
    assert index.score_document("some text") == []
    results = index.score_sentences("some text")
    assert results == [{"sentence": "some text", "source": None, "matched_sentence": None, "similarity": 0.0}]
