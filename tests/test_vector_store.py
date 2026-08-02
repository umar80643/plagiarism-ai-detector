import numpy as np

from models.vector_store import FaissVectorStore


def _normalized(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_search_returns_nearest_vector_first():
    store = FaissVectorStore(dim=4)
    vectors = _normalized(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32))
    ids = store.add(vectors)
    assert ids == [0, 1, 2]

    query = _normalized(np.array([[1, 0, 0, 0]], dtype=np.float32))
    scores, result_ids = store.search(query, top_k=2)
    assert result_ids[0][0] == 0  # exact match ranked first
    assert scores[0][0] > scores[0][1]


def test_incremental_add_does_not_require_rebuilding():
    store = FaissVectorStore(dim=4)
    first_batch = _normalized(np.array([[1, 0, 0, 0]], dtype=np.float32))
    store.add(first_batch)
    assert store.size == 1

    second_batch = _normalized(np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
    new_ids = store.add(second_batch)
    assert new_ids == [1, 2]
    assert store.size == 3


def test_search_on_empty_store_returns_padded_results_not_a_crash():
    store = FaissVectorStore(dim=4)
    query = _normalized(np.array([[1, 0, 0, 0]], dtype=np.float32))
    scores, ids = store.search(query, top_k=5)
    assert scores.shape == (1, 5)
    assert (ids == -1).all()


def test_top_k_larger_than_store_size_is_padded():
    store = FaissVectorStore(dim=4)
    store.add(_normalized(np.array([[1, 0, 0, 0]], dtype=np.float32)))
    query = _normalized(np.array([[1, 0, 0, 0]], dtype=np.float32))
    scores, ids = store.search(query, top_k=5)
    assert ids[0][0] == 0
    assert (ids[0][1:] == -1).all()


def test_save_and_load_roundtrip(tmp_path):
    store = FaissVectorStore(dim=4)
    store.add(_normalized(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)))
    path = tmp_path / "index.faiss"
    store.save(path)

    loaded = FaissVectorStore.load(path)
    assert loaded.size == 2
    query = _normalized(np.array([[1, 0, 0, 0]], dtype=np.float32))
    scores, ids = loaded.search(query, top_k=1)
    assert ids[0][0] == 0

    # ids assigned after loading must continue from where the saved store
    # left off, not restart at 0 -- otherwise incremental indexing after a
    # restart would silently collide with existing ids.
    more_ids = loaded.add(_normalized(np.array([[0, 0, 1, 0]], dtype=np.float32)))
    assert more_ids == [2]
