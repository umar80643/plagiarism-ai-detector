# Plagiarism & AI Content Detector

A corpus-based plagiarism checker and AI-text detector, served behind a
versioned FastAPI backend with a thin Streamlit UI on top. This is the
second rebuild of an earlier project — the first rebuild fixed the core
conceptual bugs (see below); this pass moves it from "single-script demo"
toward the shape a real backend service would take: a real embedding index,
a real held-out evaluation on a real dataset, an authenticated/rate-limited
API, and containerized deployment.

## What changed, and why, across both rebuilds

**Rebuild 1** fixed two structural problems: `detect_plagiarism(text)` used
to compare a document's sentences to *itself*, which can never detect
copying from an external source — it's now checked against an actual
reference corpus. And the AI-text detector claimed to be "ML-based" while
running on hand-coded thresholds — it now says so honestly, with a real
trainable classifier path.

**Rebuild 2** (this one) addresses three things a portfolio demo gets away
with that a product can't:

1. **No real dataset.** The AI detector's "trainable" path only had a
   24-row, hand-written toy CSV. It now trains on the **AI-GA dataset**
   (Anagnostou et al., 28,662 real, human-labeled title/abstract pairs, half
   original and half GPT-generated — `scripts/fetch_ai_ga_dataset.py`), with
   a genuine held-out test split, not just cross-validation.
2. **No API.** Everything lived inside one Streamlit process. Detection
   logic is now behind a versioned FastAPI backend (`api/`), with API-key
   auth and rate limiting, so it's usable from a script, another service, or
   a mobile client — not just this UI.
3. **Similarity was exact-word-only.** Plagiarism matching used raw TF-IDF,
   which misses paraphrased copying entirely. It now runs in a TF-IDF+LSA
   embedding space (`embeddings/`), and — because re-embedding the whole
   corpus on every request doesn't scale — the corpus is embedded **once**
   into a cached index (`build_index.py`), not per request.

## Real evaluation numbers

Trained via `python train_ai_detector.py --data data/ai_ga_dataset.csv`,
80/20 stratified split, logistic regression over TF-IDF features, evaluated
on the **held-out 5,733-row test set** (never seen during training):

| Metric | Value |
|---|---|
| Accuracy | 99.16% |
| Precision (AI class) | 98.79% |
| Recall (AI class) | 99.55% |
| F1 (AI class) | 99.17% |
| ROC-AUC | 0.9997 |

**Read this number honestly, not as "solved AI detection":** the AI-GA
dataset is GPT-3-generated academic abstracts vs. real PubMed abstracts —
a domain where AI and human writing differ in fairly mechanical, easy-to-learn
ways (structure, hedging language, citation patterns). This does not mean
99% accuracy on essays, social posts, or output from newer/more human-like
models. Full metrics + confusion matrix: `models/artifacts/ai_detector_metrics.json`
after training.

## Architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit UI       │  (app.py — thin client)
                    └──────────┬───────────┘
                               │ HTTP + X-API-Key
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI (api/)     │  auth, rate limiting, CORS,
                    │   /v1/detect/...     │  structured errors
                    └──────────┬───────────┘
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   models/plagiarism_model.py          models/ai_text_detector.py
   → models/plagiarism_index.py         heuristic, or a trained
     (cached embedding index,           classifier if one exists
      built via build_index.py)         (train_ai_detector.py)
              │
              ▼
   embeddings/tfidf_lsa.py   (swappable — see embeddings/base.py)
              │
              ▼
   utils/similarity.py, utils/preprocess.py, utils/file_reader.py
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
pytest

# Precompute the plagiarism index from data/corpus/
python build_index.py

# Terminal 1: the API
uvicorn api.main:app --reload

# Terminal 2: the UI (talks to the API over HTTP)
streamlit run app.py
```

Or with Docker:

```bash
docker compose up --build
```

(Dockerfile/docker-compose.yml follow standard patterns but were written and
reviewed without a Docker daemon available in the environment this was built
in — validate `docker compose up` locally before relying on it.)

### Calling the API directly

```bash
curl -X POST http://localhost:8000/v1/detect/plagiarism \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"text": "Machine learning is a branch of artificial intelligence."}'

curl -X POST http://localhost:8000/v1/detect/ai-text \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"text": "Artificial intelligence is transforming industries worldwide."}'
```

Interactive docs: `http://localhost:8000/docs` (auto-generated by FastAPI).

Set a real `API_KEY` env var in any deployment — `dev-key` is a local-only
default, not a secret.

### Training the AI detector on the real dataset

```bash
python scripts/fetch_ai_ga_dataset.py         # downloads ~41MB from GitHub
python train_ai_detector.py --data data/ai_ga_dataset.csv
```

`models/ai_text_detector.py` picks up the trained classifier automatically;
delete `models/artifacts/ai_detector.pkl` to fall back to heuristic scoring.

### Expanding what plagiarism detection can catch

Drop more `.txt`/`.pdf`/`.docx` files into `data/corpus/`, then:

```bash
python build_index.py
```

## Layout

```text
api/                    FastAPI app: routes, auth, rate limiting, DI
embeddings/
  base.py                 TextEmbedder interface
  tfidf_lsa.py             default, fully-offline embedder
  sentence_transformer.py  optional Sentence-Transformers embedder
  factory.py               builds the embedder config.py selects
models/
  plagiarism_model.py     detect_plagiarism() — public entry point
  plagiarism_index.py      FAISS-backed embedding index, incremental add
  vector_store.py          FAISS IndexHNSWFlat wrapper
  reranker.py              cross-encoder + dependency-free rerankers
  corpus.py                loads data/corpus/ from disk
  ai_text_detector.py      public entry point (backward-compatible)
  ai_detectors/
    base.py                 AIDetectionResult + AIDetectorBackend interface
    heuristic.py             lexical diversity + burstiness scorer
    sklearn_backend.py       TF-IDF + Logistic Regression classifier
    transformer_backend.py   HuggingFace transformer classifier
    calibration.py           temperature-scaling math (unit-tested standalone)
    factory.py               builds the backend config.py selects
utils/
  file_reading, preprocessing, text stats, similarity, explanation.py (why signals)
data/
  corpus/                  reference documents checked against
  ai_detector_demo.csv      tiny toy dataset (pipeline smoke test only)
scripts/fetch_ai_ga_dataset.py     downloads the real 28,662-row dataset
build_index.py                     builds/incrementally updates the plagiarism index
train_ai_detector.py                trains + evaluates the sklearn AI-text classifier
train_transformer_ai_detector.py    fine-tunes a transformer AI-text classifier
calibrate_temperature.py            fits temperature scaling for the transformer
tests/                    pytest — unit + FASTAPI TestClient + network-gated live_model tests
Dockerfile, docker-compose.yml
requirements.txt              core, fully offline
requirements-semantic.txt      optional: Sentence-Transformers/cross-encoder/transformer AI-detector
```

## Milestone: modern semantic retrieval (FAISS + optional Sentence-Transformers/cross-encoder)

Requested upgrade path: replace ad hoc cosine-similarity search with an
industry-standard retrieve-then-rerank pipeline. Delivered incrementally,
without touching the public API contract:

- **FAISS vector store** (`models/vector_store.py`) replaces the dense
  NumPy/sklearn cosine-similarity matrix. `IndexHNSWFlat` gives approximate
  nearest-neighbor search (sub-linear, not O(n) per query) *and* supports
  adding vectors after the fact with no retraining — the actual prerequisite
  for real incremental indexing, not just "call it incremental."
- **Pluggable embedding backend** (`config.EMBEDDING_BACKEND`): TF-IDF+LSA
  remains the default (fully offline, zero model download), and
  `sentence_transformer` is available as a drop-in swap
  (`embeddings/sentence_transformer.py`, `all-mpnet-base-v2` by default,
  `BAAI/bge-small-en-v1.5` via `SENTENCE_TRANSFORMER_MODEL_PRESET=small` for
  memory-constrained deployments). Both satisfy the same `TextEmbedder`
  interface, so nothing downstream branches on which one is active.
- **Why TF-IDF+LSA is still the default, not just a legacy fallback:** unlike
  a fixed-space transformer embedding, `TfidfLsaEmbedder.fit()` must see the
  whole corpus vocabulary — so `fit()` on a Sentence-Transformer is a
  deliberate no-op, which is what makes `PlagiarismIndex.add_documents()`
  genuinely incremental only with that backend. This is documented directly
  in the code, not hidden.
- **Cross-encoder reranking** (`models/reranker.py`, `config.RERANK_ENABLED`):
  FAISS retrieves the top-K candidate sentences cheaply, then — if enabled —
  `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores just those K precisely. A
  bi-encoder (the embedder) alone can't relate query and candidate words
  directly; a cross-encoder can, but is too slow to run against the whole
  corpus — hence retrieve-then-rerank rather than either alone. Cross-encoder
  logits are squashed through a sigmoid so "similarity" stays a comparable
  0–100-ish scale regardless of which reranker (or none) is active.
- **Dependency-free reranker fallback** (`LexicalOverlapReranker`): keeps the
  two-stage pipeline itself testable and usable without any model download,
  at real accuracy cost — an explicit trade-off, not a hidden shortcut.
- **True incremental indexing**: `python build_index.py --incremental` now
  only embeds documents not already in the saved index, rather than
  rebuilding from scratch (`PlagiarismIndex.add_documents()`).

**What I could and couldn't verify here:** FAISS and the retrieve/rerank
*pipeline* were fully built and tested in this environment (`tests/test_vector_store.py`,
`tests/test_plagiarism_index.py`, `tests/test_reranker.py`) using the offline
TF-IDF+LSA embedder and the lexical-overlap reranker. The real
Sentence-Transformers/cross-encoder classes are written and reviewed, but
this environment's network policy blocks `huggingface.co`, so I couldn't
execute the actual model downloads to verify them end to end here — that's
gated behind `tests/test_live_models.py`, marked `live_model` and excluded
from the default test run (see `pyproject.toml`), self-skipping if weights
are unreachable. Run `pytest -m live_model` on a machine with normal
internet access to verify those two classes for real; install
`requirements-semantic.txt` first.

**Enabling the semantic backend:**

```bash
pip install -r requirements-semantic.txt
export EMBEDDING_BACKEND=sentence_transformer
export RERANK_ENABLED=true          # optional
export RERANKER_BACKEND=cross_encoder  # or "lexical" to stay fully offline
python build_index.py
```

**Trade-offs, stated plainly:**
- HNSW is *approximate* — for a corpus in the low thousands, exact search
  (`IndexFlatIP`) would still be fast enough and simpler; HNSW is chosen
  because the requirement is scaling well beyond that.
- Sentence-Transformers + cross-encoder reranking is more accurate at
  catching paraphrase/semantic plagiarism, at the cost of a ~400MB+ model
  download, materially slower CPU inference than TF-IDF+LSA, and a genuine
  network dependency this project didn't have before.
- `sentence-transformers` (and the PyTorch it pulls in) is deliberately kept
  out of `requirements.txt` and split into `requirements-semantic.txt`, so
  the default install stays small and fully offline.

## Milestone: real transformer-based AI-text detection

Requested upgrade path: replace heuristic scoring with a real ML model
(RoBERTa/DeBERTa), with calibration, confidence, and explainability, keeping
existing API endpoints unchanged. Delivered as a third pluggable backend
alongside the existing heuristic and sklearn ones:

- **Why not `roberta-base`/`microsoft/deberta-v3-base` directly, out of the
  box:** used for classification without fine-tuning, they load with a
  *randomly initialized* classification head — they're pretrained language
  models, not detectors. Predicting with one untrained would score at
  chance, worse than the existing heuristic. The honest default is
  `openai-community/roberta-base-openai-detector` — OpenAI's own real,
  publicly released, fine-tuned checkpoint. Its real limitation, stated
  plainly: it was trained years ago to detect **GPT-2** output specifically,
  so it's meaningfully weaker against modern LLM-generated text than a model
  fine-tuned on current generations would be.
- **Supporting future fine-tuning, for real:** `train_transformer_ai_detector.py`
  fine-tunes `roberta-base` or `microsoft/deberta-v3-base` on your own
  labeled data via HuggingFace's `Trainer`, with a genuine held-out
  evaluation. Point `TRANSFORMER_AI_DETECTOR_MODEL` at the resulting
  checkpoint and nothing else — not the API, not the UI, not
  `TransformerDetector` itself — needs to change.
- **Probability calibration:** raw softmax probabilities from a fine-tuned
  transformer are frequently overconfident (Guo et al., 2017). `calibrate_temperature.py`
  fits a single temperature scalar on a held-out labeled set via
  `models/ai_detectors/calibration.py` (unit-tested with synthetic logits,
  independent of any model download) and `TransformerDetector` applies it
  automatically once saved.
- **Explainability** (`utils/explanation.py`): every backend, not just the
  transformer, now returns human-readable reasons — low lexical diversity,
  uniform sentence-length ("burstiness"), and detected internal repetition.
  Scope note, stated honestly: "excessive predictability" and "unusual
  fluency" were also requested but need a language-model perplexity
  estimate, which isn't implemented here — three real, computed signals
  beat five where two would be faked.
- **Backward compatibility:** `detect_ai_text(text) -> (score, label)` keeps
  its exact original signature and behavior; `detect_ai_text_detailed(text)`
  is new and additive. `AITextResponse` gained `human_probability`,
  `ai_probability`, `confidence`, `explanation`, and `backend` fields —
  `score` and `label` are unchanged, so nothing that reads the old shape
  breaks.
- **Auto-upgrade preserved:** `AI_DETECTOR_BACKEND` defaults to `"auto"`,
  which keeps this project's original behavior — pick up a trained sklearn
  model automatically if `train_ai_detector.py` has been run, otherwise fall
  back to heuristic — with zero configuration needed. `transformer` is never
  auto-selected (unlike sklearn), since it needs a large model download and
  heavy optional dependencies; that's an explicit opt-in
  (`AI_DETECTOR_BACKEND=transformer`), consistent with `EMBEDDING_BACKEND`'s
  same offline-by-default philosophy.

**A real bug this caught, worth naming honestly:** early in this milestone,
the AI-detector's auto-upgrade behavior silently broke — `SklearnDetector`'s
`model_path` was a default argument bound once at class-definition time, so
monkeypatching the module-level path in tests (and, more importantly, any
config change at runtime) had no effect. The existing tests still passed,
because their assertions (`0 <= score <= 100`, `label in {"AI","Human"}`)
were generic enough to pass against the *wrong* backend too. Fixed by
reading the path inside `__init__`'s body instead of as a default value, and
by adding an explicit `result.backend == "sklearn"` assertion so this
specific regression can't silently pass again.

**What I could and couldn't verify here:** the calibration *math* is fully
tested (synthetic logits, no model needed). `TransformerDetector` itself —
actually downloading and running `roberta-base-openai-detector` — could not
be executed in this environment (`huggingface.co` blocked); it's gated
behind `tests/test_live_models.py`'s `live_model` marker, self-skipping on
failure. `calibrate_temperature.py` and `train_transformer_ai_detector.py`
are written and reviewed but likewise unexecuted here — the latter also
needs meaningfully more compute than this project has otherwise required
(realistically a GPU for anything beyond a tiny dataset). Run
`pytest -m live_model`, then the two scripts, on a machine with real
internet/compute to verify all three for real.

## What's still missing for a genuine production system

Being direct about the remaining gap to "MNC-grade," since that's the
question that prompted this rebuild:

- **No real vector datastore.** FAISS now handles the retrieval math, but the
  index itself is still a local pickle + on-disk FAISS file; a production
  version needs FAISS behind a proper service (or a managed vector DB) so the
  index survives restarts and scales past one process's memory.
- **No auth beyond a single shared API key.** Fine for a demo; a real
  product needs per-tenant keys/OAuth, not one shared secret.
- **No observability.** No metrics/tracing (Prometheus, OpenTelemetry), no
  structured log shipping, no alerting on error-rate or latency regressions.
- **No CI/CD.** Tests run locally; there's no pipeline gating merges or
  automating deploys.
- **Sentence-Transformers/cross-encoder are implemented but unverified live
  in this environment** (network-blocked here, see the milestone section
  above) — verify with `pytest -m live_model` before treating them as proven.
- **Single-instance, synchronous.** No horizontal scaling, no background job
  queue for large batch/document jobs — everything blocks the request.

None of these are hard blockers to demoing or discussing the project; they're
exactly what "here's what I'd build next, and why" looks like in an
interview.
