# Development, retraining and configuration

Moved verbatim from the README front page (local setup, the retraining script order, the environment variables, the opt-in response cache, the remaining endpoints and the reports index). Numbers are re-read by `scripts/readme_numbers_check.py` (manifest `docs/numbers/development.json`).

## Local Development

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier works)

### Setup

```bash
git clone https://github.com/gaurav-gandhi-2411/triage-iq.git
cd triage-iq

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e .

export GROQ_API_KEY=your_groq_key
```

> The API requires trained model artifacts in `data/models/` to serve `/triage`. Tests mock the model store so they run without artifacts.

### Run tests

```bash
pytest tests/
# With coverage:
pytest tests/ --cov=src/triage_iq --cov-report=term-missing
```

### Run the server locally

```bash
uvicorn triage_iq.api.app:app --reload
# → http://localhost:8000
```

---

## Retraining from Scratch

Scripts are numbered in execution order. Run from the repo root with `GITHUB_TOKEN` set for scraping.

```bash
python scripts/01_scrape_issues.py        # scrape from GitHub API
python scripts/02_preprocess.py           # clean bodies, extract features
python scripts/03_split.py                # temporal train/val/test splits
python scripts/04_train_classifier.py     # original single-label TF-IDF classifier (superseded: the shipped
                                          # 47-class model is 13_train_multilabel_classifier.py + calibrate_multilabel_classifier.py)
python scripts/07_extract_related_pairs.py   # extract related-issue pairs for training/eval
python scripts/08_build_similar_issue_index.py # BGE+FAISS index
python scripts/09_train_resolution.py     # LightGBM resolution predictor
python scripts/10_curate_triage_gold.py   # gold triage examples for eval
python scripts/11_evaluate_triage.py      # full pipeline evaluation

# Optional: verify LLM priority calibration against test cases
python scripts/11b_verify_priority_calibration.py
```

> Scripts `05_train_distilbert.py` and `06_eval_llm_fewshot.py` explored alternative architectures that are not used in the production pipeline — DistilBERT loses to TF-IDF+LR on the product's real metric (top-3), on both repos; see the DistilBERT dismissal correction in [`EVALUATION_HISTORY.md`](EVALUATION_HISTORY.md) for the corrected numbers and why the original top-1-based dismissal was measuring the wrong thing.

After retraining, upload artifacts to GCS:

```bash
gsutil -m cp data/models/component_classifier_*.pkl gs://triageiq-prod-260812-models/models/
gsutil -m cp data/models/resolution_predictor_*.pkl gs://triageiq-prod-260812-models/models/
gsutil -m cp -r data/models/similar_issue_index_*_bge gs://triageiq-prod-260812-models/models/
gsutil -m cp data/processed/*_temporal_train.parquet gs://triageiq-prod-260812-models/processed/
```

---

## Configuration

All values from environment variables or `.env` file. Managed by `src/triage_iq/config.py`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Groq API key. Empty string raises `ValidationError` at startup. |
| `ENVIRONMENT` | No | `prod` | `dev` / `test` / `prod`. Controls `/metrics` fail-closed behavior. |
| `METRICS_TOKEN` | No | — | Bearer token for `/metrics`. Unset in prod → 503 (fail-closed). |
| `DATA_DIR` | No | `<repo-root>/data` | Path to `models/` and `processed/` subdirectories. |
| `PORT` | No | `8080` | Uvicorn listen port (Cloud Run sets this automatically). |
| `LOG_LEVEL` | No | `INFO` | Python logging level. |
| `RATE_LIMIT_ENABLED` | No | `true` | Set `false` to disable rate limiting (tests use this). |
| `LLM_CACHE_ENABLED` | No | `false` | Set `true` to enable the SQLite LLM response cache. |
| `LLM_CACHE_PATH` | No | `<repo-root>/data/llm_cache.sqlite` | Path to the SQLite cache DB. |

### Response cache

An opt-in SQLite-backed cache (`LLM_CACHE_ENABLED=true`) stores LLM responses
keyed on SHA-256 of the canonical request (provider + model + messages + temperature +
max\_tokens). Cache hits are returned in <5 ms without a Groq call.

Useful for:
- Eval re-runs: a 64-issue re-run against a warm cache costs 0 Groq tokens for triage
  and near-0 for judge calls.
- Development loops: identical `/triage` requests during testing skip the LLM.

Admin: `python scripts/13_cache_admin.py stats|clear|clear-provider|clear-model`.

On Cloud Run the cache is per-instance (ephemeral disk); each cold start begins empty.
This is acceptable for Stage A — warmup is fast and correctness never depends on the cache.

---

## Other endpoints

#### `GET /`

Service discovery. No auth required. Returns name, version, and endpoint links.

#### `GET /health`

Returns `{"status": "ok", "repos_loaded": [...], "groq_key_present": bool, "uptime_s": float, "dependencies": null}`. `GET /health?deps=1` additionally checks the model store and makes one zero-token authenticated Groq call (503 if either is unhealthy); the keep-warm monitor uses the bare `/health`.

#### `GET /metrics`

Prometheus text format. Auth behavior — see [Monitoring](OPERATIONS.md#monitoring).

**Interactive docs:** `/docs` (Swagger UI, auto-generated).

---

## Reports

| Report | Contents |
|---|---|
| [`reports/01_data_card.md`](../reports/01_data_card.md) | Dataset, preprocessing, known data quality issues |
| [`reports/03_classifier_baseline.md`](../reports/03_classifier_baseline.md) | TF-IDF baseline evaluation |
| [`reports/03_classifier_comparison.md`](../reports/03_classifier_comparison.md) | Classifier ablation (TF-IDF vs LLM few-shot) |
| [`reports/04_duplicate_detection.md`](../reports/04_duplicate_detection.md) | BGE vs MiniLM retrieval comparison |
| [`reports/05_resolution_results.md`](../reports/05_resolution_results.md) | LightGBM resolution predictor evaluation |
| [`reports/06_triage_assistant.md`](../reports/06_triage_assistant.md) | LLM triage pipeline evaluation + priority calibration |
