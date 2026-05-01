# Docker Local Validation — TriageIQ Production Image

**Date:** 2026-05-02  
**Image:** `triageiq:prod` built from `docker/Dockerfile.prod`  
**Host:** Windows 11, Docker Desktop  
**Status: PASS — all 4 systems active, zero warnings**

---

## Image Size

| Metric | Value |
|---|---|
| Uncompressed (local) | 4.47 GB |
| Main contributors | CPU-only PyTorch (~700 MB), BGE model (~420 MB), sklearn + LightGBM models, FAISS indexes, parquet training sets |

Cloud Run supports images up to 32 GB. 4.47 GB is acceptable for a stateful ML API.

---

## Timing

| Metric | Value | Notes |
|---|---|---|
| Cold start (container start → /health 200) | 11.3s | Fresh container, loads 2× sklearn+LightGBM classifiers + 2× BGE FAISS indexes |
| First /triage request | 2.1s | BGE embed + FAISS + Groq LLM call |
| Second /triage request (warm) | 2.5s | Network variance on Groq side |

---

## Memory

| Metric | Value |
|---|---|
| Steady state after warm requests | ~553 MiB |

Cloud Run `--memory 2Gi` has ample headroom.

---

## Health Check Response

```json
{
  "status": "ok",
  "repos_loaded": ["microsoft/vscode", "kubernetes/kubernetes"],
  "groq_key_present": true,
  "uptime_s": 11.3
}
```

---

## Sample /triage Response — All 4 Systems Active

**Input:** `{"title": "Editor crashes on large JSON", "body": "VS Code becomes unresponsive when opening 50MB+ JSON files...", "repo": "microsoft/vscode"}`

| System | Output | Value |
|---|---|---|
| System 1: classifier | `predicted_component` | `json` (conf: 0.18) |
| System 2: dup detector | `similar_issues` | 5 results (top: 0.796 similarity) |
| System 3: resolution predictor | `expected_resolution_lower/upper_days` | 0.0 – 1.3 days |
| System 4: LLM | `priority_guess`, `triage_summary`, `suggested_next_steps` | high, 3 steps |

---

## Bugs Found and Fixed (5 total across 2 sessions)

### Session 1 (initial Docker build)

| # | File | Error | Fix |
|---|---|---|---|
| 1 | `pyproject.toml` | `ModuleNotFoundError: No module named 'setuptools.backends'` — build failure | Reverted to `setuptools.build_meta` |
| 2 | `app.py:88` | `TypeError: _log_request() got multiple values for keyword argument 'total_latency_ms'` — every /triage returned 500 | Exclude `total_latency_ms` from `**meta` expansion |
| 3 | `requirements.txt` | `InconsistentVersionWarning` on all classifiers (trained on sklearn 1.6.1, Docker resolved 1.8.0) | Pinned `scikit-learn>=1.6,<1.7` |

### Session 2 (resolution predictor validation)

| # | File | Error | Fix |
|---|---|---|---|
| 4 | `resolution.py:59` | `KeyError: 'created_at'` — resolution predictor silently degraded on every API call | Added `created_at` as optional field (default: `datetime.now(UTC)`) to request schema and issue Series; added column-existence guard in `engineer_features` |
| 5 | `resolution.py:73` | `KeyError: 'component'` — same silent degradation path | Extracted `_component` Series with `pd.NA` fallback for missing column |

### Additional: HuggingFace network calls at startup

BGE model is baked into the image but `sentence-transformers` was still making HEAD requests to `huggingface.co` on every cold start, timing out after 10s each and inflating cold start from ~11s to ~48s.

**Fix:** Added `ENV HF_HUB_OFFLINE=1` and `ENV TRANSFORMERS_OFFLINE=1` to `Dockerfile.prod`.  
**Result:** Cold start 48s → 11s.

---

## Tests Added

`tests/test_api.py` — 4 new tests (suite now 8 total, all pass):

- `test_triage_includes_resolution_prediction` — asserts all 4 systems return output, `_request_id` present
- `test_triage_accepts_created_at` — optional `created_at` field in ISO 8601 format doesn't cause 500
- `test_triage_without_created_at` — omitting `created_at` defaults to `now()` without error
- Fixed existing `test_triage_returns_plan` and `test_triage_propagates_assistant_error` — mock had wrong method name (`triage` vs `triage_with_metadata`)

---

## Final Verdict

**PASS — production-ready for GCP deploy.**

No known broken systems. No silent degradation paths. No network dependency at runtime except Groq API.
