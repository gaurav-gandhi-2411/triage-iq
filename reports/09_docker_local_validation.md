# Docker Local Validation — TriageIQ Production Image

**Date:** 2026-05-02  
**Image:** `triageiq:prod` built from `docker/Dockerfile.prod`  
**Host:** Windows 11, Docker Desktop  

---

## Image Size

| Metric | Value |
|---|---|
| Uncompressed (local) | 4.47 GB |
| Main contributors | CPU-only PyTorch (~700 MB installed), BGE model (~420 MB), sklearn + LightGBM models, parquet training sets |

Note: Cloud Run supports images up to 32 GB. Cold-start latency scales with image size; 4.47 GB is in the acceptable range for a stateful ML API.

---

## Timing

| Metric | Value | Notes |
|---|---|---|
| Cold start (container start → /health 200) | 15.7s | Fresh container, loads 2× sklearn+LightGBM classifiers + 2× BGE FAISS indexes |
| First /triage request (vscode) | 1.38s | Includes Groq LLM call |
| Second /triage request (kubernetes) | ~11s | Groq-side latency spike (no retry logged, network variance) |
| Typical warm /triage | ~1.3–1.5s | Expected for BGE embed + FAISS + Groq |

---

## Memory

| Metric | Value |
|---|---|
| Peak after 2 warm requests | ~590 MiB |
| Steady state | ~545 MiB |

Cloud Run `--memory 2Gi` is sufficient with headroom.

---

## Health Check

```json
{
  "status": "ok",
  "repos_loaded": ["microsoft/vscode", "kubernetes/kubernetes"],
  "groq_key_present": true,
  "uptime_s": 10.8
}
```

---

## Bugs Found and Fixed

### 1. `_log_request` duplicate keyword argument (app crash)
- **Error:** `TypeError: _log_request() got multiple values for keyword argument 'total_latency_ms'`
- **Root cause:** `meta` dict returned by `triage_with_metadata()` already contains `total_latency_ms`; app.py also passed it explicitly, then did `**meta`.
- **Fix:** `**{k: v for k, v in meta.items() if k != "total_latency_ms"}` in the success log path (`app.py:91`).

### 2. `pyproject.toml` build backend incompatibility (Docker build failure)
- **Error:** `ModuleNotFoundError: No module named 'setuptools.backends'`
- **Root cause:** `setuptools.backends.legacy:build` is a setuptools 67.2+ internal path; pip's build isolation env downloaded an older setuptools.
- **Fix:** Reverted build-backend to stable `setuptools.build_meta`.

### 3. scikit-learn version mismatch (silent accuracy risk)
- **Warning:** `InconsistentVersionWarning: Trying to unpickle estimator from version 1.6.1 when using version 1.8.0`
- **Root cause:** `requirements.txt` had `scikit-learn>=1.3` (resolved to 1.8.0 in Docker); models trained locally on 1.6.1.
- **Fix:** Pinned `scikit-learn>=1.6,<1.7` in `requirements.txt`. No warnings in final build.

---

## Non-Fatal Warning

```
Resolution predictor failed: 'created_at'
```

The resolution predictor expects a `created_at` field in the issue payload; API requests don't include it. The predictor fails gracefully — the field is absent from the response but the triage plan (component, duplicates, LLM summary) returns 200 OK.

**Action:** Either pass `created_at` in the API schema or make the resolution predictor handle missing timestamps. Defer until GCP validation.

---

## Sample Response — microsoft/vscode

**Input:** `"Editor crashes on large JSON"` / `"VS Code becomes unresponsive when opening 50MB+ JSON files"`

| Field | Value |
|---|---|
| `predicted_component` | `json` |
| `component_confidence` | 0.177 |
| `priority_guess` | high |
| `similar_issues` | #1001 (0.796), #523 (0.794), #404 (0.780), #177 (0.779) |
| `suggested_assignee_class` | large-file team |

---

## Verdict

**Local validation: PASS with two caveats**

1. Image is 4.47 GB — acceptable for Cloud Run but worth noting in GCP cost analysis.
2. `Resolution predictor failed: 'created_at'` is a known gap — resolution ETA values will be absent until fixed.

Ready to proceed to GCP setup (Cloud Run deploy).
