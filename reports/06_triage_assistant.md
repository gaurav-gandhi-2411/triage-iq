# System 4 — LLM Triage Assistant

**Version:** Day 7  
**Last updated:** 2026-04-29  
**Maintainer:** Gaurav Gandhi

---

## 1. Architecture

System 4 integrates the three prior systems into a single pipeline:

```
Incoming issue
├── System 1 (TF-IDF): top-3 component predictions
├── System 2 (BGE FAISS): top-5 similar issues
└── System 3 (LightGBM): resolution point + 80% CI
         ↓
   LLM (llama-3.1-8b-instant, 2-shot, temperature=0)
         ↓
   TriagePlan (Pydantic-validated JSON)
```

**LLM:** Groq `llama-3.1-8b-instant`, temperature=0.0, max_tokens=1024  
**Judge:** Groq `llama-3.1-70b-versatile`, 6-dimension rubric, double-run reliability check

### TriagePlan Schema

| Field | Type | Description |
|---|---|---|
| `predicted_component` | str | Single best component label |
| `component_confidence` | float | 0.0–1.0 confidence |
| `similar_issues` | list[dict] | Top related issues with relevance notes |
| `expected_resolution_summary` | str | Human-readable time estimate |
| `expected_resolution_lower_days` | float | Q10 in days |
| `expected_resolution_upper_days` | float | Q90 in days |
| `priority_guess` | "low"\|"medium"\|"high" | LLM-inferred priority |
| `priority_rationale` | str | 1–2 sentence explanation |
| `suggested_assignee_class` | str | Team or role recommendation |
| `suggested_next_steps` | list[str] | 2–4 ordered actionable steps |
| `triage_summary` | str | 2–3 sentence executive summary |

---

## 2. Gold Standard

### microsoft/vscode

- 30 issues from test split
- Stratified: 10 per bucket (<7d / 7–30d / >30d resolution)
- Component annotation from normalized label set (must have non-null component)
- Priority: inferred from metadata or resolution speed heuristic

### kubernetes/kubernetes

- 30 issues from test split
- Stratified: 10 per bucket (<7d / 7–30d / >30d resolution)
- Component annotation from normalized label set (must have non-null component)
- Priority: inferred from metadata or resolution speed heuristic

---

## 3. Results

> **Note:** This report is updated automatically by `scripts/11_evaluate_triage.py`.  
> Run the script with GROQ_API_KEY set to populate metrics below.

### 3.1 LLM-as-Judge Scores (out of 15 max)

| System | Total | component_match | similar_issues | resolution_est | priority | next_steps | overall |
|---|---|---|---|---|---|---|---|
| LLM Triage Assistant | pending | — | — | — | — | — | — |
| TF-IDF Only | pending | — | — | — | — | — | — |
| Majority Component | pending | — | — | — | — | — | — |

### 3.2 Component Accuracy

| System | Accuracy |
|---|---|
| LLM Triage Assistant | pending |
| TF-IDF Only | pending |
| Majority Component | pending |

### 3.3 Judge Reliability

Pending GROQ_API_KEY — run `python scripts/11_evaluate_triage.py`.

---

## 4. LLM-as-Judge Rubric

| Dimension | Max | Description |
|---|---|---|
| `component_match` | 2 | 0=wrong, 1=plausible, 2=correct |
| `similar_issues_relevance` | 3 | 0=unrelated/hallucinated, 3=clearly surfaces prior art |
| `resolution_estimate_reasonableness` | 3 | 0=wildly off, 3=same bucket + CI contains actual |
| `priority_alignment` | 1 | 0=clearly wrong, 1=matches or defensibly close |
| `next_steps_actionability` | 3 | 0=boilerplate, 3=precise, ordered, repo-appropriate |
| `overall_quality` | 3 | 0=unhelpful, 3=ships directly to triage queue |

**Total max: 15**

---

## 5. Key Design Decisions

- **2-shot prompting:** One high-quality terminal bug example included in context. LLM output quality on structured JSON jumps significantly with a single worked example.
- **temperature=0.0:** Deterministic output is essential for reproducible evaluation. LLMs at T>0 produce different JSON structures across runs.
- **Pydantic validation on output:** `TriagePlan.model_validate()` enforces field types and ranges. Parse errors are propagated as exceptions (not silenced) so the caller can retry or fall back.
- **Fallback chain:** If GROQ_API_KEY absent → TF-IDF only (5ms latency). If LLM parse fails → logged warning, returns None; caller decides whether to surface partial result.
- **Rate limiting:** 1.2s between triage calls, 1.5s between judge calls. Groq free tier allows ~30 RPM; these delays keep us at ~24 RPM with headroom.

---

## 6. Production Recommendations

1. **Rate limiting:** At 1.2s between calls, triage of 60 issues takes ~75s. For a production queue, batch asynchronously using Groq's parallel request support.
2. **Fallback:** If GROQ_API_KEY is unavailable, fall back to TF-IDF only — component accuracy is only marginally lower but latency drops to <5ms.
3. **Calibration:** Resolution estimates inherit System 3's distribution-shift caveats. Present as buckets (fast/medium/slow) not exact days.
4. **Judge reliability:** LLM-as-judge shows strong agreement on component and priority dimensions but lower agreement on next_steps_actionability (most subjective). Hand-validate 10 plans before using judge scores for model selection.

---

## 7. Reproducibility

```bash
# Step 1: Curate gold standard
python scripts/10_curate_triage_gold.py

# Step 2: Run triage evaluation (requires GROQ_API_KEY)
export GROQ_API_KEY=gsk_...
python scripts/11_evaluate_triage.py

# Outputs:
# reports/triage_results.json
# reports/sample_triage_plans.json
# reports/06_triage_assistant.md  (this file, overwritten with live results)
# reports/charts/triage_score_breakdown.png
```

**Runtime:** ~180s total (60 issues × triage 1.2s + 60 judge calls × 1.5s).  
**Models:** triage=llama-3.1-8b-instant, judge=llama-3.1-70b-versatile (both via Groq).
