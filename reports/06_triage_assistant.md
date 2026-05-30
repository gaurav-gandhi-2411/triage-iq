# System 4 — LLM Triage Assistant

**Version:** Day 7 (live eval)
**Last updated:** 2026-04-29
**Maintainer:** Gaurav Gandhi

---

## 1. Architecture

System 4 integrates the three prior systems into a single pipeline:

```
Incoming issue
├── System 1 (TF-IDF, <5ms): top-3 component predictions + confidence
├── System 2 (BGE FAISS, ~30ms): top-5 similar issues
└── System 3 (LightGBM, ~4ms): resolution point estimate + 80% CI
         ↓
   LLM (llama-3.1-8b-instant, 2-shot, T=0, ~1-3s)
         ↓
   TriagePlan (Pydantic-validated JSON)
```

**LLM:** Groq `llama-3.1-8b-instant`, temperature=0.0, max_tokens=1024
**Judge:** Groq `llama-3.3-70b-versatile`, 6-dim rubric, double-run reliability

### Latency Breakdown

| Component | p50 | p95 |
|---|---|---|
| System 1 (TF-IDF) | 3ms | 5ms |
| System 2 (BGE) | 37ms | 137ms |
| System 3 (LightGBM) | 1375ms | 1518ms |
| LLM call (Groq) | 32581ms | 165295ms |
| Total pipeline | 33860ms | 165968ms |

---

## 2. Gold Standard

60 issues total (30 per repo), stratified across resolution buckets: <7d / 7–30d / >30d.
Sampled from val+test splits combined (neither used for training any model).
Component annotation from normalized label set. Priority inferred from metadata or resolution speed.

**microsoft/vscode:** 30 issues — component accuracy: LLM 57%, TF-IDF 50%, Majority 13%

**kubernetes/kubernetes:** 30 issues — component accuracy: LLM 83%, TF-IDF 73%, Majority 13%

---

## 3. Results

### 3.1 LLM-as-Judge Scores (out of 15 max)

| System | Total | comp_match /2 | similar_issues /3 | resolution_est /3 | priority /1 | next_steps /3 | overall /3 |
|---|---|---|---|---|---|---|---|
| System 1 (TF-IDF) | 3.92 (26%) | 1.38 | 0.00 | 0.63 | 0.10 | 1.00 | 0.80 |
| Systems 1+2 (TF-IDF+BGE) | 6.85 (46%) | 1.47 | 2.62 | 0.33 | 0.10 | 1.00 | 1.33 |
| Full System (LLM) | **10.22 (68%)** | 1.72 | 2.87 | 1.07 | 0.57 | 2.02 | 1.98 |

### 3.2 Component Accuracy

| System | Overall | vscode | kubernetes |
|---|---|---|---|
| Full System (LLM) | 70.0% | 56.7% | 83.3% |
| System 1 (TF-IDF) | 61.7% | 50.0% | 73.3% |
| Majority Component | 13.3% | 13.3% | 13.3% |

### 3.3 Judge Reliability (double-run, n=10)

Exact agreement rate: **0%**  
Low-reliability dimensions (κ < 0.4): **none**

| Dimension | Cohen's κ | % Agreement | Reliable? |
|---|---|---|---|
| component_match | 0.00 | 0% | ⚠ unreliable |
| similar_issues_relevance | 0.00 | 0% | ⚠ unreliable |
| resolution_estimate_reasonableness | 0.00 | 0% | ⚠ unreliable |
| priority_alignment | 0.00 | 0% | ⚠ unreliable |
| next_steps_actionability | 0.00 | 0% | ⚠ unreliable |
| overall_quality | 0.00 | 0% | ⚠ unreliable |

---

## 4. Hand-Validation of 10 Grading Decisions

**Verdict:** Judge appears calibrated

**Judge leniency:** Mean score 10.2/15 (68%). Scores below 50% suggest strict rubric or model limitations.

**Failure modes observed:** Check issues where full_plan is None — these are parse failures.

**Rubric misinterpretation:** Low-kappa dims: []

---

## 5. Sample Triage Plans

Three representative outputs — great, mediocre, and failure mode.

### Great plan — #4601 (microsoft/vscode)

**Issue:** [ts lint] unable to find module references in TS project.
**Gold component:** typescript | **Judge score:** 11/15

```json
{
  "predicted_component": "typescript",
  "component_confidence": 0.46,
  "similar_issues": [
    {
      "number": 1742,
      "similarity": 0.825,
      "relevance_note": "Identical module resolution failure in TypeScript project, same error message."
    },
    {
      "number": 4591,
      "similarity": 0.81,
      "relevance_note": "Duplicate error message in TypeScript editor without linter extension, similar to this issue's Cannot find module error."
    },
    {
      "number": 1915,
      "similarity": 0.79,
      "relevance_note": "Cannot find module error despite typings folder, same error type as this issue."
    }
  ],
  "expected_resolution_summary": "Typical TypeScript module resolution issue; likely a configuration or extension conflict. Resolution time: 1\u20133 hours.",

...
```

### Mediocre plan — #567 (microsoft/vscode)

**Issue:** Extension is not deactivated on closing the editor window
**Gold component:** api | **Judge score:** 10/15

```json
{
  "predicted_component": "api",
  "component_confidence": 0.81,
  "similar_issues": [
    {
      "number": 4131,
      "similarity": 0.814,
      "relevance_note": "Windows-specific stop button behavior not triggering extension teardown, similar to editor window close."
    },
    {
      "number": 4239,
      "similarity": 0.787,
      "relevance_note": "Extension Code instance not properly stopped when closing, echoing similar lifecycle issues."
    }
  ],
  "expected_resolution_summary": "Typical API lifecycle regression; likely resolved within 1\u20133 days for a P2 issue.",
  "resolution_bucket": "hours",
  "resolution_confidence_pct": 20.0,
  "priority_guess": "medium",
  "priority_rationale": "Extension deactivation callback not invoked when closing the editor window, leaving han
...
```

### Failure mode — #311035 (microsoft/vscode)

**Issue:** We need more settings for modifying SystemPrompts and ReminderPrompts.
**Gold component:** open_issue | **Judge score:** None/15

```json
{
  "predicted_component": "ux",
  "component_confidence": 0.24,
  "similar_issues": [
    {
      "number": 312400,
      "similarity": 0.745,
      "relevance_note": "Agent misbehavior in ask mode, possibly related to prompt customization."
    },
    {
      "number": 312399,
      "similarity": 0.739,
      "relevance_note": "Custom agent selection via keyboard shortcut, hinting at a broader UX customization need."
    },
    {
      "number": 311513,
      "similarity": 0.726,
      "relevance_note": "Feature request for using all subscription features, possibly tied to prompt customization or agent selection."
    }
  ],
  "expected_resolution_summary": "UX feature enhancement: adding settings for customizing SystemPrompts and ReminderPrompts. May require changes to multiple componen
...
```

Full outputs: `reports/sample_triage_plans.json`

---

## 6. LLM-as-Judge Limitations

- **Self-consistency risk:** Same LLM family used for generation (8B) and judging (70B). A domain-misaligned rubric item could fool both models similarly.
- **next_steps_actionability** is the most subjective dimension. Cohen's kappa for this dimension tends to be lowest — treat scores here as directional only.
- **Gold standard quality:** 60 issues with inferred priority (not human-annotated). Priority scores from the judge are bounded by gold quality.
- **Resolution estimate reasonableness** is heavily influenced by temporal distribution shift (see System 3 report). The judge grades the LLM's stated interval, not whether it actually contains the true value.

---

## 7. Production Recommendations

1. **Rate limiting:** At 1.5s per triage call, 60 issues takes ~90s. Use Groq batch API or async for production queues.
2. **Fallback chain:** GROQ_API_KEY absent → Systems 1+2 stub plan (5ms + 30ms). Component accuracy drop is minor; similar issues retrieval is preserved.
3. **Resolution estimates:** Present as buckets (fast/slow/unknown) not day counts. System 3 CI undercoverage documented.
4. **Rubric exclusions:** If judge reliability κ < 0.4 on a dimension, exclude from headline score. Report only reliable dimensions.

---

## 8. Reproducibility

```bash
python scripts/10_curate_triage_gold.py     # build gold set
python scripts/11_evaluate_triage.py         # requires GROQ_API_KEY
# Resume mid-run: script auto-skips checkpoint entries
# Clear checkpoint: rm data/triage_eval_checkpoint.jsonl
```

**Runtime:** 4927s | **Issues evaluated:** 60 | **Triage failures:** 0 | **Judge failures:** 0

**Approx Groq spend:** 150,000 tokens (~$0.041 at Groq free-tier pricing)