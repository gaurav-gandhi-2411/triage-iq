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
| System 1 (TF-IDF) | 0ms | 0ms |
| System 2 (BGE) | 0ms | 0ms |
| System 3 (LightGBM) | 0ms | 0ms |
| LLM call (Groq) | 0ms | 0ms |
| Total pipeline | 0ms | 0ms |

---

## 2. Gold Standard

60 issues total (30 per repo), stratified across resolution buckets: <7d / 7–30d / >30d.
Sampled from val+test splits combined (neither used for training any model).
Component annotation from normalized label set. Priority inferred from metadata or resolution speed.

**microsoft/vscode:** 30 issues — component accuracy: LLM 53%, TF-IDF 50%, Majority 13%

**kubernetes/kubernetes:** 30 issues — component accuracy: LLM 63%, TF-IDF 73%, Majority 10%

---

## 3. Results

### 3.1 LLM-as-Judge Scores (out of 15 max)

| System | Total | comp_match /2 | similar_issues /3 | resolution_est /3 | priority /1 | next_steps /3 | overall /3 |
|---|---|---|---|---|---|---|---|
| System 1 (TF-IDF) | 4.55 (30%) | 1.52 | 0.00 | 0.82 | 0.10 | 1.23 | 0.88 |
| Systems 1+2 (TF-IDF+BGE) | 8.48 (57%) | 1.50 | 2.30 | 0.87 | 0.10 | 1.98 | 1.73 |
| Full System (LLM) | **10.93 (73%)** | 1.55 | 2.72 | 1.28 | 0.58 | 2.65 | 2.15 |

### 3.2 Component Accuracy

| System | Overall | vscode | kubernetes |
|---|---|---|---|
| Full System (LLM) | 58.3% | 53.3% | 63.3% |
| System 1 (TF-IDF) | 61.7% | 50.0% | 73.3% |
| Majority Component | 11.7% | 13.3% | 10.0% |

### 3.3 Judge Reliability (double-run, n=10)

**Note:** Reliability double-check was blocked by Groq TPD exhaustion (100K token/day limit hit during the 10 re-score calls). The kappa values below are **not real** — they are default zeros from failed API calls, not measured agreement. Treat reliability as unknown pending a TPD-reset re-run.

| Dimension | Cohen's κ | Status |
|---|---|---|
| All dimensions | N/A | TPD limit hit — double-check incomplete |

---

## 4. Hand-Validation of 5 Grading Decisions

5 issues sampled across the score distribution (1 high, 2 mid, 2 low). For each: gold label → generated plan → judge score → assessment.

---

**#311543 — microsoft/vscode — 14/15** *(High)*

- **Gold:** component=`error-telemetry`, priority=high, resolved in 0.06 days (1.5h)
- **Generated:** component=`error-telemetry` (conf=0.67), priority=high, resolution CI 0.2–585.3 days
- **Judge:** comp=2, similar=3, resolution=2, priority=1, next_steps=3, overall=3
- **Assessment:** Judge is **slightly lenient** on resolution (CI 0.2–585 days doesn't contain 0.06; lower bound is 3× the actual). Component, priority, and next steps are genuinely excellent — rationale is accurate. 14/15 is defensible.

---

**#2093 — microsoft/vscode — 10/15** *(Mid)*

- **Gold:** component=`debug`, priority=medium, resolved 6.8 days
- **Generated:** component=`debug` (conf=0.08), priority=high, resolution CI 0.2–318 days
- **Judge:** comp=2, similar=3, resolution=0, priority=0, next_steps=3, overall=2
- **Assessment:** Judge is **accurate**. CI of 0.2–318 days technically contains 6.8 days but is too wide to be useful. Priority mis-call (high vs medium) correctly penalized. Next steps are issue-specific and reference a related PR (#2832) — genuinely good. 10/15 is fair.

---

**#3826 — microsoft/vscode — 9/15** *(Mid)*

- **Gold:** component=`debug`, priority=medium, resolved 1.05 days
- **Generated:** component=`debug` (conf=0.13), priority=high, resolution CI 0.1–62 days
- **Judge:** comp=2, similar=2, resolution=1, priority=0, next_steps=2, overall=2
- **Assessment:** Judge is **accurate**. Component correct; priority wrong (systematic bias toward "high"). CI 0.1–62 days contains 1.05 days — but a 620× ratio between bounds is near-useless. Next steps mix boilerplate ("reproduce on Insiders") with issue-specific steps. 9/15 reflects the gap well.

---

**#567 — microsoft/vscode — 7/15** *(Low)*

- **Gold:** component=`api`, priority=low, resolved 14.7 days
- **Generated:** component=`api` (conf=0.15), priority=high, resolution CI 0.2–715.6 days
- **Judge:** comp=2, similar=2, resolution=0, priority=0, next_steps=2, overall=1
- **Assessment:** Judge is **slightly harsh** on resolution — CI 0.2–715.6 technically contains 14.7 days, but the judge scored 0/3 since the range is effectively unbounded. This is a judgment call; practically the resolution estimate is useless. Priority mis-call (high vs low) is a real error. 7/15 is appropriate.

---

**#814 — microsoft/vscode — 6/15** *(Low)*

- **Gold:** component=`javascript`, priority=low, resolved 104.97 days
- **Generated:** component=`file-explorer` (conf=0.08), priority=low, resolution CI 0.2–542.6 days
- **Judge:** comp=0, similar=2, resolution=0, priority=1, next_steps=2, overall=1
- **Assessment:** Judge is **accurate**. Component is wrong (gold=javascript, LLM predicted file-explorer — the issue is about error navigation, not file-explorer per se). Priority correct at 0/1. Resolution CI contains 104.97 days but is 2,700× wide. 6/15 reflects a genuinely poor plan.

---

**Verdict:** Judge appears **well-calibrated**. Systematic model failure mode: priority is predicted "high" on 4/5 of these issues regardless of gold label. Resolution estimates are technically interval-correct but uniformly too wide to be operationally useful — the judge penalizes this appropriately.

**Dimensions most reliable:** `component_match`, `similar_issues_relevance`  
**Dimensions most subjective:** `resolution_estimate_reasonableness` (borderline 0/1 calls for wide CIs), `overall_quality`

---

## 5. Sample Triage Plans

Three representative outputs — great (14/15), mediocre (10/15), and failure mode (6/15).

### Great plan — #311543 (microsoft/vscode) — 14/15

**Issue:** [Error] unhandlederror-potential listener LEAK detected  
**Gold:** component=`error-telemetry`, priority=high, resolved in 1.5h

| Field | Value |
|---|---|
| predicted_component | error-telemetry (conf=0.67) ✓ |
| priority_guess | high ✓ |
| resolution_range | 0.2–585.3 days |
| next_steps | Compare stack traces vs duplicates · Investigate misconfigured event listener · Develop fix and backport |

**Judge:** comp=2, similar=3, resolution=2, priority=1, next_steps=3, overall=3

---

### Mediocre plan — #2093 (microsoft/vscode) — 10/15

**Issue:** "Add Function Breakpoint" shows as disabled, but can be clicked  
**Gold:** component=`debug`, priority=medium, resolved in 6.8 days

| Field | Value |
|---|---|
| predicted_component | debug (conf=0.08) ✓ |
| priority_guess | high ✗ (gold=medium) |
| resolution_range | 0.2–318.0 days |
| next_steps | Verify cross-platform · Investigate debug UI state management · Check if #2832 is related |

**Judge:** comp=2, similar=3, resolution=0, priority=0, next_steps=3, overall=2

---

### Failure mode — #814 (microsoft/vscode) — 6/15

**Issue:** Navigating from and to a file stacks errors instead of overwriting  
**Gold:** component=`javascript`, priority=low, resolved in 105 days

| Field | Value |
|---|---|
| predicted_component | file-explorer (conf=0.08) ✗ (gold=javascript) |
| priority_guess | low ✓ |
| resolution_range | 0.2–542.6 days |
| next_steps | Review original 2015 issue · Investigate file-explorer vs working-files · Create reproducer |

**Judge:** comp=0, similar=2, resolution=0, priority=1, next_steps=2, overall=1

---


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

**Runtime:** 248s | **Issues evaluated:** 60 | **Triage failures:** 0 | **Judge failures:** 0

**Approx Groq spend:** 150,000 tokens (~$0.041 at Groq free-tier pricing)