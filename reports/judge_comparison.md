# Judge Comparison: Cohere Command A vs llama-3.3-70b-versatile

**Date:** 2026-05-19  
**Gold set:** 60 issues (30 microsoft/vscode + 30 kubernetes/kubernetes)  
**Systems evaluated:** System 1 (TF-IDF), Systems 1+2 (TF-IDF+BGE), Full System (LLM)  
**Rubric max:** 15 points (6 dimensions)

---

## 1. Per-Dimension Score Table

### System 1 (TF-IDF only) — n=60

| Dimension | Max | Cohere Command A | llama-3.3-70b | Diff (C−L) |
|---|---|---|---|---|
| component_match | 2 | 1.417 | 1.517 | −0.100 |
| similar_issues_relevance | 3 | 0.000 | 0.000 | 0.000 |
| resolution_estimate_reasonableness | 3 | 0.983 | 0.817 | +0.167 |
| priority_alignment | 1 | 0.100 | 0.100 | 0.000 |
| next_steps_actionability | 3 | 1.000 | 1.233 | −0.233 |
| overall_quality | 3 | 0.983 | 0.883 | +0.100 |
| **TOTAL** | **15** | **4.483 (29.9%)** | **4.550 (30.3%)** | **−0.067** |

### Systems 1+2 (TF-IDF + BGE retrieval) — n=60

| Dimension | Max | Cohere Command A | llama-3.3-70b | Diff (C−L) |
|---|---|---|---|---|
| component_match | 2 | 1.467 | 1.500 | −0.033 |
| similar_issues_relevance | 3 | 2.617 | 2.300 | +0.317 |
| resolution_estimate_reasonableness | 3 | 1.117 | 0.867 | +0.250 |
| priority_alignment | 1 | 0.100 | 0.100 | 0.000 |
| next_steps_actionability | 3 | 1.000 | 1.983 | **−0.983** |
| overall_quality | 3 | 1.700 | 1.733 | −0.033 |
| **TOTAL** | **15** | **8.000 (53.3%)** | **8.483 (56.6%)** | **−0.483** |

### Full System (LLM) — n=60

| Dimension | Max | Cohere Command A | llama-3.3-70b | Diff (C−L) |
|---|---|---|---|---|
| component_match | 2 | 1.583 | 1.550 | +0.033 |
| similar_issues_relevance | 3 | 2.750 | 2.717 | +0.033 |
| resolution_estimate_reasonableness | 3 | 1.450 | 1.283 | +0.167 |
| priority_alignment | 1 | 0.600 | 0.583 | +0.017 |
| next_steps_actionability | 3 | 1.983 | 2.650 | **−0.667** |
| overall_quality | 3 | 2.033 | 2.150 | −0.117 |
| **TOTAL** | **15** | **10.400 (69.3%)** | **10.933 (72.9%)** | **−0.533** |

---

## 2. Per-Issue Agreement Rate (Full System, within 1 point per dimension)

| Dimension | Max | Agreement ≤1 pt | % |
|---|---|---|---|
| component_match | 2 | 60/60 | 100.0% |
| similar_issues_relevance | 3 | 60/60 | 100.0% |
| resolution_estimate_reasonableness | 3 | 59/60 | 98.3% |
| priority_alignment | 1 | 60/60 | 100.0% |
| next_steps_actionability | 3 | 60/60 | 100.0% |
| overall_quality | 3 | 60/60 | 100.0% |
| **Total score within 1** | **15** | **44/60** | **73.3%** |

---

## 3. Pearson Correlation (Full System, per-issue totals)

**r = 0.729, p < 0.0001** (n=60)

Both judges agree on which issues are hard to triage well and which are easy. The correlation exceeds the 0.70 threshold set in ADR-0003 planning.

---

## 4. Interpretation

Cohere Command A scores the Full System at **10.40/15 (69.3%)**, versus llama-3.3-70b-versatile at **10.93/15 (72.9%)** — a gap of **−0.53 points (−3.6 percentage points)**. The Pearson correlation of **r = 0.729** indicates the two judges rank issues similarly: they agree on the relative ordering of triage quality even when they disagree on the exact point count. Per-dimension within-1-point agreement is near-perfect across five of six dimensions (98–100%), with total-score within-1-point agreement at 73%. The one meaningful structural divergence is `next_steps_actionability`: Cohere awards 1.98/3 versus llama's 2.65/3 on the Full System (−0.67 points), and 1.00/3 versus 1.98/3 on Systems 1+2 (−0.98 points). This is a consistent rubric-interpretation gap — Cohere applies a stricter "precise, ordered, repo-appropriate" bar on step quality — rather than evidence of same-family leniency on the part of llama-70b. The overall −3.6pp gap is below the 5pp threshold used to flag systematic inflation. The family-bias hypothesis is not supported at a material effect size: the baseline 73% llama-70b result is a mild overestimate at best, with the true score closer to 69–73% depending on which judge is treated as ground truth.

---

## 5. Supporting Evidence: qwen3-32b Partial Run

The qwen/qwen3-32b Groq judge (Alibaba, cross-family) scored the Full System at **9.63/15 (64.2%)** on a sample of **38/60** issues before Groq TPD exhaustion blocked further scoring. This −9pp gap versus llama-70b is directionally consistent with Cohere's −3.6pp gap, both pointing toward the llama-70b judge being marginally more lenient. However, the qwen3 sample is too small (38/60 = 63% of the gold set) and the Groq TPD wall may have introduced a biased subsample (the issues that were scored earliest in the run). The qwen3 partial data is archived at `reports/_archive/2026-05-19-qwen3-partial-tpd-blocked.json` and should not be cited as a standalone result. Combined with the Cohere full-N data, the signal points consistently in one direction: same-family leniency exists but is small (3–9pp range, lower bound more credible given full N).
