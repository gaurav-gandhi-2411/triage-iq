# ADR-0023 — Diagnostic Resolution Uncertainty: Signal Search

**Status:** Accepted — documented negative finding. Closes the resolution-abstention thread
opened by ADR-0021.
**Date:** 2026-07-10
**Decider:** Gaurav Gandhi

## Context

ADR-0021 established that the resolution model's conformal interval WIDTH does not predict
whether it covers the true resolution time (mean width statistically indistinguishable between
covered and missed issues on k8s: 90.45d vs 91.01d) — despite the width varying substantially
per issue (6.91–153.72 days, CV=0.463). That closed off width as a resolution-abstention signal
but left an open thread: is there *any other* available per-issue signal that's diagnostic of
resolution error, which would make resolution abstention viable after all?

This ADR is that open search — analysis only, no LLM calls, no retraining, no schema or pipeline
change, zero live impact. The honest possible outcomes were stated up front: a real, corrected,
meaningful correlation is a POSITIVE (propose, don't build, resolution-abstention v2); no signal
clearing that bar is a NEGATIVE, and a valid, publishable finding in its own right — not a
failure to find something.

## Method

**Target labels**, computed per issue from existing ground truth (`eval_set.jsonl`'s
`actual_resolution_days`), replayed over the clean cassette (zero live calls):

- `coverage_failure` (binary): did the CQR-adjusted interval (same formula as
  `src/triage_iq/api/app.py`'s `/triage` handler) fail to contain the actual resolution time?
- `error_magnitude` (continuous): `|point_estimate_days − actual_resolution_days|`.

**11 candidate signals** in the corrected family (all already available per-issue, no new
models): classifier confidence (TF-IDF top-1, ADR-0004), retrieval similarity mean/max/spread
(top-5 BGE scores), title/body length, code-block presence, raw pre-conformal quantile spread,
resolution-bucket ordinal rank, grounding status (`component_grounded`, ADR-0015), and quantile
asymmetry (`|(point−lo) − (hi−point)|` — a zero-cost "ensemble disagreement" proxy from the
already-fitted quantile model, no retraining). **Component identity** (predicted_component) was
tested but kept **exploratory-only**, excluded from correction and the verdict: 27 distinct
values across 54 k8s issues, median group size ~2 — chi-square/Kruskal-Wallis sample-size
assumptions are too badly violated to trust regardless of the p-value it returns.

**Statistics:** `pointbiserialr(signal, coverage_failure)` and `spearmanr(signal, error_magnitude)`,
applied uniformly (valid for binary signals too — point-biserial r on two binary variables
equals the phi coefficient). 11 signals × 2 targets = **22 tests, one family, k8s only**
(n=54, the powered repo). **Multiple-comparison correction: Benjamini-Hochberg FDR (q=0.05)**,
with Bonferroni (α/22 = 0.00227) reported as the more conservative cross-check. **Meaningful
effect-size bar: |r| ≥ 0.3** (Cohen's "medium" convention), stated explicitly so a
significant-but-tiny correlation cannot pass as a finding on its own. vscode (n=11) computed and
reported in full, **excluded from the corrected family and from the verdict** — indicative-only,
same discipline as every other vscode result in this project (ADR-0017).

## Results — full per-signal table (k8s, n=54, the powered/corrected family)

| Signal | r vs coverage_failure | p (raw) | ρ vs error_magnitude | p (raw) | Survives BH-FDR (q=0.05) | Survives Bonferroni |
|---|---|---|---|---|---|---|
| classifier_confidence | −0.106 | 0.445 | −0.117 | 0.401 | No | No |
| retrieval_mean_similarity | −0.085 | 0.543 | **−0.288** | **0.035** | No | No |
| retrieval_max_similarity | −0.119 | 0.392 | −0.252 | 0.066 | No | No |
| retrieval_similarity_spread | −0.079 | 0.570 | −0.004 | 0.980 | No | No |
| title_length_chars | −0.162 | 0.242 | −0.039 | 0.779 | No | No |
| body_length_chars | −0.004 | 0.974 | **+0.284** | **0.038** | No | No |
| code_block_present | — | — | — | — | untestable — zero variance (no issue body in this gold set contains a ` ``` ` code fence) |
| raw_quantile_spread_days | +0.023 | 0.871 | **+0.273** | **0.046** | No | No |
| resolution_bucket_rank | −0.192 | 0.165 | −0.054 | 0.696 | No | No |
| component_grounded | −0.203 | 0.142 | −0.190 | 0.170 | No | No |
| quantile_asymmetry_days | +0.016 | 0.909 | **+0.273** | **0.044** | No | No |

Bolded rows are the 4/22 tests with raw p<0.05. **Zero of the 22 tests survive BH-FDR
correction or Bonferroni; all four raw hits also sit below the |r|≥0.3 meaningful-effect bar
(0.273–0.288).** With 22 tests, ~1.1 false positives are expected under a true null by chance
alone — observing 4 uncorrected hits, none of which survive correction or clear the effect-size
bar, is consistent with exactly that: noise, not signal. This is precisely why the
multiple-comparison correction is the deciding instrument here, not the raw p-values — the raw
hits looked plausible in isolation (retrieval quality, quantile behavior, body length all have
a defensible causal story) and would have been an easy, wrong place to stop.

**Component identity (exploratory, k8s):** Kruskal-Wallis H-test on `error_magnitude` across the
27 distinct predicted components, p=0.345 (uncorrected) — not included in the verdict regardless
of this value, per the sample-size caveat above.

### vscode (n=11) — indicative only, not part of the verdict

No vscode signal reaches even uncorrected p<0.05 on either target. The closest is
`classifier_confidence` vs `error_magnitude`: **r=0.509, p=0.110** — numerically the largest
point-estimate effect size anywhere in this entire search, larger than any k8s result. **This is
not a finding.** On n=11, a correlation this large not reaching significance is exactly what
small-sample noise looks like (ADR-0017's data-ceiling pattern, repeated) — a single differently
weird issue could produce or erase an r this size. Not promoted, not pooled with k8s, not
treated as suggestive of where to look next without independent confirmation on a larger sample.

## Verdict

**NEGATIVE — 0/22 k8s tests are diagnostic** (raw p<0.05 AND survives BH-FDR q=0.05 AND
|r|≥0.3, on the powered repo). No signal among classifier confidence, retrieval quality,
text features, code-block presence, raw quantile spread, resolution bucket, grounding status,
or quantile asymmetry correlates with resolution coverage failure or error magnitude in a way
that is both real and meaningful on this data.

**This is a finding, not a failure.** A systematic search across 11 signals with proper
multiple-comparison correction returning zero survivors is exactly what a rigorous negative
result looks like — the same honest-negative pattern as ADR-0006 (cross-encoder reranker) and
ADR-0021 (selective-prediction v1). The alternative — reporting one of the four uncorrected
hits as "the answer" — would have been the spurious positive this entire methodology exists to
prevent.

### This definitively closes the resolution-abstention thread

Combined with ADR-0021 (interval width is not diagnostic) and this ADR (no other tested
per-issue signal is diagnostic either), **resolution abstention is not merely unbuilt — it is
shown unbuildable on the features currently extracted by this pipeline.** This is a stronger and
more useful conclusion than ADR-0021's alone: ADR-0021 left open "maybe a different signal would
work"; this ADR closes that door for every signal available to search, systematically, with
correction. Resolution abstention should not be revisited by testing more thresholds or
re-deriving defaults on the *same* features — that would repeat ADR-0021's mistake at a smaller
scale. It would require either:

- **New feature sources not currently extracted** — e.g., issue age/staleness at triage time,
  reporter/assignee historical velocity, linked-PR or cross-reference signals, repository
  activity level at the time of filing, or NLP-derived issue complexity beyond raw length (none
  of which this pipeline currently computes; each would be a real feature-engineering
  investment, not a quick add).
- **More data** — k8s's n=54 is powered enough to trust a null result at |r|≥0.3, but a smaller,
  real effect (e.g. r≈0.15–0.2) could exist and simply be underpowered to detect here; a larger
  gold set (ADR-0017's W5 expansion direction) could resolve that ambiguity either way.

Neither direction is committed to here — both are noted as what a future attempt would need,
not a queued build.

## Consequences

- **Nothing prod-facing.** This is pure analysis: no `src/` changes, no schema changes, no
  cassette re-record, no new dependency (scipy, already a project dependency via
  `component_classifier.py`, provided every statistic used — `pointbiserialr`, `spearmanr`,
  `kruskal`, `false_discovery_control`).
- `scripts/analyze_resolution_diagnosticity.py` and `reports/resolution_diagnosticity.json` are
  retained as the audit trail for this search — re-running the script reproduces the report
  **byte-identically** (verified twice, independently, on this branch).
- **ADR-0021's selective-prediction-v1 rejection stands, now on firmer ground.** Resolution
  abstention is closed as a direction until new features or more data change the premise — not
  "still investigating," a definitive negative.
- Generalization caveat: this search covers 11 signals judged sensible for this pipeline's
  existing outputs, on this exact n=65 gold set and this exact model/prompt pair. It does not
  prove no signal could ever be diagnostic — only that none of these 11, on this data, clear the
  bar this ADR set in advance.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Report the 4 uncorrected p<0.05 hits as a finding | Exactly the spurious-positive risk multiple-comparison correction exists to catch — none survive BH-FDR or Bonferroni, and all sit under the stated meaningful-effect bar. Would have repeated ADR-0021's own lesson about noise vs. signal. |
| Promote vscode's `classifier_confidence` r=0.509 result | p=0.110 on n=11 — not significant, and the largest-effect-size-in-the-search framing is exactly the kind of "impressive-looking number, wrong sample size" result ADR-0017 already warned about generally. |
| Use only Bonferroni (skip BH-FDR) | Bonferroni is reported as the conservative cross-check, but BH-FDR is the primary correction — appropriate for an exploratory multi-signal search where the goal is honest ranking, not the single-most-conservative bar. Both are reported so the reader can apply either standard; the verdict requires surviving BOTH corrections is not gatekept on, but neither correction survives any test regardless, so the choice doesn't change the outcome here. |
| Retrain a new resolution model or add a heavier ensemble | Explicitly out of scope — this ADR searches existing signals; a new model is a different, larger, unmeasured investment that this search gives no evidence would even help (the raw quantile spread and its asymmetry — outputs of the existing model — already show no meaningful signal). |
