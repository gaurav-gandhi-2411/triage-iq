# Investigation: gold-set train-data leakage

**Date:** 2026-07-06
**Status:** CONFIRMED, then **REMEDIATED same day** (phases 1–2 + phase-4 invariant — see
§5). Phase 3 (baseline re-record) remains blocked on the ADR-0019 Ollama judge switch.
Pre-remediation finding: 54/119 gold rows (45.4%) overlapped training data; 54/60 rows (90%)
of the frozen CI-baseline eval set were contaminated.
**Scope:** triage gold set (`data/gold_triage_plans.parquet`, n=119) and every metric derived
from it. Reproduce all numbers: `python scripts/verify_gold_train_overlap.py`
(writes `reports/gold_leakage_overlap.json`).
**Verdict on reported metrics:** README test-split metrics (classifier, retrieval, CQR) are
CLEAN. The LLM-judge eval baseline (`reports/eval_baseline.json`) and the CI quality gate
built on it are CONTAMINATED.

---

## 1. Mechanism — how the leakage occurred

### Data lineage

```
scripts/01_scrape_issues.py
  └─ data/processed/issues_{repo}.parquet          (full corpus per repo)

scripts/03_split.py  — creates TWO INDEPENDENT partitions of the SAME corpus:
  ├─ time_based_split (sort by created_at, 80/10/10)
  │    └─ {repo}_temporal_{train,val,test}.parquet   → trains LightGBM resolution
  │                                                     predictor (09) + CQR calib (10)
  └─ stratified_classifier_split (random, seed 42, 80/10/10)
       └─ {repo}_classifier_{train,val,test}.parquet → trains TF-IDF+LR classifier (04)

scripts/10_curate_triage_gold.py  — original 60 gold rows
  └─ samples from UNION(temporal_val, temporal_test, classifier_val, classifier_test)
     with NO exclusion against either scheme's train split          ← THE DEFECT

scripts/w5_t3_generate_candidates.py + w5_ingest_labeled.py — W5 expansion (60 → 119)
  └─ candidates filtered against classifier_train + temporal_train + retrieval-train,
     assert_gold_disjoint_from_train() hard-fails on violations
     — but ONLY on the newly accepted rows; the legacy 60 were never re-checked

eval/eval_set.jsonl (frozen from the OLD n=60 gold) → eval/record_cassettes.py
  → reports/eval_baseline.json → eval/test_quality_regression.py (CI gate)
```

### The defect

`scripts/10_curate_triage_gold.py::load_eval_splits` (lines 38–53) treats
"val + test of *any* split scheme" as held-out. That is only true within a scheme. The
temporal and classifier splits are **independent partitions of the same issues**: membership
in one scheme's val/test says nothing about the other scheme's train. A row sampled because
it sits in `classifier_val` (random split, spans all eras) has roughly a `train_pct` = 80%
prior of sitting in `temporal_train` (earliest 80% by `created_at`) — higher in practice for
vscode because the gold stratification favored old (2014–2016) issues, which are almost all
inside the temporal-train era (ADR-0017 measured a 92.2% vscode val/test↔temporal_train
base-rate overlap). The script's own docstring — "Sampled from test split only (no training
leakage)" — is false across schemes.

The W5 expansion (2026) added exactly the right guard
(`w5_ingest_labeled.py::assert_gold_disjoint_from_train`, three-way, hard-fail) but ran it
only over the **delta** (newly accepted rows), never over the merged artifact. The
contaminated legacy rows persisted into the n=119 gold set, and the frozen n=60
`eval_set.jsonl.bak60` — the set behind the CI baseline — is 90% contaminated.

### Why existing checks didn't catch it

- The disjointness assertion is ingest-time, delta-scoped; there is no invariant over the
  gold artifact itself, so pre-existing contamination is invisible to it.
- `scripts/verify_t1_overlap.py` audits a *different* gold set (`gold_related.parquet`,
  retrieval pairs vs `w3_split.parquet`) — it never looks at `gold_triage_plans.parquet`.
- The curation script's docstring asserted safety, so nothing prompted a re-check.

---

## 2. Overlap quantification (n stated everywhere)

Method: three channels, all in `scripts/verify_gold_train_overlap.py`, evidence in
`reports/gold_leakage_overlap.json`.
1. **ID overlap** — exact `(repo, number)` intersection.
2. **Content hash** — sha256 of lowercased whitespace-collapsed `title+body_clean`
   (catches re-filed identical text under a different number).
3. **Embedding near-dup** — max BGE cosine of each gold row vs all train rows, vectors
   reconstructed from `data/models/dup_index_{repo}_bge` (same normalized vectors production
   retrieval uses; thresholds 0.90 / 0.95).

Cohorts: `original_60` = curated by `10_curate_triage_gold.py` (never checked);
`w5_added` (n=59) = passed ingest-time disjointness.

### ID overlap (gold n=119: vscode 41 = 30 orig + 11 w5; k8s 78 = 30 orig + 48 w5)

| Training set | vscode (n set) | vscode overlap | k8s (n set) | k8s overlap | Cohort |
|---|---|---|---|---|---|
| classifier_train | 1,488 | **5** | 2,284 | **15** | all original_60 |
| temporal_train | 4,923 | **27** | 11,974 | **14** | all original_60 |
| retrieval_train (w3) | 193 | 2 | 1,300 | 3 | all original_60; moot — fine-tune rejected, base BGE shipped |
| classifier_val (model selection) | 187 | 12 | 286 | 28 | mixed |
| temporal_val | 615 | 10 | 1,496 | 29 | mixed |
| CQR calibration slice (first 30% of temporal_test) | 184 | 2 | 449 | 4 | mixed |

**Union against shipped-model training data (classifier_train ∪ temporal_train):
vscode 30/30 original rows, k8s 24/30 — 54/60 of the original cohort, 54/119 (45.4%) of
current gold. The w5_added cohort has 0 ID overlaps with any train set.**
Full issue-number lists: `reports/gold_leakage_overlap.json → repos.*.id_overlap`.

### Content-hash exact duplicates (cross-number)

- vscode: 6 gold↔classifier_train pairs with byte-identical normalized text under different
  issue numbers (e.g. gold #311565 ≡ train #311562/311563/311564). All 6 involve gold rows
  already in the ID union — adds 0 new contaminated rows.
- k8s: 0 cross-number hash matches.

### Embedding near-duplicates (cross-number, gold vs train)

| Pair | Cosine | Cohort | Already ID-contaminated? |
|---|---|---|---|
| vsc gold #311565 ~ train #311562 | 1.000 | original_60 | yes |
| vsc gold #311543 ~ train #311544 | 1.000 | original_60 | yes |
| k8s gold **#14398** ~ train #14399 | 0.907 | **w5_added** | **no — new finding** |
| k8s gold #14598 ~ train #8943 | 0.920 | original_60 | yes |

Distribution context: max-cosine p50 ≈ 0.78–0.80, p90 ≈ 0.83–0.85 across all four
gold×train scans — the ≥0.90 pairs are far outliers, not threshold artifacts.
**Net near-dup addition: 1 row (k8s #14398, W5 cohort)** — ID-level ingest checks
structurally cannot catch re-filed duplicate content; needs manual review.

### Cross-check: in-flight eval set

The dirty working tree's `eval/eval_set.jsonl` (65 rows = 54 k8s + 11 vsc, mid-re-record for
the ADR-0019 judge switch) contains **0 contaminated rows** and 119 − 54 = 65 — it is exactly
the clean subset of the current gold. The frozen `eval_set.jsonl.bak60` contains 54/60
contaminated rows. (Verified by key intersection; not an assumption.)

---

## 3. Blast radius — verdict per reported metric

| Metric (where reported) | Value | Evaluated on | Verdict | Evidence |
|---|---|---|---|---|
| Classifier top-1 acc (README): vsc 69.0%, k8s 51.4%; macro-F1 0.585 | test split | `{repo}_classifier_test` | **CLEAN** (caveat) | ID-disjoint from train by construction. Residual near-dup channel measured: 1 exact-text dup + 3 near-dups ≥0.95 of 187 vsc test rows (≤2.1%), 1 + 2 of 286 k8s rows (≤1.0%) — bounded below the noise floor for these ns (rule 93). |
| Retrieval MRR 0.294, R@5 36.7%, R@10 52.1% (README) | W3 test pairs | base BGE, zero-shot | **CLEAN** | ADR-0016 fine-tune was REJECTED; shipped retriever is pretrained `BAAI/bge-base-en-v1.5`, never trained on repo data — no train set exists to leak from. `w3_split` train pairs are used only as a disjointness reference. |
| Resolution MAE (README): k8s 104.8d, vsc 116.1d; CI coverage 77.5%/76.5% | temporal true-test | last 70% of `temporal_test` | **CLEAN** | Temporal split disjoint by construction (ADR-0009 de-leak); calibration slice (first 30%) disjoint from true-test. Note: `reports/eval_summary.json` carries different vsc numbers (MAE 6.02d, cov 0.416) than README — source discrepancy flagged, separate issue. |
| LLM-judge baseline 10.03/15 (vsc), 10.87/15 (k8s), 10.45 overall (`reports/eval_baseline.json`) | frozen `eval_set.jsonl.bak60`, n=60 | **CONTAMINATED** | **54/60 rows (90%)** in ≥1 shipped-model train set. Per dimension: `component_match` biased on 20/60 rows (classifier trained on the row + its component label); `resolution_estimate_reasonableness` biased on 41/60 rows (LightGBM trained on the row's actual resolution time); `priority_alignment` derivatively biased (gold_priority is inferred from resolution speed — the leaked target) ; `similar_issues_relevance` channel clean (zero-shot BGE). |
| CI quality gate (`eval/test_quality_regression.py`: zero-tolerance drop vs baseline) | same n=60 baseline | **CONTAMINATED** (inherited) | Gate compares against a baseline measured 90% on training rows. Currently `continue-on-error: true` in `eval-gate.yml` (informational), so no CI decisions were wrongly enforced — but the numbers it reports are invalid. |
| `eval/test_invariants.py` CQR-coverage / retrieval-overlap assertions over eval rows | n=60 rows | **CONTAMINATED** (CQR-related) | CQR coverage asserted over rows where 41/60 were in `temporal_train`; retrieval-overlap invariant unaffected (zero-shot). |
| Judge-model comparison checkpoints (`data/judge_scores_checkpoint_*.jsonl`, W11/W12 runs) | same eval set | **CONTAMINATED** (inherited) | Any cross-model comparison ran on the same 90%-contaminated set. Relative rankings may survive (all models judged on the same rows); absolute scores do not. |
| CQR conformal validity (production intervals) | calibration slice | **CLEAN** (minor caveat) | Calibration uses temporal_test[:30%], disjoint from temporal_train — the model artifact and its intervals are sound. 6 gold rows sit inside calibration slices (2 vsc, 4 k8s), which only biases coverage *measured on the gold set*, not production behavior. |

**Bottom line:** the models and the README test-split metrics stand. Everything derived from
the gold set — the judge baseline, the CI gate, judge-model comparisons — is invalid until
the gold set is rebuilt. No production/deploy action needed (revision
`triageiq-api-00053-59v` untouched; model artifacts are not the problem).

---

## 4. Remediation plan (PLAN ONLY — not executed)

### Phase 1 — rebuild the gold set

1. Drop the 54 ID-contaminated rows (lists in `reports/gold_leakage_overlap.json`).
   Post-drop clean core = 65 rows (54 k8s / 11 vsc) — exactly the in-flight
   `eval_set.jsonl`, so the current W5 re-record work already targets the right subset.
2. Manually review k8s **#14398** (w5_added, cosine 0.907 to classifier_train #14399,
   adjacent numbers → likely re-filed dup). Recommend drop; document either way.
3. Backfill — vscode is the gap (11 clean rows vs 41 target). Use
   `w5_t3_generate_candidates.py` (already excludes all three train sets) + the existing
   labeling protocol + `w5_ingest_labeled.py`. Restore ADR-0017's stratification targets;
   note the 2014–2016 era stratum is structurally hard to fill for vscode (that era is
   mostly temporal_train) — accept a shifted era distribution and document it rather than
   quietly relaxing disjointness.

### Phase 2 — re-derive eval artifacts (order matters)

4. Land the in-flight ADR-0019 Ollama judge switch **first**, then re-freeze
   `similar_issues` (`eval/freeze_similar_issues.py`) and re-record cassettes once —
   avoids paying for two full re-records.
5. Regenerate `reports/eval_baseline.json` from the clean set. **Expect the judge means to
   drop** — the old 10.45/15 was measured mostly on rows the models memorized; report the
   honest number (rules 41/53).

### Phase 3 — reset the CI gate

6. Update `eval_set_hash` / `cassette_hash` and the per-repo baselines in
   `eval_baseline.json`. Keep the zero-tolerance drop gate, but only against the new
   baseline. n caveat: at n=65 (interim) a per-repo mean shift of ±0.3/15 is within noise
   (rule 93) — do not treat early deltas as regressions; revisit when backfill restores n.
7. Decide whether to promote `eval-gate.yml` jobs from `continue-on-error: true` to
   blocking — a gate that can't fail is documentation, not a gate.

### Phase 4 — fix the class, not the instance (rule 85)

8. Fix or retire `scripts/10_curate_triage_gold.py`: exclude both train splits (and w3
   train numbers) in `load_eval_splits`, and correct the false docstring.
9. Add a **permanent whole-artifact invariant** to `eval/test_invariants.py`:
   gold ∩ (classifier_train ∪ temporal_train ∪ w3-train) = ∅, plus the cross-number
   content-hash check. This turns future contamination into a CI failure instead of a
   one-time audit. (Near-dup ≥0.95 screening belongs at ingest in
   `assert_gold_disjoint_from_train`; too slow for every CI run.)
10. Re-run `scripts/verify_gold_train_overlap.py` after rebuild; require zeros across all
    three channels before re-freezing the baseline.

### What does NOT need re-running

- README classifier / retrieval / resolution metrics (clean, per §3).
- No model retraining, no redeploy — the training pipeline and artifacts are sound; the
  defect is entirely in evaluation-set construction.

---

## 5. Remediation executed (2026-07-06)

Phases 1–2 and the phase-4 invariant were executed via `scripts/remediate_gold_leakage.py`
(idempotent, dry-run by default, `--write` applied). Evidence: `reports/gold_remediation.json`.
Phase 3 (re-freeze eval_set, re-record cassettes, re-derive `eval_baseline.json`, reset CI
gate hashes/thresholds) is **deliberately not executed** — blocked on the ADR-0019 Ollama
judge switch so the re-record happens once.

### Drop (before → after, per repo)

| Repo | Before | Dropped (ID) | Dropped (near-dup) | After | Cohorts after |
|---|---|---|---|---|---|
| microsoft/vscode | 41 (30 orig + 11 w5) | 30 | 0 | **11** | 11 w5_added |
| kubernetes/kubernetes | 78 (30 orig + 48 w5) | 24 | 1 (#14398) | **53** | 47 w5_added + 6 original_60 |
| **Total** | **119** | **54** | **1** | **64** | |

k8s **#14398** dropped per explicit decision (BGE cosine 0.907 to classifier_train #14399 —
re-filed duplicate). Near-dup admission threshold fixed at **cosine 0.90**: measured
non-duplicate background tops out at 0.85–0.89 (p90 0.83–0.85) while confirmed re-filed
duplicates sit at 0.907–1.0 — the bands do not overlap. Contamination sets were
**recomputed live** from the split parquets at drop time, not read from this report.

### Reconciliation with the in-flight eval set (verified, not assumed)

`eval/eval_set.jsonl` (65 rows, in-flight for the ADR-0019 re-record) minus clean gold
(64 rows) = exactly `{kubernetes/kubernetes #14398}`; gold − eval_set = ∅
(`reports/gold_remediation.json → reconciliation.reconciles_as_expected: true`).
**Action required before phase-3 re-record:** remove #14398 from `eval/eval_set.jsonl`.

### Post-remediation audit (all zeros)

`scripts/verify_gold_train_overlap.py` re-run on the clean set: ID overlap 0, hash overlap 0
(same- and cross-number), near-dups ≥0.90 = 0 against classifier_train, temporal_train, and
retrieval-train for both repos (max residual cosine: vsc 0.824, k8s 0.890). Residual
secondary overlaps remain by design and are documented: val-split overlaps (model selection,
not training) and 2+2 rows in the CQR calibration slices — these bias only
coverage-measured-on-gold, not the conformal model itself.

### vscode backfill: structurally impossible from the current corpus (n stated)

Enumerating ALL remaining vscode candidates from the held-out eval-split union with the dual
admission checks (ID-disjoint from all three training sources AND max BGE cosine < 0.90),
excluding GG's 6 W5 rejections: of 1,592 unique pooled rows, 410 have component +
resolution labels; 393 of those overlap training IDs; 11 are already in gold; 6 were
rejected → **0 eligible candidates** (`reports/gold_remediation.json → vscode_backfill`).
This is consistent with the W5 round (17 candidates found then = today's 11 accepts + 6
rejects) and makes ADR-0017's vscode data-ceiling finding absolute: per-repo balance cannot
be restored from the existing corpus. Options, for a future decision: (a) scrape additional
vscode eras/issues (extend `01_scrape_issues.py` window) and route new candidates through
the W5 labeling + ingest flow; (b) accept the 11/53 imbalance and report per-repo metrics
only (never pooled); (c) relax nothing — the disjointness discipline stays.
`data/gold_backfill_candidates_vscode.csv` was written with headers and 0 rows as evidence.

### Regression guard now active

`eval/test_invariants.py::test_gold_disjoint_from_training_ids` and
`::test_gold_no_near_duplicate_of_training_text` assert over the FULL gold artifact on every
run (not delta-scoped). Demonstrated failing on the pre-remediation set — ID test: all 54
rows across 6 repo×source pairs; near-dup test: all 4 pairs including #14398 — and passing
post-remediation (2 passed, 1.5s; near-dup test reconstructs vectors from the saved BGE
index, no model load). `scripts/10_curate_triage_gold.py` docstrings corrected and the
script marked DEPRECATED for gold regeneration.

### Metric status after this remediation

Unchanged from §3 until phase 3 runs: the judge baseline and CI gate remain invalid (they
still reference the contaminated frozen n=60 set) — they become valid only after re-record
against the clean 64-row set (or its labeled successor). README metrics were and remain
CLEAN.

---

## Appendix: contaminated gold issue numbers (ID union, shipped models)

- **microsoft/vscode (30/30 original):** 161, 567, 814, 1045, 1057, 1502, 1508, 2067, 2093,
  2115, 2468, 2496, 2636, 3047, 3077, 3299, 3360, 3486, 3655, 3671, 3826, 4223, 4338, 4601,
  4741, 4759, 4760, 311414, 311543, 311565
- **kubernetes/kubernetes (24/30 original):** 140, 1548, 1678, 3121, 3481, 3606, 4746, 4947,
  5634, 5963, 8362, 10497, 11079, 11243, 13190, 13878, 13890, 14228, 14284, 14598, 14669,
  14743, 14781, 14921
- **Manual review:** kubernetes/kubernetes #14398 (near-dup, w5_added cohort)

Per-training-set breakdowns, hash pairs, and near-dup similarity scores:
`reports/gold_leakage_overlap.json`.
