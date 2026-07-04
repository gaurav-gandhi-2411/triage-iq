# ADR-0017 — Expand gold eval set: n=60 → n=150 (W5)

**Status:** In progress (labeling phase — results pending, see placeholder in Consequences)
**Date:** 2026-05-31 (rebased from `feat/w5-eval-expansion`, 2026-07-04)
**Decider:** Gaurav Gandhi

---

## Rebase note (2026-07-04)

This ADR was originally drafted as `0011-eval-set-expansion.md` on the stale
`feat/w5-eval-expansion` branch (47 commits behind main, PR #8, failing CI, never merged — no
labeling was done on that branch). That branch's number collided with main's current
`ADR-0011-eval-regression-gate.md` (a different topic — the eval-quality-regression CI gate).
Renumbered to 0017, the next free slot.

The stratification rationale, bucket targets, component-diversity caps, and CI-width
justification below are carried over as-is from the original draft — they remain valid and were
not rewritten.

**This iteration now ALSO grows `data/gold_related.parquet`** (the retrieval-pair eval set), not
just `data/gold_triage_plans.parquet` (the judge-eval gold set) — this is new scope beyond the
original draft, added per GG's explicit decision: expanding `gold_triage_plans` alone would do
nothing to unblock the ADR-0016 W3 fine-tune rejection, since W3's insufficient-n problem was in
the retrieval *pairs* (`gold_related.parquet`, 1,435 pairs — both repos' recall@k 95% CI crossed
zero), not the judge-eval issue count. Growing the judge-eval gold set sharpens the grounding/
quality-regression metrics; growing the retrieval-pair set is the actual unblock for a future
honest retry of ADR-0016.

Additionally, two guarantees not present in the original draft's scripts were added during the
port, matching current main's disjointness discipline (see `scripts/w3_t5_eval.py`'s
`assert_eval_disjoint_from_train`):
- The T3 candidate generator (`scripts/w5_t3_generate_candidates.py`) now also excludes any
  candidate issue number present in `data/w3_split.parquet`'s train split (the ADR-0016 retrieval
  fine-tune's train/val/test assignment), so no candidate can later collide with a future
  retrieval-train split.
- The T3 ingestion script (`scripts/w5_ingest_labeled.py`) now asserts three-way disjointness
  (classifier-train, temporal-train, retrieval-train) on every accepted row before merging into
  the canonical gold set, and extracts a `related_issue_numbers` ground-truth column per accepted
  row using ONLY the high-confidence `body_ref` pattern strategy from
  `scripts/07_extract_related_pairs.py` (duplicate/dup-of/same-as/closing-as-dup-of #N). See
  `docs/eval/gold_labeling_protocol.md` for the labeling-protocol writeup of this new label type.

---

## Context

The current gold eval set (`data/gold_triage_plans.parquet`) has 60 issues (30 per repo), used to measure all four judge dimensions: component accuracy, similar-issue relevance, resolution estimate reasonableness, and priority alignment. With n=30 per repo, the 95% CI on any proportion metric (component accuracy, judge pass/fail rates) is ±18 percentage points — wide enough to make most deltas statistically ambiguous. This is the limiting factor on every metric comparison in W4 and W3.

**Specific motivations for expansion now:**
1. W3 fine-tuning produced +13 pp R@5 on a clean retrieval eval (n=152 test pairs) but only n=30 vscode / n=30 k8s for the judge eval. The similar-issues dimension (2.87/3) is near-ceiling and the CI is wide enough that even a real regression from a future model change could be masked.
2. The W4 data-card audit flagged two systematic biases in the current gold set (see T1 below). Both affect metric validity, not just CI width.
3. Future workstreams (W6 reranking, multi-repo expansion) will each need to measure delta effects on the same eval set. A larger, better-stratified set makes each subsequent eval more useful.

---

## T1 — Current Gold Set Audit

### Sampling methodology (existing)

`scripts/10_curate_triage_gold.py`:
- Source: temporal_val + temporal_test + classifier_val + classifier_test splits (no training data)
- Filter: component label non-null + resolution_hours > 0
- Stratification: 10 issues per resolution bucket (\<7d, 7–30d, \>30d), 30 per repo
- Seed: 42

### Composition

| Dimension | Current (n=60) |
|---|---|
| Per repo | 30 vscode / 30 k8s |
| Unique components | 28 (13 vscode + 16 k8s, 1 shared: api) |
| Resolution buckets | hours=8, days=12, weeks=20, months=8, long=12 |
| Priority | low=35, medium=6, high=19 |
| Era (year) | 2014=4, 2015=33, 2016=20, 2026=3 |

### Gaps identified

1. **Coarse resolution bucket** (critical for resolution estimate eval): The existing ">30d" bucket covers everything from 30 days to 8.5 years. Within it: months (30–180d)=8 issues, long (>180d)=12. The resolution predictor's CI coverage metric cannot distinguish model quality in the medium-slow range.

2. **Component concentration**: api=9/60, test-infra=8/60. Together = 28% of the gold set. The 28 unique components cover only about half of the classifiable component space: vscode has 50+ label categories, k8s has 54+ area/ labels. Many real-world component distributions are invisible in the current eval.

3. **Priority imbalance**: medium is 10% (6/60). Priority is inferred from resolution speed (fast=high, medium=within-a-week, slow=low), not from explicit label. The inferred priority skews toward the extremes, masking the system's ability to handle true medium-priority issues.

4. **Era representativeness** (acknowledged but not fixable without new scraping): 95% of issues from 2014–2016. The 2026 vscode slice (3 issues) is the only recent representation. This is an artifact of the corpus composition (k8s corpus = oldest 15k issues; vscode corpus = 2015–2016 + 2025–2026 with a 9-year gap). The expansion cannot fix the era gap for k8s — it can add more 2015–2016 coverage with better component diversity.

---

## Decision

Expand to n=150 (75 per repo) by adding 90 new issues (45 per repo) from held-out eval splits not currently in the gold set. The expansion uses 5-bucket resolution stratification (hours / days / weeks / months / long) to address gap #1, and component diversity constraints to address gap #2.

### T2 — Stratification targets

**Sampling pool available** (held-out eval splits, not in current gold):

| Repo | hours | days | weeks | months | long | total | unique_components |
|---|---|---|---|---|---|---|---|
| microsoft/vscode | 105 | 75 | 39 | 47 | 116 | 382 | ~30 |
| kubernetes/kubernetes | 144 | 210 | 183 | 211 | 321 | 1,069 | ~35 |

**Target allocation (+45 per repo = 9 per bucket × 5 buckets):**

| Repo | hours | days | weeks | months | long | total new |
|---|---|---|---|---|---|---|
| microsoft/vscode | +9 | +9 | +9 | +9 | +9 | +45 |
| kubernetes/kubernetes | +9 | +9 | +9 | +9 | +9 | +45 |

**Component diversity constraints:**
- Max 3 issues from any single component in the new 45 (per repo).
- Prefer components absent or underrepresented in current gold (≤ 1 existing issue).
- Specifically target: vscode — accessibility, file-explorer, languages-basic, install-update, themes, extensions; k8s — usability, kubelet, introspection, apiserver, controller-manager, kube-proxy.

**Final target (n=150):**

| Dimension | Current n=60 | Target n=150 | Change |
|---|---|---|---|
| Per repo | 30 | 75 | +45 each |
| Unique components | 28 | ≥50 | +22 min |
| Resolution months | 8 | ≥20 | +12 |
| Resolution hours | 8 | ≥20 | +12 |
| CI width (prop) | ±18 pp | ±11 pp | ≈40% tighter |

### Why 50/50 repo split

Both repos are equally important for the system comparison. vscode provides a more diverse component landscape; k8s provides a more uniform technical vocabulary. Keeping 50/50 prevents aggregate metrics from being dominated by one repo's characteristics.

---

## T3 — Candidate Pool

`scripts/w5_t3_generate_candidates.py` generates a 120-issue labeling worklist (60 per repo, 12 × 5 buckets), giving GG a 33% margin above the 45-issue target per repo to reject unsuitable candidates.

Each candidate is pre-populated with:
- Issue metadata: title, body_excerpt (300 chars), repo, component (from label), resolution bucket
- TF-IDF top-3 component predictions + confidence (System 1 output)
- BGE top-3 similar issue numbers + cosine scores (System 2 output)
- `stratum` label: e.g. `k8s-months-kubelet`
- `label_status`: "pending"

**System 4 (LLM) plans are NOT pre-generated** in the candidate pool to avoid spending Groq quota on issues that GG may reject. LLM plans should be generated in a focused run after GG selects the final 90 issues. See labeling protocol for the generation command.

Outputs:
- `data/gold_expansion_candidates.parquet` — 120 rows, full fields
- `data/gold_expansion_candidates.csv` — 120 rows, human-readable (body_clean dropped, body_excerpt kept)
- `reports/w5_gold_audit.json` — structured T1 audit + candidate pool summary

---

## Consequences

**RESULTS: PENDING.** Labeling of the ~90 new judge-eval issues and the `gold_related.parquet`
expansion have not happened yet (this port/rediff pass only produced the 120-candidate pool — see
T3 below — and the disjointness/related-issue-extraction machinery in `w5_ingest_labeled.py`). The
figures immediately below are the *targets* this expansion is designed to hit, not measured
outcomes. This section will be updated with actual n=150 per-repo means, CI widths, and the
`gold_related.parquet` pair-count delta once labeling completes and `--write` is run.

**When complete (after labeling):**
- Component accuracy CI: ±18 pp → ±11 pp (n=30→75 per repo)
- Judge delta detectability: minimum detectable effect ≈ 0.5 judge points (vs ~0.7 currently at α=0.05)
- Resolution bucket coverage: months + hours each get 20+ samples (vs 8 each currently)
- 36 new component categories tested (28→64 unique components in gold)

**What changes in the eval pipeline** (W5 follow-up, not this PR):
- `scripts/10_curate_triage_gold.py` gains a `--extend` flag to add labeled issues without regenerating the base set
- The gold set uses 5-bucket stratification (`hours/days/weeks/months/long`) instead of the coarse 3-bucket scheme
- The eval report reports per-bucket resolution estimate accuracy

**Risks:**
- New components may have harder-to-judge issues (more domain-specific), which could lower component accuracy metrics — this would be a real measurement, not a regression
- The era constraint (k8s = 2014–2015 only) cannot be resolved without new scraping; the expansion improves component and bucket coverage but not temporal diversity for k8s

**What is NOT done in this PR:**
- No changes to `scripts/11_evaluate_triage.py`
- No changes to `data/gold_triage_plans.parquet`
- No LLM triage plans generated for candidates

---

## Labeling instructions

See `docs/eval/gold_labeling_protocol.md` for the per-issue rubric and the exact criteria for accepting/rejecting candidates.

After labeling, the follow-up session wires the labeled additions into the gold set and re-runs the evaluation at n=150.

## Scripts

- `scripts/10_curate_triage_gold.py` — existing gold set curation
- `scripts/w5_t3_generate_candidates.py` — candidate pool generation, now with an added
  retrieval-train disjointness filter against `data/w3_split.parquet`'s train split (per-repo drop
  counts logged and reported in `reports/w5_gold_audit.json["pool_filter_stats"]`)
- `scripts/w5_ingest_labeled.py` — labeled-CSV ingestion, now with a three-way training
  disjointness hard-fail and `related_issue_numbers` (body_ref-only) extraction — dry-run default,
  `--write` gated

## Data artifacts

- `data/gold_triage_plans.parquet` — UNCHANGED (existing gold, n=60) until labeling + `--write`
- `data/w3_split.parquet` — ported reference data (ADR-0016 retrieval fine-tune's train/val/test
  split assignments), used by both scripts above for disjointness filtering/assertion
- `data/gold_expansion_candidates.parquet` — candidate pool (120 issues: 60/repo, confirmed full
  12-per-bucket×5 after both the current-gold filter and the new retrieval-train filter — see T3)
- `data/gold_expansion_candidates.csv` — human-readable view
- `reports/w5_gold_audit.json` — T1 audit + T2 sampling summary + `pool_filter_stats` (drop counts
  per repo per filter)
