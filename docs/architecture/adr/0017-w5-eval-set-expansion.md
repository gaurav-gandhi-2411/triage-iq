# ADR-0017 — Expand gold eval set: n=60 → n=150 (W5)

**Status:** Accepted (partial result — n=119 achieved, not the targeted n=150; see Consequences)
**Date:** 2026-05-31 (rebased from `feat/w5-eval-expansion`, 2026-07-04; results 2026-07-04)
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

**RESULTS: COMPLETE.** Labeling finished 2026-07-04 (GG, 76 candidates) and the merge into
`data/gold_triage_plans.parquet` ran the same day. The outcome differs from this ADR's original
n=150 target. Reported honestly below, including the parts that did not work out — this is an
honest, thorough negative-and-partial result, not a failure to hide (same framing as ADR-0016's
"rejected on current evidence, not failed").

### (a) Final composition

The gold set expanded from n=60 to **n=119** — not the originally-targeted n=150. After adding the
missing `classifier_train`/`temporal_train` disjointness filters to the candidate generator
(`scripts/w5_t3_generate_candidates.py`) — a real bug fix; the original filter only excluded
`retrieval_train` overlap — vscode's eligible candidate pool collapsed from a 60-candidate target
to **17 clean candidates**; 11 of those 17 passed quality labeling. k8s's pool held up: **138
eligible candidates** survived the same three-way filter (from an original 60-candidate target
pool), of which 59 were sampled into the labeling worklist and **48 were accepted**.

Final composition: **78 kubernetes/kubernetes (65.5%) / 41 microsoft/vscode (34.5%)** — a shift
from the existing set's clean 50/50 split (see "Why 50/50 repo split" above, now superseded by
this result).

GG's explicit decision: **do not trim k8s to force parity.** Discarding valid, disjoint,
quality-checked k8s data to hit an arbitrary ratio would invert the "quality over volume"
principle this whole workstream is built on. Condition attached to that decision: **every
downstream gate or metric must report per-repo, never a pooled number that could be misread as
balanced.**

vscode's per-bucket targets are missed explicitly, not silently absorbed — a visible, acknowledged
gap:

| Bucket | vscode target (new) | vscode actual (new) |
|---|---|---|
| days | 9 | 6 |
| long | 9 | 8 |
| months | 9 | 6 |
| hours | 9 | 11 |
| weeks | 9 | 10 |

### (b) W3 unblock — dead end this iteration

The hypothesis that growing the gold set would also grow `gold_related.parquet`'s retrieval-eval
pairs (unblocking ADR-0016's rejected W3 fine-tune) did not materialize. This was verified
analytically before attempting any mining pass, not discovered after the fact:
`scripts/07_extract_related_pairs.py`'s `body_ref` pattern already scans the **entire corpus**
(not a sample) and found exactly **4 hits total**, historically, across both repos
(`data/gold_related.parquet`: 4 of 1,435 pairs tagged `source == "body_ref"`) — and all 4 are
already inside `w3_split.parquet`'s **TRAIN** split (i.e., already spent as training data, not
available as new test signal).

Corpus size is unchanged since that extraction ran (kubernetes/kubernetes n=15,000,
microsoft/vscode n=7,028 — identical to the W3 eval run). Applying the same `BODY_REF_PATTERNS`
directly to the new 76-candidate W5 pool (title + body_clean, before the existence/predate checks)
yields **0/76 hits** — confirming there is no additional signal to mine at this corpus size. This
is not a sampling problem; the signal is structurally exhausted.

**Conclusion: W5 does not, and structurally cannot, unblock W3 with the current corpus. Corpus
growth (new scraping) is the actual prerequisite for any future W3 retry**, and that is out of
scope for this iteration.

### (c) vscode data ceiling — a project-level finding, not a W5 footnote

This is not a W5-specific gap. The pattern recurs three times independently across three different
workstreams:

1. **W3's retrieval pairs**: microsoft/vscode has 411 total `gold_related.parquet` pairs vs.
   kubernetes/kubernetes's 1,024 — vscode's fine-tune arm couldn't reach n=300 for the
   ADR-0006-style robustness escalation.
2. **Resolution-predictor CQR calibration** (ADR-0010): vscode's true-test set is n=370 under the
   selected 40/60 calibration/test split (n=432 under the 30/70 split documented for comparison) —
   see `docs/architecture/adr/0010-conformal-quantile-regression.md` for the full split table.
3. **W5 (this ADR)**: vscode — only 17 of an original 60-candidate target pool survived
   triple-disjointness filtering, vs. kubernetes/kubernetes's 138 of 60.

**Root cause, verified directly this session:** vscode's `classifier_train/val/test` split (1,862
issues total) and `temporal_train/val/test` split (6,154 issues total) are built over a much
smaller absolute corpus (7,028 issues total) than kubernetes/kubernetes's (15,000). Any small
held-out slice from one split overlaps the other split's much larger train allocation by base
rate — measured: **92.2%** overlap between vscode's `classifier_val`+`classifier_test` (n=374) and
`temporal_train` (n=4,923). kubernetes/kubernetes shows the same base-rate dynamic but at a lower
measured rate (76.9%; `classifier_val`+`classifier_test` n=572 vs. `temporal_train` n=11,974) — it
survives not because the mechanism differs, but because its absolute corpus is 2.1x larger, giving
enough headroom even with substantial overlap.

**State plainly: vscode is the binding data constraint on TriageIQ, not an incidental gap in any
one workstream.** Future work must either (i) grow the vscode corpus (re-scrape a larger or more
contiguous history), or (ii) explicitly treat kubernetes/kubernetes as the primary evaluated repo
and vscode as a data-limited secondary. Do not let any future claim imply 41 vscode issues
supports the same statistical confidence as 78 k8s issues.

### (d) Deferred: human-confirmed `body_related` pairs

GG's explicit decision: **defer, do not attempt now, do not reject either.** `body_related`
(Closes/Fixes/#N patterns, ~1,010 pairs total in `gold_related.parquet`) was excluded from the
automatic/bulk extraction because ADR-0007 found ~70% are PR→issue references, not genuine
issue-to-issue relatedness — that exclusion is correct and unchanged for any bulk/automatic use.

But per-pair **manual confirmation** (a human or careful agent reading each hit to filter the ~30%
ADR-0007 estimated as genuine) is a different, not-yet-attempted mechanism — the same discipline
already applied to validate the 4 existing `body_ref` pairs in ADR-0007's own manual spot-check
table. This remains a real, viable, but labor-intensive path to eventually growing
`gold_related.parquet` beyond what corpus growth alone would give. Noted here as a candidate
follow-up task for whoever eventually attempts the W3 retry — not committed to now.

---

**What changed in the eval pipeline (this PR):**
- `scripts/w5_t3_generate_candidates.py` gained `classifier_train`/`temporal_train` disjointness
  filters (previously only excluded `retrieval_train` overlap) — the bug fix that produced the
  138-vs-17 eligible-pool split in (a) above.
- `scripts/w5_ingest_labeled.py` merges labeled candidates into `data/gold_triage_plans.parquet`,
  asserting three-way training disjointness before every write, and extracts
  `related_issue_numbers` (body_ref-only) per accepted row.
- `data/gold_triage_plans.parquet` is now n=119 (78 k8s / 41 vscode), up from n=60 (30/30).

**Risks (carried forward, now with measured outcomes):**
- New components introduce harder-to-judge issues in the accepted set (24 new component
  categories across both repos) — any component-accuracy delta against the old n=60 baseline
  should be read as a real measurement against a broader, harder distribution, not a like-for-like
  regression.
- The era constraint (k8s = 2014–2016 only) remains unresolved, as anticipated — the expansion
  improved component and bucket coverage but not temporal diversity for k8s.
- vscode's resolution-bucket coverage remains uneven post-expansion (days/long/months under
  target — see (a)) — this is now a durable, documented property of the gold set, not a transient
  gap to be silently closed later.

**What is NOT done in this PR:**
- No changes to `scripts/11_evaluate_triage.py` (re-running the eval suite against n=119 is a
  separate, later step).
- No re-record of `eval/eval_set.jsonl`, any cassette, or `reports/eval_baseline.json` — pending a
  further GG decision on the Groq re-record.
- `gold_related.parquet` is unchanged (see (b) — W3 does not unblock this iteration).
- No `body_related` manual-confirmation pass (see (d) — deferred, not rejected).

---

## Labeling instructions

See `docs/eval/gold_labeling_protocol.md` for the per-issue rubric and the exact criteria for accepting/rejecting candidates.

Labeling completed 2026-07-04 (GG, 76 candidates: 59 accept / 17 reject). The labeled additions
were merged into the gold set at n=119, not n=150 — see Consequences (a) for why.

## Scripts

- `scripts/10_curate_triage_gold.py` — existing gold set curation
- `scripts/w5_t3_generate_candidates.py` — candidate pool generation. Originally added a
  retrieval-train disjointness filter against `data/w3_split.parquet`'s train split; a later fix in
  this iteration (see Consequences (a)) added the missing `classifier_train`/`temporal_train`
  disjointness filters as well, which is what shrank the pool from 120 (60/repo) to 76 (59 k8s / 17
  vscode) — per-repo drop counts logged and reported in
  `reports/w5_gold_audit.json["pool_filter_stats"]`.
- `scripts/w5_ingest_labeled.py` — labeled-CSV ingestion, with a three-way training disjointness
  hard-fail and `related_issue_numbers` (body_ref-only) extraction — dry-run default, `--write`
  gated. Also fixed in this iteration: CSV-sourced `created_at` (a plain string from `pd.read_csv`)
  is now coerced to `Timestamp` in `build_gold_rows`, since the mixed str/Timestamp column it
  produced previously was rejected by pyarrow on `to_parquet`.

## Data artifacts

- `data/gold_triage_plans.parquet` — **CHANGED**: n=60 (30/30) → n=119 (78 kubernetes/kubernetes /
  41 microsoft/vscode)
- `data/gold_expansion_candidates_labeled.csv` — GG's 76 labeled candidates (59 accept, 17 reject),
  the source input to the merge above
- `data/w3_split.parquet` — ported reference data (ADR-0016 retrieval fine-tune's train/val/test
  split assignments), used by both scripts above for disjointness filtering/assertion
- `data/gold_expansion_candidates.parquet` — candidate pool, regenerated after the
  classifier/temporal disjointness fix: 76 total (59 kubernetes/kubernetes / 17 microsoft/vscode),
  down from the original 120 (60/repo) — see T3 and Consequences (a)
- `data/gold_expansion_candidates.csv` — human-readable view
- `data/gold_related.parquet` — UNCHANGED (1,435 pairs) — see Consequences (b)
- `reports/w5_gold_audit.json` — T1 audit + T2 sampling summary + `pool_filter_stats` (drop counts
  per repo per filter)
