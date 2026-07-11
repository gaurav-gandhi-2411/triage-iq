# ADR-0030 — Phase C: Product-Task Gold Feasibility (Data Decision)

**Status:** Proposed — go/no-go escalated to Gaurav Gandhi (data-collection scope is a human call)
**Date:** 2026-07-12
**Decider:** Gaurav Gandhi (analysis by CC, autonomous, analysis-only per spec.md)

---

## Context

Two things are blocked on the same missing data — genuine product-task (issue→related-issue)
gold pairs, at powered scale, per repo:

1. The k8s retriever's LIVE product-task recall@5 is UNMEASURABLE (ADR-0028): zero product-task
   test pairs exist against the live k8s index, per the w3-retry split.
2. The Phase 2 retrieval fine-tune is HELD (ADR-0027): its product-task improvement is
   directionally positive on both repos but crosses zero, underpowered to gate on.

This ADR decides whether mining more issue→issue pairs at scale is worth it, with numbers,
before committing to a collection build. Analysis only — no mining at scale, no fine-tune
cutover, no re-indexing. Reproducible via `scripts/phaseC_channel_mining.py`,
`scripts/phaseC_live_probe.py`, `scripts/phaseC_power_and_liveindex.py`,
`scripts/phaseC_assemble_report.py` → `reports/phaseC_feasibility.json`. Manual precision
judging recorded in `reports/phaseC_precision_review.json`.

## Decision — the headline finding: MEASURING k8s doesn't need new mining at all

The most consequential finding upends the framing ADR-0028 escalated under: k8s's live
product-task recall@5 was called "unmeasurable" because zero product-task pairs fell in the
w3-retry **test** split against the live index. But the live-serving retriever
(`dup_index_kubernetes_kubernetes_bge`, `BAAI/bge-base-en-v1.5`, loaded by
`src/triage_iq/api/loader.py`) is an **off-the-shelf pretrained embedder, never trained on any
gold pair** — only the separate, unshipped `bge_finetuned_*_v2` artifact is. The w3-retry
train/val/test split exists solely to prevent leakage for *that* fine-tuned model's training; it
carries **zero leakage risk** for the live model. Every product-stratum pair whose query and
target both fall in the live index's number range is usable for measurement right now,
regardless of its split label.

Concretely: the live k8s index is exactly `#1–15,002` (15,000 records, confirmed from
`data/models/dup_index_kubernetes_kubernetes_bge/meta.pkl`). 277 of k8s's 776 product-stratum
pairs fall in that range — not the 0 the test-split framing implied. At a plausible live-index
recall prior (p=0.25–0.30, informed by the v2-index product baseline of 0.228–0.263, and the
v1 index being smaller/easier per ADR-0027's own note), n=277 already sits at or near a ±5pp CI
half-width (±5.1–5.4pp); only the maximally conservative p=0.5 assumption falls short (108 more
needed). **This is a re-eval action — select product-stratum pairs by live-index membership
instead of split label and re-run `scripts/08_build_similar_issue_index.py`'s evaluation against
the v1 index — not a new mining or scraping effort.**

### 1. Channels — yield + precision, sampled not assumed

| Channel | k8s | vscode | Verdict |
|---|---|---|---|
| **A. Extended body** ("related to/similar to/refs #N", issue URLs) | 9 new candidates (mined out; same class as the already-used 779-pair `k8s_extended_mine` channel), precision 78–89% on n=9 | 405 candidates, precision **30–43%** on n=30 — materially noisier than vscode's existing narrower patterns | k8s: exhausted. vscode: not recommended unreviewed. |
| **B. Comments** ("related to/see also #N") | Not locally mineable — k8s has **0% comments_data coverage** (neither scrape fetched comments). Live probe (n=25): 8% hit rate. | 94 local candidates (67 in live index), precision **76–84%** after excluding ~20% dup-contamination (`/duplicate` triage-bot comments regex-matched as false positives) — net ~75 usable pairs, zero new scraping | **Best channel found.** vscode: usable now. k8s: needs an unproven scrape. |
| **C. Native linked issues** (`connected`/`disconnected` timeline events) | 0/25 sampled | 0/25 sampled | **Dead channel**, both repos — matches ADR-0026's prior finding that structured GitHub link events go unused by these communities. No further investment. |
| **D. Timeline cross-referenced** | 44% of sampled issues have ≥1 issue-sourced cross-ref, but **71% of raw pairs target issue #1** (a generic "unit test coverage" meta-issue) — a noise magnet. Excluding it, precision is 47–73% on n=15. | 48% issue-sourced rate; same #1-hub noise, but a genuine hub also exists (#98479, a real cluster of related keybinding-conflict reports) | Real signal, **needs a validated hub-exclusion filter before scale mining** — not ready today. |
| **E. Label-cluster** (same component, ±14 days) | 11,793 candidates, precision **3% strict / 30% lenient** on n=30 | 8,409 candidates, same failure mode expected | **Not viable standalone** — confirms the spec's own caution: high raw yield here is noise, not signal. |

### 2. Power — pairs needed

| Target | k8s | vscode |
|---|---|---|
| **Measure live recall (±5pp CI)** | 289–385 needed (prior-dependent); **277 already usable, gap 0–108** | Already done (ADR-0028: 22.4% CI[17.7,28.0]) |
| **Gate fine-tune, 80% power** | 466 test pairs (+409) / **~6,075 total pairs** at the current 7.67% test-fraction — ~8x the current 776-pair product stratum, >1.5x the *entire* current gold set (4,030 pairs, all strata) | 709 test pairs (+428) / **~1,169 total pairs** at the current 60.7% test-fraction — ~2.3x the current 505-pair product stratum |
| Point-estimate fragility | +3.51pp — near the 0.03 near-zero flag; if true effect is half, ask nearly quadruples (1,864 test pairs) | +3.20pp — same caveat; half-effect ask is 2,836 test pairs |

The test-fraction figures above are **not a fixed design ratio** — ADR-0027's split-correction
lets product pairs "ride along" with the gate-stratum's chronological walk, producing k8s's
unusually low 7.67% and vscode's unusually high 60.7% test shares as algorithm artifacts, not
targets. Projecting total-pairs-needed by dividing through these ratios assumes future mining
splits the same way; flagged, not silently assumed.

### 3. Live-index quantification

| | k8s | vscode |
|---|---|---|
| Live index size | 15,000 (`#1–15,002`, a strict number range) | 7,028 (a specific issue-number set, not a range) |
| Product pairs usable now | 277 / 776 (35.7%) | 292 / 505 (57.8%, already realized in ADR-0028) |
| New-channel candidates in-range | A: 6/9. E: 5,576/11,793 (low value — E is noise) | B: 67/94 (high value). A: 242/405 (low precision). E: 4,983/8,409 (noise) |

## Go/no-go (mixed, as anticipated)

1. **MEASURE k8s live recall: GO, near-zero cost.** Re-eval with existing data (277 in-range
   pairs, zero leakage risk); optionally supplement with ~50–100 more pairs for a tighter CI
   under a conservative prior. This is the single highest-leverage, lowest-cost action in this
   analysis and does not require this ADR's mining question to be answered "yes" at all.
2. **SHIP the k8s fine-tune (gate the product task): NO-GO at current scope.** The ask (~6,075
   total product pairs) is disproportionate to the corpus; no channel found here gets close.
   Stays HELD per ADR-0027, now with a number attached to *why* it's not a near-term target.
3. **SHIP the vscode fine-tune (gate the product task): MIXED, not closed by this analysis.**
   The gap (+428 test pairs / ~664 total beyond channel B's ~75) is a *bounded, proportionate*
   ask relative to vscode's current data — unlike k8s's. Channel B (comments) is immediately
   usable for a partial contribution; channel D (cross-referenced, hub-filtered) is a promising
   but unvalidated next step. **Recommend a scoped Phase C-build spec focused on vscode**, using
   channel B directly and a channel-D hub-filter design + validation pilot as its first task —
   not a blanket "mine everything" build.
4. **Channels C and E: no further investment.** Dead / too noisy respectively.

## Consequences

- **What changes:** nothing executed yet — analysis only, per spec. `reports/phaseC_feasibility.json`
  is the reference artifact for a Phase C-build spec if GG approves the vscode-focused scope, and
  for an immediate small re-eval task for k8s live measurement regardless.
- **What becomes easier:** the k8s live-measurability question is resolved as "already
  near-answerable," removing the apparent blocker ADR-0028 escalated. The vscode fine-tune's
  data gap is now a concrete, bounded number instead of an open-ended "needs more data."
- **What becomes harder:** nothing new; the k8s fine-tune gate is confirmed harder to reach than
  ADR-0027 could tell at the time, with a specific number explaining why.
- Every precision number above came from a fixed-seed sample, manually judged, recorded in
  `reports/phaseC_precision_review.json` — no channel's yield is reported without its measured
  precision alongside it, per the spec's hard rule.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Recommend a full-scale mining build across all 5 channels | Channels C and E are shown not viable (0% and 3% precision); bulk-mining them would inflate gold with noise, repeating the exact mistake the spec warned against. |
| Treat k8s as "unmeasurable, revisit later" (ADR-0028's framing) | The live-index/split-leakage analysis shows this was an artifact of applying a leakage-prevention rule where it doesn't apply — re-checking the assumption found a near-free unblock. |
| Scope a k8s fine-tune-gating mining build anyway | The ~6,075-total-pair ask is disproportionate (>1.5x the entire current k8s gold set) with no channel found that gets meaningfully close; recommending it would set up another underpowered retry. |
| Treat vscode's gap as closed by channel B alone | Channel B yields ~75 pairs against a ~428-pair (test) gap — real progress, not sufficient; reporting it as "solved" would be the kind of foregone-conclusion the spec explicitly rules out. |
