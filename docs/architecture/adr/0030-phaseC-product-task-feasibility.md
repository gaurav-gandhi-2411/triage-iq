# ADR-0030 — Phase C: Product-Task Gold Feasibility (Data Decision)

**Status:** Accepted. Action 1 (measure k8s live recall) executed 2026-07-12 — result:
product-task retrieval is the weakest model in this pipeline (~23% R@5, both repos, first
honest measurement on the live index). Actions 2–3 (fine-tune gating, either repo) and 4
(channel mining): **NO-GO, decided on value grounds, not data-availability grounds** — proving
a ~3pp fine-tune lift against a ~23% base rate isn't worth mining for, on either repo. Phase 2's
fine-tune (ADR-0027) stays HELD with this same reasoning: not a near-miss awaiting data, but
marginal against a weak baseline. Action 5 (retrieval-quality investment) is a real, separate
decision — not made in this ADR (see Go/no-go).

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

## Decision

### The headline finding: product-task retrieval is the weakest model in this pipeline

Now that Action 1 has run (see Decision record below), the most consequential fact in this ADR
isn't a mining-feasibility number — it's the first honest, apples-to-apples measurement of
product-task Recall@5 against the LIVE, actually-deployed retriever, on both repos:
**k8s 23.5% CI[18.4,28.5]** (n=277), **vscode 22.4% CI[17.7,28.0]** (n=254) — statistically
indistinguishable. The retriever finds the genuinely-related issue in the top-5 roughly a
quarter of the time, on both repos. Set against the rest of this project's per-model audit
(ADR-0028: classifier top-3 82.5%/90.4%, resolution's real k8s bucket gains, synthesis's
floor-fail rates), this is the lowest absolute performance number measured anywhere in the
system on its primary, decision-relevant metric — and it was invisible until this ADR because
neither repo had ever been measured on the actual product task, against the actual live index:
k8s was measured on a PR→issue proxy (ADR-0016/ADR-0027, +11.84pp/+14.29pp that never
represented "given an issue, find related issues"), and vscode on a proxy-inflated
duplicate-comment number (36.7% vs. the honest 22.4%, ADR-0028). This finding, not the mining
feasibility question below, is why this ADR settles on NO-GO for fine-tune gating on both
repos — see Go/no-go.

### How the k8s half of the measurement became possible without new mining

The framing ADR-0028 escalated under was itself wrong, independent of the result above: k8s's
live product-task recall@5 was called "unmeasurable" because zero product-task pairs fell in the
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

## Go/no-go — decided on value, not just feasibility

1. **MEASURE k8s live recall: GO, executed.** Re-eval with existing data (277 in-range pairs,
   zero leakage risk) — done, see Decision record. Result: 23.5% CI[18.4,28.5], statistically
   indistinguishable from vscode's 22.4% CI[17.7,28.0]. This was the single highest-leverage,
   lowest-cost action in this analysis, and its result is what drives the two calls below.
2. **SHIP the k8s fine-tune (gate the product task): NO-GO.** The mining ask (~6,075 total
   product pairs) is disproportionate to the corpus — that was already true — but that is not
   the operative reason anymore. The operative reason: a successful gate would prove a ~3.5pp
   lift (ADR-0027) on top of a 23.5% base rate, shipping a retriever that would still miss the
   related issue roughly 3 times out of 4. That is not worth mining ~8x the current product
   stratum to prove. Stays HELD per ADR-0027, reasoning corrected: not a near-miss awaiting
   data, but marginal against a weak baseline.
3. **SHIP the vscode fine-tune (gate the product task): NO-GO, same reasoning — supersedes this
   ADR's earlier MIXED call.** The data-feasibility part of the original analysis is unchanged
   and still true: vscode's gap (+428 test pairs / ~664 total beyond channel B's ~75) is a
   *bounded, proportionate* ask, unlike k8s's — channel B (comments) is immediately usable, and
   a channel-D hub-filter is a promising unvalidated next step. But closing that gap would only
   prove a +3.2pp lift (ADR-0027, CI already crosses zero) against a 22.4% base rate — the same
   weak-baseline problem as k8s, just with an easier data path to prove it on. **Not
   recommending the scoped Phase C-build spec for vscode mining now.** Spending the vscode
   comment-channel effort to gate a fine-tune against a retriever this weak is premature ahead
   of a deliberate decision on whether retrieval quality itself is the actual problem.
4. **Channels C and E: no further investment.** Dead / too noisy respectively.
5. **Retrieval-quality improvement (hybrid BM25+dense, reranking, a stronger base embedder,
   or something else): NOT DECIDED HERE.** The ~23% base rate on both repos is this ADR's
   headline finding, and it reframes the fine-tune-vs-mining question this ADR was scoped to
   answer — but choosing *how* to close that gap is a real, separate decision with its own
   trade-offs (cost, latency, engineering effort, which lever actually moves the number). Out
   of scope for this ADR; explicitly not started here.

## Consequences

- **What changes:** the k8s "unmeasurable" framing is retracted everywhere it appeared
  (`docs/architecture/adr/0028-per-model-eval-audit.md` audit table + escalations,
  `README.md` evaluation table + footnote) and replaced with the measured number. The Phase 2
  fine-tune's HELD status (ADR-0027) is unchanged in outcome but its reasoning is corrected: it
  was framed as underpowered-pending-data; it is now understood as marginal-against-a-weak-base
  regardless of data. No scoped Phase C-build spec is opened for either repo.
- **What becomes easier:** the k8s live-measurability question is resolved — no future
  escalation needed. Both repos' fine-tune-gating question is closed the same way (NO-GO on
  value grounds), removing the asymmetric "k8s blocked / vscode maybe" framing this ADR opened
  with. The actual next decision (retrieval-quality investment) is now clearly separated from
  the fine-tune-data question and not conflated with it.
- **What becomes harder:** nothing new. The retrieval-quality question this finding surfaces is
  explicitly *not* answered here — a future phase has to scope that work (which lever: hybrid
  BM25+dense, reranking, a stronger base embedder, or something else) from a cold start.
- Every precision number above came from a fixed-seed sample, manually judged, recorded in
  `reports/phaseC_precision_review.json` — no channel's yield is reported without its measured
  precision alongside it, per the spec's hard rule.

## Decision record (2026-07-12, GG) — Action 1 executed; reasoning for Actions 2–4 above

**Action 1 (MEASURE k8s live recall) executed.** Re-ran the live-serving retriever
(`scripts/08_build_similar_issue_index.py`'s evaluation method) on the 277 in-range
product-stratum pairs identified above, selected by live-index membership rather than
w3-retry split label per this ADR's zero-leakage reasoning
(`scripts/phaseC_k8s_live_product_eval.py`, `reports/phaseC_k8s_live_product_eval.json`).
Same bootstrap method as the vscode number in ADR-0028 (percentile, 2000 resamples, seed
42) so the two are directly comparable:

| Repo | Index | n | Recall@1 | Recall@5 | Recall@10 | Recall@20 | 95% CI (R@5) |
|---|---|---|---|---|---|---|---|
| kubernetes | live v1 (15,000 records) | 277 | 10.5% | **23.5%** | 30.3% | 36.5% | [18.4, 28.5] |
| vscode (ADR-0028, for comparison) | live v1 (7,028 records) | 254 | 7.5% | 22.4% | 43.7% | 71.3% | [17.7, 28.0] |

Ported into `docs/architecture/adr/0028-per-model-eval-audit.md` (audit table + escalations
correction) and `README.md` (evaluation table + footnote), replacing the "unmeasurable" framing
everywhere it appeared. The reasoning for Actions 2–5 (NO-GO on both fine-tunes, retrieval
quality deliberately not started) is stated above under Go/no-go — not repeated here.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Recommend a full-scale mining build across all 5 channels | Channels C and E are shown not viable (0% and 3% precision); bulk-mining them would inflate gold with noise, repeating the exact mistake the spec warned against. |
| Treat k8s as "unmeasurable, revisit later" (ADR-0028's framing) | The live-index/split-leakage analysis shows this was an artifact of applying a leakage-prevention rule where it doesn't apply — re-checking the assumption found a near-free unblock. |
| Scope a k8s fine-tune-gating mining build anyway | The ~6,075-total-pair ask is disproportionate (>1.5x the entire current k8s gold set) with no channel found that gets meaningfully close; recommending it would set up another underpowered retry. |
| Treat vscode's gap as closed by channel B alone | Channel B yields ~75 pairs against a ~428-pair (test) gap — real progress, not sufficient; reporting it as "solved" would be the kind of foregone-conclusion the spec explicitly rules out. |
| Open the scoped Phase C-build spec for vscode mining (channel B + hub-filtered channel D) since the data gap is bounded and closeable | The gap being closeable was never the blocker — the closed result would still gate a +3.2pp lift against a 22.4% base rate. Closing a data gap to prove a marginal number isn't the lever; would repeat the mistake this ADR exists to catch, one level removed. |
| Start scoping retrieval-quality work (hybrid BM25+dense, reranking, stronger embedder) in this ADR | The ~23% finding just surfaced; which lever actually moves it is unknown and deserves its own deliberate analysis, not a same-session extension of a data-feasibility ADR. Explicitly deferred per GG (2026-07-12). |
