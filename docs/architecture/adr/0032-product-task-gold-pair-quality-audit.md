# ADR-0032 — Product-Task Gold Pair Quality Audit (before accepting a ~23-27% ceiling)

**Status:** Accepted (finding, not a mining/fine-tune decision)
**Date:** 2026-07-16
**Decider:** Gaurav Gandhi (analysis by CC, autonomous)

---

## Context

ADR-0031 tried three untried, zero-training retrieval-quality levers (hybrid BM25+dense,
a pretrained cross-encoder reranker, a stronger pretrained embedder) against ADR-0030's
corrected product-task Recall@5 baseline (**k8s 23.5% [18.4, 28.5]**, **vscode 26.7%
[21.6, 31.9]**). All three rejected. Three independent levers all failing to move the
number is itself evidence the ~23-27% base rate isn't an easy algorithmic fix — but before
treating it as a durable ceiling, there's a cheaper, unexamined alternative: **the gold
pairs themselves might be noisy.** The product-task pairs are mined from "related to
#N"-style signals across several channels (ADR-0030); if a meaningful fraction of the
pairs used to *measure* R@5 aren't genuinely related, a low score reflects noisy ground
truth, not retriever failure.

This ADR does the audit: hand-sample the pairs, judge genuine vs. incidental, recompute
R@5 on the genuine-only subset, and check whether the retriever's misses on genuinely-related
pairs are explainable by zero lexical overlap (an honest ceiling) or not (a real retriever
gap). Reproducible via `scripts/phaseC_pair_quality_audit.py`. Manual judgments recorded
in `reports/phaseC_pair_quality_review.json`; full results in
`reports/phaseC_pair_quality_audit.json`.

## Method

`scripts/phaseC_pair_quality_audit.py --sample` reproduces the *exact* pair sets ADR-0030/
ADR-0031 measure R@5 against (`select_live_product_pairs`, same function used by
`phaseC_{k8s,vscode}_live_product_eval.py`: product-stratum pairs from
`gold_related_v2.parquet` filtered to live-index membership — k8s n=277, vscode n=292) and
draws a fixed-seed (`seed=42`) sample of 25 pairs per repo (50 total) for hand judging. Each
pair's full query/target title+body was read and classified **genuine** (same root cause,
same subsystem, or a triager would genuinely want to see the other) or **incidental**
(loose/coincidental link, or no real shared content) — see `reports/phaseC_pair_quality_review.json`
for every verdict and its one-line reason. `--analyze` then computes precision, reruns the
live retriever restricted to the genuine subset, and measures token overlap
(lowercased, stopword-filtered, ≥3 chars) between query and target for every genuinely-related
pair the retriever misses at R@5.

## Finding 1 — pair-set precision is 46% overall, but the two repos are not the same problem

| Repo | n sampled | Genuine | Precision | Wilson 95% CI |
|---|---|---|---|---|
| kubernetes | 25 | 18 | **72.0%** | [52.4, 85.7] |
| vscode | 25 | 5 | **20.0%** | [8.9, 39.1] |
| Overall | 50 | 23 | 46.0% | [33.0, 59.6] |

The CIs don't overlap — this asymmetry is real at n=25/repo, not sampling noise between
the repos. It traces directly to **channel/source composition** of the live-evaluated
pair sets (`reports/phaseC_pair_quality_audit.json` → `channel_composition_of_live_product_sets`):

| Repo | Reference-mined (body-ref channels) | Title-similarity-mined (`title_sim`) |
|---|---|---|
| k8s (n=277) | 239 (86.3%) — `k8s_extended_mine` (ADR-0030 measured 78-89% precision) + `legacy_gold_v1`'s `body_related`/`body_ref` | 38 (13.7%) |
| vscode (n=292) | 11 (3.8%) | **277 (94.9%)** |

vscode's live-evaluated product set is almost entirely (94.9%) sourced from `title_sim` —
pairs mined by title-text similarity, a channel **never precision-audited before this ADR**
(it isn't in ADR-0030's channel A-E table at all). k8s's set is 86% reference-mined
(explicit "See #N" / "Related to #N" / "Forked from #N" links), a channel class ADR-0030
already measured at 78-89% precision — consistent with this ADR's 72% hand-count.

**Concrete false-positive classes found in the sample** (all in `phaseC_pair_quality_review.json`):
- **Boilerplate-template collisions (vscode, 20/25 incidental cases):** both issues are
  unfilled GitHub bug-report templates ("Does this issue occur when all extensions are
  disabled?... Steps to Reproduce: 1. 2.") with unrelated one-word titles ("Bayou"↔"11",
  "Chat"↔"ROBLOX", "issue"↔"Ehh") — paired only because `title_sim` matched near-identical
  boilerplate. 130/288 (45.1%) of vscode's `legacy_gold_v1` pairs have this exact shape on
  *both* sides.
- **Cross-repo issue-number collisions (k8s, 2 cases):** the query explicitly references a
  *different* repo's issue (`google/cadvisor#770`, `coreos/fleet#1280`) but got linked here
  because `kubernetes/kubernetes` happens to have an issue with the same number — a mining
  bug, not noise in the underlying signal.
- **Reporter-disclaimed relation (k8s, 1 case):** query #14023 explicitly states it checked
  #4891 and "they don't appear to be the cause" — the pair is in gold despite the source
  text saying the relation doesn't hold.
- **Generic auto-bucket title collision (vscode, 1 case):** two different VS Code crash
  reports share the same auto-generated bucket title ("[Error] unhandlederror-potential
  listener LEAK detected, popular") but different specific leak sources
  (`chatEditingServiceImpl` vs `executionStatusBarItemController`) — same label, different bug.

Sample-size caveat: n=25/repo carries real width (Wilson CIs above) — this is directional
evidence of a large, real asymmetry, not a certified final precision figure. A full-scale
audit of vscode's `title_sim`/`legacy_gold_v1` channel (505 pairs, same rigor as ADR-0030's
channel table) would be needed to certify a corrected number.

## Finding 2 — clean-subset R@5 doesn't move the interpretation for k8s; vscode's clean sample is too small to read

| Repo | Full-set R@5 (ADR-0030/31) | Genuine-only R@5 (this sample) | n genuine | CI (illustrative, small-n) |
|---|---|---|---|---|
| k8s | 23.47% (n=277) | **27.78%** | 18 | [11.1, 50.0] |
| vscode | 26.71% (n=292) | **20.0%** | 5 | [0.0, 60.0] |

k8s's genuine-only point estimate (27.78%) sits inside the full-set's own CI
[18.4, 28.5] — pulling out the 28% of the k8s sample judged incidental doesn't move the
number outside noise. Combined with the 72% precision and the reference-mined channel
composition, **k8s's ~23% figure holds up as a real retrieval-quality measurement**, not
an eval-noise artifact — ADR-0030's finding and ADR-0031's lever rejections are reinforced,
not undermined.

vscode's genuine-only subsample is n=5 — a single flip changes the point estimate by 20pp;
this number cannot support any claim about vscode's "true" retrieval quality. What *can* be
said regardless of what that number would turn out to be at scale: **the pair set used to
produce vscode's 26.7% headline number is, by this hand sample, only ~20% precise.** The
ADR-0030/0031 framing that k8s and vscode are "statistically indistinguishable, both ~23-27%"
compares a mostly-genuine measurement (k8s) to a mostly-incidental one (vscode) — the two
numbers are not measuring the same thing, even though they happen to land in the same range.

## Finding 3 — genuine misses are not a vocabulary-orphaned ceiling

Of the 23 genuine pairs, 6 hit (5 k8s + 1 vscode, matching the R@5s above) and 17 missed.
**0/17 genuine misses have zero shared vocabulary** between query and target
(`genuine_misses_zero_overlap_count: 0`, `reports/phaseC_pair_quality_audit.json`). Overlap
ranges from 2 shared tokens (e.g. k8s #11089→#8448: `exec`, `kubectl`) up to 90 shared
tokens for vscode's near-verbatim duplicate crash-stack-trace pairs (#311555/#311545/#311546
→ #311022 — the same "listener LEAK" bucket, sharing dozens of identical stack-frame tokens
like `_createinstance`, `_deliver`, `_event`) that the dense retriever *still* fails to
surface in the top 5.

This refutes, for this sample, the "genuinely hard, no shared surface text, honest ceiling"
hypothesis — every miss was lexically findable in principle. It's consistent with, and adds
hand-verified genuine-relation ground truth to, ADR-0031 Lever 1's own diagnostic that BM25
recovers a subset of dense's misses via exact-term matches (dense embeddings blur distinctive
vocabulary — API names, error strings, stack-frame identifiers — that lexical methods catch
directly).

## Interpretation — which of the three pre-registered hypotheses is true

1. **"Our retriever is weak"** — **true for k8s.** Pairs are mostly genuine (72%), the
   genuine-only R@5 doesn't move outside the full-set's own CI, and genuine misses have
   real, exploitable shared vocabulary the dense-only retriever ignores. k8s's ~23% ceiling
   and ADR-0031's lever rejections are a real retrieval-quality finding.
2. **"The eval pairs are noisy, and the real number is better"** — **true of vscode's pair
   set quality** (20% precision, dominated by an unaudited channel and boilerplate
   collisions), but **not confirmed as a corrected number** — n=5 genuine pairs can't
   support one. The structural problem (untrustworthy gold set) stands regardless of what a
   properly-powered clean re-measurement would find.
3. **"The task is genuinely hard, zero shared vocabulary"** — **refuted** in this sample:
   0/17 genuine misses have no shared tokens.

## Decision

**No mining or fine-tune action reopened here** — that question was already closed on
value grounds (ADR-0030: even a much better retrieval number wouldn't justify the mining
cost given the current corpus scale) and this ADR doesn't revisit it. What this ADR does
change: **vscode's product-task R@5 should not be cited as corroborating k8s's finding, or
compared to it as "statistically indistinguishable," without the caveat that vscode's
measurement is against a pair set this audit finds only ~20% precise.** k8s's finding
stands unqualified.

**Recommended next step (not executed here, scoped like ADR-0030's channel actions):** a
full-scale precision audit of vscode's `title_sim`/`legacy_gold_v1` product-stratum pairs
(505 rows, or at minimum the 292 in the live index), with the same per-channel rigor
ADR-0030 applied to channels A-E — this is a **measurement-integrity fix on an existing
gold set**, categorically cheaper than the mining-at-scale asks ADR-0030 already NO-GO'd
(no new scraping, no new API calls — just judging pairs that already exist). Pruning or
re-weighting the confirmed-incidental pairs would let a future re-measurement report a
trustworthy vscode product-task R@5 for the first time.

## Consequences

- **What changes:** README.md's headline finding gets a caveat on the vscode number's
  pair-set precision (see diff). ADR-0030/0031 are not edited — their k8s-side conclusions
  are unaffected and this ADR's finding is additive context, not a correction to either.
- **What becomes easier:** any future retrieval-quality work on vscode now has a concrete,
  hand-verified list of the specific noise classes to filter (boilerplate-template pairs,
  generic-title-bucket collisions) before re-measuring.
- **What becomes harder:** nothing new — the recommended full-scale audit is optional future
  work, not a blocker on anything currently shipped.
- Every precision and R@5 number above has its provenance in
  `reports/phaseC_pair_quality_audit.json` and `reports/phaseC_pair_quality_review.json`
  (per-pair verdicts + reasons), reproducible via `scripts/phaseC_pair_quality_audit.py`.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Treat the ~23-27% base rate as settled after ADR-0031 without auditing pair quality | Three lever rejections plus an unaudited gold set together would leave "is the eval even measuring the right thing" as an open question before spending any more retrieval-quality effort — cheaper to check now than after a fourth lever. |
| Full-scale (505-pair) precision audit instead of a 50-pair hand sample | Scoped to match ADR-0030's own sampling discipline (n=9-30 per channel there); a 50-pair hand-judged sample is enough to detect a large, real asymmetry (non-overlapping Wilson CIs) cheaply; a full-scale audit is the explicit recommended follow-up, not skipped, just not done in this pass. |
| Re-open the vscode mining/fine-tune question given the pair-quality finding | Out of scope — ADR-0030's NO-GO was decided on value (a few pp lift against a weak baseline isn't worth the mining ask), which this finding doesn't change even if the "true" vscode number turns out higher; that's a separate decision to make later, deliberately, not a side effect of this audit. |

## Reproduce

```
python scripts/phaseC_pair_quality_audit.py --sample     # writes reports/phaseC_pair_sample_for_review.json
# hand-judge -> reports/phaseC_pair_quality_review.json (already committed with this ADR)
python scripts/phaseC_pair_quality_audit.py --analyze     # writes reports/phaseC_pair_quality_audit.json
```
