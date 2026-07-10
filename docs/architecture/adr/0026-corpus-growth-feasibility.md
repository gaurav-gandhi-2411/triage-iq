# ADR-0026 — Phase 2a: Corpus-Growth Feasibility for the W3 Fine-Tune Retry

**Status:** Proposed — GO recommendation escalated to GG (scrape/skip is a human call)
**Date:** 2026-07-10
**Decider:** Gaurav Gandhi (analysis by CC, autonomous)

---

## Context

ADR-0016 rejected the W3 BGE fine-tune because neither repo's bootstrap 95% CI on ΔR@5 excluded
zero at the available test n (the ADR-0006 bar, applied consistently): k8s +11.84pp CI[0.00,
+22.38] at n=152; vscode +10.00pp CI[−6.67, +25.00] at n=60. The diagnosis was data volume
(1,435 gold pairs: 1,024 k8s / 411 vscode), not the method. Before committing to a Phase 2b
scraping build, this analysis answers whether the corpus CAN grow enough to clear the bar —
especially for vscode, the binding constraint at every prior turn.

Analysis only: no scraping, no fine-tuning, no data changes. All numbers reproducible via
`python scripts/phase2a_corpus_feasibility.py` → `reports/corpus_feasibility.json`.
Live GitHub totals queried 2026-07-10 (metadata search counts only, no content fetched).

## Decision

**GO on both repos** — recommend proceeding to a Phase 2b scraping build. Escalated to GG for
the final scrape/skip call; nothing is executed until then.

### 1. Power analysis (test-set n needed for the CI to exclude zero)

SE per observation is backed out of the observed bootstrap CI (empirical paired variance), then
scaled 1/√n. Assumes the true effect equals the observed point estimate — see honesty caveats.

| Repo | observed Δ (n test) | n test for 80% power | n test for 90% power | test frac | total pairs needed (80/90%) | current pairs |
|---|---|---|---|---|---|---|
| k8s | +11.84pp (152) | 278 | 372 | 0.151 | **1,840 / 2,463** | 1,024 |
| vscode | +10.00pp (60) | 308 | 412 | 0.152 | **2,023 / 2,706** | 411 |

- Neither point estimate is near-zero (+11.84pp / +10.00pp) — this is not chasing a null. But
  vscode's CI includes zero: the +10pp is *unproven*, and the power calc conditions on it being
  real. If the true effect is half the point estimate, needed test n quadruples (k8s 1,110,
  vscode 1,230 → ~7,400 / ~8,100 total pairs). The vscode dup-channel ceiling (below) covers
  even that pessimistic branch; for k8s it would require most of the repo plus heavy reliance
  on the weak title-sim channel.
- k8s's implied per-query discordance rate is 51% (fine-tuned and baseline disagree on half the
  queries) — the wide CI is real re-ranking variance, not an artifact.

### 2. Ceiling analysis (mining vs scraping, measured on the actual corpus)

| Channel | k8s | vscode |
|---|---|---|
| In-corpus, current patterns (miner's first-match-only loss) | **+22** | **0** |
| In-corpus, extended patterns ("related to", "ref", issue URLs — review-grade) | **+857** | +344 |
| Out-of-corpus refs (scrape-recoverable from existing bodies) | 1 | 83 |
| Title-sim ≥0.45 uncapped (weak channel; miner caps at 300) | 8,839 | 11,838 |
| GitHub repo-wide issues ever / duplicate-labeled (2026-07-10) | 49,266 / 52 | 247,856 / **29,111** |

- **The existing corpus is mined out** on the genuine-reference channel at current patterns
  (k8s +22, vscode +0). The constraint was never under-mining of what we have.
- **k8s is scraping-ceilinged**: its corpus is a strict number prefix (#1–15,002, 2014–15), so
  backward references land in-corpus by construction (1 out-of-corpus ref total) — growth
  requires scraping forward. Historical yield: 901 ref pairs / 15,002 numbers ≈ 6 per 100.
  The extended-pattern candidates (+857, same body-reference class, needs a review pass) alone
  would take k8s to ~1,881 ≥ the 1,840 80%-power target *without scraping*.
- **vscode is scraping-ceilinged + mining-channel-mismatched**, not structurally ceilinged. Its
  corpus covers 2.8% of the repo (5,000 early issues from 2015–16 + 2,028 from 2026; the
  2016–2026 middle — ~300K numbers — was never scraped). And vscode's duplicate signal lives in
  **comments, not bodies**: across all 238 dup-labeled issues in-corpus, a dup-target reference
  is recoverable from comments in 49% and from bodies in 0% — the body-only miner is
  structurally blind to it. Repo-wide: 29,111 `*duplicate`-labeled issues × 49% comment
  recovery ≈ **14,100 candidate pairs** (floor; the timeline API's `marked_as_duplicate`
  events are structured and should recover more) vs 2,023 needed.

### 3. vscode verdict (the crux)

**vscode is NOT structurally data-limited.** Its 411-pair constraint (72% of which is capped
title-sim, only 113 genuine reference pairs) is an artifact of (a) scraping the wrong slice of
the repo and (b) mining the wrong channel. A label-targeted scrape of `*duplicate` issues plus
comment/timeline target extraction has a candidate ceiling ~7× the 80%-power requirement, with
headroom to spare even if the true effect is half the observed +10pp. The "fine-tune can only
ever ship k8s-only" outcome is not supported by the data.

### Phase 2b scraping scope (if GG says GO)

- **k8s** (~0 API risk): (1) extended-pattern re-mine of the existing corpus, +857 candidates,
  human/heuristic review pass; (2) optionally forward-scrape #15,003–30,000 (~15K issue+PR
  records + comments) → ~+900 ref pairs at historical yield. Either path clears 80% power;
  both together approach 90%.
- **vscode** (the real build): scrape ~5,000–8,000 `*duplicate`-labeled issues (GitHub search
  by label) + comments, extract dup targets from comments/timeline, fetch the ~2,500–4,000
  missing target issues. Expected yield ~2,000–3,900 pairs at the 49% floor → clears 80–90%
  power at the observed effect. Roughly 15–25K API requests ≈ 3–5 hours at authed rate limits.
  Requires extending the miner to a comment/timeline extraction channel and a labeling/review
  protocol for the new pairs (reuse W5's).

## Consequences

- W3 retry becomes viable on a grown corpus on **both** repos — the honest split verdict
  ("k8s growable, vscode ceilinged") anticipated by the spec is *not* what the data shows.
- **Caveats carried into Phase 2b, stated up front:**
  1. The power targets condition on the observed point estimates being real. vscode's CI
     includes zero; a smaller true effect means the retry can still honestly fail the bar —
     that is the bar working, not a wasted scrape (the corpus growth also serves W5/eval).
  2. Dup-channel pairs are near-identical issues — easier retrieval targets than the current
     gold mix. The retry's eval stands on the new corpus's own CI; its delta is not comparable
     to W3's +10pp. Report both the new baseline and the new delta (rule 41).
  3. Out-of-corpus targets share GitHub numbering with PRs; k8s forward-scrape yields are
     issue+PR mixed (consistent with the existing corpus under the ADR-0008 framing).
  4. Comment-recovery (49%) was measured on the 238 in-corpus dup-labeled issues, which skew
     2015–16; modern vscode bot workflows may shift the rate either way. The timeline API is
     the robust channel and should be validated on a ~100-issue probe first (riskiest
     assumption first, rule 77).
- New pairs must flow through the existing disjointness guards (`assert_eval_disjoint_from_train`,
  the three-way gold checks from the ADR-0018 remediation) before any retrain.

## Alternatives

| Alternative | Reason rejected |
|---|---|
| NO-GO, skip to Phase 3 | Ceiling analysis contradicts it: both repos have realistic paths to 80–90% power; vscode's ceiling was an artifact, not structure. |
| k8s-only growth (write vscode off) | vscode's dup-label pool (29,111 × 49%) is the *largest* pair source in the project; writing it off would repeat the original scraping mistake. |
| Raise the title-sim cap (8.8K/11.8K pairs available, zero scraping) | Weakest-confidence channel; floods gold with near-duplicate-text pairs and inflates R@5 with easy positives — a metric win by dataset dilution, not a model win. Usable only as a labeled, quality-reviewed supplement. |
| Escalate k8s to n=300 on existing pairs | Already rejected by ADR-0016: asymmetric standards; vscode can't follow at current corpus. |
