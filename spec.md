# Project Spec: TriageIQ — Phase 2a: Corpus-Growth Feasibility (investigate before scraping)

## Goal

The W3 retrieval fine-tune was rejected (ADR-0016) because recall@k improvement had a bootstrap CI
that crossed zero at available n — a DATA problem (too few gold_related pairs), not a method problem.
Phase 2 would retry the fine-tune on a grown corpus. But before committing to scraping (real effort),
this investigation answers: **CAN the corpus realistically grow enough to clear the ADR-0006 CI bar
(CI excludes zero on BOTH repos), and is that growth achievable — especially for vscode, the binding
data constraint at every prior turn?**

This is analysis-only, no scraping, no fine-tuning. The deliverable is a go/no-go with NUMBERS:
either "yes, growing to N pairs plausibly clears the bar, here's the scraping scope" (→ Phase 2b
scraping build) or "no, the data ceiling is structural, fine-tune retry isn't viable" (→ honest
finding, skip to Phase 3). Do NOT foregone-conclude either direction.

## Current state

- gold_related.parquet: ~1,435 pairs (1,024 k8s / 411 vscode). Sources: body_related=1010,
  title_sim=421, body_ref=4. (body_ref is the strict "duplicate of #N" pattern — only 4 in the
  entire corpus, ADR-0007.)
- W3 rejected: re-established fine-tune gave +11.84pp k8s CI[0.00,+22.38], +10pp vscode
  CI[-6.67,+25.00] (ADR-0016) — neither excludes zero, both need more n.
- vscode is the binding constraint everywhere: 411 pairs, couldn't reach n=300 test for W3, 17
  clean issues for W5, 11 clean for the eval baseline. 92% split-overlap by base rate.
- Corpus: k8s ~15,000 issues, vscode ~7,028 issues (the scraped issue bodies, not just pairs).
- Retrieval fine-tune trains on gold_related PAIRS; the eval needs enough disjoint TEST pairs for a
  bootstrap CI that can exclude zero.

## Scope

### In scope (analysis only)

**1. Quantify how many MORE pairs would clear the bar (power analysis):**
- From the W3 result (effect sizes + CIs at current n), estimate: at what test-set n would the k8s
  CI plausibly exclude zero? At what n for vscode? This is a power calculation from the observed
  effect size and variance — how many disjoint test pairs does each repo need for the CI to exclude
  zero if the true effect is ~the observed point estimate?
- Be honest about the assumption: this assumes the effect is real at the point estimate. If the
  point estimate itself is near-zero (vscode's +10pp with a wide CI), no n saves it — flag that.

**2. Quantify how many pairs the corpus COULD yield (ceiling analysis):**
- The current pairs come from body_related (1010), title_sim (421), body_ref (4). How many MORE
  genuine issue-to-issue related pairs could be extracted from the EXISTING scraped corpus (15K k8s
  / 7K vscode issues) that aren't already in gold_related? I.e. is the corpus under-mined, or is
  1,435 close to what's extractable?
- And: how many more ISSUES could realistically be scraped (are there unscraped issues in these
  repos — closed issues, older issues, that weren't pulled)? Estimate the addressable pool.
- The KEY question for vscode: is vscode's 411 a mining ceiling (all extractable pairs already
  found) or a scraping ceiling (more issues exist, unscraped)? These have very different scraping
  implications.

**3. The vscode-specific verdict (this is the crux):**
- vscode has been the binding constraint at every turn. Determine honestly: is there a realistic
  path to enough vscode disjoint test pairs to clear the CI bar, or is vscode structurally
  data-limited such that the fine-tune can only ever ship k8s-only?
- If vscode can't reach the bar even with maximal realistic scraping → the honest finding is "the
  fine-tune is a k8s-only prospect; vscode retrieval stays on the baseline BGE" — which is a valid
  outcome (k8s-only improvement) but must be stated, not hidden.

**4. Go/no-go recommendation with the scraping scope:**
- If GO: how many issues to scrape per repo, expected pair yield, whether it clears the bar on
  k8s / vscode / both, and the rough effort. This becomes the Phase 2b scraping spec.
- If NO-GO: which repo(s) are structurally ceilinged, why, and the recommendation to skip to Phase 3.
- MIXED (likely): "k8s is growable and worth it, vscode is ceilinged" — the honest split verdict.

### Out of scope

- No scraping (this decides WHETHER to scrape).
- No fine-tuning (W3 is rejected until the corpus grows; this is the feasibility gate).
- No new labeling (that's Phase 2b if GO).
- No model/pipeline/eval changes.

## Tech stack

- Existing Python + the scraped corpus + gold_related.parquet. scipy for the power calc. No LLM,
  no new deps.

## Autonomy & escalation

CC runs the full analysis autonomously. Escalate ONLY:
1. The go/no-go recommendation + the numbers behind it (the power calc + ceiling estimate + vscode
   verdict) — this is a strategic decision, report it for the human to make the scrape/skip call.

## Hard rules

- Honest numbers, no foregone conclusion. If the power calc says vscode's point estimate is too
  near-zero for any n to help, SAY SO — don't recommend scraping to chase a null effect.
- vscode indicative caveats apply; be explicit about what's structural vs addressable.
- Analysis only — no scraping, no fine-tuning, no data changes.
- Branch only (`analysis/corpus-feasibility`); I merge. Zero-cost, no LLM. Claude Max — never
  ANTHROPIC_API_KEY. Don't touch aetherart-497918.

## Success criteria

- Power analysis: test-set n needed per repo for the CI to exclude zero (given observed effect).
- Ceiling analysis: extractable additional pairs from existing corpus + addressable unscraped issues,
  per repo — mining-ceiling vs scraping-ceiling distinguished.
- vscode verdict: realistic path to the bar, or structurally ceilinged (k8s-only prospect).
- Go/no-go recommendation with scraping scope (if go) or skip rationale (if no-go), escalated.
- reports/corpus_feasibility.json + a short ADR-0026 documenting the analysis + recommendation.

## Build order (CC autonomous)

1. Power calc: from W3's effect sizes + CIs, the test-n needed per repo to exclude zero. Flag if
   any repo's point estimate is too near-zero for n to help.
2. Ceiling: mine the existing corpus for additional extractable pairs (not already in gold_related);
   estimate unscraped-issue pool per repo. Distinguish mining-ceiling from scraping-ceiling.
3. vscode verdict: realistic-path vs structurally-ceilinged.
4. ESCALATE the go/no-go + numbers + scraping scope.
5. ADR-0026 + corpus_feasibility.json.
```

