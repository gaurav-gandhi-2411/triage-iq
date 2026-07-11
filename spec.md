# Project Spec: TriageIQ — Phase C: Product-Task Gold Feasibility (data decision)

## Goal

Two things are currently BLOCKED on the same missing data — genuine product-task (issue→related-
issue) gold pairs, at powered scale, per repo:
1. The k8s retriever's LIVE product-task recall@5 is UNMEASURABLE (ADR-0028): zero product-task test
   pairs exist against the live k8s index. We literally cannot say how well k8s retrieval serves the
   product task.
2. The Phase 2 retrieval fine-tune is HELD (ADR-0027): its product-task improvement is directionally
   positive but crosses zero on both repos — it needs more product-task test pairs to gate on the
   task that matters.

Both unblock with the same asset: enough genuine issue→related-issue pairs (NOT PR→issue proxy, NOT
near-duplicate) to power a product-task recall@5 CI per repo. This iteration DECIDES whether mining
that asset at scale is worth it — with numbers, before committing to the collection effort. It is
analysis-only: no mining at scale, no fine-tune cutover. The deliverable is a go/no-go.

## Background (why the data is scarce)

- Product-task = issue→related-issue (a user triaging an ISSUE wants related ISSUES). This is the
  use case; PR→issue (proxy) and duplicate→canonical (Phase 2b dup channel) are NOT it.
- Current product-task gold: vscode ~254-304 (grew via Phase 2), k8s ~72-78 (mostly stranded in
  train, ~0 evaluable against the live index).
- Phase 2b's dup-mining channel (the /duplicate comment parse) yields DUPLICATES, not general related
  issues — it can't provide the related-pair asset (dups are a different stratum).
- ADR-0027's stated unblock: vscode needs ~700 product test pairs (~4,700 issue→issue total) for 80%
  power; k8s needs its live-index product pairs to exist at all.

## Scope (analysis only — no scale mining, no fine-tune)

**1. Sources of genuine issue→related-issue pairs (what channels exist?):**
- Beyond dup-links and PR-links, where do GitHub issues express "related to #N" NON-duplicate,
  NON-PR relationships? Candidates: "related to #N" / "see also #N" / "similar to #N" in bodies AND
  comments; GitHub's native "linked issues" (not PRs); cross-references in timeline; label-based
  clustering (issues sharing a fine-grained component + temporal proximity as weak-related).
- For each channel: estimate the YIELD per repo (how many genuine related-pairs mineable) and the
  PRECISION (how many are truly related vs incidental mentions). Sample-and-measure like Phase 2b's
  probe, don't assume.

**2. The power target (how many pairs are actually needed?):**
- From the held Phase 2 fine-tune result (product-task deltas + CIs), how many product-task TEST
  pairs per repo would tighten the CI to exclude zero IF the effect is ~the point estimate? (Re-use
  the Phase 2a power-calc method.) And how many for the k8s LIVE index to be measurable at all
  (a workable ±5pp CI)?
- Honest flag: if the product-task point estimate is too near-zero (like vscode's borderline case),
  no n saves the fine-tune — say so, and then the ONLY value of the data is MEASURING k8s (still
  worth something) not SHIPPING the fine-tune.

**3. The live-index problem (k8s-specific):**
- k8s product pairs must be evaluable against the LIVE index (15K records). Phase 2b's forward-scrape
  pairs (#15,003-30,000) are NOT in the live index. So either: mine product-pairs from the
  IN-LIVE-INDEX issue range (#1-15,000), or accept that measuring k8s requires re-indexing to the
  grown corpus (a bigger change). Quantify: how many mineable product-pairs fall in the live-index
  range vs require re-indexing?

**4. Go/no-go with the collection scope:**
- GO: which channel(s), expected yield + precision per repo, whether it hits the power target for
  (a) measuring k8s live retriever, (b) gating the Phase 2 fine-tune product task, and the rough
  effort. → becomes the Phase C-build (mining) spec.
- NO-GO: the data isn't mineable at sufficient yield/precision → honest finding: product-task
  retrieval stays unmeasurable / the fine-tune stays held indefinitely, and WHY.
- MIXED (likely): "enough to MEASURE k8s (worth doing), not enough to SHIP the fine-tune" OR
  "vscode gateable, k8s needs re-indexing" — the honest split.

### Out of scope

- No scale mining (this decides whether to). No fine-tune cutover. No re-indexing (this quantifies
  whether it'd be needed). No model/pipeline changes. Analysis only.

## Autonomy & escalation

CC runs the full analysis autonomously. Escalate ONLY:
1. The go/no-go + numbers (channel yields + precision, power target, live-index quantification,
   collection scope if go) — the strategic decision, human-made.

## Hard rules

- Honest numbers, no foregone conclusion. If a channel's precision is low (incidental "#N" mentions,
  not real relationships), SAY SO — don't inflate yield with noise. If the fine-tune's point estimate
  can't be saved by any n, SAY SO.
- Distinguish "worth it to MEASURE k8s" from "worth it to SHIP the fine-tune" — they have different
  data bars and one may be reachable without the other.
- Analysis only — no scale mining, no cutover, no re-indexing. Branch `analysis/phaseC-feasibility`;
  I merge. Zero-cost, no LLM at scale (small precision-sampling only). Claude Max — never
  ANTHROPIC_API_KEY. Don't touch aetherart-497918.

## Success criteria

- Related-pair channels enumerated; yield + precision estimated per channel per repo (sampled, not
  assumed).
- Power target: product-task test pairs needed per repo to (a) measure k8s live, (b) gate the fine-tune.
- Live-index quantification: mineable k8s product-pairs in-live-range vs requiring re-indexing.
- Go/no-go (likely mixed) with collection scope if go — escalated.
- reports/phaseC_feasibility.json + ADR-0030.

## Build order (CC autonomous)

1. Enumerate + sample related-pair channels (bodies/comments "related to #N", native linked-issues,
   timeline cross-refs, label-cluster weak-related). Yield + precision per channel per repo.
2. Power calc: pairs needed to measure k8s live + to gate the fine-tune, per repo. Flag if the
   fine-tune's effect is too near-zero for any n.
3. Live-index quantification (k8s in-range vs re-index-needed).
4. ESCALATE the go/no-go + collection scope.
5. ADR-0030.
```

