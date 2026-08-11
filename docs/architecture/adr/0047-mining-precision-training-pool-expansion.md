# ADR-0047 — Mining precision: training pool expansion, correcting two D1 channel-drop decisions

**Status:** Accepted
**Date:** 2026-08-11
**Decider:** Gaurav Gandhi

## Correction to the escalation this ADR follows up on

The escalation that preceded this decision framed the change as "route `dup_comment` from the
gate stratum into product-task training" — implying `dup_comment` had been walled off from
training entirely. **On code inspection (`scripts/d1_build_eval_set.py`), that framing was
wrong**: `dup_comment` was already the entire training pool for D1/D2's `vscode_duplicate` task
(`("vscode_dup_scrape", "dup_comment", "gate")` is literally in `build_training_pool`'s channel
list) — it's just that D1's own task-naming keeps `vscode_duplicate` (dup-sourced) and
`vscode_related` (title_sim/body_refs-sourced) as two separate, never-blended tasks, per
ADR-0033's design. "Gate" in `gold_related_v2.parquet`'s stratum column means "the powered,
CI-gated headline metric," not "excluded from training" — a naming collision with k8s's PR-based
`gate` stratum (a genuinely different, training-excluded concept for k8s) that produced the
mis-framing. Recorded here rather than silently corrected, per this project's own disclosure
norm (ADR-0046, rule 101c) — the escalation's *instinct* (there's untapped training value being
left out) was right, the stated *mechanism* was not.

## What's actually being corrected

D1 (`scripts/d1_assemble_clean_pool.py`, 2026-07-24-ish) explicitly **dropped** two channels from
the training pool, each on a precision estimate that this session's fresh, larger, strict-rubric
audit (`docs/investigations/2026-08-11-mining-precision-channel-characterization.md`) now
contradicts:

| Channel | D1's estimate (basis) | This session's estimate (basis) |
|---|---|---|
| `k8s_extended_mine`/`body_related_ext` | 65% (ADR-0032 partial read, n=20, D1's own looser genuine/incidental rubric) | 54.0% [40.4, 67.0] (n=50, the stricter VALID/EXCLUDE rubric used to build the 2026-08-11 k8s clean eval set) |
| `vscode_body_refs`/`body_related_ext` | 30-43% (ADR-0030, n=30, cited not re-derived) | 74.0% [60.5, 84.1] (n=50, same strict rubric) |

Both of D1's original estimates predate the strict retrieval-validity rubric this project now
uses for eval-set construction — the rubric that specifically screens for umbrella/tracking
issues and causal-only citations, the exact failure modes that make a pair a bad *test* of
content-similarity retrieval. Applying that same, more appropriate rubric fresh:

- `k8s_extended_mine` (54.0%) turns out to be in the **same band as every k8s channel D1 kept**
  (42.1-44.8%, measured the same way via the 2026-08-11 k8s clean-eval build) — arguably the
  *best* of k8s's channels, not the worst. D1's relative "meaningfully below sibling channels"
  reasoning doesn't survive being re-measured on a consistent rubric.
- `vscode_body_refs`/`body_related_ext` (74.0%) is dramatically higher than the stale ADR-0030
  estimate D1 relied on — the old estimate's own upper bound (43%) sits below this session's CI
  lower bound (60.5%), a real disagreement, not noise.

## Decision

**Include both channels in the training pool going forward.** Neither drop decision was wrong
given the evidence available at the time (ADR-0032/ADR-0030's numbers were real measurements);
both are superseded by more rigorous, more-recently-collected evidence, the same "measure the
metric, don't just carry the old number forward" discipline ADR-0046 is the standing prior for.

**Held-out eval sets are unchanged, full stop.** `reports/d1_eval_set_k8s_related.json` (150
pairs, == the 2026-08-11 clean-eval set) and `reports/d1_eval_set_vscode_duplicate.json` (200
pairs) are reused exactly as D1 froze them — no eval pair, from any channel, is added, removed,
or re-labeled by this change. GG's underlying reasoning for the *original* escalation (a metric
that inflates its own denominator by training-then-testing on the same easy channel is a
contaminated measurement; training-then-testing on disjoint issue sets is legitimate transfer)
is the correct principle even though the specific "dup pairs were eval-excluded but not
train-excluded" premise didn't hold — it already applied, and continues to apply here for the two
newly-included channels: they go into training, never into eval, and the eval sets they might
otherwise have inflated stay exactly as hand-verified.

**Disjointness enforced hard-fail, issue-level, both directions** (`scripts/
mining_precision_build_pool.py`, `scripts/d3_assert_leakage_guard.py`): a training pair touching
any eval-set issue number, in any channel, is dropped before it ever reaches training. Verified
by assertion, not by construction alone — the script raises `AssertionError` on any violation, and
both D3 pipeline scripts (`d3_mine_train_negatives.py`, `d3_train.py`, `d3_eval_finetuned.py`)
re-run the same check as a pre-flight gate before doing any work, matching D2's own established
discipline.

## Resulting pools

| Task | D1/D2 original | This session |
|---|---|---|
| `k8s_related` | 264 pairs (NO-GO'd as thin, ADR-0034) | **448 pairs** (+70%) — channels: `k8s_forward_scrape`/`body_related` (45), `k8s_forward_scrape`/`body_related_ext` (454 raw), `legacy_gold_v1`/`body_related` (37), `k8s_extended_mine`/`body_related_ext` (200 raw, newly included) |
| `vscode_duplicate` | 1,734 pairs | **1,958 pairs** (+13%) — channels: `vscode_dup_scrape`/`dup_comment` (2,242 raw), `vscode_body_refs`/`body_related` (11), `vscode_body_refs`/`body_related_ext` (206 raw, newly included), `legacy_gold_v1`/`body_related` (10) |

Pools are the full raw channel volume minus (a) any pair touching an eval-set issue number and
(b) any pair explicitly hand-reviewed and judged bad across all review passes to date (D1's
genuine/incidental review, ADR-0032's review, and this session's strict VALID/EXCLUDE review) —
same convention D1 itself used: training pools are not 100%-hand-verified, they're the channel's
measured/estimated precision as a background noise level, consistent with how D1's own
`vscode_duplicate` pool (85% D1-measured precision) was already built.

`title_sim` remains dropped everywhere, unconditionally — unaffected by this ADR.

**k8s's ceiling is real and asymmetric, correctly diagnosed, not a gap to keep chasing**: k8s has
zero `dup_comment`-equivalent channel (confirmed by a full census of all 29,994 locally-cached
issues — zero "duplicate"-labeled issues exist; see the investigation doc). k8s's 448-pair pool is
the ceiling of its existing regex-mined channels, not an artifact of an unexplored channel.

## Consequences

- **What changes:** `scripts/d3_*.py` (leakage guard, negative mining, training, eval) train on
  the expanded pools above; `scripts/d1_train_pool_*.json` and D1/D2's original artifacts are
  untouched (historical record, not superseded in place).
- **What becomes easier:** k8s_related is powered for the first time (448 vs. the 264 ADR-0034
  explicitly NO-GO'd as too thin to spend GPU time re-confirming).
- **What becomes harder:** nothing structurally — pool assembly is CPU/analysis work, same
  sequencing ADR-0046 already establishes for measurement-before-modeling.
- **Open/not addressed:** `vscode_related` (the non-dup title_sim/body_refs-only task) remains
  eval-only, unmeasured as a standalone metric (ADR-0032) — folding `vscode_body_refs`/
  `body_related_ext` into the unified `vscode_duplicate` training pool tests whether broader
  training data helps the one task with a trustworthy eval (`vscode_duplicate`), it does not
  establish a separate, powered `vscode_related` baseline. That remains a distinct, not-yet-done
  piece of work if ever prioritized.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Hand-verify all ~2,650 candidate pairs across every newly-included channel before training | Disproportionate (would need ~90 parallel labeling batches); D1's own established convention (full channel volume minus reviewed-bad pairs, not 100% hand-verification) is the precedent this project already trusts for `vscode_duplicate`'s existing 85%-precision pool. |
| Keep D1's original drop decisions, treat them as settled | The precision estimates justifying both drops are stale or under-powered relative to evidence collected this session on a more appropriate rubric — carrying forward a superseded number violates ADR-0046's own standing prior. |
| Build a fresh, separately-powered `vscode_related` eval set now | Out of scope for this pass — the immediate ask is training-pool expansion for the D2-retry fine-tune, not a new eval-construction project; flagged as future work above, not silently dropped. |
