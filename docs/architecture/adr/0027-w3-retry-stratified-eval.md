# ADR-0027 — W3 Retry on the Grown Corpus: Stratified Eval Design + Product-Task Disclosure

**Status:** Accepted (design pre-registered 2026-07-12, committed BEFORE training ran; results pending)
**Date:** 2026-07-12
**Decider:** Gaurav Gandhi (design approved explicitly; execution by CC)

---

## The headline finding first: what the k8s retrieval metric has actually measured

GitHub issue numbers are shared between issues and PRs, and the gold-pair miner was never
PR-aware. Ground-truth classification of every pair via the raw API records
(`reports/phase2b_pr_pair_breakdown.json`, reproduce with
`scripts/phase2b_pr_pair_breakdown.py`):

| k8s existing gold (n=1,024) | →issue target | →PR target |
|---|---|---|
| issue query | **78 (7.6%)** ← the product task | 23 |
| PR query | 823 (80.4%) | 100 |

**The W3 test split behind ADR-0016's k8s +11.84pp was 2.0% product-task (3 of 152 pairs).**
The k8s retrieval metric — in ADR-0008, ADR-0016, and the README — has, in effect, always
measured *"given a PR's text, find the issue it fixes"*, not the product task *"given a new
ISSUE, find related ISSUES"*. vscode is less affected (74% product-task overall) but its W3
test split was 35% product-task. This joins the calibrator, parquet-drift, and gold-set
contamination corrections as a permanent disclosure: the prior "k8s retrieval" numbers are
valid measurements of a *proxy* task and are relabeled, not retracted.

## Context

ADR-0026 (GO) + the era probe + the Phase 2b collection grew the corpus and gold:

- Corpus: k8s #1–30,000 (was #1–15,002); vscode +4,700 dup-labeled issues + 1,888 targets.
- Gold: `data/gold_related_v2.parquet`, **6,879 pairs** (was 1,435). 39 legacy pairs dropped
  because they touch judge-eval (`gold_triage_plans`) issues — retraining on them would have
  contaminated the LLM-judge eval (ADR-0018 class).
- Strata (assigned per-pair from raw-API PR flags):

| repo | gate (proxy, CI-gated) | product (directional) | train_only |
|---|---|---|---|
| k8s | PR→issue: **3,132** | issue→issue: **776** | PR-target pairs: 122 |
| vscode | dup_comment: **2,242** | non-dup issue→issue: **505** | PR-query/target: 102 |

## Decision — pre-registered eval design (locked before training)

1. **Train on everything** (all strata, all channels). PR→issue and dup pairs are legitimate
   embedding-training signal; the strata exist for *measurement*, not training.
2. **Gate per repo on the powered proxy stratum**: PASS iff ΔR@5 ≥ 3pp AND paired-bootstrap
   95% CI lower bound > 0 on the gate stratum test pairs (same 3pp bar as W1.3/ADR-0016).
   k8s gate = PR→issue; vscode gate = dup_comment.
3. **Product strata are directional secondaries, never gated** (~15% of 776/505 ≈ 115/75
   test pairs — underpowered for W3-size effects by design honesty, not by choice).
4. **Headline rule (locked, cannot be reframed after results):** the headline is the
   product-task (issue→issue) directional deltas on BOTH repos, reported alongside the gated
   proxy CIs. If proxy improves and product doesn't, the honest headline is *"reference/dup
   retrieval improved; product-task retrieval unproven at current data"* — never
   "retrieval improved +X pp".
5. **Pre-registered acceptable outcome:** "fine-tune passes proxy gates; product-task
   directionally positive but underpowered; gating the product task needs more issue→issue
   pairs." That is a valid Phase 2 result, not a failure to be reframed.
6. **`train_only` pairs never appear in any eval stratum** (val included). PR-target pairs
   are thereby purged from measurement without discarding training signal.
7. **Bootstrap correction (methodological, decided before results):** the ADR-0016-era T5
   resampled baseline and fine-tuned hit vectors with *independent* indices — an unpaired
   bootstrap described as paired, overstating CI width. v2 uses a true paired bootstrap
   (same resample indices) as primary and reports the legacy unpaired CI alongside for
   comparability with ADR-0016.
8. **Baselines are recomputed live on the v2 corpus** (both models search the same grown
   index; `dup_index_*_bge_v2`). v2 baselines and deltas are NOT comparable to ADR-0016
   numbers: the corpus doubled and the gold mix changed. No cross-ADR delta comparisons.
9. **Disjointness guards (non-negotiable, build fails on violation):** judge-eval
   ADR-0018 assert at merge; `assert_eval_disjoint_from_train` (pair-level) and an
   issue-level disjointness assert in T5 before any metric is computed.

## Consequences

- README/ADR language for retrieval gets relabeled: the gated k8s metric is
  "issue/PR-reference retrieval (92% PR-query)"; the gated vscode metric is "duplicate
  retrieval". The product-task metric is reported separately with its honest n.
- Ship/cutover decision is GG's, made on the escalated per-stratum results. Nothing ships
  from this ADR.
- Model/index artifacts are v2-suffixed (`bge_finetuned_*_v2`, `dup_index_*_bge_v2`);
  shipped v1 artifacts and `gold_related.parquet` are untouched until a cutover decision.
- The 2026-07-11/12 scrapes are recorded in `reports/phase2b_collection_report.json` and
  `reports/phase2b_merge_report.json`; raw records are body-channel for the k8s forward
  slice (`comments_skipped: true`) — comments can be backfilled if a future channel needs
  them.

## Design-stage split correction (2026-07-12, before any training — part of pre-registration)

The first T3 run on v2 used the ADR-0016 quota (state-machine thresholds on cumulative TOTAL
pairs). Because dup pairs chain to old canonical issues, gate-carrying components get early
dates and sort into train: vscode's gate stratum landed only **76** test pairs vs the ~308
the pre-registered power target requires (product got 260 — inverted). Fixed by thresholding
the same chronological component walk on cumulative **GATE-stratum** pairs per repo; other
strata ride along with their components. No model had been trained and no result existed
when this changed. Resulting test strata: k8s gate 441 / product 57; vscode gate 308 /
product 281. Issue-level leakage check passes unchanged.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Purge k8s gold to product-task-only | Collapses k8s to 776 pairs (~115 test) — CI bar unreachable; discards 3,254 valid training pairs; invalidates all prior comparisons. |
| Keep-as-is with a caveat footnote | Same data as stratify, less visibility — strictly worse. |
| Gate on the product stratum anyway | ~115/75 test pairs can only detect ≥~18pp effects at 80% power — a bar that can't resolve is theater (ADR-0016 lesson). |
| Compare v2 deltas to W3's +10/+11.84pp | Different corpus, different gold mix, different task strata — pre-registered as non-comparable. |
