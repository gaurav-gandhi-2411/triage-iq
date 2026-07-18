# Project Spec: TriageIQ — Phase D1: Clean Retrieval Data + Trustworthy Eval Foundation

## Goal

Before ANY GPU training of a better retrieval model, the data and eval must be clean — otherwise
GPU training produces a model trained on noise, evaluated on noise, giving a fake improvement (the
worst possible outcome for a project whose value is honest measurement). This phase builds that
foundation:

1. **Clean retrieval TRAINING pairs** per repo — genuinely-related issue→issue pairs, replacing the
   ~80%-noise title_sim channel (ADR-0032 found title_sim is ~20% precision; it currently feeds
   hard-negative mining + the train split).
2. **A hand-verified, leakage-safe, held-out EVAL set** per repo that stays valid AFTER training
   (disjoint from all training pairs) — the trustworthy scoreboard D2's GPU training will be judged
   on. Without this, GPU training is unmeasurable.

D1 is mostly CPU/analysis + a bounded hand-verification effort. It gates D2 (GPU training). It is
also independently valuable: it fixes the contaminated retrieval eval that ADR-0032 exposed.

## Current state (the mess D1 cleans)

- `gold_related_v2.parquet`: pairs from multiple channels — reference-mined ("See #N", 78-89%
  precision, GOOD), dup_comment (/duplicate, high precision, but DUPLICATES not general-related),
  PR-query (proxy, wrong task), and **title_sim (~20% precision, NOISE)**.
- title_sim feeds `w3_t2_mine_negatives.py` (hard negatives), `w3_t3_split.py` (train split),
  `w3_t4_train.py` (triplet training) — so the held W3 fine-tune trained partly on noise.
- Honest eval today: k8s ~23.5% [18.4,28.5] (n=277, ~72% precision — usable but not clean),
  vscode UNMEASURED (~20% precision gold — unusable).
- Zero-leakage reasoning (ADR-0030) holds ONLY for the untrained index. **Once we train (D2),
  eval MUST be disjoint from training data.** D1 must produce that disjoint split.

## Scope (D1 — data + eval only, NO training)

### 1. Characterize + clean the pair channels
- Per channel per repo: precision (hand-sample where not already known — reference-mined and
  title_sim are known; audit dup_comment and any others), and product-task relevance (is it
  issue→related-issue, the product task, or a proxy?).
- Build a CLEAN pair pool: keep genuinely-related, product-task, high-precision pairs. Drop title_sim
  noise. For each kept channel, state the precision and why it's included.
- **k8s**: reference-mined is the clean backbone (~78-89%). Quantify the clean pool size.
- **vscode**: the hard case — its clean channels are thin (ADR-0032). Determine honestly: is there a
  clean vscode pair pool large enough to train + eval on, or does vscode need the comment-mining
  channel (Phase 2b found /duplicate comments at 62% modern recovery — but those are DUPLICATES,
  a related-but-distinct stratum)? **Report vscode's realistic clean pool per stratum.** If vscode
  can't reach a trainable+evaluable clean set, say so — vscode may be train-on-k8s / eval-only, or
  deferred.

### 2. Hand-verify the held-out EVAL set (the scoreboard)
- Carve a HELD-OUT eval set per repo from the clean pool — hand-verified genuine (like the ADR-0032
  audit, but larger: target enough for a powered recall@k CI, ~150-300 pairs per repo if the clean
  pool supports it; report what's achievable).
- **This eval set is DISJOINT from the training pool by construction** — no eval issue appears in any
  training pair. Assert it (the ADR-0018 / disjointness discipline — this is the leakage guard that
  makes D2's trained-model numbers valid).
- Hand-verify a sample to confirm precision is high (target ≥90% genuine — this is the scoreboard, it
  must be clean). Report the verified precision.
- Freeze it with provenance (which channel, which issues, verification date) so D2 evaluates on a
  fixed, trustworthy target.

### 3. Re-establish the honest CURRENT baseline on the clean eval
- Re-run the current live (untrained) retriever's recall@5 (+R@1/R@10) on the new CLEAN held-out
  eval set, per repo, with bootstrap CI. This is the honest "before" number D2 must beat.
- Expect it to differ from the ~23.5% (that was measured on the 72%-precision set) — report the
  clean-eval baseline as the real current performance.

### 4. Best eval PARAMS (the "best eval params" ask)
- Determine the right eval configuration: which k for recall@k (what does the product surface —
  top-5? top-10?), whether MRR/nDCG add signal over recall@k, the right CI method. Recommend the
  eval-param set D2 will use, justified by the product use case.

### Out of scope (D1)
- NO training (that's D2). NO GPU (D1 is CPU/analysis + hand-verification).
- No shipping/cutover (D1 produces data + eval, not a model).
- No new large-scale mining unless a channel is needed for a minimum viable clean pool (escalate if
  vscode requires it).

## Autonomy & escalation
CC runs the analysis + clean-pool construction autonomously. Hand-verification is bounded labeling.
Escalate ONLY:
1. **The clean-pool sizes + vscode's realistic verdict** (can vscode be trained+evaluated cleanly, or
   is it deferred/k8s-only?) — before finalizing, since it shapes D2.
2. **The held-out eval set + its verified precision + the clean-eval baseline numbers** — before D2
   uses them as the scoreboard.
3. Any need for new large-scale mining to reach a minimum clean pool.

## Hard rules
- The held-out eval set is DISJOINT from all training pairs by construction — asserted. This is the
  leakage guard for D2; without it, D2's trained numbers are contaminated (the exact bug class this
  project has caught 4×).
- Clean pool = genuinely-related product-task pairs only. title_sim NOISE is dropped, not included.
- Eval set precision hand-verified ≥90% (it's the scoreboard).
- Honest vscode verdict — if it can't be cleanly trained+evaluated, SAY SO; don't force it.
- No training, no GPU, no cutover in D1. Branch `feat/retrieval-clean-data`; human merges.
  Zero-cost (D1 is CPU). Claude Max — never ANTHROPIC_API_KEY. Don't touch aetherart-497918.

## Success criteria
- Clean pair pool per repo, per channel, with precision stated; title_sim dropped.
- Held-out eval set per repo: hand-verified ≥90% precision, disjoint-from-training asserted, frozen
  with provenance.
- Clean-eval baseline (current untrained retriever) per repo with CI — the honest "before".
- vscode verdict: trainable+evaluable clean, or deferred/k8s-only — stated.
- Recommended eval params (k, metrics, CI) for D2.
- ADR-0033 + `reports/retrieval_clean_data.json`.

## Build order (CC autonomous, escalate at the gates)
1. Channel precision + product-task audit → clean pool per repo. ESCALATE the pool sizes + vscode verdict.
2. Carve + hand-verify the held-out eval set (≥90%, disjoint-asserted, frozen). ESCALATE it + precision.
3. Re-baseline the current untrained retriever on the clean eval (CI). Report.
4. Recommend eval params. ADR-0033.
```

