# Project Spec: TriageIQ — Phase D2: GPU Fine-Tune Retrieval on Clean Data

## Goal

D1 produced clean, hand-verified, disjoint-asserted training pools + held-out eval sets, and the
honest baselines: retrieval is genuinely weak on the tasks the data supports. D2 fine-tunes the
embedder LOCALLY on GG's RTX 3070 (8GB VRAM) on the CLEAN data and finds the best model — measured
on D1's trustworthy held-out eval, with leakage impossible by construction. $0 cost (GCP was
evaluated and ruled out — the billing account's free tier structurally bans GPU instances; local
compute is both cheaper and simpler for a job this size).

Two trainable tasks (asymmetric — different tasks, different data volumes):
- **vscode-DUPLICATE (PRIMARY)**: 2,242 clean pairs @ 85%, held-out eval n=200 (gateable),
  baseline R@5 = 43.5% [37.0, 50.5]. Most data, cleanest signal, real headroom — the likeliest real
  win and the likeliest HF-publishable model.
- **k8s-RELATED (SECONDARY, thin-data experiment)**: 264 training pairs @ 84.6%, held-out eval
  n=150 (gateable), baseline R@5 = 9.3% [5.3, 14.0]. 264 pairs is THIN for embedder fine-tuning —
  real overfit risk. Pre-register the honest expectation it may not generalize; a negative is a
  valid finding.

**Measure-first**: before any hyperparameter sweep, ONE default fine-tune run on vscode-duplicate.
If it doesn't move 43.5% meaningfully, no sweep will — stop and report. Only sweep if the single run
shows real signal.

## Local GPU environment (RTX 3070 laptop GPU, 8GB VRAM, Windows/CUDA)

- No cloud, no VM, no staging. Training data (corpus parquet, D1 full-corpus indices, mined hard
  negatives) is already local — GG runs the scripts directly, no provisioning step.
- The 3070 is shared with GG's AetherArt work. **Don't touch the AetherArt process** (don't kill it,
  don't assume its VRAM is free) — GG runs D2 once AetherArt's job releases the GPU, not before.
- BGE-base (109M params) at the scripts' default batch size (16) fits comfortably in 8GB; note in
  the escalation report if a run needs a smaller batch size to fit alongside anything else resident.
- CC gives the exact local run sequence in chat (per GG's standing rule — commands pasted in chat,
  not just files). Never ANTHROPIC_API_KEY.

## The leakage guard (NON-NEGOTIABLE — the whole reason D1 existed)

- Train ONLY on the training pool. Evaluate ONLY on the D1 held-out eval set. They are disjoint at
  the issue level (D1 asserted this). D2 RE-ASSERTS it programmatically before training and before
  eval — a trained model evaluated on any training issue is a contaminated fake number (the bug
  class this project has caught 5×).
- No eval issue in any training pair, no training issue in any eval pair. Assert, fail hard on violation.
- Hard-negative mining (if used) draws negatives ONLY from training-pool issues, never eval issues.

## The ship / publish bar

- Metric: R@5 on the D1 held-out eval, per task, bootstrap CI vs the D1 baseline (vscode-dup 43.5%,
  k8s-related 9.3%). Report R@1/R@10 for shape.
- **SHIP bar: a MEANINGFUL lift, not marginal.** A fine-tune that squeaks past significance
  (e.g. 43.5% → 46%, CI barely excluding zero) is the marginal-polish trap that got W3 held — NOT a
  ship. State the effect size; the lift must be substantial and the CI must clearly exclude zero.
- **PUBLISH bar (HF): gated on a REAL USP, assessed AFTER training** (see D3 below). A marginal
  model is NOT published — a mediocre model on HF under GG's name is worse than none.
- A fine-tune that doesn't clear the bar is a documented negative (W3 / reranker / lever pattern),
  not a failure. Do not p-hack, do not lower the bar.

## Scope

### In scope
1. **vscode-duplicate (primary)**: measure-first single fine-tune (sensible defaults: base BGE,
   contrastive/triplet loss on the clean pairs, in-batch + mined hard negatives from TRAINING pool
   only). Eval on held-out n=200. ESCALATE the single-run result.
2. If the single run shows real signal → a bounded hyperparameter sweep (LR, epochs, batch, negative
   strategy), each eval'd on the held-out set. Pick the best by held-out R@5. Report the sweep.
3. **k8s-related (secondary)**: ONE thin-data fine-tune attempt (264 pairs), strong regularization /
   early stopping to fight overfit, eval on held-out n=150. Pre-registered honest expectation: may
   not generalize. Report whatever it does — negative is valid.
4. **Best eval params (from D1's rec)**: apply the D1-recommended k / metrics / CI consistently.
5. If a model clears the ship bar: the fine-tuned embedder artifact + a reproducible training script
   + the held-out eval result — staged for a deliberate cutover decision (a retrieval change =
   re-index + re-record + re-baseline; escalate the cutover separately, don't auto-ship).

### Out of scope
- No cutover/deploy in D2 (train + measure only; cutover is a separate escalated decision).
- No training on eval issues (leakage). No lowering the ship bar. No paid APIs.
- No HF push in D2 (that's D3, gated on the USP assessment).
- vscode-related (n=22) and any task without a clean trainable pool — not trained (directional-only).

## Autonomy & escalation
CC prepares the training code + the local run sequence autonomously. Escalate:
1. **The measure-first single-run result** (vscode-dup) — before any sweep.
2. **The final per-task results** (fine-tune vs baseline, CI, effect size) — before any cutover or
   HF decision.
3. **Any leakage-guard question** — if the disjointness assertion is ambiguous, STOP.
4. The cutover (if a model ships to prod) and the HF push (D3) — separate decisions.

## Hard rules
- Leakage guard asserted before train AND before eval — fail hard on violation. Non-negotiable.
- Ship bar = meaningful lift, CI clearly excludes zero, effect size stated. Marginal = documented
  negative, not shipped.
- Measure-first: single run before sweep.
- Local run only (RTX 3070); don't touch the AetherArt process sharing the GPU.
- Branch `feat/retrieval-gpu-finetune`; human merges. Never ANTHROPIC_API_KEY.
  Don't touch aetherart-497918.

## Success criteria
- Leakage disjointness re-asserted programmatically (train vs held-out) before train + eval.
- vscode-duplicate: measure-first single run reported; sweep only if signal; best model by held-out R@5.
- k8s-related: thin-data attempt reported honestly (incl. a valid negative).
- Per-task fine-tune vs baseline: R@5 + CI + effect size, on D1's held-out eval.
- ADR-0034: the training, per-task results, ship/reject per task, and (if shipping) the USP
  assessment feeding the D3 HF decision.

## Build order
1. Prepare training code + re-assert leakage disjointness.
2. GG runs the vscode-duplicate MEASURE-FIRST single run locally. ESCALATE the result.
3. If signal → bounded sweep, pick best by held-out R@5. Then the k8s-related thin-data attempt.
4. ESCALATE final per-task results (vs baseline, CI, effect size).
5. ADR-0034 + USP assessment for the D3 HF decision.

---

## D3 (outline only — separate spec after D2, gated on the ship bar)

HuggingFace release, same as AetherArt / Mindmeld — ONLY IF D2 produces a genuinely strong model
with a REAL, honest USP:
- USP candidates: "open model for GitHub issue DUPLICATE detection, honestly per-task benchmarked"
  (narrow but real if vscode-dup trains well). NOT "related retrieval" (data can't support it).
- Model card with the HONEST benchmarks (the held-out R@5, the baseline it beats, the task scope,
  the limitations — vscode-related unmeasured, k8s-related thin). The honesty IS the differentiator.
- Gated: if D2 is only marginal, DO NOT publish. Assess the USP on D2's real numbers first.
```

