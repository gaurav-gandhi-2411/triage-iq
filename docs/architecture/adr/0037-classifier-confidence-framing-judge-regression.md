# ADR-0037 — Judge-Score Regression After ADR-0036: Confidence Framing, Not Labels

**Status:** Accepted — v3 prompt fix implemented (v1 and v2 tried and superseded, see below),
pending full validation recording
**Date:** 2026-07-30
**Decider:** Gaurav Gandhi (root-cause investigation and fix executed by CC on
`fix/record-cassette-ollama-ci`; GG directed the diagnostic procedure and the fix approach at every
iteration)

---

## Context

Re-recording `eval/cassettes/eval_cassette.json` against the ADR-0036 multi-label classifier (the
actual task this branch exists for) surfaced an unexpected result: the LLM-judge synthesis-quality
baseline **regressed** for k8s despite the classifier itself being verifiably more accurate.

| repo | n | old mean | new mean | delta |
|---|---|---|---|---|
| kubernetes/kubernetes | 53 | 10.5094/15 | 10.1887/15 | **-0.3207** |
| microsoft/vscode | 11 | 8.3636/15 | 8.4545/15 | +0.0909 |
| overall | 64 | 10.1406/15 | 9.8906/15 | -0.2500 |

k8s's -0.32 exceeds the previously-measured judge-noise band for that repo (±0.22, from the
two-recording jitter study already baked into `reports/eval_baseline.json`) — not explainable as
noise. This is surprising given ADR-0036 measured k8s top-1 +9.09pp / top-3 +4.55pp via paired
bootstrap (CI excluding zero) on the classifier's own accuracy. A more accurate classifier producing
a *worse* end-to-end synthesis-quality score needed a mechanism, not a shrug.

## Investigation

**Per-issue score deltas, split by label-correctness transition** (old vs. new prediction vs. gold,
k8s n=53):

| group | n | mean score delta |
|---|---|---|
| both correct | 24 | **-0.58** |
| both wrong | 26 | -0.23 |
| old correct → new wrong | 2 | -2.50 |
| old wrong → new correct | 1 | +1.00 |

Labels barely moved net (2 lost, 1 gained) — nowhere near enough to explain the aggregate -0.45
average delta across all 53 issues (raw, pre-repo-mean). Critically, issues where the **label was
correct both times still lost 0.58 points on average**. The regression is not explained by label
accuracy.

**Confidence variance collapsed 5x.** Old `component_confidence` (softmax, single-label): mean=0.502,
std=**0.290**. New (independent OvR sigmoids): mean=0.545, std=**0.058**. `correlation(old_confidence,
score_delta) = -0.16` — issues that used to be reported at very high confidence (0.93-0.99) lost the
most, because those are exactly the ones the new model compresses hardest toward the ~0.45-0.65 band.

**Reconstructed the actual prompt text** (loaded both classifier `.pkl`s directly — old
`TFIDFComponentClassifier`, new `MultiLabelTFIDFComponentClassifier` — and ran
`predict_proba_calibrated` on the same issue text; zero Groq calls needed):

```
k8s-13270 (gold=kube-proxy, correct both times)
  OLD: 1. kube-proxy (0.927)   2. provider/gcp (0.021)   3. ui (0.012)
  NEW: 1. kube-proxy (0.546)   2. usability   (0.459)   3. apiserver (0.448)

k8s-14756 (gold=kube-proxy, correct both times)
  OLD: 1. kube-proxy (0.987)   2. test-infra (0.002)   3. test (0.001)
  NEW: 1. kube-proxy (0.575)   2. test        (0.468)   3. test-infra (0.468)

k8s-14281 (gold=app-lifecycle — the single worst score drop, -5, and one of only two label flips)
  OLD: 1. app-lifecycle (0.498)  2. kubectl (0.075)     3. usability (0.046)
  NEW: 1. kubectl (0.520)        2. app-lifecycle (0.511) 3. usability (0.502)
```

Old top-3 reads as "one clear answer, two irrelevant long-shots." New top-3 reads as "three
components the model is nearly equally unsure about," even on issues it still gets right. On 14281
the correct answer is a near-tied coin-flip against a wrong one — exactly the failure mode OvR's
tight clustering will occasionally produce.

**The hedge propagates beyond the component field.** k8s-13270/14756/13096/14835 all kept
`component_match=2/2` unchanged but dropped on `resolution_estimate_reasonableness` and
`next_steps_actionability` — dimensions fed by an unrelated, **unchanged** LightGBM predictor. Judge
rationale, same issue, same underlying numbers, before/after:

- OLD: *"excellent... reasonable resolution estimate... well-structured summary"*
- NEW: *"solid... the resolution estimate is imprecise and overly optimistic, which lowers the
  overall quality score"*

The LLM writes a less assertive plan across dimensions it wasn't given new information about,
because its own stated component confidence dropped from "93-99% sure" to "55-58% sure, with two
other things almost as likely."

## Decision (v1): fix the prompt's interpretation of confidence, not the numbers

Two candidate fixes were considered and one rejected — see Alternatives. The chosen fix:

1. **`build_triage_prompt()`**: the SYSTEM 1 section now explains that these are independent
   per-component probabilities (not a softmax distribution), that a close spread means "several
   components plausibly apply" rather than "the classifier is unsure," and that the model should
   read rank order, not gap size. Top-3 lines are labeled `[primary pick]` / `[also plausible]`.
   Numbers are otherwise rendered identically (`.3f`, verbatim calibrated values).
2. **`SYSTEM_PROMPT_LEGACY`** (confirmed the actual live prompt — `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION`
   is unset in both the Cloud Run service env and `record-cassette.yml`, so production and this eval
   both run the legacy/pre-attribution prompt, not the newer `SYSTEM_PROMPT`) gets a new
   "CLASSIFIER CONFIDENCE GUIDANCE" paragraph: don't judge the classifier by score spread, and don't
   let the resolution estimate or next steps hedge because of it — those come from separate models.
   `SYSTEM_PROMPT` (the not-yet-enabled attribution variant) got the identical paragraph for
   consistency, so it doesn't inherit the same bug whenever attribution mode ships.
3. **A fourth few-shot example was appended** (not editing the three frozen ones) to both
   `build_few_shot_examples_legacy()` and `build_few_shot_examples()`, demonstrating a case with
   three similarly-scored components (0.579/0.531/0.492) where the plan is still written decisively
   — confident resolution estimate, concrete next steps, no hedging language. In-context
   demonstrations were judged the highest-risk gap in the instruction-only version of this fix: the
   three original examples were written against the single-label softmax classifier's confidence
   shape (one dominant score, two near-zero) and no longer demonstrate what the current multi-label
   classifier's normal output looks like. Editing their content would violate the ADR-0020 freeze
   (kept for reproducibility of what those specific examples demonstrate); appending doesn't — their
   bytes are untouched, and the freeze's purpose (stable, byte-exact demonstrations) is preserved for
   the original three.

**v1 result (full validation recording, `dry_run:true resume:false`, 2026-07-30 11:40-13:34 UTC):**
k8s mean landed at **10.1887 — identical to 4 decimal places to the pre-fix regression**, i.e. zero
net movement (still -0.3207 vs. OLD, still outside the ±0.22 band). vscode recovered strongly
(8.3636 → 9.2727). Per-dimension, the targeted mechanism partially worked:
`resolution_estimate_reasonableness` recovered *above* OLD (1.302 → 1.321) and
`next_steps_actionability` recovered close to OLD (2.547 → 2.491) — but `component_match` got worse
(1.415 → 1.245) and `floor_fail_rate` more than doubled (0.094 → 0.226), exactly offsetting the gain.
Root cause: `label == gold_component` accuracy dropped 49.1% → 41.5%, concentrated in the LLM
switching to "kubectl" (a broad, generic-sounding k8s term) on issues where the classifier's own
top-1 was correct in the prior (no-fix) recording. The v1 wording asked the LLM to treat "several
components can apply" without also anchoring it to *which one is still the prediction* — it started
treating the ranked top-3 as a menu to choose from by its own textual judgment rather than a ranked
recommendation to defer to by default.

## Decision (v2, rejected as a no-op): sharpen the wording

Tried adding explicit anchoring language: "the primary pick... should be chosen by default; deviate
ONLY if the issue text contains a specific concrete reason it doesn't fit." Validated *before* any
quota spend via a synthesis-only probe (`scripts/probe_label_anchoring_fix.py`) on the 9 k8s issues
that regressed under v1 — no judge, ~9 Groq calls instead of 128, since the label-drift question is
answerable from `predicted_component` alone.

**Result: 9/9 outputs byte-identical to v1.** The sharpened paragraph changed nothing measurable.
**Non-obvious, reusable lesson: the `[primary pick]` / `[also plausible]` structural labels on the
top-3 lines were doing 100% of the anchoring work — the surrounding instruction prose was
decorative.** At `temperature=0.0`, the model anchors hard to an explicit inline marker on the data
itself; rewording the paragraph around that marker doesn't change the anchoring strength one bit.
This generalizes beyond this fix: when a prompt marks structure directly on the data block (a label,
a bracket, a bold tag), that marker will dominate free-text instructions elsewhere in the same
prompt about how to weigh that data — don't expect prose elsewhere to override it, and don't expect
prose elsewhere to be *necessary* once the marker is there either.

## Decision (v3, chosen): remove the anchoring labels, keep de-hedging unchanged

Removed the `[primary pick]` / `[also plausible]` labels entirely and the "choose it by default,
deviate only with concrete reason" directive. Replaced with neutral language: explain the confidence
semantics without directing which entry to choose ("weigh these scores together with the issue text
and the similar issues below, the same way you always would"). The resolution/next-steps de-hedging
sentence — demonstrated to work in v1 — is untouched. The appended fourth few-shot example's user
turn was updated to match (no more inline labels, so the example stays a faithful demonstration of
what the model actually sees); its assistant turn (picking the classifier's own top-1 for that
example) was left as-is since v3 doesn't forbid agreeing with the classifier, it just stops
instructing it.

**Cheap-probe result (same 9 issues, live, no judge):** `label == gold` 3/9, up from v1's 1/9 (3x),
against a no-fix baseline of 4/9. 2 of the 3 correct→wrong flips recovered (k8s-12703, k8s-14363).
The third (k8s-14895) stayed on "kubectl" — but the classifier's own top-1 for that issue *is*
"kubectl" (gold is `client-libraries`), so this is a **classifier error, not a prompt problem**;
GG's explicit call was to not iterate the prompt to force an override of one specific classifier
mistake, since that fits one gold label without generalizing. On the non-flip cases, v3 repeatedly
reproduced the *no-fix* recording's specific answer rather than v1's — including on issues where that
answer isn't gold-correct — which is independent evidence the fix restored the LLM's original
decision process rather than coincidentally matching more gold labels.

## Consequences

- All 211 repo tests pass after each iteration. One test asserted the old few-shot count (3 assistant
  turns) and was updated to 4 — a legitimate update to a count that changed intentionally, not a
  weakened assertion.
- The classifier and its calibration are untouched throughout. ADR-0036's accuracy win (k8s +9.09pp
  top-1, vscode +7.49pp top-1, both CIs excluding zero) stands regardless of this ADR's outcome.
- This is a live-prompt change (affects production synthesis output, not just the eval harness) —
  the validation recording (`dry_run:true`) measures its effect before any baseline write, per the
  standing eval-baseline-write escalation gate.
- Approval criteria for the v3 full recording: k8s within ±0.22 of OLD (10.5094) with de-hedging
  retained (resolution/next-steps at or above OLD) → baseline write approved. If `component_match`
  recovers but the mean still sits low, that gets reported and investigated on its own terms rather
  than triggering another prompt-wording iteration.

## Separately logged, not fixed here

**Resolution-estimate text/number mismatch.** k8s-14756 and k8s-13096 kept getting dinged on
`resolution_estimate_reasonableness` across both v1 and the no-fix recording even where the
underlying LightGBM numbers were unchanged and the language wasn't obviously hedged. Reading the
actual text: k8s-14756's `expected_resolution_summary` said *"typically 1 day or less for a
straightforward configuration tweak"* against a numeric interval of `[2.8d, 21.6d]` — the prose
summary doesn't match the number it's supposed to summarize. This is a distinct, pre-existing quality
bug (the LLM's free-text resolution summary can contradict the resolution predictor's own interval)
independent of the confidence-framing regression this ADR addresses. Not in scope here per GG's
explicit instruction — logged for a future pass.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Softmax-renormalize the top-3 confidences for display only (classifier's stored/calibrated probabilities untouched) | Rejected on direct evidence from k8s-14281: the correct label (app-lifecycle, 0.511) currently loses a genuine near-tie to a wrong one (kubectl, 0.520). A display rescale can't distinguish a right-label near-tie from a wrong-label near-tie — it would render *both* as confidently decisive, manufacturing false certainty on exactly the issues where the model is honestly, correctly uncertain. Also: the clustering is real information (several components genuinely plausible — the entire point of the ADR-0036 multi-label supervision fix), so faking a spread to recover a judge score would be optimizing the metric, not the product. |
| Do nothing / accept the regression as an inherent cost of the classifier upgrade | Rejected — the evidence (both-correct group still losing 0.58 points; dimensions fed by an unchanged resolution predictor showing the same hedge) shows this is a framing artifact, not a real quality cost, so accepting it would be leaving recoverable synthesis quality on the table for no reason. |
| Edit the three frozen few-shot examples' confidence values to reflect the new model's typical spread | Rejected — violates the ADR-0020 freeze (`do not edit`, kept for byte-exact reproducibility of what those examples demonstrate). Appending a fourth achieves the same in-context-learning goal without that cost. |
