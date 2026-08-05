# 0039 — Keep the multi-label classifier; leave the synthesis-quality baseline unwritten; mark the k8s quality gate known-failing

**Status:** Accepted
**Date:** 2026-08-05
**Decider:** Gaurav Gandhi, closing the ADR-0037 investigation

## Context

ADR-0036 shipped a multi-label classifier cutover with a verified, statistically significant
ground-truth accuracy win (k8s top-1 +9.09pp / top-3 +4.55pp, vscode top-1 +7.49pp / top-3
+4.55pp, paired bootstrap, CIs excluding zero). Re-recording the eval cassette against it
surfaced a judge-scored synthesis-quality regression on kubernetes/kubernetes: -0.6226 against
the OLD baseline (10.5094 → 9.8868), 2.8x the project's own measured ±0.22 noise band. ADR-0037
root-caused the mechanism (the new classifier's independent per-class confidence scores cluster
tightly, which reads to the LLM as "the model is unsure" and induces hedged language across the
whole synthesized plan — including dimensions fed by an unrelated, unchanged resolution
predictor) and tried three prompt-wording fixes (v1: label the top-3 primary/also-plausible; v2:
sharpen the wording, proven a no-op; v3: remove the labels, keep de-hedging). None reached OLD.
v3's full 64-issue live validation recording (run `30998299699`) failed the approval bar GG set:
k8s still -0.62 outside tolerance, and de-hedging did not hold — v3 landed *below* even v1 on the
two dimensions (`resolution_estimate_reasonableness`, `next_steps_actionability`) that de-hedging
was specifically meant to protect. Investigated whether this is classifier-attributable or judge
measurement noise (replayed v1's original cassette, pulled back from GitHub Actions artifact
storage, against v1's code in an isolated worktree, compared directly against v3 on the 53 k8s
issues where the classifier's own predicted_component came out identical in both) — evidence
points predominantly to a real, reproducible synthesis-text effect, not primarily noise. Full
diagnostic trail: ADR-0037.

## Decision

**Keep the multi-label classifier. Do not roll back.**

The classifier's accuracy improvement is measured against ground truth on a clean, disjoint test
set — it is the real, primary thing this component is supposed to do well, and it does, provably.
The judge is a proxy for a downstream, harder-to-verify property (does a synthesized plan *read*
well to another LLM) and ADR-0037's evidence shows precisely what it's reacting to: hedged prose
in a resolution-estimate summary, a dimension fed by a model (the LightGBM resolution predictor)
that is completely unchanged by the classifier cutover. Rolling back a verified, ground-truth
accuracy gain to satisfy a proxy metric reacting to writing style is optimizing the metric over
the product — exactly the failure mode this project's own tie-breaker order (correctness → product
quality → eval rigor → ...) exists to prevent.

**Do not write the new baseline.** `reports/eval_baseline.json`'s `per_repo`/`overall` means stay
at OLD (k8s 10.5094, vscode 8.3636) — untouched, not overwritten with the v3 recording's 9.8868.
Folding the regression in as the new "normal" would destroy the only signal that a real, understood,
open quality gap exists. The frozen OLD baseline remains the honest reference point.

**Mark the k8s quality gate known-failing, loudly and self-explainingly, not silently.** Two
mechanical changes make this a documented, deliberate state rather than a red mystery:

1. `eval/cassettes/eval_cassette.json` **was** updated to the v3 recording (run `30998299699`) —
   this is necessary for the gate to even execute; replaying the OLD cassette against current
   (post-ADR-0036) code would hard-crash with `CassetteMissError` on the first classifier-fed
   call, which is a worse failure mode than an honest, informative regression report. `cassette_hash`
   in `reports/eval_baseline.json` was updated to match (with an explanatory `cassette_hash_note`
   field directly in the JSON) — but the means it's compared against were deliberately left alone.
   This means `test_cassette_hash_matches_baseline` passes (the gate can run) while
   `test_k8s_quality_regression` correctly and honestly reports the real gap against the real
   frozen target.
2. `eval/test_quality_regression.py::test_k8s_quality_regression` is now `@pytest.mark.xfail`,
   `strict=True`, with the reason string pointing directly at this ADR and ADR-0037. `strict=True`
   is the load-bearing choice: if a future change ever makes this test unexpectedly PASS (e.g. a
   real fix, or a classifier/prompt change that happens to close the gap), pytest reports that as
   a *failure* (`XPASS(strict)`), forcing someone to notice and either lift the marker or
   investigate why the "known-failing" case stopped failing — the marker cannot silently go stale
   in either direction. `test_vscode_quality_regression` is untouched (vscode genuinely improved,
   +0.36 over OLD, and should keep passing normally).

## UPDATE (2026-08-05, same day) — the gate had never actually run in CI. Now it does.

Wiring the `xfail` marker surfaced a bigger problem: `eval-gate.yml`'s two jobs (`Structural
invariants`, `Quality regression`) were dying at an earlier, shared pre-step —
`scripts/verify_model_manifest.py` — before ever reaching `pytest`, for every run since at
least 2026-07-11. Root cause: `data/models/cqr_conformal_adjustments.json`'s committed manifest
hash was never updated after commit `39b5d88` legitimately regenerated that artifact fixing a
CQR calibration-set leak (that commit's own message flags the regenerated artifact as needing
approval "before treating as canonical" — a follow-up that never happened). `continue-on-error:
true` on both jobs meant this never blocked a merge and never produced an alert — it just quietly
turned every run of these two jobs into a no-op for **~3.5 weeks**, invisible unless someone
opened the job log and noticed it never got past the first step.

**Fixed properly, not just re-synced blind:** confirmed the local dev copy and the GCS-published
copy of the artifact already agreed with each other (both `9ad1a63...`) — there was never an
actual content inconsistency, only a stale manifest record — so the fix is a one-line manifest
correction, not a re-publish. Verified against the live bucket: 11/11 artifacts now match.

**Unblocking it revealed a second, independent dead test.** `eval/test_invariants.py::
test_calibration_ece_in_tolerance` has an `ImportError` (`triage_iq.api.loader._load_classifier`
— never existed; the real function is `triage_iq.models.component_classifier.load_classifier`)
introduced in the *same commit* as the ADR-0036 classifier cutover (`2430490`, 2026-07-24). This
test has not passed — has not even successfully *collected* — a single time since it was written.
Fixed (corrected import path).

**So: the honest answer to "has the quality-regression gate ever run successfully in CI" is
effectively no, not since the manifest drift appeared.** We built the `xfail`/`strict=True`
discipline documented above around a check that CI was never actually executing. That's now
fixed — confirmed by re-running live (not just locally): `test_k8s_quality_regression` shows as
`XFAIL` in the actual GitHub Actions log, `test_vscode_quality_regression`/
`test_k8s_no_fabrication`/`test_cassette_hash_matches_baseline` all `PASSED`, and the one
remaining failure (`test_vscode_no_fabrication`) is the pre-existing, already-documented
informational case (vscode#311836) — matching local exactly. The required `test` CI check (a
separate, unrelated dependency-CVE failure — 3 new aiohttp CVEs, fixed by bumping to 3.14.3, a
genuine patch not a suppression) is also confirmed green.

**One more thing unblocking the gate revealed, left open on purpose:**
`test_grounding_known_cases_still_flagged` — a ratchet that pins two specific known-hallucination
issues (#13057 k8s, #311836 vscode) to make sure the grounding verifier hasn't regressed to a
no-op — now fails too, because neither pinned case reproduces under the v3 cassette. Investigated
before touching anything: the grounding verifier itself is still working correctly
(`test_grounding_ratchet_no_new_ungrounded_claims` passes — k8s has 0 ungrounded issues now,
vscode has 1, both within the ratchet's bound), it's specifically the two *named* examples that
no longer hallucinate under v3's prompt. The current actual ungrounded case for vscode is
`#311284` instead. This is the same category of decision as the quality-baseline call above —
which known-bad example to pin as the new canary is a judgment call, not a mechanical fix — so
it's left failing (already non-blocking, `continue-on-error: true`) rather than silently re-pinned.

## Consequences

- **Open, tracked, not silently absorbed.** The synthesis-quality regression on k8s is a real,
  understood, documented gap between the classifier's ground-truth accuracy improvement and its
  effect on downstream LLM-judged plan prose. Anyone running the eval suite sees an `XFAIL` with
  this ADR's number in the reason text, not a bare red `FAILED` and not a silently-passing gate.
- **The remaining untested lever, explicitly not pursued now:** ADR-0037's four variants all
  worked at the *wording* level (how the confidence numbers are described in prose). The one
  mechanism not tried is changing how confidence is *structurally represented* to the LLM (e.g.
  a different numeric transform, a categorical confidence band instead of raw probabilities, or
  restructuring the top-3 block's layout) — a genuinely different lever, not a fifth wording
  variant of the same one. Deliberately not pursued in this session per GG's explicit call to stop
  spending on this thread; logged here so a future session doesn't have to rediscover that this
  specific door was left open on purpose, distinct from the four doors already tried and closed.
- **`test_vscode_no_fabrication` also fails on this recording** (fabrication_rate=0.0909, n=11) —
  pre-existing, already informational-only (`continue-on-error: true`), already documented in its
  own docstring as a known case (vscode#311836) predating this investigation. Unrelated to this
  decision; not touched here.
- No deploy, no rollback, no re-record needed to act on this decision — it is a decision to leave
  production (already serving the multi-label classifier, verified live) and the frozen baseline
  exactly as they are, with the gap now correctly labeled instead of either hidden or silently
  blocking.
- Reversible: if a future structural (not wording) fix closes the gap, re-running the full
  recording, updating the baseline for real this time, and removing the xfail marker is the
  expected unwind path — the `strict=True` marker is specifically what makes that moment visible
  rather than requiring someone to remember to check.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Roll back to the single-label classifier | Restores the judge baseline but gives up a verified, ground-truth-anchored, statistically significant accuracy win to satisfy a proxy metric — optimizing the metric over the product. Also doesn't fix anything: the latent prompt fragility (hedging under compressed confidence spread) remains, just untriggered until the next model/calibration change with a similar confidence shape. |
| Write the new baseline (accept 9.8868 as the new normal) | Destroys the regression signal — a future reader would see a passing gate and have no reason to know synthesis quality on k8s is worse than it used to be, or why. |
| Leave the test as an unmarked, unexplained `FAILED` | Technically already non-blocking (`continue-on-error: true` on this CI job), but a bare red `FAILED` with no ADR pointer reads as "something broke" to a future contributor (or GG, months later) rather than "this is a known, deliberate, documented tradeoff" — exactly the ambiguity this ADR exists to remove. |
| A fifth prompt-wording variant | Explicitly rejected by GG — four variants (no-fix, v1, v2, v3) and ~1.5M cumulative Groq tokens is enough evidence that the framing-only hypothesis, as currently understood, does not have a wording-level fix. The untested lever is structural representation, not wording — see Consequences. |
