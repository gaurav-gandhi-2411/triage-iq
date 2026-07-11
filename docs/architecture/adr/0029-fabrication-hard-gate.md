# ADR-0029 — Fabrication Metric: Informational to Hard Gate

**Status:** Accepted
**Date:** 2026-07-12
**Decider:** Gaurav Gandhi (gate-policy decision); executed autonomously by CC per spec.md.

---

## Context

ADR-0028's audit found synthesis quality gating structurally blind to fabrication: the
local judge scores the final `TriagePlan` JSON but never sees `classifier_top3` or the
retrieved-issue set, so it cannot detect when synthesis cites a component or similar-issue
its own upstream signals never produced. A known hallucination (vscode #311836) scored
9/15 — above its repo's 8.36/15 mean.

Phase B3 (PR #31) added a deterministic fabrication check reusing the existing grounding
verifier (`grounding.py`, ADR-0015) but wired it **informational-only**: `fabrication_rate`
computed and surfaced in `reports/eval_baseline.json`, `eval/test_quality_regression.py`
asserted `== 0` per repo, but that assertion lived in a `continue-on-error: true` CI job —
visible, never blocking. ADR-0028 named this as the deliberate, conservative next step:
observe the real-world rate before promoting to a hard gate, with the promotion criteria
not yet defined.

This ADR is that promotion, gated on the measurement it was always conditioned on.

## Measurement (the gate on the gate)

Re-ran `scripts/measure_grounding.py` against the current eval set (the previously
committed `reports/grounding_measurement.json` was stale — dated to before the B1
near-duplicate quarantine that dropped k8s from 54 to 53 issues, changing `eval_set_hash`).

| Repo | n | fabrication_rate | type |
|---|---|---|---|
| microsoft/vscode | 11 | 1/11 = 9.09% | component only |
| kubernetes/kubernetes | 53 | 1/53 = 1.89% | component only |
| **overall** | 64 | **2/64 = 3.12%** | 0% similar-issue fabrication |

Both cases are the same two already known from ADR-0015/ADR-0028 (vscode #311836, k8s
#13057) — no new fabrications surfaced. Zero similar-issue-reference fabrications exist
anywhere in the 64-plan eval set; both fabrications are on the component axis.

**Artifact check (is this a top-3-vs-top-5 threshold quibble?):** No. Checked both
predicted components against each classifier's *full* label space, not just top-3:

- vscode #311836 predicted `"webview"` — not one of the classifier's 28 labels, at any rank.
- k8s #13057 predicted `"storage"` — not one of the classifier's 35 labels, at any rank.

Neither is a near-miss (not rank 4 or 5, and a case-insensitive/whitespace-normalized
match — checked directly via `scripts/measure_grounding.py`'s diagnostic — also fails for
both). The LLM invented components the classifier's label space doesn't contain. This
confirms both are genuine fabrications, not artifacts of the top-3 threshold, and that
loosening the definition to top-5 (or any N) would not have changed either verdict.

## Decision

**1. Fabrication definition:** unchanged — top-3 membership (`grounding.py`,
`verify_plan_grounding`), as already shipped. No change is justified: the measurement
shows the actual failure mode is "invented a label outside the entire space," not
"cited a real but lower-ranked label," so a looser threshold would not distinguish real
fabrication from an artifact — there is no artifact to distinguish here.

**2. Gate scope: CI-gate only, not serve-time.** The measured rate justifies blocking CI
merges on regression. Serve-time rejection/flagging in live `/triage` is a separate,
larger-blast-radius change (a production behavior change requiring its own deploy
approval per spec's hard rules) and is explicitly deferred — not actioned this iteration.
`plan.grounding_status` is already returned in every response for callers who want to
check it themselves; that visibility is unchanged.

**3. Fail mode: hard-fail in CI, ratchet semantics (not zero-tolerance).** The gate fails
the workflow on any fabrication **beyond** the measured, pinned baseline (the 2 known
cases), not on the mere existence of those 2 cases. Absolute zero-tolerance would fail
CI unconditionally on every run given the current committed cassette, which is out of
scope to fix this iteration (no retraining, no verifier-logic change, no re-recording —
spec's explicit exclusions). This mirrors every other ratchet already in this codebase
(mean-band judge gate, model-artifact-drift guard, ECE tolerance) — ceiling-at-baseline,
not an absolute floor, tied to `eval_set_hash` so eval-set drift is caught loudly rather
than silently comparing across different sets.

## What was wired

- **`eval/test_fabrication_gate.py`** (new file): the ratchet + known-case-pin tests
  (`test_grounding_ratchet_no_new_ungrounded_claims`,
  `test_grounding_known_cases_still_flagged`), moved from `eval/test_invariants.py`
  where they already existed with correct ratchet semantics but lived inside an
  informational (`continue-on-error: true`) job alongside unrelated invariants
  (calibration, conformal coverage, retrieval, model-manifest drift). The check logic
  itself is unchanged — only its CI blast radius.
- **`.github/workflows/eval-gate.yml`**: new `fabrication-gate` job. Same setup steps
  (checkout, GCS model download, manifest drift verification) as the existing two jobs,
  but **no `continue-on-error`** — the only hard-blocking job in this workflow.
- **`eval/test_quality_regression.py`**: removed the redundant, contradictory
  zero-tolerance `test_{vscode,k8s}_no_fabrication` assertions (`fabrication_rate == 0`)
  added by B3. They were already informational-only and, unlike the ratchet, would fail
  unconditionally against the current cassette — keeping both a zero-tolerance check and
  a ratchet check for the same underlying signal would send two different, conflicting
  gate signals for the same failure mode. `run_eval.py`'s `fabrication_rate` computation
  and its entry in `reports/eval_baseline.json` are unchanged and still reported — only
  the redundant gating assertion was removed. The mean-band judge gate in this file is
  untouched (still a soft regression detector, per spec's explicit out-of-scope note).
- **`reports/eval_baseline.json`**: `synthesis_quality_floor.fabrication_rate.gate`
  text updated to describe the new hard, blocking policy and point to
  `eval/test_fabrication_gate.py`. `floor_fail_rate` is untouched (report-only, out of
  scope).
- **`eval/README.md`**: documents the new file and that `fabrication-gate` is the one
  hard-blocking job in `eval-gate.yml`, in contrast to the other two, still-informational
  jobs.

No baseline re-establishment was needed: the new gate's ceiling is the same 2-case
baseline that was already pinned in the moved code, so nothing that currently passes or
fails changed.

## TEST THE TEST

1. Ran `pytest eval/test_fabrication_gate.py -v --no-cov` against the real, unmodified
   cassette: **2 passed** (current state is within the approved baseline).
2. Temporarily tightened the k8s ratchet ceiling from `"ungrounded_count": 1` to `0`
   in `eval/test_fabrication_gate.py` (simulating a fabrication beyond what's approved —
   equivalent, from the ratchet's perspective, to a new fabrication appearing in the
   data, since the check is a relative comparison between current and baseline counts).
   Re-ran: **`test_grounding_ratchet_no_new_ungrounded_claims` FAILED** —
   `AssertionError: kubernetes/kubernetes: ungrounded claim count regressed: 1 > baseline 0`.
3. Reverted the edit (back to `1`). Re-ran: **2 passed** again.

## Consequences

- **What changes:** a fabricated component or similar-issue claim beyond the measured
  baseline now fails CI outright, instead of being visible-but-ignorable. The two known
  pre-existing cases remain pinned and do not block merges by themselves.
- **What becomes easier:** any future synthesis regression that produces a fabricated
  claim is caught immediately, not just observed in a report someone has to remember to
  check.
- **What becomes harder:** fixing either of the 2 known cases (or genuinely improving the
  classifier/prompt in a way that changes which components get predicted) requires a
  deliberate baseline update in `eval/test_fabrication_gate.py`'s `_GROUNDING_BASELINE`,
  same as any other ratchet in this project — not a silent pass.
- **What's explicitly deferred:** serve-time rejection/flagging of fabricated claims in
  live `/triage`. `plan.grounding_status` already exposes the signal to callers; turning
  it into an active reject/flag at serve time is a separate scoping and deploy decision,
  not bundled into this CI-only iteration.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Zero-tolerance gate (fail on any fabrication, including the 2 known cases) | Would fail CI unconditionally on every run against the current committed cassette; fixing those 2 cases requires prompt/model work explicitly out of scope this iteration (spec: no retraining, no verifier-logic change). |
| Loosen definition to top-5 classifier membership | Measurement shows this wouldn't change either known case — both predicted labels aren't in the label space at all, not merely outside top-3. A looser threshold would only reduce sensitivity to genuine future fabrications for no corresponding gain in this iteration. |
| Ship serve-time reject/flag live this iteration | Larger blast radius (production response-shape change) for a signal that's already visible via `grounding_status`; spec requires explicit prod-deploy approval, and CI-gate alone already closes ADR-0028's core finding (a fabricating plan can no longer silently pass evaluation). |
| Keep both the zero-tolerance assertion and the new ratchet | Two gates for the same underlying signal with contradictory semantics (one always red, one green) is confusing and not more protective than the ratchet alone. |
