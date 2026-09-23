# ADR-0058 — Proving the grounding/fabrication gate with a negative control, and sizing the vscode ratchet honestly

Status: Accepted
Date: 2026-09-05
Decider: Gaurav Gandhi

## Context

ADR-0052/0057 closed out a genuine, clean 64/64 baseline. Before treating that baseline as
trustworthy going forward, three things needed checking that hadn't been: whether the
grounding gate had ever been observed actually failing (as opposed to only ever observed
passing), whether vscode's ungrounded count moving 4 → 1 → 0 across this engagement's
measurements was sampling noise or a real metric defect, and whether the two residual gate
failures found while closing ADR-0052 (`test_calibration_ece_in_tolerance`,
`test_model_manifest_clean`) were genuine problems or artifacts of a stale constant.

## Phase 1 — vscode #311836 and the negative control

**1a.** Replayed the final recording's actual cassette entry for vscode #311836: predicted
`api`, which IS inside the new classifier's top-3 (`[ux, extensions, api]`), and is NOT
correct against gold (`perf`). The model has never been correct on this issue in any draw
observed across this engagement.

**1b. Both things are true, and they are not in tension.** Six independent live redraws of
this exact issue under the FINAL shipping config (same model, classifier, schema, prompt)
produced `extensions` (4/6) or `api` (2/6) — never `webview` (the answer that was flagged as
ungrounded earlier in this engagement) — and landed inside the classifier's top-3 on **6/6**
draws, correct against gold on **0/6**. This is genuine sampling variance in *which* wrong
answer the model gives (extensions vs. api), but it is **not** a case of "got lucky avoiding
detection once" — under the current config, detection failure is the *stable, repeatable*
outcome for this specific issue, not a fluke. The grounding metric is doing exactly what it
was designed to do (ADR-0015/0056: consistency with the classifier's own top-3, not
correctness against gold) — for this specific issue, the new classifier's top-3 happens to
be broad enough (`ux`/`extensions`/`api`) that essentially any plausible guess the LLM makes
lands inside it, making the metric structurally uninformative for this one case. That is a
known, documented scope limitation of the metric (not new), now demonstrated directly rather
than inferred.

**1c/1d. Negative control constructed and the gate passes it.**
`scripts/scratch/negative_control_fabrication.py`: corrupted a scratch copy of the #311836
cassette entry's `predicted_component` to a guaranteed-fabricated string, then ran the REAL
production functions — `scripts/measure_grounding.py`'s `compute_grounding_reports()` and
`eval/run_eval.py`'s `compute_scores()` (the exact function
`eval/test_quality_regression.py`'s fabrication-rate assertion consumes) — against it. A
genuine live local-Ollama judge score (zero cost) was obtained for the corrupted plan so the
full pipeline ran end to end with nothing synthetic. Result:

- Grounding gate: `component_grounded=False`, `all_grounded=False` — **caught**.
- Fabrication gate: `fabrication_rate=0.0909` (1/11) — **caught**.
- Real committed `eval_cassette.json`: confirmed byte-identical throughout (sha256 verified,
  never opened for writing — `compute_scores()` takes an explicit `cassette_path` parameter).

**The gate has now been observed both passing (0/64, this engagement's whole history) and
correctly failing (this control) — it is a proven gate, not an assumed one.**

## Phase 2 — Sizing the vscode ratchet honestly

**2a. Run-to-run variance, reported precisely, not conflated:**
- *Cross-configuration counts* (4 → 1 → 0): NOT pure noise — each number reflects a
  different, real methodological change (old classifier vs. a diagnostic decoupling vs. the
  final classifier+prompt), not repeated draws of one fixed configuration.
- *True same-config repeat draws* (this session, 6 draws of #311836 alone, current final
  config held fixed): 0/6 would register as ungrounded — low observed variance in the
  *outcome*, though the specific wrong label varies (extensions/api). This is direct evidence
  for one issue; no equivalent repeat-draw data exists for vscode's other 10 issues.
- *Structural framing*: Wilson 95% CI on 1/11 is [1.6%, 37.7%] — a true rate of 2% and a true
  rate of 30% are indistinguishable at this n regardless of what any single draw shows.

**2b. Recommendation: remove the hard gate for vscode's grounding/fabrication specifically,
mirroring this exact codebase's own existing precedent for the identical problem
(`floor_fail_rate`, already "REPORT ONLY, no gate" for vscode at this same n=11, per
`reports/eval_baseline.json`'s own stated reasoning: "vscode's n=11 gives a [21,72]% CI on
this proportion -- too underpowered to gate reliably"). Explicitly rejected: a rate-based
tolerance band (any nonzero count at n=11 is already close to the practical ceiling — a
"tolerate 1" rule has no principled derivation beyond "that's what we just saw"), k repeats
at recording time (multiplies every future live re-record's Groq quota cost by k×, forever,
to solve a coverage problem that repeat-drawing one issue doesn't fix — the other 10 issues
still have n=1 draw each), and pooling onto n=64 (this project's own `_GROUNDING_BASELINE`
comment already rejected this: "a pooled count on an 83%-k8s-weighted gold set could mask a
vscode-only regression going 0 -> N ungrounded underneath k8s's volume" — reopening a
deliberately-avoided design flaw). k8s (n=53, Wilson upper bound ~9.9% at 0 observed) keeps
its hard gate unchanged — no statistical basis to loosen it.**

**Explicitly NOT continue-on-error.** This project's own CI history (`.github/workflows/
eval-gate.yml`'s `structural-invariants` job comment) documents `continue-on-error` masking
real regressions silently for weeks on three separate prior occasions before being banned at
the job level. Removing an assertion that was never gate-worthy at this n (floor_fail_rate's
precedent) is a different, safer choice than suppressing an assertion that IS gate-worthy —
the former discloses "no check exists here," the latter hides "a check exists and is
broken." Recommending continue-on-error again, for any reason, would repeat a mistake this
exact repo already paid for.

**2c. Implemented:** `eval/test_invariants.py`'s single `test_grounding_ratchet_no_new_ungrounded_claims`
split into `test_grounding_ratchet_k8s` (hard-gated, unchanged behavior) and
`test_grounding_ratchet_vscode` (computes and prints the count, no assertion).
`eval/test_quality_regression.py`'s `test_vscode_no_fabrication` similarly de-gated (was
already a separate function from `test_k8s_no_fabrication`, which stays hard-gated). Both
still run every time and print a visible warning on any regression — nothing is silent.
**vscode's eval arm cannot support a zero-tolerance gate at n=11; n≈73 is required to bound
the true rate to a 5% ceiling at 95% confidence with zero observed failures** (Wilson upper
bound calculation, `reports/old_classifier_on_fresh_taxonomy.json`-adjacent work, ADR-0057
Phase 2 addendum) — this is the same figure already computed there, restated here as the
concrete precondition for ever re-gating vscode's grounding/fabrication as blocking again.

## Phase 3 — The two residual gate failures from ADR-0052

**3a/3b. ECE: confirmed comparable population, re-derived.** Unlike the top-3 accuracy
comparison (ADR-0057), which had to correct for the old classifier's cited figures coming
from a *different, smaller, stale* test population, `_RECORDED_ECE`'s values were already
measured against `eval/eval_set.jsonl` (n=11/n=53, unchanged) — confirmed by reproducing the
old classifier's ECE via this exact method (`scripts/scratch/ece_comparability_check.py`) and
getting an exact match to the recorded constants (0.3781/0.1299). Same population both
times; the improvement (0.3781→0.1875 vscode, 0.1299→0.1210 k8s) is genuine, not a
comparability artifact, and is mechanistically explained by the resolved taxonomy gap: 7/11
vscode gold labels (63.6%) and 7/53 k8s gold labels (13.2%) were outside the OLD classifier's
class list — automatic top-1 misses regardless of confidence, which mechanically inflates
ECE. `_RECORDED_ECE` updated to the new measured values.

**3c. `test_model_manifest_clean`: confirmed it needs no code change, will resolve on the
Phase 5 GCS publish.** The test only compares local file hashes against the committed
`MANIFEST.sha256` — no fabricated manifest was written, and none is needed. Once
`scripts/publish_models.py` runs (hashes the now-current local classifier files, writes
those hashes to `MANIFEST.sha256`, uploads to GCS), local and manifest will match again by
construction.

## Consequences

- vscode's grounding/fabrication ratchet is now honestly not-a-gate, visibly so (both in the
  test docstrings and in printed output on every run), rather than a gate that periodically
  produces a false-positive block on ordinary sampling noise.
- k8s's grounding/fabrication gates are unchanged and remain hard-blocking.
- `_RECORDED_ECE` reflects the current classifier's true calibration; the ECE gate should
  pass cleanly on the next run for both repos.
- `test_model_manifest_clean` remains failing until Phase 4's GCS publish — expected, not a
  new problem.
- **What would re-open this:** the eval-set-expansion work already flagged as urgent across
  ADR-0056/0057 reaching vscode n≈73 is the stated precondition for restoring a hard
  zero-tolerance gate on vscode's grounding/fabrication.

## Alternatives considered

See Phase 2b above for the three rejected ratchet-sizing alternatives (rate-based tolerance,
k-repeat recording, pooled-n gating) with reasoning for each.

## 2026-09-23 correction — Phase 1 ran against the old-classifier recording; re-run on the superseding one

**Text above kept unedited.** The "clean 64/64 baseline" this ADR builds on (ADR-0052,
2026-09-05) had the pre-retrain classifier in the loop (ADR-0059 incident; see ADR-0052's
2026-09-23 correction). Re-checked on the superseding recording (cassette `444ef64`):

- **Negative control re-run: still CAUGHT.** `scripts/scratch/negative_control_fabrication.py`
  (key now resolved from the checkpoint rather than hardcoded) on vscode #311836: grounding
  gate `component_grounded=False, all_grounded=False`; `run_eval.compute_scores()`
  `fabrication_rate=0.0909` (1/11); real cassette byte-identical before/after (sha256
  `f08e296d…`).
- **Phase 3 ECE: independent of the cassette, VERIFIED.** `test_calibration_ece_in_tolerance`
  reads only `eval_set.jsonl` + the classifier. Recomputed 0.1875 / 0.1210 before and after the
  cassette changed (`cd198978…` → `f08e296d…`), identical to `_RECORDED_ECE`. ADR-0057's top-3
  comparison (`scripts/measure_old_classifier_on_fresh_taxonomy.py`) reads only
  `*_classifier_test.parquet` + the pkls; it regenerated
  `reports/old_classifier_on_fresh_taxonomy.json` byte-identical at both cassette states
  (68.09% / 83.61% old-classifier top-3).
- **k8s grounding hard gate fails: 1 > 0** (`k8s-12665`, declared `model_override`, wrong vs
  gold) — see ADR-0052's correction. Gate not modified.

## 2026-09-23 — the grounding metric's #311836 weakness, and a stricter variant (PROPOSAL, not implemented)

**Weakness (recorded, not new, now quantified):** "grounded" means the predicted component is
in the classifier's top-3. When top-3 is flat, almost any plausible wrong guess passes. #311836:
top-3 `[ux .345, extensions .322, api .305]`, gold `perf`. **6/6 live redraws were wrong vs gold
and all 6 scored grounded** (Phase 1b above), and the current recording (`extensions`) is the
same. More broadly, on the current recording **19 of the 20 gold-wrong predictions are scored
grounded**; only `k8s-12665` (outside top-3) is flagged.

**Proposed variant — `component_departed_from_top1`: predicted ≠ classifier top-1, with no
declared-override exemption.** Effect on the current recording (zero-call replay; per-issue
rows in `reports/eval_baseline_candidate_2026-09-23.json`):

| | current rule (∉ top-3) | proposed (≠ top-1) |
|---|---:|---:|
| vscode flagged | 0/11 | 1/11 (#311836) |
| k8s flagged | 1/53 | 5/53 (#14723, #12665, #12784, #14281, #14895) |
| pooled flagged | 1/64 | 6/64 |
| of flagged, gold-wrong | 1/1 | 6/6 |
| of flagged, gold-correct (false alarm) | 0 | 0 (all 44 gold-correct predictions are top-1) |
| gold-wrong but not flagged | 19 | 14 |

In 4 of the 5 newly flagged cases, gold **was** the classifier's top-1 and the LLM moved off it.
This is a distinct and useful failure signal: the LLM overriding a correct classifier.

**Rejected alternatives, measured:**
- "top-1 or declared override" would un-flag `k8s-12665`, the one case the current rule catches.
- "≥ 50% of top-1's confidence" flags nothing new (1/64): top-3 confidences are flat by
  construction.

**Recommendation:** add it as a **report-only** metric beside `fabrication_rate`, and do not
replace or gate on it yet.
- It measures classifier *agreement*, not fabrication.
- A correct top-2 pick would count as a false alarm, and 0 false alarms at n=64 bounds that
  rate only to ≲ 5.6% (rule of three).
- Gating it would need its own ratchet sized per Phase 2 (vscode n≈73).
