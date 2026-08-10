# ADR-0043 — Combined retrieval+resolution cutover: k8s synthesis quality recovers 61% of the ADR-0037 gap; confidence-framing was never the whole story

**Status:** Accepted (findings + baseline write)
**Date:** 2026-08-06 (findings); 2026-08-10 (baseline write)
**Decider:** Gaurav Gandhi

## Context

ADR-0037/ADR-0039 diagnosed and accepted a k8s synthesis-quality regression (-0.6226 vs the
frozen OLD baseline, 2.8x the measured noise band) as a side effect of the ADR-0036 classifier
cutover: independent per-class confidence scores clustering tighter than the old softmax's,
reading to the LLM as "the model is unsure" and inducing hedged language. Four prompt-wording
fixes (v1-v3) failed to close the gap. ADR-0039's decision — keep the classifier (a verified
ground-truth accuracy win), leave the baseline frozen, mark the gate `xfail` — was made on the
working assumption that the regression was primarily a confidence-framing/hedging effect.

Since then, two independent, unrelated fixes shipped that both improve signals synthesis
consumes, neither touching the classifier or its confidence framing at all:
- **ADR-0040 (retrieval)**: k8s product-task R@5 18.0% → 24.7% (corpus truncation fix + query
  instruction, both cutover and live-verified).
- **ADR-0041 (resolution)**: k8s bucket-classifier accuracy delta vs naive +3.27pp → +6.35pp
  (stale-split fix, cutover and live-verified).

GG's framing going in: if the judge means recover toward OLD, the original regression was partly
upstream-signal-driven, not purely confidence-framing; if they don't recover, the ADR-0037
diagnosis stands unchanged. Ran one combined cassette re-record (`dry_run:true`, run
`31103655399`) to answer this directly, on the current cassette (`996a02da...`) after both
cutovers.

## Result: both are true, for different dimensions — not either/or

| Repo | OLD | v3 (post-ADR-0036, pre-cutover) | **NEW (post both cutovers)** | Δ vs OLD | Δ vs v3 |
|---|---|---|---|---|---|
| k8s | 10.5094 | 9.8868 | **10.2642** | -0.2452 | **+0.3774** |
| vscode | 8.3636 | 8.7273 | 8.6364 | +0.2728 | -0.0909 (within its own ±0.45 noise band) |

**k8s recovered 60.6% of the original -0.6226 gap** (+0.3774 of it). The residual -0.2452 sits
just outside k8s's own ±0.22 tolerance band — real, not noise, but roughly 40% the size of what
ADR-0037 originally measured.

### Per-dimension causal mapping — each fix landed almost exactly on the dimension it feeds

| k8s dimension | OLD | v3 | NEW | Read |
|---|---|---|---|---|
| `similar_issues_relevance` | 2.6038 | — | **2.8491** | **Beats OLD by +0.2453** — direct payoff of the retrieval fix, the dimension it most directly feeds. |
| `resolution_estimate_reasonableness` | 1.3019 | 1.1887 | 1.2264 | **Half-recovers** (+0.0377 of a -0.1132 gap) — consistent with a better resolution model improving the numbers fed into synthesis, but not fully. |
| `component_match` | 1.4151 | 1.2830 | 1.3019 | **Barely moves** (+0.0189 of -0.1321) — neither cutover touches the classifier or its confidence output; this dimension had no reason to recover, and didn't. |
| `next_steps_actionability` | 2.5472 | 2.3774 | 2.3962 | **Barely moves** (+0.0188 of -0.1510) — same story; ADR-0037 traced this dimension's hedging specifically to confidence-framing, and it's exactly the one that stayed flat. |
| `floor_fail_rate` | 0.0943 | 0.1887 | 0.1887 | **Unchanged from v3, still 2x OLD** — see the vscode-adjacent investigation below; this is judge-scoring-jitter-shaped, not further diagnosed for k8s here. |

This is the clean part of the finding: **the two dimensions ADR-0037 explicitly traced to
confidence-framing/hedging (`component_match`, `next_steps_actionability`) did not move at all**
despite materially better upstream signal quality, while **the two dimensions that directly
consume the improved signals (`similar_issues_relevance`, `resolution_estimate_reasonableness`)
moved substantially or fully recovered.** Neither cutover could plausibly have changed the
classifier's confidence output — they didn't touch it — so a dimension moving only when its own
upstream signal improved, and not otherwise, is exactly the causal signature you'd expect if both
mechanisms are real and independent, not competing explanations for the same gap.

**Conclusion: the original -0.6226 regression was a mix — roughly 61% attributable to weak
upstream signal quality (now fixed), 39% consistent with ADR-0037's confidence-semantics
diagnosis (unchanged, still present).** ADR-0039's decision is unaffected by this — the classifier
was never going to be rolled back for a proxy metric, and the residual gap is still real and still
best explained by confidence-framing. What changed is understanding *how much* of the original gap
that explains: not all of it, as originally framed, but a genuine ~39% share, with the other ~61%
now shown to be a separate, already-fixed cause.

## vscode: investigated a counterintuitive floor_fail_rate jump, resolved as small-n noise

vscode's mean is flat within its own noise band (v3→NEW: -0.0909, band is ±0.45), but
`floor_fail_rate` moved 0.4545→0.6364 (v3, not directly measured, vs frozen OLD 0.4545, to
0.6364 now) — a large-looking jump that doesn't follow from the change (Lever 1's corpus
truncation should *help*, not hurt: 43.2% of vscode issues were truncated pre-fix, losing ~48%
of their content on average).

**Investigated directly** (`scripts/investigate_vscode_floor_fail_shift.py`, replaying v3's
actual cassette against the pre-Cutover-A `eval_set.jsonl` it was really recorded against, vs.
the new recording, per-issue): v3 floor-fails 5/11, NEW floor-fails 7/11, **exactly 2 issues
moved (#4996, #311284), zero issues moved the other direction.**

- **#4996**: `predicted_component` identical, gold identical, `similar_issues_relevance`
  *improved* 2→3. The floor-fail is caused entirely by the judge's `component_match` score
  flipping 1→0 on the *same* prediction — a borderline "editor-core vs. editor-multicursor"
  partial-credit call, both times explicitly called out as a near-miss in the judge's own
  rationale.
- **#311284**: `similar_issues_relevance` *unchanged*, still 3 ("highly relevant") both times.
  `predicted_component` changed (extension-host→api) but both are wrong against gold='scm' — a
  different wrong guess, not a worse one. Same `component_match` 1→0 flip pattern.

**Neither newly-failing issue shows retrieval getting worse** — one improved, one held steady,
directly consistent with the aggregate R@5 finding (50.5%→53.5%, Lever 1 alone, ADR-0040). The
jump is fully explained by judge-scoring variance on already-borderline `component_match=1`
calls flipping to 0 — the same qwen3:8b judge jitter ADR-0019 already measured (std=0.748/15) and
built the mean-band tolerance around, just landing on a binary threshold metric
(`floor_fail_rate`) that tolerance band doesn't cover. **At n=11, 2 issues is the whole
signal** — this is unreadable as a trend, not a regression from the index change.

## Other measurements from the same recording

- **Label accuracy** (`predicted_component == gold_component`): overall 40.6% (26/64), k8s
  43.4%, vscode 27.3% — flat vs v3 (42.2% overall, k8s 45.3%, vscode 27.3% unchanged) within a
  1-issue margin. Expected: the classifier is untouched by both cutovers, same weights, same
  input text. Not a new finding.
- **Fabrication rate**: 0/53 k8s, 0/11 vscode — clean on both repos, consistent with the
  no-fix→v1→v3 pattern of named cases not reproducing recording-to-recording.
- **Prose/number contradiction rate** (ADR-0042, LEVER 4): 0/53 k8s, 0/11 vscode — clean, same
  as the pre-cutover measurement.

## Decision

**Findings accepted and recorded here regardless of the baseline call**, per instruction. Writing
the new baseline (k8s 10.2642, vscode 8.6364) required a separate, explicit approval step — this
ADR was originally the informative-result writeup that approval decision was contingent on, not
the approval itself.

**2026-08-10 update: the baseline write was approved and executed** (run 31103655399's dry-run
recording, committed via PR #56). `reports/eval_baseline.json` and `eval/cassettes/eval_cassette.json`
now hold the NEW values from this ADR's table (k8s 10.2642, vscode 8.6364, overall 9.9844),
replacing the frozen pre-cutover baseline (k8s 10.5094, vscode 8.3636, overall 10.1406) that
`test_k8s_quality_regression` had been pinned against via `xfail(strict=True)` since 2026-08-05.

**The -0.2452 k8s residual (vs. the OLD frozen baseline) is now BAKED INTO the new baseline as a
deliberately accepted tradeoff — it is not a ceiling, ratchet, or open item this gate tracks going
forward.** Concretely: `test_k8s_quality_regression` now compares future recordings against
10.2642, not against 10.5094 — so a future reader should not read 10.2642 as "the regression we're
still trying to close." It's the floor the ADR-0036 classifier cutover's verified ground-truth
accuracy win (k8s top-1 +9.09pp / top-3 +4.55pp) costs in this proxy metric, accepted per ADR-0039,
with 61% of the *original* -0.6226 gap independently recovered by the unrelated upstream-signal
fixes documented above (ADR-0040/ADR-0042) and the remaining ~39% left as the confidence-semantics
residual this decision explicitly keeps rather than chases further at the prompt-wording level
(ADR-0037 already exhausted that lever). The xfail marker on `test_k8s_quality_regression` was
removed in the same change — the test now protects against regressions *below* 10.2642, not
against the already-accepted gap relative to 10.5094.

## Consequences

- **ADR-0039's decision is unchanged**: keep the classifier, don't roll back for a proxy metric.
  What's new is the causal attribution — confidence-framing explains ~39% of the original gap,
  not all of it, and the other ~61% is now independently confirmed and fixed.
- **If the baseline is approved and written**: k8s's gate would need re-evaluation against a
  -0.2452 gap (marginally outside the ±0.22 band) instead of -0.6226 — a materially smaller,
  better-understood residual, still real, still likely confidence-framing per the per-dimension
  evidence above.
- **The remaining ~39% (confidence-framing) is not re-attempted here** — ADR-0037 already tried
  four prompt-wording variants; the untested lever it identified (structural confidence
  representation, not wording) remains open and untouched by this ADR.
- **vscode's floor_fail_rate is confirmed noisy at n=11**, not a new problem introduced by either
  cutover — worth remembering the next time this metric moves for vscode specifically.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Treat the k8s recovery as full vindication of "it was always upstream signal" | Contradicted directly by the per-dimension data — `component_match`/`next_steps_actionability` didn't move at all, which they should have if confidence-framing weren't still a real, independent factor. |
| Treat vscode's floor_fail_rate jump as a real regression and hold the whole recording | Investigated first, per the standing "counterintuitive result usually means a bug" instinct — resolved to 2 issues, judge-scoring variance, retrieval unambiguously not worse for either. Holding on an explained, noise-shaped n=11 movement would be the wrong call. |
| Write the baseline now, given the story is coherent | Explicit instruction: report straight, no baseline write without approval — this ADR is that report. |
