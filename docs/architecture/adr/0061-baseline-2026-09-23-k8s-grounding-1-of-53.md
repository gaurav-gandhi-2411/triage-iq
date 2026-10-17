# ADR-0061 — The 2026-09-23 baseline, and accepting k8s grounding at 1/53

Status: Accepted
Date: 2026-09-23
Decider: Gaurav Gandhi

## Context

The first recording with the retrained classifier actually in the loop finished 2026-09-23.
Cassette commit `444ef64`, sha256 `f08e296d…`; synthesis `openai/gpt-oss-120b`, prompt_hash
`5f845ce8697e3e7f` (attribution ON), artifact_hash `55559dd93dc2a65d`; judge `qwen3:8b`. It
supersedes the 2026-09-05 "clean baseline" (ADR-0052), which was recorded against the
pre-retrain classifier (ADR-0059 incident) and is void.

On this recording `test_grounding_ratchet_k8s` failed, 1 > 0. The case is `k8s-12665`: the LLM
predicted `networking` (gold `ha`), outside the classifier's top-3
`[kubectl, usability, app-lifecycle]`, with a declared `model_override` and a stated reason.
The ratchet's 0 came from the void recording.

## Decision

1. **Accept 1/53 as the k8s grounding baseline** (`_GROUNDING_BASELINE`), with vscode unchanged
   at 0/11. The gate is correctly flagging a real, disclosed error; it is not a regression.
   - The only comparator is void.
   - k8s has read 0 → 1 → 0 → 1 across this engagement's recordings (different configs), all
     inside overlapping intervals.
2. **n=53 cannot distinguish 0 from 2 ungrounded.** Wilson 95% intervals:

   | observed | interval |
   |---|---|
   | 0/53 | [0.0%, 6.8%] |
   | 1/53 | [0.3%, 9.9%] |
   | 2/53 | [1.0%, 12.8%] |

   The ratchet stays a hard `<=` gate at 1: it catches any movement upward from here. A single
   extra ungrounded case will fail it, and that failure means "look at the case", not "the
   model got worse". Deciding the latter needs the eval-set expansion ADR-0058 already calls
   for.
3. **Promote the recording to `reports/eval_baseline.json`** via
   `eval/run_eval.py --update-baseline` (strict replay, attribution-ON default).

   | | vscode | k8s | overall |
   |---|---:|---:|---:|
   | n | 11 | 53 | 64 |
   | judge mean /15 | 12.2727 | 11.8679 | 11.9375 |
   | fabrication_rate | 0.0 | 0.0189 | — |
   | floor_fail_rate | 0.0 | 0.0566 | — |
   | component_departed_from_top1_rate (report only) | 0.0909 | 0.0943 | — |

4. **Adopt `component_departed_from_top1_rate` as a report-only metric** in `run_eval.py`'s
   per-repo scores and the baseline's `synthesis_quality_floor` block (ADR-0058, 2026-09-23
   addendum).

## Consequences

- **No quality-improvement claim.** Paired against the void old-classifier run, the judge mean
  moved −0.156 (95% CI [−0.51, +0.20], n=64), which cannot be distinguished from zero. The
  retrain's measured gains are taxonomy coverage and top-3 on a common population (ADR-0057),
  not judge quality.
- The mean-band regression gate now compares against a baseline recorded under the shipping
  config. That config is pinned by `tests/test_prompt_config_parity.py`, and artifact identity
  by ADR-0059's gate.
- **What would reopen this:** eval-set expansion (vscode n≈73 per ADR-0058); any re-record, whose
  grounding count should be read against the intervals above, not against the point value 1.

## Alternatives considered

- **Keep the ratchet at 0 and treat `k8s-12665` as a regression to fix first.** Rejected: the 0
  was measured on a configuration that never shipped, and fixing one sampled wrong override
  does not change the underlying rate at this n.
- **Report-only for k8s, like vscode.** Rejected: n=53 still bounds the rate usefully (≲10% at
  1/53), and ADR-0058 kept k8s hard-gated on exactly that basis.
