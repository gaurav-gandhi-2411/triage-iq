# ADR-0075 — The 150-issue set becomes the CI eval set; the 64-issue set is retired from the LLM gates

Status: Accepted for the set switch by GG on 2026-10-29 (D94.2); its three open questions were decided by GG on 2026-10-30 (D95).
Date: 2026-10-29
Decider: Gaurav Gandhi (D94.2); drafted by the unattended run

Partly supersedes: ADR-0019 (the tolerance-band sample size), ADR-0058 (the vscode ratchet sizing statement), ADR-0061 (the 2026-09-23 baseline as the
canonical baseline). Everything else in those ADRs stands.

## Context

The committed 64-issue set (`eval/eval_set.jsonl`, 53 kubernetes + 11 vscode) no longer measures what production runs.

- **k8s rows are from one old window.** All 53 k8s rows were created between 2015-08-01 and 2015-10-01 (measured over `eval/eval_set.jsonl`
  `created_at`). The served k8s index (bge v2, built from the `derived_v2` scrape, 7,733 rows) covers 2023-2026 issues. A 2015 query has no
  neighbours that were created before it, or, without the created-before filter (serving has none), only neighbours from its future. The frozen
  neighbours stored in the 64-set are the old index's, so the recording measures a retrieval configuration prod no longer runs. The index date range
  itself was not re-measured for this ADR.
- **Prod serves a different configuration than the 64-set was recorded under.** Revision `triageiq-api-00108-mod` runs family_v2 for vscode, the k8s v2
  index and the resolution-evidence block (RUN_STATUS 2026-10-29). The 64-set cassette was recorded under the earlier served config.
- **vscode n=11 cannot gate.** ADR-0058 sized the vscode grounding ratchet at n=11 as report-only and said n of about 73 is needed to bound the true
  rate to a 5% ceiling at 95% confidence with zero observed failures.
- **A 150-issue set exists:** `eval/sets/quality_push_2026-10` (75 kubernetes + 75 vscode, created 2026-01-05 to 2026-10-19, with per-row
  `meta.weight`), with its own frozen neighbours and query embeddings, and `RECORDING_CONFIG.json` = `{vscode_classifier: family_v2, k8s_index: v2,
  resolution_evidence: true}`. Preflight (PR #364) assembles all 150 prompts with no LLM call: 918,124 estimated prompt tokens, 0 degraded, 150/150
  match `prose_interval_check._PROMPT_RE` with the calibrated-interval marker.

## Decision

1. **The CI eval set is `eval/sets/quality_push_2026-10`.** Its `eval_set.jsonl` has sha256
   `151df79cb1239511e7c04581876fe49bf148e47f2f4c08a6a7c7086edf6daba2` (computed as `run_eval._file_sha256` does, sha256 of the file bytes). Its cassette
   lives at `eval/sets/quality_push_2026-10/cassettes/`, next to the config (`eval/recording_config.py` finds a set's config from its cassette dir
   since PR #360). Only this set is re-recorded (Groq, by GG, later).
2. **The canonical baseline path stays `reports/eval_baseline.json`.** At baseline time the current 64-set baseline is archived to `reports/archive/`
   and the new one is written there. A missing baseline now fails the quality gate instead of skipping it (PR #360), so the archive step cannot leave
   the gate silently off. Writing the baseline is a separate, GG-approved step with before/after reported first.
3. **The 64-set files stay in place** (`eval/eval_set.jsonl`, `eval/cassettes/` and, once archived, its baseline) as a historical
   record and as the fixture for the no-LLM tripwires (ECE, conformal coverage, retrieval, which need no recording). No LLM gate references them after
   the switch.
4. **The tolerance band is recomputed for n = 75 per repo:** 2 x 0.748 / sqrt(75) = 0.17 (was 0.22 for k8s at n=46 and 0.45 for vscode at n=11,
   ADR-0019). The per-issue jitter std 0.748 is the ADR-0019 measurement (two local-judge recordings of the old set). That it transfers to the new
   set, config and prompt is an ASSUMPTION, not a measurement on this set; it is revisited when a second recording of the 150 set exists.
5. **Judge numbers are not comparable across sets.** The 64-set judge mean (12.0938 / 15, `reports/judge_mean_ci.json`) and the 150-set means answer
   different populations; no delta between them is reported as a regression or improvement.
6. **Both an unweighted and a weighted headline are published.** Unweighted is the headline. `meta.weight` weighting is reported alongside with its
   Kish effective n (`scripts/judge_mean_ci.py --detail`, PR #361). On the committed weights Kish n_eff is 70.6 overall, 14.1 for k8s (n=75) and 64.3
   for vscode (n=75) (computed over `meta.weight` of the 150 rows): the weighted k8s figure rests on about 14 issues' worth of information, and the
   pooled weighted mean is dominated by vscode (sum of weights 9,480 vs 527).
7. **Known biases, stated up front** (they bound what the set can show):
   - k8s gold components are `sig/*` labels while the served k8s classifier emits its 47-class labels; `component_match` is structurally low for k8s
     and is not read as classifier quality.
   - 60 of 150 rows have no `actual_resolution_days` (32 k8s, 28 vscode) and 52 have no `gold_priority` (24 k8s, 28 vscode); the judge sees "None"
     for them and its resolution/priority dimensions are less informative on those rows.
8. **`record-cassette.yml` is to be retired.** That is a workflow change, drafted by GG; it is not done here.
9. **Prompt-code freeze during recording.** Nothing under `src/triage_iq/prompts/` or the prompt assembly in `triage.py` changes while the 150-set is
   being recorded; a change mid-run changes the prompt hash and orphans recorded entries (ADR-0059).

Wiring done so far: replay, provenance and grounding paths read the set's config (PR #360); expected artifact hashes for the set's config, 15 entries
checked against `MANIFEST.sha256` (PR #363); preflight (PR #364); judge CIs for any set (PR #361). The switch itself (repointing `EVAL_SET_PATH` /
`CASSETTE_PATH`, the grounding baseline hash in `eval/test_invariants.py`, the quality-gate tolerance) follows the recording and is not part of this ADR.

## GG decisions 2026-10-30 (D95) on the three open questions

1. **Recording plan: A.** One recording of all 150 issues on #320's code (the final configuration). The 35.1-day change was decided on
   ground-truth resolution error; a judge before/after is a proxy and does not justify about 460K extra tokens. Before any token is spent, every
   neighbour in a prompt (similar issues and resolution evidence) is checked to be created strictly before its issue (D95.2).
2. **vscode fabrication and grounding at n=75: report-only.** No hard gate.
3. **Missing gold in the judge rubric:** a row with no resolution is shown as "unknown - issue still open", a row with no priority as "not
   prioritised", never "None". This applies before the judge pass.

The original options are kept below for the record.

## Open for GG (as written 2026-10-29, now decided above)

1. **Recording plan.** A: one recording of all 150 (about 919K tokens, estimate 918,124) on #320's code (k8s point 35.1 d, recalibrated intervals).
   B: stage 1 records all 150 on main = current prod (k8s point 3.0 d, `train_median`), then only the 75 k8s rows are re-recorded on #320
   (about +460K; the k8s share of the estimate is 471,006), giving a paired before/after on the same issues at the cost of a second k8s pass.
2. **Hard-gating vscode fabrication and grounding at n=75.** ADR-0058 said n of about 73 is needed for a 5% ceiling with zero observed failures;
   n=75 just reaches that, with no margin for one observed failure.
3. **Judge rubric wording for rows with missing gold** (resolution days, priority): whether "None" is scored, skipped, or worded as unknown in the
   rubric.

## Consequences

- The gate measures the configuration prod serves and the issues it is asked about (2026 issues against a 2023-2026 index, family_v2, evidence).
- The k8s tolerance band (0.17) is tighter than the old 0.22 and the vscode band much tighter than 0.45; that is real extra power only if the jitter
  assumption in Decision 4 holds. If it does not, the gate gives false alarms or misses; it is measured on the first re-record.
- The first 150-set baseline has no prior to compare with: there is no regression history for it, and the 64-set numbers cannot stand in.
- Until the switch lands, CI keeps gating the 64-set; the 150-set has no LLM gate. Recording under the default config would stop at the recorder's
  artifact check by design (PR #363 describes the 150-set config).
- Weighted metrics on k8s are low-information (Kish n_eff 14.1); they are shown, not gated.
- Cost: one recording of about 919K Groq tokens (plan A) or about 1.39M (plan B), plus local judge time; nothing is recorded by this ADR.

## Alternatives

- **Re-record the 64-set too:** rejected by GG; it would spend quota to refresh a set whose k8s half cannot represent the served index.
- **Keep the 64-set with its old neighbours:** rejected; it measures a retrieval configuration prod no longer runs.
- **Keep both as LLM gates:** rejected; two baselines to maintain and two recordings per prompt change, for a set whose k8s half is stale.
- **Use the pre-D92 recording as the before-state:** rejected as a comparator; it is confounded, because its frozen neighbours differ from the new
  set's, so any difference mixes the configuration change with a different retrieval input.
