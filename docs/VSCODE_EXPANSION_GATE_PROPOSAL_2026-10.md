# Proposal for GG: gate and baseline for microsoft/vscode after the 63-issue expansion (2026-10-19)

Status: PARKED. GG declined this proposal on 2026-10-20: gates and baselines are re-proposed only after the quality push (spec-quality-push.md), on the new numbers. The measurements below remain valid as measurements; the proposal section is not scheduled. Original status: PROPOSAL ONLY. No threshold, band, baseline, `eval/eval_set.jsonl` or committed cassette was changed. Every number below comes from `reports/vscode_expansion_results_2026-10.json` (produced by `scripts/vscode_expansion_results.py` from the recorded judge entries; a test fails if the report goes stale). Recording artifacts: `reports/vscode_expansion_2026-10-12/` (cassette, checkpoint). Synthesis `openai/gpt-oss-120b`, judge `qwen3:8b` (local, zero Groq), 63/63 synthesized and judged, 0 permanently dead.

## What was measured

| Arm | n | Judge mean /15 | 95% CI (bootstrap over issues) | Component exact match | Floor-fail | Fabrication |
|---|---|---|---|---|---|---|
| Committed vscode | 11 | 12.727 | [11.64, 13.64] | 10/11 = 90.9% (Wilson 62.3-98.4%) | 0/11 | 0/11 |
| Expansion | 63 | 12.524 | [12.08, 12.92] | 48/63 = 76.2% (Wilson 64.4-85.0%) | 2/63 | 1/63 |
| Pooled vscode | 74 | 12.554 | [12.14, 12.92] | 58/74 = 78.4% (Wilson 67.7-86.2%) | 2/74 (0.7-9.3%) | 1/74 (0.2-7.3%) |

Expansion minus committed-11 judge mean: -0.20, bootstrap 95% CI [-1.26, +0.93]. The two arms are not distinguishable on the judge total; the gap in component exact match (90.9% vs 76.2%) is inside the n=11 interval too. What the expansion does show is that the n=11 point estimate of 90.9% was optimistic: the better estimate for vscode is 78% (68-86%).

Where the 16 component misses are (pooled): git->scm 2, tasks->error-telemetry 2, and 12 singletons (perf->api, editor-find->keybindings, workbench-os-integration->electron, themes->ux, javascript->typescript, extensions<->settings-sync, debug->error-telemetry, extensions->menus, workbench-tabs->workbench-editors, file-io->editor-core, error-list->ux). The 63-issue set has error-telemetry 14, extensions 8, accessibility 7, git 7 as its largest gold components. The two floor-fails are vscode-195980 and vscode-206425; the single ungrounded plan is vscode-175236.

## Current gate on vscode (for reference)

`reports/eval_baseline.json`: vscode n=11, mean 12.7273, band 0.45 (2 x SEM with the measured per-issue judge jitter 0.748). It is the coarsest gate in the suite and is a regression detector on a fixed, replayed set, not a quality floor.

## Proposal

1. **Adopt pooled n=74 as the vscode gate set.** Baseline mean 12.554; band = 2 x 0.748 / sqrt(74) = 0.174 (about 2.6x tighter than today's 0.45, same derivation, same jitter source). The band stays one-directional (fires only on a drop below baseline minus band). Reason: the n=11 band could not see a 0.4-point regression; n=74 can see 0.2.
2. **Keep floor-fail, fabrication and departure-from-top-1 as report-only** (as decided 2026-07-11 and 2026-09-23). At n=74 the floor-fail interval is still 0.7-9.3%, too wide to gate; revisit after the next full re-recording.
3. **Do not change the README headline yet.** The 12.09/15 pooled judge mean (n=64) and the 85.82% vscode classifier top-3 stay as they are until the baseline is written; if you adopt the proposal the README manifest entries for the judge mean and its CI move with the new baseline in the same PR (rule 65c).

## What adopting it takes (steps for a later PR, none done)

- Append the 63 rows to `eval/eval_set.jsonl` and merge the 63 synthesis + 63 judge cassette entries into `eval/cassettes/eval_cassette.json` (it changes `eval_set_hash` and `cassette_hash` in the baseline and in the tests that pin them). I have not tested the replay of the merged files, and I have not checked whether `eval/frozen_query_embeddings.npz` / `frozen_retrieval_provenance.json` need the 63 new queries added; the expansion rows carry their frozen `similar_issues`, so treat the freeze step as unverified until a dry run.
- Write the new baseline with `python eval/run_eval.py --update-baseline` after you have seen the before/after means above (your call: this is a new baseline).
- Whole-set impact: pooled overall mean over 127 issues would move from 12.094 (n=64) to the value that run prints; I did not compute it because it requires the merged cassette.

## Caveats

Intervals cover issue sampling only; they exclude judge noise and synthesis sampling noise. The judge is a local 8B model whose `component_match` is more lenient than string equality (D74), so the exact-match column above is the cleaner component figure. The 63 issues were drawn by the 2026-10-12 sampling (`eval/expansions/vscode_2026-10-12/provenance.json`, real repo labels only). The synthesis ran against production-equivalent artifacts (fingerprint `14ac2b3c50f3a675`), and the 11 committed issues were recorded under the earlier recording, so the arms differ in recording date as well as issue set.
