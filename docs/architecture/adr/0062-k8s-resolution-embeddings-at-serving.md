# ADR-0062 — Compute k8s resolution `emb_*` features at serving from the retrieval query embedding

Status: Accepted (code on a draft PR; not merged, not deployed)
Date: 2026-10-07
Decider: Gaurav Gandhi (decision made from the offline study below)

## Context

The production resolution predictors were trained with 64 PCA-of-BGE `emb_0..emb_63` features
(`engineer_features(..., embeddings=..., pca=...)`, PCA fitted on training embeddings and stored
on `ResolutionTimePredictor.pca`). Serving never passed embeddings:
`TriageAssistant._collect_signals` called `engineer_features(issue_df, train_df=self.train_df)`
and then filled every missing model column with `0.0`. So in production every `emb_*` was 0 — a
train/serve skew. Retrieval already computes a 768-d BGE query embedding on every request
(`SimilarIssueRetriever.retrieve`, with ADR-0040's k8s-only query instruction); it was thrown away.

An offline study (scratchpad `p3skew/`: `eval_variants.py`, `build_frames.py`, `served_feats.py`,
outputs in `p3skew/out/`; production artifacts, sklearn 1.6.1 image) measured the served
predictor under feature variants. Held-out frame for k8s: the ADR-0041 re-split test, n=2992.
Naive baseline = train median (3.0365 d), MAE 104.229 d.

| k8s variant (served predictor) | MAE (d) | vs naive | bucket vs majority | raw 80% interval coverage |
|---|---|---|---|---|
| Z: served today (emb=0, dsrs=0, no author) | 103.491 | +0.71% | +5.05pp [3.94, 6.22] | 80.48% |
| B: served-mode + retrieval-query embedding | 102.086 | +2.06% | +6.95pp [5.55, 8.36] | 83.26% |
| A: training-path features, doc embedding | 101.981 | +2.16% | +6.35pp | 80.75% |
| C: retrain without `emb_*`, seed 42, served-mode | 102.739 | +1.43% | +6.35pp | 86.20% |
| C2: retrain without `emb_*` and dsrs/author, seed 42 | 102.890 | +1.28% | +5.98pp | 81.22% |

Findings that shape the decision:

1. **k8s improves** when `emb_*` are computed (Z → B: MAE −1.405 d, bucket +1.90pp), using the
   exact vector retrieval already produced. The query-instruction embedding is slightly different
   from the doc-side embedding the PCA was fitted on (mean cosine 0.957 between `query_inst` and
   `doc_old`), and still beats zero-fill; it is within 0.1 d MAE of the doc-embedding variants.
2. **microsoft/vscode gets worse** with embeddings: MAE vs naive −54.19% (Z) → −86.51% (B);
   bucket delta −15.4pp → −24.0pp (`p3skew/out/table_vscode.txt`). (The task brief quoted −82%;
   the study table reads −86.51% for variant B. The direction and conclusion are the same.) vscode
   is already worse than naive on this window and embeddings amplify it. vscode is therefore left
   exactly as it is.
3. **`days_since_repo_start` (dsrs) must not be touched.** Serving computes `created.min()` over a
   one-row frame, so dsrs is always 0; the trained value is days since the repo's first issue. The
   "fix" (Zd/S2: dsrs = days since train start) improves point MAE (+1.51% / +2.86% vs naive) but
   collapses raw interval coverage from 80.5% to 65.7% (Zd) / 72.5% (S2), because the quantile
   models were fit on a different dsrs distribution. Author features (`author_prior_*`) are likewise
   left as served.
4. **The retrain-without-embeddings alternative was measured on one seed** (`seed=42`, one
   untuned run; the k8s script ran only `seed=42`, the 5-seed sweep ran for vscode only). C and C2
   land between Z and B (+1.43% / +1.28% vs +2.06%), but a single seed cannot separate a ~0.65 d
   MAE difference from trainer noise; it is not evidence that retraining is worse, only that it
   was not shown better. It also needs a GCS artifact overwrite, which is out of scope here.
5. **The stored CQR adjustment is stale for k8s, and this change does not fix it.** For variant B
   on the split-30 held-out portion the fresh conformal Q is −1.02 h while the stored value in
   `cqr_conformal_adjustments.json` is +0.2835 h. Held-out coverage is 83.82% with the stored Q and
   79.71% with a fresh one (target 80%). That is a pre-existing miscalibration of the stored
   artifact (sign and size), independent of this PR. Correcting it means overwriting a GCS
   artifact; that is queued for GG and is deliberately **not** done here. With this PR the stored Q
   stays +0.2835 h.

## Decision

For `kubernetes/kubernetes` **only**, reuse the retrieval query embedding as the `embeddings`
input of `engineer_features`, together with the loaded predictor's own `predictor.pca`:

- `SimilarIssueRetriever.retrieve_with_embedding()` returns `(hits, query_embedding)`; `retrieve()`
  delegates to it with unchanged return type. `retrieve_batch` is untouched. One encode serves both
  System 2 and System 3.
- `triage.RESOLUTION_EMBEDDINGS_REPOS = frozenset({"kubernetes/kubernetes"})` gates the behaviour.
  Any other repo takes today's code path exactly: `engineer_features` is called without the
  `embeddings`/`pca` arguments and `retrieve()` (not the new method) is used, so vscode output is
  bit-identical (verified, see Consequences).
- For an enabled repo, if the embedding is missing (retrieval raised, or the retriever returns
  none), the predictor has no fitted `pca`, the dimension differs from `pca.n_features_in_`, or
  the vector is non-finite, the stage falls back to zero-filled `emb_*` and logs a WARNING with
  structured fields (`repo`, `reason`, `embedding_dim`, `pca_n_features_in`). It never raises.
- The eval/cassette path runs the same `TriageAssistant`, but its retriever is the eval-only
  `FrozenRetriever`, which has no embedding. `eval/freeze_query_embeddings.py` freezes the live
  query vectors on CPU float32 for the 53 k8s eval issues into `eval/frozen_query_embeddings.npz`
  (aborts if live top-5 differs from the frozen top-5; all 53 matched), and
  `FrozenRetriever.retrieve_with_embedding()` serves them. Eval therefore exercises the same
  feature path as production and stays deterministic across hardware.
- Not changed: `days_since_repo_start` / author handling, `cqr_conformal_adjustments.json`, any
  GCS artifact, vscode behaviour.

## Consequences

Verified (real production artifacts, Docker image with the lock environment sklearn 1.7.2):

- k8s, 120 test-frame rows (stride 25): pipeline `emb_*` vs the study's variant-B
  `pca.transform(query_inst embeddings)`: max abs diff 2.1e-07; pipeline predictions equal the
  study-B-feature predictions exactly (0.0 days diff); non-`emb_*` features identical to
  origin/main; retrieval hits identical. MAE on those rows: naive 99.922 d, origin/main 99.142 d,
  branch 97.987 d (paired gain 1.155 d, bootstrap 95% [0.575, 1.793]; better on 82 rows, worse on
  38; n=120 is a sample, the n=2992 numbers above are the claim).
- vscode, 62 test-frame rows (stride 10): SHA-256 over every row (features, predictions,
  intervals, bucket, similar-issue hits and scores, full prompt) identical between origin/main and
  this branch; `emb_*` still exactly 0.
- Latency: resolution stage (feature build + 3 predictions) median 63.7 ms zero-fill vs 66.0 ms
  with embeddings (+2.3 ms, n=400 each); `pca.transform` of 1x768 alone is 0.06 ms. The remainder
  is DataFrame assembly of 64 extra columns. No extra encode.

Costs and risks:

- **Prompt inputs change.** The LLM prompt carries the predicted resolution point and interval, so
  every k8s eval issue's synthesis prompt changes (120/120 sampled rows changed prompt text), the
  cassette keys for k8s change, and **the cassette-replay gate will fail on this PR until the 53
  k8s synthesis entries (and their 53 judge entries) are re-recorded.** vscode keys are unchanged.
  The recording checkpoint key is `(issue_id, model, prompt_hash, artifact_hash)`; neither the
  prompt hash nor the artifact hash covers this change, so a resume would treat the existing k8s
  checkpoint entries as done. Re-recording requires removing the 53 k8s entries from
  `recording_checkpoint.json` first (see the PR description).
- The new `frozen_query_embeddings.npz` is not part of the artifact fingerprint
  (`eval/artifact_fingerprint.py`); a stale file would surface as a cassette miss rather than a
  provenance failure. Adding it to the fingerprint would change the combined hash for every repo
  and force a full re-record, so it is not done here.
- Interval coverage moves. With the stored Q, split-30 held-out coverage is 80.81% today (Z) and
  83.82% with this change (B), against an 80% target (raw intervals: 80.48% -> 83.26%; median raw
  width 233.7 d -> 232.2 d, i.e. coverage rises without wider intervals). Over-coverage is the
  safer direction for a triage ETA, but it is a measured change, and the stored Q is
  miscalibrated either way (finding 5); the CQR refresh queued for GG should land with or soon
  after this.
- The training PCA was fitted on doc-side embeddings; serving feeds a query-instruction embedding.
  Measured effect is small and favourable (finding 1), but it is a distribution shift that a
  refit-with-query-embeddings would remove.

## Alternatives

1. **Do nothing.** Keeps +0.71% vs naive on k8s; leaves a known train/serve skew.
2. **Retrain without `emb_*` and ship that predictor.** Removes the skew at the source and works for
   vscode too (C2 on vscode is worse than naive either way), but needs a GCS artifact overwrite and
   a new CQR, and its k8s gain (+1.28% to +1.43%, one seed) was not shown to beat B (+2.06%).
3. **Compute `emb_*` for vscode too.** Rejected: measurably worse (finding 2).
4. **Fix dsrs / author features at serving.** Rejected for now: improves MAE, breaks interval
   coverage (finding 3); needs recalibration first.
5. **Encode a second, doc-style embedding for the PCA input.** Slightly closer to the training
   distribution but adds a second BGE encode (hundreds of ms on CPU) for ~0.1 d MAE; rejected.
6. **Refresh the CQR Q for k8s now.** Correct direction, but needs a GCS overwrite; queued for GG.

## Addendum 2026-10-07/08: k8s cassette re-record and new judge baseline

Because the k8s resolution numbers are inputs to the synthesis prompt, the 53 `k8s-*` synthesis
entries (Groq `openai/gpt-oss-120b`) and their 53 judge entries (local Ollama `qwen3:8b`) were
re-recorded; the 11 vscode entries are unchanged (their checkpoint entries were kept). Recording
ran unattended through `scripts/run_recording_unattended.py` (synthesis, then judge), with Groq
rate-limit waits handled by the launcher; 0 permanently dead entries, 0 degraded or truncated
completions (the launcher hard-stops on both). Old k8s cassette entries remain in
`eval_cassette.json` as unreferenced orphans (the file only grew); they are harmless to replay and
were not pruned, to keep the change reversible.

**Judge drift control (run before trusting the new k8s judge scores).** Six unchanged vscode plans
were re-scored live (no cassette; same warm-up protocol, temperature 0, seed 42, qwen3:8b digest
`500a1f067a9f`): 6 of 6 matched the recorded dimension scores exactly (mean |delta| 0.000). The new
k8s judge scores are therefore comparable with the old ones.

**Result** (`reports/eval_baseline.json`, cassette `f08e296d52bf` -> `dc03f6c0fa94`):

| | Before | After |
|---|---|---|
| vscode mean (n=11) | 12.2727 | 12.2727 (identical, as expected: its entries are unchanged) |
| kubernetes mean (n=53) | 11.8679 | 11.9811 (+0.113) |
| overall (n=64) | 11.9375 | 12.0312 |
| k8s resolution_estimate_reasonableness | 1.585 | 1.679 |
| k8s component_match / next_steps_actionability | 1.660 / 2.962 | 1.623 / 2.925 |
| k8s floor-fail rate | 5.66% (3/53) | 7.55% (4/53) |
| k8s fabrication (grounding) | 1.89% (1/53) | 0.0% (0/53) |

**Reading, honestly.** The k8s mean moved +0.113, inside the +/-0.22 regression band, and every k8s
plan was re-synthesized by a sampling LLM, so part of any per-dimension movement (for example
component_match 1.660 -> 1.623, which this change cannot affect) is re-synthesis variance, not the
change. The one dimension that reads the resolution estimate moved up (1.585 -> 1.679), which is the
direction the offline study predicts, but no quality-improvement claim is made from n=53 on one run.
The grounding ratchet stays at its ADR-0061 bound (k8s <= 1/53): one run at 0/53 is inside the same
Wilson interval as 1/53, so tightening it would make the deterministic gate brittle for no evidence.
`reports/eval_summary.json`'s judge block was regenerated from the new baseline (the drift test in
`tests/test_api.py` checks it). Full unit suite: 366 passed; eval suites: 21 passed (the manifest
test needs cloud credentials and was run in CI instead).

**Still open (separate changes):** the Eval page's resolution table and the README rows describe the
superseded model; they are updated after this PR deploys. The stored CQR Q for k8s is still the old
model's (see Consequences).

## Coupling: k8s resolution features depend on retrieval's query recipe

**What is coupled.** Since this ADR, the kubernetes resolution predictor's `emb_*` features are
`PCA(64)` of the query embedding that retrieval computes (`SimilarIssueRetriever.retrieve_with_embedding`).
The predictor was validated offline, and the k8s cassette recorded, against exactly one recipe:

| Part of the recipe | Pinned value | Where it lives |
|---|---|---|
| Query text | `f"{title}. {body_clean}"`, no character truncation | `triage._collect_signals` |
| Embedding model | `BAAI/bge-base-en-v1.5` (768-d) | `similar_issues.SUPPORTED_MODELS` |
| Query instruction | `"Represent this sentence for searching relevant passages: "`, ON for k8s, OFF for vscode (ADR-0040) | `QUERY_INSTRUCTIONS`, `QUERY_INSTRUCTION_REPO_OVERRIDE` |
| Encode call | `encode([text], normalize_embeddings=True, convert_to_numpy=True)`, float32 cast, no `max_length` | `retrieve_with_embedding` |
| Truncation | the model's own `max_seq_length` (512), never overridden | sentence-transformers |
| PCA input | `n_features_in_ == 768`, 64 components | the served `predictor.pca` |

**Why it matters.** The recipe is shared by two consumers that used to be independent: retrieval
(which wanted the best neighbours) and the resolution model (which wants the vector it was fit on).
Changing retrieval, for example a better embedder, a new instruction or a longer query, would
silently shift every k8s `emb_*` feature, hence the resolution numbers embedded in every k8s
synthesis prompt, hence every cassette key, with no test failing. The PCA fail-soft path (zero-fill
on a dimension mismatch) would hide even a dimension change.

**What pins it.**
- `tests/test_resolution_retrieval_coupling.py` (network-free; `encode` is a spy): the constants
  above, the exact arguments reaching `encode`, query-text assembly, and the PCA dimension
  contract. Five single-constant mutations (query text, instruction, k8s override, normalisation,
  model name) were each confirmed to fail it.
- `eval/test_resolution_query_coupling.py` (eval-gate quality job, where the BGE model is cached
  and the served index is downloaded): encodes 3 fixed k8s eval issues live through the production
  class and requires cosine >= 0.9999 against `eval/frozen_query_embeddings.npz`; also asserts
  `max_seq_length == 512` and the served PCA shape. A changed model, weights or recipe fails the
  gate even though the offline replay (which uses the frozen vectors) would stay green.

**When changing the retrieval recipe.** Do not just update the pinned constant or re-freeze.
(1) Re-validate the resolution metrics offline against the new vectors (the ADR-0062 study:
MAE vs naive, interval coverage), and refit the PCA / predictor if the distribution moved.
(2) Re-freeze `eval/frozen_query_embeddings.npz` (`eval/freeze_query_embeddings.py`).
(3) Re-record the affected cassette entries: `python eval/record_cassettes.py --validate-checkpoint`
lists exactly which `done` entries no longer replay (it recomputes each request through the current
code), and a plain `--mode synthesis` then `--mode judge` resume re-records only those. Before
this check existed, the checkpoint (keyed by issue, model, prompt hash and artifact hash) could not
see a code change that altered a request, and 53 keys were deleted by hand.
(4) Update the pins in the same PR, citing the validation.
