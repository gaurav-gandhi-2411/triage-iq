# Pre-registration: vscode similar-issue index v2 with a created-before filter (GG item 3, 2026-11-01)

Written before the filtered scoring is run. Committed to main before any number below is produced.

## Disclosure: this is not blind

On 2026-10-31 the v2 candidate was already scored offline without this filter in serving code. One variant counted
only neighbours created strictly before the query (`%TEMP%\claude\triage-iq\20261031\e_vsidx\scores.json`):

- **served:** R@5 3.5% over all 1,705 queries;
- **v2:** 30.9%;
- **paired difference:** +27.4 pts [+25.1, +29.7].

On the 65 queries answerable by both indexes, v2 was worse (30 hits vs 45). The criteria below were set by GG on 2026-11-01
with those numbers known. The filtered run checks that the shipped filter reproduces them. It is not a fresh test of the
idea.

## Candidate

- **Index:** `D:\triageiq-data\models\candidates\similar_issue_index_microsoft_vscode_bge_v2`, plus a per-issue `created_at`
  array added to `meta.pkl` for the filter.
- **Contents:** 109,537 vscode issues created 2023-01-01 to 2026-10-22, embedded with bge-base-en-v1.5 rev a5beb1e3.
- **Vectors:** from the private Kaggle run of 2026-10-28; parity with production `build_index`, min cosine 0.9999988.
- `index.faiss` sha256 b5be4bb1…; the meta sha changes when dates are added and is recorded at freeze time.

## Filter under test

- Retrieval keeps only neighbours whose `created_at` is strictly before the query issue's `created_at`.
- Applied always in the recording and evaluation path. In serving it is applied whenever the request carries
  `created_at`; `api/app.py` defaults it to now, so for new issues the filter is a no-op.
- If a neighbour has no date, the evaluation and recording path fails closed; serving keeps the neighbour.
- Over-fetch so that k=5 is still returned after filtering.

## Data

- **Queries:** the 2026 vscode test split, 1,705 duplicate queries (`data/splits/microsoft__vscode/dup_task_pairs.parquet`,
  `split_of_source == "test"`).
- **Scoring:** the same definitions as `scripts/served_artifacts_offline_scoring.py` (#384), except for the filter.
  - A query is *answerable* when at least one canonical is in the index.
  - R@5 counts a canonical among the top 5 after the filter, with the query itself excluded.
- **Both indexes** are scored with the filter. The served index's dates come from `derived_v2` `issues.parquet`, falling back
  to raw JSON. The 2 served-index issues without dates are dropped from the served index for this comparison, and that is
  reported.

## Criteria: all must pass to proceed to the cutover (item 3d)

1. **Leakage (3b).** On the 150-set's 75 vscode issues, with v2 and the filter, every similar-issue and evidence neighbour
   in the prompt is created strictly before its issue. `scripts/check_prompt_neighbour_leakage.py` must exit 0 with 0
   violations and 0 unknown dates.
2. **Answerable share.** Over 1,705 queries, a paired bootstrap 95% CI (seed 42, 10,000 resamples) of v2 minus served has a
   lower bound > 0.
3. **R@5 over all 1,705 queries, filtered.** The paired bootstrap 95% CI of v2 minus served has a lower bound > 0.
4. **Memory.** The serving process with production flags and v2 is measured with both repos' models and indexes loaded,
   after loading and 20 `/triage` requests.
   - Peak RSS is <= 3.4 GiB, leaving 0.6 GiB headroom under the 4 GiB service.
   - On the tagged no-traffic candidate revision, Cloud Run memory utilisation during the smoke stays < 85%.
5. **Latency.** CPU retrieval for vscode, at k=5 with the filter, is measured over 100 queries after warm-up, in the
   serving process. The p50 must be <= 300 ms, the spec's per-model budget. p95 is reported.

## Reported, not gated

- **Cold-start impact.** Container startup to the first healthy `/health?deps=1` on the candidate revision against the
  current revision, plus the local load time of v2 against the served index.
- **The 65-query overlap.** R@5 on queries answerable by both indexes. v2 was worse there before, and that is reported
  whatever it shows.
- **Image size** change.

## If any gated criterion fails

There is no cutover; the served index stays. The 75 vscode issues are then recorded on the current configuration, with
the filter in the recording path (item 4).
