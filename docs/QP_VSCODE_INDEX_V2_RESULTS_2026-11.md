# Results: vscode similar-issue index v2 with a created-before filter (prereg items 3b, 3c)

Pre-registration: `docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md` (binding; criteria, data and definitions unchanged).
Machine-readable numbers: `reports/vscode_index_v2_prereg_results_2026-11.json` (scripts commit `c5fad948`, base `0466b65`).

## Not blind

The unfiltered/older-only scoring of 2026-10-31 was known to everyone when the criteria were set, as the prereg discloses.
This run checks that the shipped filter reproduces it; it is not a fresh test of the idea. Reproduction: served R@5 over all
queries 59/1,705 now and before (identical); v2 525/1,705 now against 526/1,705 before (see deviation 4).

## Verdict

**Criteria 1-3 and 4 (memory half) pass. Criterion 5 (latency) FAILS as measured: v2 p50 = 611.6 ms against the 300 ms limit.**
Criterion 4's Cloud Run half is pending. Under the prereg's rule ("all must pass"), a failed gated criterion means no
cutover and the 75 vscode issues get recorded on the current configuration (item 4).

**Decision (orchestrator, 2026-11-01): no cutover.** The served vscode index stays, applied as pre-registered and not
re-interpreted after the fact. Any re-measurement of criterion 5 on Cloud Run would be a new pre-registration, and that is
GG's call. See "Reading criterion 5" below for what the number does and does not show.

| # | Criterion | Result | Label |
|---|---|---|---|
| 1 | Leakage, 75 vscode issues: `check_prompt_neighbour_leakage.py` exit 0, 0 violations, 0 unknown | exit 0; 0 violations; 0 unknown; 0 prompt-vs-signal mismatches; 150/150 prompts assembled | PASS (VERIFIED) |
| 2 | Answerable share, paired bootstrap 95% CI of v2 - served, lower bound > 0 | served 95/1,705 = 5.57% [4.58, 6.76]; v2 1,634/1,705 = 95.84% [94.78, 96.69]; diff +90.26 pts, CI [+88.50, +91.91] | PASS (VERIFIED) |
| 3 | R@5 over all 1,705 queries, filtered, CI lower bound > 0 | served 59/1,705 = 3.46% [2.69, 4.44]; v2 525/1,705 = 30.79% [28.65, 33.02]; diff +27.33 pts, CI [+25.04, +29.62] | PASS (VERIFIED) |
| 4a | Peak RSS after load + 20 `/triage`-path calls <= 3.4 GiB | v2 1.980 GiB (served config 1.598 GiB; delta +0.38 GiB) | PASS (VERIFIED for this harness; production comparability BELIEVED) |
| 4b | Cloud Run memory utilisation < 85% on the tagged candidate revision | not measured | PENDING |
| 5 | vscode retrieval p50 <= 300 ms (k=5, filter, 100 queries after 10 warm-up, 2 threads) | v2 p50 611.6 ms, p95 774.0 ms | **FAIL (VERIFIED as measured; laptop, see below)** |

Bootstrap: seed 42, 10,000 resamples, paired over the 1,705 queries; the same resample indices serve criteria 2 and 3.
Wilson 95% intervals for each rate in brackets. Test split: `dup_task_pairs.parquet`, `split_of_source == "test"`, 1,705 queries.

## 3b: leakage detail (VERIFIED)

The 150-set was copied to scratch (source read only). The 75 vscode rows were re-frozen against the dated candidate
(`index.faiss` sha b5be4bb1..., `meta.pkl` sha ae52dcf5..., 109,537 issues) with `created_before=<issue created_at>,
strict_dates=True`; the 75 k8s rows are byte-identical to the live set (`diff` of the k8s lines is empty). The query-embedding
npz was copied unchanged: only k8s carries one (`RESOLUTION_EMBEDDINGS_REPOS`).
Prompts were assembled with the real path (`recording_config.build_assistant`, frozen retrievers, sockets blocked).

| Repo | Neighbour type | Total | Before | Equal | After | Unknown | Self |
|---|---|---|---|---|---|---|---|
| vscode | similar | 375 | 375 | 0 | 0 | 0 | 0 |
| vscode | evidence neighbours / items | 0 / 0 | - | - | - | - | - |
| k8s | similar | 375 | 375 | 0 | 0 | 0 | 0 |
| k8s | evidence neighbours | 165 | 165 | 0 | 0 | 0 | 0 |
| k8s | evidence items | 210 | 210 | 0 | 0 | 0 | 0 |

vscode has no resolution-evidence artifact, so it has no evidence lines. 75 of 75 vscode rows changed neighbours against the
current set (the v2 index holds 109,537 issues against 13,315). The as-of filter dropped 27 k8s
evidence items, as before. Neighbour dates: 373 distinct vscode and 348 k8s neighbours, all from the issue parquet, 0
parquet/raw disagreements.

## 3c: other reported items

- **Served index undated rows (prereg: dropped and reported):** #5123 and #29238 (of 13,315). The scored served copy has 13,313 vectors.
  Dropping them changes neither the served answerable count (95) nor its hits (59).
- **R@5 over answerable:** served 59/95 = 62.1% [52.1, 71.2]; v2 525/1,634 = 32.1% [29.9, 34.4].
- **65-query overlap (answerable by both; reported, not gated):** served 45/65 = 69.2% [57.2, 79.1]; v2 30/65 = 46.2% [34.6, 58.1];
  paired v2 - served -23.1 pts, CI [-33.8, -13.8] (n = 65). v2 is worse here, as before, and it is the honest caveat on criterion 3:
  the gain comes from answering 1,539 more queries, at a lower hit rate per answerable query.
- **Index files (bytes):** v2 `index.faiss` 336,497,709 + `meta.pkl` 112,488,186; served 40,903,725 + 9,499,789.
  Image size delta of the index files: +398,582,381 bytes (+380.1 MiB). The Docker image size itself was not built or measured.
- **Local load time** (fresh process each, file cache probably warm, `HF_HUB_OFFLINE=1`): `meta.pkl` + `faiss` read v2 0.257 s + 0.215 s,
  served 0.035 s + 0.021 s; whole retriever incl. BGE model v2 0.861 s, served 0.458 s; `ModelStore.load_all` (both repos) v2 3.80 s,
  served 3.11 s. Container cold start on Cloud Run: not measured (candidate revision).
- **Memory:** after load v2 1.269 GiB, served 0.877 GiB; after 20 calls v2 1.980, served 1.593 (current working set; peaks in the JSON).
  The whole-process peak at the end of the run was 3.378 GiB (v2) and 3.009 GiB (served), from the latency stage, which loads
  the full vscode issue text table in the harness. That attribution is BELIEVED (not isolated); it is the figure to cite if the
  criterion is read as "peak of the whole harness", and it is 0.022 GiB under the 3.4 limit.

## Reading criterion 5 (latency)

| Detector | p50 ms | p95 ms | mean ms | Dated |
|---|---|---|---|---|
| v2 (109,537 vectors) | 611.6 | 774.0 | 520.3 | yes |
| served, production index (13,315) | 585.3 | 751.7 | 495.3 | no (filter is a no-op) |
| served, dated copy (13,313) | 609.6 | 758.7 | 510.4 | yes |

The served index also misses 300 ms in this environment (585 ms), and v2 is within noise of the dated served copy (+2 ms p50),
so the failure is dominated by something both share, most plausibly the 512-token BGE encode on 2 threads (BELIEVED; encode and
search were not timed separately). The laptop was not idle: a live recording was running and CPU load read 48% in one sample taken
after the runs (not during them). The prereg names the laptop as BELIEVED-comparable to Cloud Run's 2 vCPU; this result shows
the local measurement does not say whether the 300 ms limit holds on Cloud Run. The gate as written is failed; a re-measurement on
the tagged candidate revision is the way to settle it, and that is a decision for GG, not applied here.

## Deviations from the prereg and harness choices

1. **vscode index swap.** There is no vscode index flag in the loader, so the harness wraps `loader._load_detector` to load the
   dated candidate for vscode at load time (the served vscode index is never loaded in that config, so the peak never holds both).
2. **`HF_HUB_OFFLINE=1`** added to the prescribed prod flags: the prod image runs offline (`ci.yml` comment); without it the embedder
   load tried the network and the blocked-socket guard stopped it.
3. **LLM stage.** `_groq_completion` raises `groq.APIConnectionError`, so all 20 calls took the production provider-outage degrade path
   (`degraded_provider_error` x20). Sockets blocked, 0 connection attempts counted. Memory of a real LLM response is not in the number.
   The 20 calls are the first 10 vscode and first 10 k8s rows of the re-frozen copy, with `created_at` = now (as `api/app.py`).
4. **Scoring path.** Both indexes are scored through the shipped `retrieve(...)` (one encode per query per index), not the earlier vectorised
   matmul. v2 hits 525 vs 526 before: one query differs; cause not diagnosed (BELIEVED floating-point near-tie ordering between the
   BLAS matmul and FAISS). The served index reproduces exactly (59).
5. **Served dated copy** was rebuilt in scratch from the served vectors minus the 2 undated rows (`add_dates_to_index.py` fails closed on them
   and is unchanged). Its shas in the JSON are of that scratch copy, not of `data/models`.
6. **Peak RSS** is Windows `PeakWorkingSetSize` (psutil is not installed). The harness has no FastAPI/uvicorn layer, so production RSS is
   higher by that overhead (not measured).
7. **Latency queries**: the first 110 sorted test-split sources (10 warm-up, 100 timed); retrieval is non-strict, as in serving.
8. **Scripts commit.** The numbers were produced at `c5fad948` on branch `feat/vscode-index-v2-prereg-results` (kept). For review the
   work was split into three PRs: the re-freeze and scoring scripts, the serving harness, and this doc with the report.
