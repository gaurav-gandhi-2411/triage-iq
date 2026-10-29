# Q5a part 2: vscode duplicate retrieval, BM25 variants, dense BGE and RRF (pre-registration)

Written and committed 2026-10-29 BEFORE any retrieval number of this part is computed. Extends the
design in `scripts/qp2_vscode_dup_eval.py` (PR #309); everything not restated here (task, query /
index construction, time-respecting prefix, tie rule, train-only sparse statistics, primary metric,
Wilson / bootstrap CIs, ceiling) is inherited unchanged. Target (ADR-0066): vscode duplicate
Recall@5 >= 80% as a test POINT estimate. Validation baseline (#309, `reports/qp2_vscode_dup_validation_2026-10.json`):
B2 BM25 R@5 29.2%, B1 TF-IDF 27.3%, ceiling 92.8% (1,073 of 1,156 queries have an indexed canonical).

## Input and recipe parity (measured before this document)

Dense vectors: Kaggle BGE of all 109,537 vscode issues (`D:\triageiq-data\kaggle_out\vscode`,
BAAI/bge-base-en-v1.5 rev a5beb1e3, cls+l2, max_seq_length 512, no document instruction, float16
storage), the production text recipe (#288). Parity check (D86 lesson): 200 issues (seed 42 sample
of row positions) re-encoded on CPU with the production path (`similar_issues._build_text` with the
model tokenizer + `SentenceTransformer.encode`, sentence-transformers 2.7.0, torch 2.15.0+cpu,
`CUDA_VISIBLE_DEVICES=-1`) vs the Kaggle vectors. Pre-set bar: min cosine >= 0.999.
Result (VERIFIED, own run): min 0.99995, mean 0.9999994, 0 of 200 below 0.999. PASS; vectors usable.

## Candidates (all on VALIDATION; fixed grid, no additions after seeing numbers)

Text variants for the sparse (BM25) side, BM25 parameters fixed at the qp2 values (k1 1.2, b 0.75),
documents and queries built identically, vocabulary / IDF / avgdl from train issues only:

* Title boost `w` in {1, 2, 3}: the title followed by ". " is repeated `w` times before the
  production text (w = 1 is exactly baseline B2).
* Identifier extraction `ids` in {off, on}: from the RAW title + body (code blocks included, first
  20,000 chars), every code-like token (contains `_`, a lower-to-upper case transition, or is a
  dotted path with >= 2 parts) is appended as extra terms: the whole token (dots mapped to `_`)
  and its sub-words (split on `_`, `.`, case transitions; length >= 2).
* Error-message extraction `errs` in {off, on}: from the same raw text, up to the first 5 lines
  (each <= 300 chars) matching a fixed error regex (error, exception, failed, failure, cannot,
  can't, unable, unhandled, fatal, traceback, ENOENT, EACCES; case-insensitive) are appended once
  more (a second copy, i.e. a term-frequency boost).

That is 3 x 2 x 2 = 12 BM25 variants (V1 ... V12, includes B2 as w=1/off/off). Plus:

* B1 TF-IDF (as #309), kept in the table only.
* D1: dense cosine of the Kaggle BGE document vectors; query = the later issue's own vector
  (document-to-document, no query instruction; the production vscode setting is
  `QUERY_INSTRUCTION_REPO_OVERRIDE = False`).
* H(k): reciprocal rank fusion of the SELECTED BM25 variant (best validation R@5 among V1..V12,
  rule below) and D1, `score = 1/(k + rank_bm25) + 1/(k + rank_dense)` over the full time-respecting
  index (ranks 1-based, full lists, no depth cut), k in {10, 30, 60}. Ties in fused score go to
  the more recent issue.

## Selection rule (validation only)

Primary metric: Recall@5 over ALL validation queries (unindexed = miss), Wilson 95% and
bootstrap 95% (2000 resamples, seed 42). Winner = highest validation R@5 among {B1, V1..V12, D1,
H(10), H(30), H(60)}; ties -> higher validation MRR -> the simpler candidate, simplicity order
B1 < V (by w - 1 + ids + errs, then w) < D1 < H (smaller k first). The best BM25 variant fed to H
is chosen by the same rule among V1..V12 before fusion. Differences between candidates are NOT
tested for significance on validation; selection is a point rule, the validation CIs are reported.

## Freeze and the single test evaluation

After the grid, the winner's full config (candidate id, w, ids, errs, k, BM25 constants, split
names, embedding file sha256, script sha256) is written to
`reports/q5a_vscode_dup_dense_frozen_2026-10.json` with its own SHA-256 over the canonical JSON
(sorted keys) and committed BEFORE the test split is read for scoring. The test run (guarded by
`--final-test` and `--frozen-sha256`, which must equal the committed hash) evaluates exactly two
candidates on the SAME test queries and index: the frozen winner and the baseline B2 (#309). It
is run ONCE; no test number is looked at before the freeze commit. Reported: Recall@1/5/10/50
and MRR with Wilson / bootstrap CIs, indexed-only variants, pair-level variants, the ceiling, and
the paired bootstrap difference (winner - B2) for every metric. Success per ADR-0066: winner test
R@5 point estimate >= 0.80. If the winner is B2 itself the run reports that and the paired
difference is zero by construction. Whatever the outcome it is reported as is; no second test run
and no re-selection.

## Scope

Experiment only: no serving change, no index build, no cutover (GG's). Deterministic (seed 42,
CPU, `CUDA_VISIBLE_DEVICES=-1`, <= 4 threads).
