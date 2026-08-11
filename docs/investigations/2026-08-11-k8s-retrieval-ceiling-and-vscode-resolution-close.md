# Investigation: k8s retrieval R@5 ceiling + closing out vscode resolution's naive-loss question

**Date:** 2026-08-11
**Status:** Investigation only — no model/prod change. Reports before proposing any build, per
standing instruction.

## Context

k8s R@5 is 24.67% (ADR-0040, currently deployed — verified below, not assumed). The project's
own record going into this session: 8 modeling attempts (W3/D2 fine-tunes, hybrid BM25, cross-
encoder reranker, stronger embedder, DistilBERT, DeBERTa x2) all failed or came back negative;
5 measurement/implementation fixes (leakage removal, eval-harness correction, pooling-mismatch
fix, truncation fix, query-instruction fix) all won. This investigation follows that pattern:
measure the actual failure mode before proposing anything to build.

## A — What's the ceiling? Hand-examined 30 of the 113 k8s R@5 misses

Reproduced the eval against the **production-served** index
(`data/models/dup_index_kubernetes_kubernetes_bge` — confirmed byte-identical to the ADR-0040
verified candidate via `sha256sum` on both `index.faiss` and `meta.pkl`) with the real,
un-overridden `retrieve()` code path. Exact reproduction: **37/150 hits, R@5 = 24.67%**,
matching ADR-0040's number precisely (`scripts/track2_k8s_miss_analysis.py`,
`reports/track2_k8s_miss_analysis.json`).

Sampled 30 of the 113 misses (fixed seed=42) and hand-categorized each by reading the query,
the labeled target, and the actual top-5 retrieved neighbors:

| Category | Count | % |
|---|---|---|
| (i) Genuinely findable — real lexical/semantic overlap a better retriever should catch | 10 | 33% |
| (ii) Needs context a text embedding can't recover (causal/provenance reference, near-empty query body) | 8 | 27% |
| (iii) Arguably mislabeled or structurally unwinnable for a single-vector retriever | 12 | 40% |

**67% of sampled misses are not fixable by a better retriever.** The dominant pattern in (iii) —
present in 5 of the 12 — is **umbrella/tracking issues**: an issue that lists many other issues
as a checklist or "see also" summary (e.g. #21699 "v1.0 upgrades to v1.2: outstanding issues",
#21931 "Detailed Design for Volume Mount/Unmount Redesign", #25716 "Umbrella issue for swagger
related improvements"). Each such issue contributes **multiple pairs** to the eval set (one per
referenced sub-issue) — #21699 and #21931 each appear twice in this 30-pair sample alone — and
no single embedding of the umbrella issue's full body can be simultaneously close to N distinct,
only-loosely-related sub-issues. This is a measurement-population artifact, not a retrieval
quality gap, and it's structurally guaranteed to keep costing R@5 points regardless of embedder
quality.

The rest of (iii): cases where the retriever's actual top-5 looks **more topically relevant**
than the labeled target (e.g. #23742: query about kubelet never terminating a pod when the image
registry is unavailable retrieved "A pod never terminated if a container image registry was
unavailable" — nearly a restatement — while the labeled target is an unrelated e2e-flake ticket),
and cases with several equally-valid near-duplicates where recall@5 penalizes picking the "wrong"
one among many correct ones (e.g. #17468, tight client-gen/go2idl cluster; #21809, a swarm of
near-identically-worded e2e-flake tickets).

Category (i) includes several **near-misses**, not total failures — ranks 7, 8, 14, 15 (just
outside the k=5 cutoff), where the topic is clearly right and a moderately better retriever could
plausibly close the gap (e.g. #21937→#20613, kubernetes-test-go timeout flake, target at rank 8
with all 5 retrieved results also being test-timeout flakes).

**Read:** 24.67% is closer to the real ceiling than it looks. Naively "fixing" this eval
population (the (iii) cases) might mechanically raise the measured number without reflecting any
retrieval improvement at all — an eval-cleanup opportunity in its own right, but a different
piece of work than model improvement, and worth being honest that it would inflate the metric
without inflating the product.

## B — Is the corpus complete? Clean negative.

Of the 150 eval pairs, **0 have the target OR the query issue missing from the live index**
(both are always present; k8s's index carries 29,994 issues). This rules out a data-coverage bug
for the CURRENT k8s R@5 measurement — the earlier "k8s retriever unmeasurable, zero product-task
pairs survive against the live index" state (ADR-0028, 2026-07-13) was resolved by the Phase 2b
corpus growth and forward-scrape work already completed. Re-checked, not assumed.

## C — What does the query actually see? Confirmed fixed, deployed, and symmetric.

Verified directly against current code (`src/triage_iq/models/similar_issues.py::_build_text()`),
not from memory: the corpus side truncates by **token count** via the model's own tokenizer, not
a fixed 512-char cut — the char-based path is now an explicit legacy/no-tokenizer fallback only.
Query-side (`triage.py::_collect_signals`) builds `f"{title}. {body}"` fully untruncated, matching
`_retrieval_eval_common.py`'s eval-side construction exactly.

This is ADR-0040 (2026-08-06), already measured, verified (200/200 byte-identical index/query
text reconstruction, both repos, zero mismatches — `reports/lever12_candidate_verification.json`),
and **confirmed deployed**: `git log` shows `ad96700 chore(models): re-publish dup_index_*_bge`
merged to `main`, and the locally re-derived hash of the currently-served
`dup_index_kubernetes_kubernetes_bge/{index.faiss,meta.pkl}` is byte-identical to the verified
`_candidate/` directory. No remaining asymmetry — this lever is fully banked, not a live gap.

## D — Training-data scale: volume is solved, precision at the product stratum is the open question

`data/gold_related_v2.parquet` (the canonical corpus, post Phase 2b) currently holds:

| Repo | Total pairs | Product stratum (issue→issue) | Gate stratum |
|---|---|---|---|
| k8s | 4,030 | 776 | 3,132 |
| vscode | 2,849 | (not re-checked this session) | — |

This clears the 80%-power target identified in July (k8s ~1,840 total) by more than 2x, and the
product-stratum population specifically grew 10x over the original 78 pairs (ADR-0028's
"unmeasurable" finding). **Raw training volume is no longer the blocker it was when W3's original
fine-tune (ADR-0016) was rejected.** D2's retry on this same corpus, bug-fixed (ADR-0035), still
came back a null result (+2.0pp, CI[-4.5,+8.5]) — but that CI width comes from an underpowered
**product-task test set** (~150-200 pairs), not the training set.

The new information this session adds: Part A's categorization applies to this exact mining
channel (`k8s_forward_scrape`/`body_related_ext` — the "#N cross-reference in body" method that
built the product stratum). If 40% of a sample mined this way turns out arguably mislabeled and
another 27% needs non-textual context, a next fine-tune attempt trained on more of the same
channel would likely inherit that noise rate, not just add clean volume. **The honest
recommendation is not "mine more" — it's that any next attempt should (a) grow the held-out
product-task TEST population specifically (the actual CI-width bottleneck) and (b) add a
precision filter to the mining method itself (e.g., drop umbrella-issue sources — issues with
>3-5 outbound "#N" references — and near-duplicate-cluster sources) before training on more
pairs from this channel.** Not attempted here — reporting the shape of the gap only, per
standing instruction not to start modeling work without escalating first.

## vscode resolution: duplicate-wave/bot-filtering hypothesis tested — does NOT explain the naive loss

ADR-0041 (2026-08-06) disclosed but didn't act on: the vscode lever3 test window (Feb
2025-Apr 2026) is 83.8% "hours"-bucket, and a cluster of near-duplicate "Terminal not working"
reports (#240070-240145) plus `vs-code-engineering[bot]` activity were visible in it. This
session did the follow-up ADR-0041 flagged as open (`scripts/track2_vscode_resolution_filter_check.py`,
`reports/track2_vscode_resolution_filter_check.json`):

Filtered the test set (1,225 rows) by bot authorship (`vs-code-engineering[bot]`, 63 rows,
5.1%) plus a disclosed rule-based duplicate-wave filter (title contains "terminal" + a generic-
complaint phrase, resolution < 24h; 13 rows, 1.1%) — **76 rows removed, 6.2% of the test set.**

| | Unfiltered (n=1225) | Filtered (n=1149) |
|---|---|---|
| "hours"-bucket share | 83.76% | 83.20% |
| Naive bucket accuracy | 83.76% | 83.20% |
| Trained model accuracy | 82.69% | 82.07% |
| Model − naive delta | **-1.06pp CI[-1.63,-0.57]** | **-1.13pp CI[-1.74,-0.61]** |

**Filtering the exact cluster ADR-0041 flagged changes essentially nothing** — the "hours"-bucket
share barely moves (83.76%→83.20%) and the model-vs-naive gap is unchanged within noise (both
CIs firmly exclude zero, both still losses). This is a concrete answer, not a directional one:
the duplicate-report-wave hypothesis, tested directly rather than left as an open finding, does
**not** explain vscode's naive-beats-model result. The near-instant-closure skew is a pervasive,
not a localized, property of vscode's current issue-resolution behavior.

This doesn't fully close the "eval methodology vs. genuine ceiling" question — a broader
duplicate-detection pass (semantic near-dup clustering, not the title-keyword rule used here)
or a wider/differently-sampled test window are still untested — but the specific, most concrete
hypothesis on record has now been measured and rejected. `BUCKET_CLASSIFIER_TRUSTED["microsoft_vscode"]
= False` continues to be the right call on the evidence available, with one fewer plausible
"actually it's a data artifact" explanation left to check before concluding this is a genuine
unpredictability ceiling for vscode with current features.

## Bottom line

Nothing here proposes a build. Summary for the next decision:

- **k8s retrieval**: ~24.7% may be close to the practical ceiling for a single-vector retriever
  against this eval population; the clearest real lever left is cleaning the eval/mining
  population's umbrella-issue and near-duplicate-cluster noise (a measurement fix, matching this
  project's 5-for-5 record), not another embedder/fine-tune attempt in isolation.
- **k8s training data**: volume is no longer the blocker; a next fine-tune attempt (if ever
  pursued) should prioritize a bigger product-task TEST set and a precision-filtered mining
  method over just mining more pairs.
- **vscode resolution**: the duplicate-wave hypothesis is tested and rejected as the explanation
  for the naive loss. Current naive-fallback serving stays correct. One narrower follow-up
  (semantic near-dup clustering instead of title-keyword filtering) would fully close this out,
  but the highest-value, cheapest hypothesis has already been checked and didn't pan out.
