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

**Read:** 24.67% is closer to the real ceiling than it looks — confirmed below by actually
building the clean subset and re-scoring it, not left as a guess.

## A2 — Follow-up: full-150 clean eval set, hand-verified blind to retrieval outcome

The 30-pair sample in Part A suggested most misses aren't fixable by a better retriever, but a
sample of misses can't answer "what does the retriever score on a fair population" — and
excluding pairs *because* they were misses would be circular (an easy pair a retriever also
missed by bad luck would get excluded for the wrong reason; a hard pair it happened to hit would
survive for the wrong reason). Fixed by pre-registering exclusion criteria on the pair's own
content and applying them **blind to hit/miss** to all 150 pairs, then only afterward joining
the labels to the already-computed retrieval results.

**Pre-registered criteria (defined before any pair was scored against retrieval outcomes):**

- `VALID` — query and target share genuine, substantive topical content overlap: the same
  specific bug, feature, or proposal. Default label; the bar to exclude is "real, articulable
  reason," not "any tangential imperfection."
- `EXCLUDE_UMBRELLA` — the query is a checklist/tracking issue enumerating multiple distinct
  sub-issues, such that no single embedding of it could fairly be expected to be close to any
  ONE referenced item.
- `EXCLUDE_CAUSAL_ONLY` — the target is cited only as background/precedent/motivating-example
  for a topic the query is actually, substantively about something else.
- `EXCLUDE_OTHER` — any other reason the pair isn't a fair content-similarity test (near-empty
  query/target body, etc.), with the specific reason stated.

**Execution** (`scripts/track2_k8s_clean_eval_build.py` inputs preserved in
`reports/track2_k8s_clean_eval.json`): extracted all 150 pairs' query+target title/body (target
body joined from the corpus parquet) into a form with **no hit/miss or retrieval-score field at
all**. Dispatched to parallel reviewers in 5 batches of 30, matching this project's own D1
precedent (ADR-0033 used two parallel agents for the same kind of hand-verification at a
comparable scale) — each batch reviewed once against the written rubric above, blind to
retrieval outcome and to the other batches. All 150 pair_ids returned exactly once; spot-checked
for internal consistency (e.g. the same umbrella query paired with 4 different targets — #21699,
"v1.0 upgrades to v1.2: outstanding issues" — was independently labeled `EXCLUDE_UMBRELLA` all 4
times it appeared, without the reviewer being told these were the same source issue).

**Results:**

| Label | n | Hit rate |
|---|---|---|
| VALID (clean subset) | 66 | 39.4% (26/66) |
| EXCLUDE_UMBRELLA | 27 | 14.8% |
| EXCLUDE_CAUSAL_ONLY | 48 | 8.3% |
| EXCLUDE_OTHER | 9 | 33.3% |
| **All 150 (unfiltered, for reference)** | 150 | **24.67% (37/150)** — exact match to ADR-0040 |

**Clean-subset R@5 = 39.39% [27.27%, 51.52%] (95% CI, percentile bootstrap, 2000 resamples,
seed=42 — same method as `_retrieval_eval_common.py`/D1).** The unfiltered 24.67% point estimate
sits just below the clean subset's CI lower bound — the clean number is distinguishably higher,
not just directionally higher, though n=66 keeps the CI fairly wide (±12pp half-width) and this
should be read as informative, not fully powered in the way the 150-pair gate set is.

**Addressing the selection-bias risk directly, as instructed:** excluded pairs do have a lower
hit rate (13.1%) than valid pairs (39.4%) — but this is the expected, correct shape for a
principled filter, not evidence of cherry-picking. The exclusion labels were assigned with zero
visibility into whether the retriever hit or missed each pair; the gap exists because the
exclusion criteria target genuine properties (diluted/multi-topic content, citation-only
references) that independently make a pair both *invalid as a content-similarity test* and
*harder for a content-similarity retriever to win* — the same underlying reason, not two
coincidentally-correlated ones. If exclusion had been outcome-driven, the excluded-pair hit rate
would trivially be ~0% by construction; 13.1% (not 0%) is consistent with a validity-based filter
that happens to correlate with difficulty, not a difficulty-based filter dressed up as validity.

**Disclosed limitation**: each pair was reviewed once (5 non-overlapping batches), not
D1's full double-review-with-reconciliation. The rubric here is more mechanical/checkable
(checklist-structure, background-citation-vs-core-topic) than D1's harder "is this genuinely the
same issue at all" judgment, so single-review is a reasonable trade for this pass, but a second
independent pass reconciling disagreements would be the next rigor increment if this number
becomes decision-load-bearing (e.g. gating a future fine-tune attempt).

**Read: the "weak retriever" framing was partly an eval artifact.** On pairs that are actually a
fair test of content-based retrieval, the current off-the-shelf BGE embedder scores ~39%, not
25% — a real, substantial difference in how the product's retrieval quality should be described,
even though nothing about the retriever itself changed. The remaining ~60% miss rate on the
clean subset is still real headroom (this is not "retrieval is secretly great"), but the honest
starting point for judging any future retrieval work is 39%, not 25%.

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

- **k8s retrieval: the "weak retriever" framing was substantially an eval artifact.** Building
  and hand-verifying a clean eval subset (66 of 150 pairs survive a pre-registered, outcome-blind
  validity filter) puts the current, unmodified off-the-shelf BGE embedder at **R@5 = 39.4%
  [27.3%, 51.5%]**, not 24.7%. This changes the shape of the problem: the priority is a permanent
  clean(er) eval set (and, separately, a precision-filtered mining method for any future training
  data — the umbrella-issue/causal-reference noise found here is the same noise the product-
  stratum training pairs are mined with) over another embedder/fine-tune attempt aimed at closing
  a gap that was partly measurement, not model quality. The remaining ~60% miss rate on the clean
  subset is still real headroom, not "problem solved" — but 39%, not 25%, is the honest baseline
  any future retrieval work should be measured against.
- **k8s training data**: volume is no longer the blocker; a next fine-tune attempt (if ever
  pursued) should prioritize a bigger, similarly-cleaned product-task TEST set and the same
  precision filter used here for mining, over just mining more pairs.
- **vscode resolution**: the duplicate-wave hypothesis is tested and rejected as the explanation
  for the naive loss. Current naive-fallback serving stays correct. One narrower follow-up
  (semantic near-dup clustering instead of title-keyword filtering) would fully close this out,
  but the highest-value, cheapest hypothesis has already been checked and didn't pan out.
- **CORS bug (#78)**: fix proposed and tested (PR #80, draft) — not merged/deployed, escalated
  per standing instruction on security-surface changes.
