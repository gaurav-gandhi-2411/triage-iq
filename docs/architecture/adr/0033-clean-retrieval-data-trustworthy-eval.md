# ADR-0033 — Clean Retrieval Data + Trustworthy Eval Foundation (Phase D1)

**Status:** Baselines SUPERSEDED by [ADR-0035](0035-retrieval-harness-correction.md) (2026-07-24)
— the R@5 numbers below (k8s 9.3%, vscode 43.5%) were measured with title-only queries; production
embeds title+body untruncated. Corrected, prod-matching canonical baselines: **k8s 18.0%
[12.0, 24.0], vscode 50.5% [43.0, 57.5]**. D1's actual contribution (clean, hand-verified,
issue-level-disjoint pairs; the eval-set/train-pool construction; the leakage guard) is unaffected
and still the foundation everything downstream relies on — only the query-construction bug in the
baseline measurement is corrected.
**Original status:** Accepted (data + eval foundation; no training, no cutover)
**Date:** 2026-07-19
**Decider:** Gaurav Gandhi (analysis + bounded hand-verification by CC, autonomous, per spec.md)

---

## Context

ADR-0032 found the product-task gold pairs used to measure retrieval quality were themselves
dominated by `title_sim` (title-text cosine similarity) — a channel that is only ~20% precision
on vscode (94.9% of its live-evaluated pairs) and retired vscode's headline number entirely.
`title_sim` also feeds `w3_t2_mine_negatives.py` (hard-negative mining), `w3_t3_split.py`
(train/val/test split), and the HELD W3 fine-tune's own training data, with no stratum filter
in the train-split loader.

GPU credits are available for a future retrieval fine-tune (D2), but training on the current
data would mean training on noise and evaluating on noise/leaked pairs — a fake improvement,
the worst possible outcome for a project whose value is honest measurement. D1 is the
prerequisite: build (1) a clean product-task TRAINING pool per repo, dropping `title_sim`,
(2) a hand-verified, leakage-safe, HELD-OUT eval set disjoint from training by construction,
(3) the honest current baseline on that clean eval, and (4) recommended eval params for D2.
D1 is CPU/analysis + bounded hand-verification — no training, no cutover.

Reproducible via `scripts/d1_channel_precision_audit.py`, `scripts/d1_eval_carve_sample.py`,
`scripts/d1_assemble_clean_pool.py`, `scripts/d1_build_eval_set.py`,
`scripts/d1_build_full_corpus_index.py`, `scripts/d1_baseline_eval.py`,
`scripts/d1_assemble_final_report.py` → `reports/retrieval_clean_data.json`. Every hand
judgment (283 pairs total, two rounds) is recorded with a one-sentence reason in
`reports/d1_pair_quality_review.json` and `reports/d1_eval_carve_review_{k8s,vscode}.json`.

## Method

### 1. Channel precision audit (clean training pool)

ADR-0032's existing 25-pair-per-repo hand review only sampled the "live product-eval" subset
(pairs both usable against the then-current live index) — it never touched two channels that
matter for a TRAINING pool: k8s's forward-scraped reference-mined pairs (issues #15003+,
outside the live index at ADR-0032 time) and vscode's `dup_comment` channel (excluded by
construction, since that sample only covered `stratum=="product"` and `dup_comment` is
`stratum=="gate"`). Drew fixed-seed (SEED=42) samples from both, plus full censuses of two
tiny channels, and dispatched hand-judging to two parallel agents (139 pairs, same
genuine/incidental methodology as ADR-0032). Combined with ADR-0032's review resliced by
`source` (previously only reported as one mixed number per repo) and ADR-0030's already-measured
vscode "Channel A" precision (30-43%, cited not re-derived).

**Escalated finding**: vscode's non-`title_sim` `body_related_ext` channel (`vscode_body_refs`,
206 pairs) is noisy — ADR-0030 already measured it at 30-43%, the same tier as `title_sim`
despite not being `title_sim`. Non-`title_sim` alone is not sufficient for "clean."

**GG decisions** (escalated, approved):
- k8s's `k8s_extended_mine` channel (200 pairs, 65.0% precision, n=20) — meaningfully below its
  sibling channels (~83-84%) — **dropped**. Pool shrinks from 736 to 536 pairs, precision rises
  from 79.3% to 84.6% (post-drop totals in the table below).
- vscode's two strata — **duplicate** (`dup_comment`, genuinely-duplicate issues) and
  **related** (narrower cross-reference channels) — are reported **separately, never blended**.
  Conflating them would hide that the product-valuable related-issue task remains effectively
  unmeasured on vscode — the same proxy-vs-product trap this project has caught 4 times.
- k8s has **no duplicate/comment channel at all** (0% comment-scrape coverage, ADR-0030) — its
  clean pool is the related task only, by construction, not by choice.

**Final clean training pool** (post-decision, `reports/d1_clean_pool_checkpoint1.json`):

| Repo | Task | Pool size | Weighted precision |
|---|---|---|---|
| k8s | related (only task available) | 536 | 84.6% |
| vscode | duplicate | 2,242 | 85.0% |
| vscode | related | 22 (eval-only, see below) | 86.4% |

vscode's clean pool (2,264 pairs combined) reverses the spec's working assumption that vscode
was "the hard case." `dup_comment` was simply never precision-audited before this phase.

### 2. Held-out eval set (the scoreboard)

The measured channel precisions (83-86%) sit below the spec's ≥90% hard rule for the eval set
specifically — a plain random draw would not clear it. The only rigorous fix: build the eval
set entirely from **individually hand-verified genuine pairs** (the same principle ADR-0032
used for its "clean subset" recall), which clears ≥90% trivially by construction. A second
hand-judging round (344 pairs, two parallel agents, fixed-seed samples excluding every
already-reviewed pair) topped up the bank to comfortably clear per-task targets.

| Eval set | Target | Available genuine | Frozen size | Status |
|---|---|---|---|---|
| k8s_related | 150 | 167 | **150** | gateable |
| vscode_duplicate | 200 | 224 | **200** | gateable |
| vscode_related | n/a (full pool) | 19 | **19** | directional-only, never gated |

**Disjointness** is enforced at the ISSUE level, not just the pair level (ADR-0018 discipline):
once an issue number appears on either side of an eval pair, every training pair touching that
issue number — in any capacity, in any channel — is dropped from the training pool. A pair-level
dedup alone would still let a fine-tune see issue X in training (paired with Y) and be evaluated
on X (paired with Z), leaking through X's pulled embedding. Asserted programmatically
(`scripts/d1_build_eval_set.py`). Resulting disjoint training pools: **k8s 264 pairs**,
**vscode_duplicate 1,734 pairs** (vscode_related has no training pool — held out entirely per
GG's decision).

**Bug caught before this landed**: the first training-pool builder matched clean channels by
`(channel, source)` only. k8s's `k8s_forward_scrape` channel has both `gate` (PR-query proxy,
explicitly excluded) and `product` rows under the same `(channel, source)` key — the bug
silently pulled the excluded proxy-task pairs back into the "clean" pool (2,762 pairs instead of
the correct 264). Fixed to match on `(channel, source, stratum)`; verified against the
checkpoint-1 pool sizes before proceeding.

### 3. Honest clean-eval baseline

Blocked initially: `torch` failed to import in the environment (`c10_cuda.dll` load error, a
CUDA-13.0-driver vs. cu124-build mismatch unrelated to this phase's changes). Fixed with GG's
approval via `pip install --force-reinstall torch==2.6.0+cu124`; verified working before
proceeding.

The eval sets are dominated by pairs mined from *outside* the currently-served index's coverage
(k8s's served index is a stale `#1-15,002`, built before the Phase 2b forward-scrape to
`#30,000`; vscode's served index covers only 7,028 of the corpus's issues) — only 16/150 k8s
eval pairs and 4/200 vscode-duplicate eval pairs fall within the served index at all. Rather
than report a severely underpowered number, built a **separate, D1-scoped index**
(`data/models/d1_full_corpus_index_{repo}_bge`) over the full currently-processed corpus
(k8s: 29,994 issues #1-30,000; vscode: 13,315 issues) using the same off-the-shelf, never
fine-tuned `BAAI/bge-base-en-v1.5` embedder already in production — pure inference, zero
gradient updates, the identical zero-leakage reasoning ADR-0030 established for the served
index. The served production artifact was never read for writing and stays untouched.

| Eval set | n | Recall@1 | Recall@5 | Recall@10 | 95% CI (R@5) | MRR |
|---|---|---|---|---|---|---|
| k8s_related | 150 | 3.3% | **9.3%** | 12.0% | [5.3, 14.0] | 0.059 |
| vscode_duplicate | 200 | 22.5% | **43.5%** | 47.5% | [37.0, 50.5] | 0.318 |
| vscode_related | 19 | 36.8% | **57.9%** | 63.2% | [36.8, 78.9] | 0.445 (directional) |

**k8s's honest R@5 is 9.3% [5.3, 14.0] — far below the previously-reported 23.5% (ADR-0030).**
Two factors both plausibly contribute, not disentangled at D1's power: (1) genuinely
harder/cleaner pairs — `title_sim` pairs had trivially high lexical overlap that inflated recall
on incidental matches, now excluded; (2) a ~2x larger candidate corpus (30,000 vs. the old live
index's 15,000) mechanically lowers recall by adding distractors, independent of pair quality.
Only 16 of the 150 eval pairs fall within the old 15k range — too few to cleanly separate the
two effects. Recommendation: treat 9.3% as the real, current, honest baseline — it reflects the
actual corpus scale the product should be benchmarked against, not a measurement artifact worth
re-litigating.

vscode's `duplicate` and `related` numbers are not comparable to each other or to k8s — three
different tasks, three different corpora, reported separately by design (per GG's checkpoint-1
decision).

### 4. Recommended eval params for D2

- **Primary gate metric**: Recall@5 (matches the product surface — top-5 similar issues shown
  to the triager).
- **Secondary**: Recall@1, Recall@10 for shape; MRR for rank-position signal beyond hit/miss.
- **CI method**: percentile bootstrap (2000 resamples, seed=42) for single-arm numbers (as
  above); **true paired bootstrap** (same resample indices for both arms,
  `scripts/_retrieval_eval_common.py::paired_bootstrap_ci`, ADR-0027's corrected method) for any
  D2 trained-vs-baseline delta — ships only if the paired CI on the improvement excludes zero,
  this project's established bar.
- **Gateable tasks**: `k8s_related`, `vscode_duplicate`. **Directional-only**: `vscode_related`
  (n=19, underpowered by construction, never gated, never blended with `vscode_duplicate`).
- **Corpus consistency**: D2 must evaluate against the CURRENT full-corpus index (this report's
  `d1_full_corpus_index_*` or its successor), never a stale subset — corpus size directly
  affects recall, so baseline and trained-model numbers must share the same candidate pool.

## Decision

D1 is complete: clean training pools (k8s 536 related-only; vscode 2,242 duplicate + 22
related-eval-only), frozen disjoint held-out eval sets (150 / 200 / 19), the honest clean-eval
baseline (9.3% / 43.5% / 57.9% R@5, per task), and recommended eval params are all in place and
committed. **vscode verdict: trainable + evaluable on the duplicate task (not deferred, not
k8s-only) — a reversal of this phase's working assumption.** k8s is trainable + evaluable on the
related task only (no duplicate channel exists). No training, no GPU beyond embedding inference,
no index cutover — the served production index and model are untouched.

## Consequences

- **What changes**: D2 (GPU training) now has a clean, disjoint, honestly-precision-stated
  foundation to build on. The prior 23.5% k8s baseline is superseded — the honest number is
  materially lower (9.3%), which raises the bar for what D2 needs to prove, not lowers it.
- **What becomes easier**: D2 can start immediately without its own data-cleaning detour; the
  disjointness guarantee (issue-level, asserted) means D2's trained-model eval numbers are valid
  by construction, not something D2 has to re-derive.
- **What becomes harder**: k8s's usable related-task training pool (264 pairs after the eval
  carve) is thin for a fine-tune — D2 should treat this as a real constraint, not force volume
  by reintroducing lower-precision channels.
- The corpus-size confound (9.3% vs. the old 23.5%) is disclosed, not resolved — a future
  session could re-run the old 277-pair k8s set against the new full-corpus index to isolate
  the corpus-size effect specifically, if that distinction becomes decision-relevant.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Report eval-set precision as the channel-level sampled estimate (83-86%) | Fails the spec's ≥90% hard rule; the eval set is the scoreboard and must be clean, not merely representative. |
| Restrict the eval set to pairs within the (stale) served index | Would leave k8s at 16 pairs and vscode-duplicate at 4 — nowhere near powered; repeats the exact "unmeasurable" trap ADR-0030 already solved once by re-checking a leakage assumption. |
| Keep the k8s `k8s_extended_mine` channel (65% precision) in the training pool | GG's explicit call: meaningfully below sibling channels, traded 200 pairs of size for a precision gain (79.3%→84.6%). |
| Blend vscode's duplicate and related strata into one "vscode retrieval" number | Would hide that the related task stays effectively unmeasured (n=22) behind a duplicate-dominated headline — the proxy-vs-product conflation this project has caught 4 times. |
| Reinstall torch without asking, or skip task 7 and report no baseline | Environment mutation to a shared conda env warrants confirmation (CLAUDE.md); skipping would leave D2 without the one number this whole phase exists to produce. |
