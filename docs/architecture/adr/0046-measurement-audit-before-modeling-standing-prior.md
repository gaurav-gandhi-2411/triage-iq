# ADR-0046 — Standing prior: audit the metric before touching the model

**Status:** Accepted
**Date:** 2026-08-11
**Decider:** Gaurav Gandhi

## Correction (2026-08-11, same day): "eight attempts, zero wins" was itself imprecise

The Context section below (as originally written) treated all eight modeling attempts as
uniformly valid, uniformly negative measurements. On audit — applying this ADR's own standing
prior to itself — that framing doesn't hold. Of the roughly thirteen distinct measurement *runs*
underlying those eight attempts, **seven ran against a broken harness and taught nothing** (the
same class of defect item 3, 4, and 6 above document for eval sets); the remaining runs are real,
trustworthy negatives. Conflating the two understates how much genuine headroom this project has
left unexamined, and — worse — risks citing an invalid run as if it were evidence.

**Per-attempt audit (generation-by-generation, not attempt-by-attempt, because three of the eight
were re-measured multiple times with different validity each time):**

| # | Attempt | Generation | Verdict | Category |
|---|---|---|---|---|
| 1 | W3 bi-encoder fine-tune | ADR-0016, honest re-run (2026-07-04) | +11.84pp k8s / +10.00pp vsc, neither CI excludes zero | **(b) REAL LIMIT** — self-audited (GPU-state-leak and eval-contamination bugs caught and fixed *within the ADR, before trusting the result*); underpowered at 1,435 total gold pairs, not a broken measurement |
| 2a | D2 fine-tune | ADR-0034, 5ep run + 2ep diagnostic leg | −5.0pp then −10.0pp R@5 (CI excludes zero on the diagnostic leg) | **(a) INVALID** — MEAN pooling trained against BGE's native CLS pooling, plus 65.73% of training examples silently truncated at 128 tokens. WITHDRAWN by ADR-0035; the "overfitting rejected" conclusion had no support since the diagnostic never touched either confound |
| 2b | D2 fine-tune | ADR-0035, corrected (CLS pooling, seq_len 256) | +2.0pp R@5, CI[−4.5, +8.5] | **(c) VALID, inconclusive** — "underpowered, not disproven" per the ADR's own language, n=200 |
| 3a | Hybrid BM25+dense | ADR-0031 Lever 1 (2026-07-12) | rejected both repos | **(a) INVALID** — title-only queries (query_body never populated) + unaudited pair population, 72%/20% genuine per ADR-0032 |
| 3b | Hybrid BM25+dense | ADR-0035 retest (2026-07-24) | rejected, vscode flipped to significant regression | **(a) STILL INVALID** — fixed queries + D1's clean population, but confirmed (2026-08-10 investigation) to have run against the stale, char-truncated (not token-truncated) corpus index |
| 3c | Hybrid BM25+dense | 2026-08-10 investigation, token-truncated live-matching index | k8s directionally negative (CI crosses zero, n=150); vscode significantly negative (CI excludes zero on the harmful side, −11.0pp RRF / −6.0pp weighted) | **(c) VALID NEGATIVE** — decisive on vscode, underpowered-negative on k8s |
| 4a | Cross-encoder reranker | ADR-0006, W1.3 + Phase 2 T2 (n=300, 2026-05-30) | CI crosses zero [−0.037,+0.053] | **(c) VALID** — measured against the PR→issue proxy task, but ADR-0008 explicitly certified this doesn't invalidate the screening data ("all R@5 numbers above are valid measurements") |
| 4b | Cross-encoder reranker | ADR-0031 Lever 2 (2026-07-12) | rejected both repos | **(a) INVALID** — same title-only-query + unaudited-population bug as 3a |
| 4c | Cross-encoder reranker | ADR-0035 retest (2026-07-24) | rejected, −2.67pp k8s / −3.0pp vscode | **(a) STILL PARTIALLY INVALID** — corrected queries + D1 population, but **never** re-verified against the token-truncated corpus the way hybrid BM25 was in 3c. Genuinely untested at full correctness — though moot for shipping either way: 190–330× CPU latency regression disqualifies independent of quality |
| 5a | Stronger embedder (bge-large) | ADR-0031 Lever 3 (2026-07-12) | rejected both repos | **(a) INVALID** — same title-only-query + unaudited-population bug |
| 5b | Stronger embedder (bge-large) | ADR-0035 retest (2026-07-24) | rejected, CI crosses zero both repos | **(a) STILL INVALID** — same stale-corpus gap as 4c, never re-verified against the token-truncated corpus. **This lever has never once been correctly measured** — zero valid generations exist |
| 6 | DistilBERT (component classifier) | Day-4 original + later top-3 re-eval | loses TF-IDF+LR on accuracy/macro-F1 and on top-3 | **(b) REAL LIMIT, valid but stale-baseline** — the top-3 re-eval compared against the *pre*-ADR-0036 single-label-collapsed baseline (now superseded); a multi-label-supervision variant of DistilBERT itself was never tried, but DeBERTa's own multi-label arm (7b below) already lost by 21pp under the *current* baseline with that exact fix applied, making this a low-expected-value gap, not a live one |
| 7a | DeBERTa-v3-base Phase B | ARM 1, single-label softmax | −20.3pp k8s / −13.9pp vscode vs current baseline | **(b) REAL LIMIT** — measured against the current, correct multi-label baseline (87.06%/89.84% top-3, exact match to ADR-0036) |
| 7b | DeBERTa-v3-base Phase B | ARM 2, multi-label BCE, diagnosed + pos_weight-corrected | −21.0pp k8s vs current baseline after fix recovered +9.8pp; tail-class recall 0.000 in every config | **(b) REAL LIMIT, mechanism confirmed from two directions** — own supervision bug found and fixed within the same experiment, still lost by a wide margin; and a *bigger* model (184M) underperforms the *smaller* DistilBERT (66M) on the same data, the classic small-data capacity-mismatch signature |

**Corrected tally: of 8 attempts / 13 measurement generations, 7 generations were invalid and
taught nothing (D2's first run, both hybrid-BM25 pre-2026-08-10 generations, both reranker
post-ADR-0006 generations, and both stronger-embedder generations). The stronger-embedder lever
(5) has literally never been correctly measured — every generation of it ran against a broken
harness.** The genuinely trustworthy negative-result count is 7 (W3, D2-corrected, hybrid-BM25-
final, reranker-original-ADR-0006, DistilBERT, DeBERTa ARM1, DeBERTa ARM2), not 8, and none of
those 7 is a blanket "modeling doesn't work here" finding — each has a stated mechanism (data
volume for W3/D2, latency+corpus-mismatch for reranker, data-scale ceiling for the classifiers).

**What it would cost to re-run the still-open invalid generations, now that the eval is fixed:**

- **Stronger embedder (5b) — cheapest, highest-priority gap.** No training: re-embed the corpus
  with `bge-large-en-v1.5` via the current `_build_text()` (token-truncated) and re-run
  `scripts/lever3_stronger_embedder.py` against the live-serving index construction. ~2–5 minutes
  per repo. This is a free, never-actually-run experiment sitting in the backlog — worth doing
  opportunistically, though not gating the mining-precision work below.
- **Reranker (4c)** — same fix pattern (repoint at the token-truncated index), but the latency
  disqualification (190–330×) means a clean rerun only sharpens the quality diagnosis, not the
  ship decision. Low priority.
- **D2/W3 fine-tune, properly powered** — this is not a quick script fix; it needs a bigger, high-
  precision clean training pool (D2's 1,734 pairs, and especially k8s's 264, are the actual
  constraint) and a bigger held-out product-task test set. This is exactly the retrain scoped
  later in this session (mining-precision work, then a fine-tune on the resulting clean pool with
  the already-correct CLS-pooling/256-seq-len config from 2b) — the D2 retry D2 never actually
  got, not a repeat of a prior run.

This correction doesn't change ADR-0046's Decision (below) — if anything it strengthens the
prior, since even the "corrected" retests turned out to still be running on stale corpora in two
of three cases. It changes what the *record* should say happened: not "eight tries, eight honest
failures," but "eight attempts, seven contaminated-or-partial measurement generations, and seven
trustworthy negatives with stated mechanisms" — audit the metric before touching the model,
including when auditing your own prior modeling attempts.

## Context

This project's history now has six documented instances where a metric that looked like a
model-quality problem turned out to be a measurement or eval-validity defect:

1. **Proxy-task framing** (ADR-0008) — the retrieval task was originally scored as "duplicate
   detection" when the product surface is "similar-issue retrieval." Reframing (not retraining)
   corrected every downstream metric.
2. **`title_sim` channel noise** (ADR-0032/ADR-0033) — a large share of the retrieval gold pairs
   were mined by title-text cosine similarity, a channel later measured at ~20-30% precision on
   vscode. Both the training pool and the eval set were rebuilt clean; nothing about the model
   changed.
3. **Title-only eval queries** (ADR-0035) — the retrieval eval harness embedded only issue titles
   while production embeds title+body untruncated. Fixing the query construction moved k8s's
   canonical baseline from 9.3% to 18.0% R@5 with zero model changes.
4. **Character-based corpus truncation** (ADR-0040) — the corpus side of the retrieval index
   truncated at 512 characters while the embedder's real limit is 512 tokens, silently dropping
   real content on 17.8-43.2% of issues. Switching to token-based truncation (still the same
   off-the-shelf embedder) moved k8s to 24.67% R@5.
5. **Stale temporal split** (ADR-0041) — the shipped resolution models were trained on a split
   generated before a 99-100% corpus growth, so the bulk of newly-available training rows were
   simply invisible to them. Re-splitting from the current corpus (still the same LightGBM
   config) roughly doubled k8s's bucket-classifier advantage over naive (+3.27pp → +6.35pp).
6. **Structurally-invalid eval pairs** (2026-08-11, this session,
   [investigation](../../investigations/2026-08-11-k8s-retrieval-ceiling-and-vscode-resolution-close.md)) —
   hand-categorizing k8s R@5 misses found ~56% of the 150-pair eval population isn't a fair test
   of content-based retrieval at all: checklist/umbrella issues that reference many unrelated
   sub-issues, and citations where the referenced issue is background/precedent for a different
   topic than the query is actually about. A clean, pre-registered, outcome-blind subset (still
   the same off-the-shelf BGE embedder, zero model changes) scores 39.4% [27.3, 51.5], not 24.7%.

Over the same period, **eight distinct modeling attempts to improve retrieval or classification
quality produced zero shipped wins**: the original W3 bi-encoder fine-tune (ADR-0016, rejected —
GPU-state-leak-inflated result, honest re-run didn't clear the bar), the D2 retry fine-tune
(ADR-0034/0035, corrected harness, still a null result — CI crossing zero), hybrid BM25+dense
fusion (ADR-0031's Lever 1, rejected on both repos; re-tested against the corrected corpus
2026-08-10, rejected again), a pretrained cross-encoder reranker (ADR-0006, rejected at n=300;
re-tested, rejected again on quality+latency), a stronger off-the-shelf embedder (ADR-0031's
Lever 3, rejected), DistilBERT for component classification (loses to TF-IDF+LR on the correct
top-3 metric), and DeBERTa-v3-base Phase B, two arms (2026-08-10, negative result — TF-IDF+LR
remains champion).

Every meaningful "the model isn't good enough" signal on this project has, on inspection, been
partly or wholly a "the way we're measuring it is broken" signal. Every attempt to fix the
measurement instead of the model has paid off. Every attempt to fix the model without first
auditing the measurement has failed.

## Decision

**Standing prior for this project: when a metric looks bad, audit the metric before touching the
model.** Concretely, before starting any modeling work (fine-tune, architecture swap, new
features, hyperparameter search) in response to a metric that looks worse than expected:

1. Re-derive the metric's construction from first principles — query text, corpus text, label
   provenance, train/eval split — and diff it against what production actually does, not what
   the eval code is assumed to do.
2. Hand-sample the failure population (not just the aggregate number) and categorize failures by
   whether they're genuinely winnable by a better model, versus artifacts of the measurement
   itself (mislabeled pairs, proxy-task mismatch, stale/leaky splits, truncation asymmetries,
   structurally-unfair test cases).
3. If a measurement fix is found, ship and re-measure BEFORE any modeling attempt — the model
   that "failed" against a broken metric may already be fine.
4. Only invest in modeling work against a metric that has survived this audit — i.e., a metric
   where the hand-sampled failure population is genuinely dominated by cases a better model
   could plausibly fix.

This is a prior, not an absolute rule: a metric can survive audit and still need real modeling
work (the ~60% remaining miss rate on the clean k8s retrieval subset is exactly this case — real
headroom, not another eval bug, though it hasn't been re-audited as exhaustively as the 24.7%
number was). The point is sequencing and burden of proof: measurement gets audited first, cheaply,
and modeling is the second move, not the first — not "never model."

## Consequences

- **What changes:** future sessions facing a disappointing metric on this project should default
  to a measurement audit (query/corpus construction, split freshness, label provenance,
  hand-sampled failure categorization) as the first diagnostic step, not the last resort after
  modeling attempts fail. This ADR is the citable justification for that ordering — a future
  session shouldn't have to re-argue it from scratch or re-discover the 6-measurement-fixes-won /
  7-of-13-modeling-runs-invalid track record (see the Correction section above for the full
  breakdown — the "8-vs-6" framing undercounted how many of the eight were unmeasurable, not
  measured-and-failed).
- **What becomes easier:** any future "should we fine-tune X" decision has a fast, cheap first
  gate (has the metric been audited?) before committing GPU time, data-mining effort, or a
  multi-session modeling arc.
- **What becomes harder:** nothing structurally — a measurement audit is CPU/analysis work,
  strictly cheaper than a fine-tune attempt, so this doesn't trade away modeling capacity, it
  sequences it.
- **What this doesn't claim:** that modeling never helps here, or that every future bad metric is
  a measurement bug. The six-measurement-fixes-won record, and the seven trustworthy (of thirteen
  total) modeling-run negatives per the Correction section above, are this project's own history,
  not a law of nature — but it's a strong enough prior to require: name the specific modeling
  hypothesis, and explain why it survives "have we actually looked at the measurement first."

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Leave this as tribal knowledge / repeat the finding informally each time | Exactly the failure mode this ADR exists to prevent — the pattern already recurred six times without being written down as a standing rule, costing real sessions of modeling effort against broken metrics each time. |
| State it as an absolute rule ("never model without an audit") | Too strong — a metric that has already survived audit (e.g. a metric this ADR itself validates) shouldn't require re-litigating the audit every time it's touched. Framed as a prior/sequencing rule with a clear survives-audit exit condition instead. |
| Fold this into an existing ADR (e.g. ADR-0040 or ADR-0033) | Those ADRs document specific instances; this is the pattern across all of them, and needs to be findable on its own, not buried inside one instance's rationale. |
