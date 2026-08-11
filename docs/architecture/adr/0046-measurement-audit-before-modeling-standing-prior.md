# ADR-0046 — Standing prior: audit the metric before touching the model

**Status:** Accepted
**Date:** 2026-08-11
**Decider:** Gaurav Gandhi

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
  session shouldn't have to re-argue it from scratch or re-discover the 8-vs-6 track record.
- **What becomes easier:** any future "should we fine-tune X" decision has a fast, cheap first
  gate (has the metric been audited?) before committing GPU time, data-mining effort, or a
  multi-session modeling arc.
- **What becomes harder:** nothing structurally — a measurement audit is CPU/analysis work,
  strictly cheaper than a fine-tune attempt, so this doesn't trade away modeling capacity, it
  sequences it.
- **What this doesn't claim:** that modeling never helps here, or that every future bad metric is
  a measurement bug. The six-for-six / eight-for-zero record is this project's own history, not a
  law of nature — but it's a strong enough prior to require: name the specific modeling
  hypothesis, and explain why it survives "have we actually looked at the measurement first."

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Leave this as tribal knowledge / repeat the finding informally each time | Exactly the failure mode this ADR exists to prevent — the pattern already recurred six times without being written down as a standing rule, costing real sessions of modeling effort against broken metrics each time. |
| State it as an absolute rule ("never model without an audit") | Too strong — a metric that has already survived audit (e.g. a metric this ADR itself validates) shouldn't require re-litigating the audit every time it's touched. Framed as a prior/sequencing rule with a clear survives-audit exit condition instead. |
| Fold this into an existing ADR (e.g. ADR-0040 or ADR-0033) | Those ADRs document specific instances; this is the pattern across all of them, and needs to be findable on its own, not buried inside one instance's rationale. |
