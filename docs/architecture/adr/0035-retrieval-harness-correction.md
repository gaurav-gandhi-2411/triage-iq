# ADR-0035 — Retrieval Harness Correction: Three Measurement Bugs, One Lever-Population Error

**Status:** Accepted (measurement-error correction; supersedes ADR-0031's lever numbers and
ADR-0033's baselines; withdraws ADR-0034's conclusion)
**Date:** 2026-07-24
**Decider:** Gaurav Gandhi (audit + corrections executed autonomously by CC, escalated at every step)

---

## Context

D2's vscode_duplicate fine-tune (ADR-0034) reported a confirmed regression, with an overfitting
diagnostic apparently ruling out the leading alternative explanation. Rather than accept "five
independent retrieval-improvement techniques all failed" (hybrid BM25 fusion, a pretrained
reranker, a stronger embedder, and two fine-tune configurations — ADR-0027, ADR-0031, ADR-0034 —
across two separate phases) as evidence the task is uniquely hard, GG called for a harness audit
first: "five established techniques failing is stronger evidence of a broken harness than of a
uniquely hard task." The audit found four distinct, independently-confirmed bugs.

## What was wrong

**1. Eval queries were title-only; production embeds title+body, untruncated.** Every retrieval
eval script (`d1_baseline_eval.py`, `d2_eval_finetuned.py`, all three ADR-0031 levers) built query
text as `f"{query_title}. {row.get('query_body', '')[:512]}"` — but the eval-set JSON schema never
populated `query_body` at all (confirmed: 0/200 vscode, 0/150 k8s pairs had it), so every query was
effectively title-only. Production (`triage.py:354-376`, `_collect_signals`) has always built
`f"{title}. {body}"` from the full incoming issue, **untruncated**. Verified via the exact code
path: `api/app.py` → `triage()` → `_collect_signals()` → `detector.retrieve()`. This was a
measurement bug, not a live product bug — production was never affected, only every retrieval
number this project has ever reported.

**2. The fine-tune trained and saved MEAN pooling; BGE-base's native config is CLS-token
pooling.** Compared the cached `BAAI/bge-base-en-v1.5` config
(`1_Pooling/config.json`: `pooling_mode_cls_token: true, pooling_mode_mean_tokens: false`)
against the D2 fine-tune's saved config (`pooling_mode_cls_token: false,
pooling_mode_mean_tokens: true`). `d2_train.py`'s `_save_st_model()` constructed
`st_models.Pooling(dim)` with no `pooling_mode` override — defaults to mean pooling in
sentence-transformers — while training itself used a hand-written `mean_pool()`. Both the ADR-0034
runs (5ep/lr2e-5 and the 2ep/lr1e-5 diagnostic) trained against the grain of what BGE-base was
actually pretrained to produce.

**3. Training truncated at 128 tokens, cutting 65.73% of examples.** Measured the real
BGE-tokenizer length distribution over all 25,980 anchor/positive/negative training texts:
p50=146, p90=217, p95=230, p99=272, max=314 tokens. `MAX_LEN=128` silently truncated more than
managed to hit the ceiling — 17,076 of 25,980 texts (65.73%) — while BGE-base's own native max is
512. Neither the original run's logs nor ADR-0034 reported this; it surfaced only when Step 4's
gradient-accumulation debugging required measuring actual token lengths.

**4. All three ADR-0031 levers were evaluated against the wrong pair population.** They read
`data/gold_related_v2.parquet` via `select_live_product_pairs()` — 277 (k8s) / 292 (vscode)
pairs, filtered only by live-index membership, **72% (k8s) / 20% (vscode) genuine** per ADR-0032's
own hand audit — against the stale served `dup_index_*_bge` (k8s covers only #1-15002, vscode only
7,028 issues). This is the exact unaudited population D1 (ADR-0033) was built to replace. The
levers never used D1's clean, hand-verified, issue-level-disjoint eval sets (150/200 pairs, 100%
genuine by construction) at all.

## What it invalidated (SUPERSEDED, not deleted)

### ADR-0033's baselines

| Eval set | Old (title-only) | **Corrected (prod-matching, untruncated)** |
|---|---|---|
| k8s_related R@5 | 9.3% [5.3, 14.0] | **18.0% [12.0, 24.0]** |
| vscode_duplicate R@5 | 43.5% [37.0, 50.5] | **50.5% [43.0, 57.5]** |

k8s nearly doubles; vscode gains 7 points. D1's actual contribution — the clean, hand-verified,
disjoint pairs, the eval-set/train-pool construction, the leakage guard — is unaffected; only the
query-construction bug in how the baseline was *measured* is corrected. Old baseline archived at
`reports/d1_clean_eval_baseline_PRE_QUERY_FIX.json`.

### ADR-0031's levers — re-run on D1's canonical eval sets + full-corpus index + fixed queries

Sanity check: each lever's dense-only arm reproduces the corrected canonical baseline exactly
(k8s R@5=0.1800, vscode R@5=0.5050, to the decimal, all three scripts) — proof the corrected
harness is aligned, not just plausible-looking.

| Repo | System | R@5 | Δ vs dense | 95% CI | Ships? |
|---|---|---|---|---|---|
| k8s (n=150) | BM25-only *(as-configured, see caveat)* | 12.7% | −5.33pp | [−10.67, 0.0] | — |
| k8s | Dense-only (canonical) | **18.0%** | — | — | — |
| k8s | RRF fusion | 14.7% | −3.33pp | [−8.0, +0.67] | No |
| k8s | Weighted fusion | 18.7% | +0.67pp | [−2.67, +4.0] | No |
| k8s | Reranker (bge-reranker-v2-m3) | 15.3% | −2.67pp | [−7.33, +2.67] | No |
| k8s | Stronger embedder (bge-large) | 18.0% | 0.0pp | [−4.67, +4.67] | No |
| vscode (n=200) | BM25-only *(as-configured)* | 23.5% | **−27.0pp** | **[−34.0, −20.0]** | — |
| vscode | Dense-only (canonical) | **50.5%** | — | — | — |
| vscode | RRF fusion | 40.5% | **−10.0pp** | **[−17.5, −3.0]** | **No (regression)** |
| vscode | Weighted fusion | 42.5% | **−8.0pp** | **[−13.5, −3.0]** | **No (regression)** |
| vscode | Reranker | 47.5% | −3.0pp | [−10.0, +3.5] | No |
| vscode | Stronger embedder | 47.5% | −3.0pp | [−7.5, +1.5] | No |

**All four levers (RRF, weighted, reranker, stronger embedder) are still rejected** — same
directional conclusion as ADR-0031 — but **weighted fusion flipped sign**: ADR-0031 originally
reported it as marginally *shipping* on both repos (k8s +3.25pp CI[0.35,6.5], vscode +4.11pp
CI[1.03,7.19], both excluding zero on the positive side). On the corrected harness it's flat on
k8s (+0.67pp, crosses zero) and a **statistically significant regression on vscode** (−8.0pp,
CI clearly excludes zero on the negative side). This is not a smaller version of the same result —
it's the opposite conclusion.

**BM25-alone answers the "diluting a strong signal or fusing two weak ones" question the original
ADR-0031 never measured** (the ranking was computed but never scored standalone): BM25 is the
**weaker** system on both repos, dramatically so on vscode (23.5% vs. dense's 50.5%, CI clearly
excludes zero). Hybrid fusion drags a strong dense signal down with a weaker lexical one — most
damaging where the strength gap is largest.

**Caveat, not acted on**: BM25's numbers above are *as-configured* — its tokenization
(`re.compile(r"[a-z0-9]+")`, plain alphanumeric splitting) was never inspected for identifier
fragmentation (e.g. `CrashLoopBackOff` → does it split into a token BM25 can match, or fragment
into pieces that don't overlap the query's tokenization of the same term?). If tokenization
handicaps BM25, its true ceiling could be higher than 12.7%/23.5%. Flagged, not re-run.

### ADR-0034's conclusion — WITHDRAWN

Both runs analyzed in ADR-0034 (5ep/lr2e-5 and the 2ep/lr1e-5 diagnostic leg) trained with the
mean-pooling and 128-token-truncation bugs above. The diagnostic leg's anti-overfit correction
(fewer epochs, lower LR) never touched either confound — it could not have ruled out overfitting
as *the* mechanism, because it never isolated pooling or truncation as variables. The ADR's
"overfitting rejected → distribution-mismatch hypothesis" reasoning had no support: the actual
cause of the original regression was two mechanistic bugs, not a data/approach limitation.

## The corrected finding

With CLS pooling, `max_seq_length=256` (covers p95, truncates only 2.26%), and gradient
accumulation (per-device batch=8 × 2 steps, VRAM-safe) — same 5 epochs, lr=2e-5, seed=42 —
evaluated against the corrected canonical baseline:

| Metric | Baseline | Fine-tuned | Δ | 95% CI (paired) |
|---|---|---|---|---|
| R@1 | 27.0% | 29.5% | +2.5pp | [−3.0, +8.0] |
| **R@5** | **50.5%** | **52.5%** | **+2.0pp** | **[−4.5, +8.5]** |
| R@10 | 59.5% | 61.5% | +2.0pp | [−4.5, +8.0] |
| MRR | 0.367 | 0.391 | +0.024 | — |

**NO SIGNAL — every metric moved positive (a complete sign reversal from ADR-0034's regression),
but no CI excludes zero.** This does not clear the ship bar.

**The power limit must be stated explicitly, not glossed over**: at n=200, the R@5 CI half-width
is ≈±6.5pp. A true underlying effect of, say, +4pp would produce a confidence interval that still
crosses zero at this sample size — this eval is **not powered to detect an effect of the size
that would plausibly matter**. "No signal" here means **underpowered, not disproven**. Whether a
larger training set, more epochs, or different hyperparameters would produce a real, detectable
lift is genuinely **open and untested** — this ADR does not claim fine-tuning doesn't work on this
data, only that this run didn't show a statistically distinguishable effect.

## The process lesson

Five consecutive negative results across independent techniques (BM25 hybrid, cross-encoder
reranker, stronger embedder, two fine-tune configs) should have triggered a harness audit
immediately, not after GG explicitly asked for one. A pretrained cross-encoder reranker
*regressing* retrieval quality is nearly unheard of in the IR literature — that alone was a strong
signal something upstream was wrong, not evidence the corpus "defeats standard IR." This project
had already learned this exact class of lesson three times before this thread — the proxy-metric
trap (ADR-0028/0030), the `title_sim` noise contamination (ADR-0032), the gold-pair audit
(ADR-0032) — and still didn't point the same skepticism at the retrieval eval harness itself until
asked. The generalizable rule: when every lever against a baseline fails, audit the measurement
before concluding the task is hard.

## Still open — flagged, not fixed

- **Corpus/query truncation asymmetry**: `_build_text()` (corpus) truncates body to 512 chars;
  production queries (`triage.py`) are untruncated. Unintentional — no comment, ADR, or spec
  language justifies it; the two code paths were evidently written independently. Not fixed —
  changing corpus construction invalidates the index, out of scope for this correction pass.
- **BGE's query instruction prefix is unused.** BAAI's documented best practice for `bge-base-en-v1.5`
  is to prefix *queries* (not documents) with `"Represent this sentence for searching relevant
  passages: "` for retrieval tasks. Neither production nor any eval script does this. This would be
  a **paired change** — production and eval would need to change together, since adding the prefix
  only to eval would reopen exactly the class of measurement/prod mismatch this ADR just closed.
  Not attempted here.
- **BM25 tokenization uninspected** (see caveat above).

## Consequences

- **What changes**: nothing in production. `src/triage_iq/api/loader.py` loads only
  `dup_index_{slug}_bge`; `SUPPORTED_MODELS["bge"]` resolves to the HF hub id
  `BAAI/bge-base-en-v1.5`, never a local fine-tuned path. Re-confirmed after this correction pass —
  no fine-tuned artifact, no D1/D2 index, is referenced anywhere in serving code.
- **What becomes easier**: every retrieval number in this project is now measured through a harness
  that's been verified against a ground-truth sanity check (self-retrieval R@1 ≈100%) and confirmed
  to reproduce identically across four independent scripts. Future retrieval work has a trustworthy
  floor to build on.
- **What becomes harder**: the README's retrieval story is now a corrected-baseline-plus-still-all-
  rejected-levers narrative — accurate, but it means five ADRs (0027, 0028, 0030, 0031, 0032) and
  this correction all sit in the retrieval history, a heavier provenance trail than a clean win
  would have left.
- **Cost**: $0 additional (all correction work ran locally; the corrected D2 training run cost
  ~70 minutes of RTX 3070 time, same hardware as before).

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Accept the original five-technique failure as evidence of task difficulty | This was GG's explicit objection: a pretrained reranker regressing is a stronger signal of harness breakage than task difficulty. Not investigating would have shipped a false negative into the project's permanent record. |
| Fix only the bug that was asked about (pooling) and leave query construction alone | Each fix surfaced during verification of the previous one (seq-length during pooling verification, lever population during the "vs corrected baseline" framing, do_lower_case during the pooling diff) — stopping at the first found bug would have left known-broken comparisons standing. |
| Silently accept `do_lower_case: False` vs `True` as a fourth bug | Verified functionally inert first: BGE's own HF tokenizer normalizer applies `lowercase=True` unconditionally regardless of the sentence-transformers wrapper flag (confirmed via `tok.backend_tokenizer.normalizer`). Reported as found-and-verified-harmless, not silently ignored or wrongly escalated as a bug. |
| Treat the +2.0pp non-significant R@5 lift as a small win worth shipping | Directly against this project's ship bar (meaningful lift, CI clearly excluding zero) and the explicit lesson of the W3 hold (ADR-0027) — a smaller, still-non-significant effect is not a weaker version of a win, it's the same "don't ship" call with more honest uncertainty language. |
