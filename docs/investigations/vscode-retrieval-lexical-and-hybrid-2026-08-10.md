# Investigation: vscode_duplicate lexical structure vs k8s_related, and BM25/hybrid re-test

**Date:** 2026-08-10
**Status:** CONCLUDED — negative result, nothing to ship
**Scope:** ADR-0040's working hypothesis for why the BGE query instruction helps k8s_related but
hurts vscode_duplicate; a re-test of BM25/hybrid retrieval (ADR-0031, previously rejected) against
the current, corpus-truncation-fixed index.
**Reproduce:**
- `PYTHONPATH=src python scripts/investigate_lexical_similarity_k8s_vs_vscode.py` →
  `reports/lexical_similarity_k8s_vs_vscode.json`
- `PYTHONPATH=src python scripts/lever1_hybrid_bm25_rrf.py` → `reports/lever1_hybrid_bm25_rrf.json`

---

## 0. Reconciling the baseline number: 53.50%, not 50.50%

The task brief that spawned this investigation cited vscode_duplicate current R@5 as "50.5%".
ADR-0040 (2026-08-06) reports 53.50% [46.5, 60.5] as the reproduced current number. Both are real
numbers that exist in this codebase's history — they are not the same measurement.

**Mechanism:** `reports/d1_clean_eval_baseline.json` / the `d1_full_corpus_index_{repo}_bge`
artifact (no suffix) was built by `scripts/d1_build_full_corpus_index.py` **before** ADR-0040's
Lever 1 fix landed — its `_build_text()` corpus-side truncation is the old **512-character** cut,
not the tokenizer-based 512-**token** cut. That's the artifact behind the "50.50%" figure (see
`scripts/lever12_eval.py`'s `BASELINE` row, which reads exactly this path). It is stale.

`data/models/d1_full_corpus_index_{repo}_bge_lever1` is the corpus-truncation-fixed rebuild.
I verified by SHA-256 that its `index.faiss` and `meta.pkl` are **byte-identical** to the
currently live-serving `data/models/similar_issue_index_{repo}_bge` (both repos) — same hashes as
recorded for the old `dup_index_{repo}_bge` name in the tracked `data/models/MANIFEST.sha256`:

```
3b825bd0...  d1_full_corpus_index_microsoft_vscode_bge_lever1/index.faiss
3b825bd0...  similar_issue_index_microsoft_vscode_bge/index.faiss          (== MANIFEST's dup_index_microsoft_vscode_bge)
cd74345b...  d1_full_corpus_index_microsoft_vscode_bge_lever1/meta.pkl
cd74345b...  similar_issue_index_microsoft_vscode_bge/meta.pkl             (== MANIFEST's dup_index_microsoft_vscode_bge)
```

So `_lever1` is not a separate, D1-only artifact — it's the same content as what's actually live in
production today, just built via a different script invocation. Running `d1_baseline_eval.py`-style
measurement against it, through the real un-overridden `retrieve()` code path (letting
`QUERY_INSTRUCTION_REPO_OVERRIDE["microsoft_vscode"] = False` resolve on its own), reproduces
ADR-0040's own confirmed number exactly (also reproduced fresh in this investigation's hybrid run,
§2 below, as the `dense_only` row).

**Reconciled: current vscode_duplicate R@5 = 53.50% [46.5, 60.5] (95% CI, n=200). The "50.5%" figure
is the pre-corpus-truncation-fix baseline and should not be cited as current going forward.**

---

## 1. Lexical characterization: k8s_related vs vscode_duplicate

**Method:** for each eval set's (query, true target) pair, three lexical-similarity metrics, all in
[0, 1], higher = more similar:

1. **Jaccard token overlap, title-only** — lowercased `[a-z0-9]+` tokenization.
2. **Jaccard token overlap, title+body** — same tokenization, full text.
3. **Normalized Levenshtein similarity, title+body** (`rapidfuzz.distance.Levenshtein.normalized_similarity`)
   — the edit-distance-family metric. Chosen over TF-IDF cosine because Jaccard already covers the
   "bag of tokens" view; Levenshtein adds an orthogonal, character-level/word-order signal that
   near-duplicate issues (same bug re-titled or re-worded) plausibly share more of than genuinely
   related-but-distinct issues, without doubling up on what Jaccard already measures.

Target-side text is pulled from the **same current full-corpus index** used for retrieval
(`d1_full_corpus_index_{repo}_bge_lever1`, confirmed identical to the live-serving index, §0) —
i.e. exactly what the retriever has indexed for that issue (title + tokenizer-truncated body), not
a separately-scraped raw body. The eval JSON itself only carries `original_title`, not
`original_body`, so this is also the only source of full target text available.

### Distributions (n=150 k8s, n=200 vscode; script: `scripts/investigate_lexical_similarity_k8s_vs_vscode.py`)

| Metric | Eval set | mean | median | IQR (Q1–Q3) | std |
|---|---|---|---|---|---|
| Title Jaccard | k8s_related | 0.130 | 0.071 | 0.000–0.154 | 0.194 |
| Title Jaccard | vscode_duplicate | 0.178 | 0.121 | 0.062–0.204 | 0.233 |
| Full-text Jaccard | k8s_related | 0.137 | 0.121 | 0.086–0.170 | 0.074 |
| Full-text Jaccard | vscode_duplicate | 0.187 | 0.149 | 0.102–0.214 | 0.153 |
| Levenshtein normalized similarity | k8s_related | 0.210 | 0.226 | 0.167–0.253 | 0.072 |
| Levenshtein normalized similarity | vscode_duplicate | 0.225 | 0.217 | 0.167–0.260 | 0.104 |

### vscode − k8s delta (two-sample percentile bootstrap, independent resampling per arm, seed=42, 2000 resamples)

| Metric | Δ (vscode − k8s) | 95% CI | Excludes zero? |
|---|---|---|---|
| Title Jaccard | +0.047 | [0.003, 0.094] | **Yes** |
| Full-text Jaccard | +0.050 | [0.027, 0.076] | **Yes** |
| Levenshtein normalized similarity | +0.015 | [-0.001, 0.035] | No |

### Tail check (not in the original plan, added because the mean/median comparison alone was
ambiguous — see below)

| Metric | frac. k8s pairs > 0.5 | frac. vscode pairs > 0.5 |
|---|---|---|
| Full-text Jaccard | 0.0% | 5.0% |
| Levenshtein normalized similarity | 0.0% | 2.5% |

### Verdict: partially confirmed, weak-to-moderate — not a clean confirm or kill

Two of three metrics show a statistically significant (CI excludes zero) mean difference, and all
three point the same direction (vscode higher). But the effect is modest in absolute terms — a
~0.05 absolute bump on a base of ~0.13–0.19, i.e. the *bulk* of both distributions (medians, IQRs)
look broadly similar. What's real and distinctive is the **tail**: vscode has a genuine minority of
near-duplicate-looking pairs (5% with full-Jaccard > 0.5, 2.5% with Levenshtein > 0.5) that k8s
structurally lacks (0% in both). 95% of vscode pairs, though, sit below that threshold — most
"duplicate" pairs in this eval set are *not* lexically near-identical text; they're the same bug
reported in different words, which is a genuinely harder retrieval problem than pure string
matching, more evidence against the "near-duplicate = lexical overlap dominant" framing than for
it, when taken at face value.

I set my "meaningfully higher" bar in advance as: at least 2/3 metrics statistically significant
AND consistent direction across all 3. That bar was met, so per the task's own conditional
structure I proceeded to Part 2 rather than declaring the hypothesis killed off indirect stats
alone — a direct retrieval measurement is strictly more decisive than distributional proxies, and
the tail effect specifically is exactly the kind of thing that could plausibly move BM25/hybrid
behavior on a subset of queries without moving the aggregate mean much.

---

## 2. BM25 / hybrid RRF re-test on the current, corrected index

**Bug found and fixed first:** `scripts/lever1_hybrid_bm25_rrf.py` (last touched under ADR-0031,
"corrected" once already for eval-set/query-text issues) still pointed its `REPOS` config at
`data/models/d1_full_corpus_index_{repo}_bge` — the **pre**-Lever-1-fix, char-truncated index (the
same stale artifact behind the "50.5%" figure in §0), not the corrected one. Fixed to point at
`_bge_lever1` (confirmed byte-identical to the live-serving index, §0). Query text was already
correct (untruncated title+body via `_retrieval_eval_common.py::query_text`, matching production).
This is a real, reproducible correctness bug independent of this investigation — anyone running
this script before today's fix would silently re-measure against the wrong, stale corpus.

**Method (unchanged from ADR-0031's design, `scripts/lever1_hybrid_bm25_rrf.py`):** BM25Okapi built
over the exact same corpus text and issue-number universe as the loaded dense index
(`detector.texts`/`detector.issue_numbers`, so BM25 and dense see an identical document set). Top-100
candidate pool from each system, fused via Reciprocal Rank Fusion (k=60, untuned — no clean
tuning slice exists without either shrinking the eval set or tuning on the test pairs themselves,
same reasoning ADR-0031 used). A secondary min-max-normalized weighted fusion (0.5/0.5) is also
reported. Paired bootstrap CI (`_retrieval_eval_common.py::paired_bootstrap_ci`, same seed/N as
everywhere else in this project) on the R@5 delta vs. dense-only.

### Full recall table (n=150 k8s, n=200 vscode)

| Repo | System | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| k8s_related | dense-only | 10.67% | **24.67%** | 30.00% | 34.67% |
| k8s_related | BM25-only | 6.67% | 18.67% | 23.33% | 28.67% |
| k8s_related | hybrid RRF | 10.00% | 20.00% | 27.33% | 38.00% |
| k8s_related | hybrid weighted | 11.33% | 21.33% | 32.00% | 41.33% |
| vscode_duplicate | dense-only | 32.00% | **53.50%** | 61.00% | 67.50% |
| vscode_duplicate | BM25-only | 15.00% | 24.00% | 29.50% | 36.00% |
| vscode_duplicate | hybrid RRF | 20.00% | 42.50% | 51.00% | 64.00% |
| vscode_duplicate | hybrid weighted | 23.50% | 47.50% | 56.00% | 66.50% |

The dense-only R@5 rows reproduce §0's reconciled numbers exactly (24.67% k8s, 53.50% vscode),
confirming this run is against the correct, current index.

### R@5 deltas vs. dense-only (paired bootstrap, 2000 resamples, seed 42)

| Repo | Comparison | Δ (pp) | 95% CI (pp) | Ships (CI excludes zero, positive)? |
|---|---|---|---|---|
| k8s_related | BM25-only vs dense | -6.00 | [-12.67, +0.67] | No — crosses zero, negative point estimate |
| k8s_related | hybrid RRF vs dense | -4.67 | [-10.00, +0.67] | No — crosses zero, negative point estimate |
| k8s_related | hybrid weighted vs dense | -3.33 | [-8.67, +1.33] | No — crosses zero, negative point estimate |
| vscode_duplicate | BM25-only vs dense | **-29.50** | **[-36.50, -22.50]** | **No — CI excludes zero, strongly negative** |
| vscode_duplicate | hybrid RRF vs dense | **-11.00** | **[-17.50, -4.50]** | **No — CI excludes zero, negative** |
| vscode_duplicate | hybrid weighted vs dense | **-6.00** | **[-10.50, -2.00]** | **No — CI excludes zero, negative** |

vscode's hybrid deltas aren't just "doesn't clear the ship bar" (crosses zero, ambiguous) — the CI
excludes zero **on the harmful side**. This is a statistically significant regression, not a null
result. k8s's deltas are directionally negative too, just underpowered to be significant at n=150.

Diagnostic: BM25 recovers a dense miss (hits @5 where dense doesn't) on only 6/150 (4.0%) k8s
pairs and 11/200 (5.5%) vscode pairs — a small minority, and nowhere near enough to offset how much
worse BM25/hybrid does on the much larger set of pairs where dense already wins and adding BM25
into the fusion demotes the correct answer.

### This directly bears on ADR-0040's working hypothesis — and doesn't support it

ADR-0040's hypothesis was that vscode's near-duplicate task is lexical-overlap-dominant, which
would predict BM25 alone should perform reasonably and hybrid should at least match dense. The
opposite happened: BM25-alone gets less than half of dense's R@5 on vscode (24.0% vs 53.5%), a
much larger gap than on k8s (18.7% vs 24.7%). If anything, vscode's ranking task is **more**
dense-embedding-dominant than k8s's, not less. §1's lexical tail effect (a real minority of
near-identical pairs) is not large or consistent enough to make lexical retrieval competitive at
the aggregate level. **The evidence here actively weakens, not confirms, ADR-0040's stated
hypothesis for the query-instruction asymmetry** — the instruction effect (helps k8s, hurts
vscode) has some other cause, still unidentified. That question remains genuinely open; this
investigation answers "is hybrid/BM25 the fix" (no), not "why does the instruction hurt vscode."

---

## 3. Recommendation

**Don't ship hybrid BM25+dense retrieval for either repo.** Both the point estimates and CIs are
clean on this: vscode's hybrid/BM25 deltas are significantly *negative* (CI excludes zero on the
harmful side), and k8s's are directionally negative and don't clear the ship bar either. This
reconfirms ADR-0031's original rejection of hybrid — even after fixing the corpus-truncation bug,
using the current live-serving index, and using full (not title-only) query text, the qualitative
conclusion is unchanged: **dense-only is the stronger retriever for both tasks**, and RRF/weighted
fusion with an untuned BM25 arm actively hurts vscode's numerically best-performing eval set.

No further hybrid-retrieval lever is worth pursuing on this axis without a fundamentally different
approach (e.g. a learned/tuned fusion weight, or BM25 as a re-ranking signal only on the small
tail of high-lexical-overlap candidates rather than a full-corpus fusion) — and given the
magnitude of the regression here, that's a "would need much stronger evidence of a narrower,
targeted benefit" bar, not a quick re-tune.

**The ADR-0040 query-instruction asymmetry (helps k8s, hurts vscode) remains unexplained.** This
investigation was designed to test one specific candidate explanation (near-duplicate/lexical-
overlap dominance) and that explanation does not hold up under a direct test — vscode is not more
lexically-tractable than k8s in a way that would predict its behavior under the BGE instruction.
ADR-0040's per-repo override (`QUERY_INSTRUCTION_REPO_OVERRIDE["microsoft_vscode"] = False`) should
stay as configured — nothing here changes that decision — but its own explicit invitation to revisit
the hypothesis if new evidence emerges is now partially exercised: the specific mechanism proposed
is not supported by lexical-similarity or BM25/hybrid data. A future investigation into the real
cause would need a different angle (e.g. corpus-side text-length effects, embedding-space geometry
differences between the two repos' corpora, or a per-query breakdown of which vscode pairs the
instruction actually hurts) — not lexical-overlap or hybrid-retrieval framing, both now tested and
found wanting.

### What does NOT need re-running

- ADR-0040's shipped config (`QUERY_INSTRUCTION_REPO_OVERRIDE`) — unaffected, no production code
  touched by this investigation.
- The current baseline numbers (53.50% vscode, 24.67% k8s) — reconfirmed here via an independent
  script run, not just cited from ADR-0040.

### Artifacts

- `scripts/investigate_lexical_similarity_k8s_vs_vscode.py` → `reports/lexical_similarity_k8s_vs_vscode.json`
- `scripts/lever1_hybrid_bm25_rrf.py` (index-path bug fixed) → `reports/lever1_hybrid_bm25_rrf.json`
- `requirements-dev.txt` — added `rapidfuzz` (Levenshtein metric, eval-only dependency)
