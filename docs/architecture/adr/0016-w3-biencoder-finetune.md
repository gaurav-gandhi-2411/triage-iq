# ADR-0016 — W3: Fine-tune BGE bi-encoder for similar-issue retrieval

**Status:** Rejected (current evidence) — not a failure, a not-yet-demonstrated result. See Consequences.
**Date:** 2026-07-04 (rebased from a draft originally numbered ADR-0010 on `feat/w3-finetune`, 2026-05-31)

---

## Rebase note (2026-07-04)

This ADR was originally drafted as `0010-w3-biencoder-finetune.md` on `feat/w3-finetune`, a branch
that diverged from main 47 commits ago (before CQR, the eval-regression-gate, the model-artifact
drift guard, and the grounding verifier landed). That branch's number collided with main's current
`0010-conformal-quantile-regression.md`. Renumbered to 0016, the next free slot.

More importantly: the branch's fine-tuned model weights were never committed
(`data/models/**` is gitignored except the manifest), so the original +13.16pp k8s / +13.33pp vsc
R@5 result could not simply be carried over — it had to be reproduced end-to-end on current main
before being claimed or shipped. The scripts (`scripts/w3_t2_mine_negatives.py`, `w3_t3_split.py`,
`w3_t4_train.py`, `w3_t5_eval.py`) and their tracked inputs (`data/w3_hard_negatives.parquet`,
`data/w3_split.parquet`) were ported as-is — `data/gold_related.parquet` was confirmed unchanged
(1,435 pairs) since the original run, so those mining/split outputs remain valid.

**Re-verification result: the honest re-run does NOT match the stale branch's claim.** Retraining on
current main (same seed, same hyperparameters, same ported hard negatives/split) produced smaller,
statistically weaker deltas — see Consequences below. This is the expected outcome of the "verify
against current, don't trust the stale report" discipline: the original PASS verdict does not survive
reproduction. **Final decision: Rejected on current evidence** — no index cutover, no cassette
re-record, no merge. The branch stays documented-and-rejected.

---

## Context

Baseline BAAI/bge-base-en-v1.5 (zero-shot) achieved R@5 of 0.4102 (k8s) and 0.3674 (vsc) on the full
gold similar-issue pairs (ADR-0008). ADR-0006 rejected cross-encoder reranking as a latency-neutral path
to improvement. The next hypothesis: domain-specific fine-tuning of the bi-encoder itself should
lift recall with zero added inference latency.

**Gold dataset**: 1,435 positive pairs (`data/gold_related.parquet`) built from PR→issue references
and text-similarity labelling across two repos (1,024 k8s, 411 vsc). Sources: body_ref (4), body_related
(1,010), title_sim (421). Confidence: 1,431 medium, 4 high.

## Decision

Fine-tune BAAI/bge-base-en-v1.5 on domain gold pairs using MultipleNegativesRankingLoss with
explicit hard negatives.

**T2 — Hard-negative mining**: For each positive pair, query the pre-built BGE FAISS top-50 and
collect the rank-1 non-positive result as the hard negative. 1,028 training triplets after
connected-component split (35 cross-repo-contaminated pairs dropped). Spot-check: 15% soft-positive
rate — below the 20% abort threshold.

**T3 — Split**: Connected-component temporal split (networkx). Components sorted chronologically
by earliest created_at. Greedy state-machine assignment at 70/15/15. k8s: 704/150/152,
vsc: 324/10/60. Zero-leakage verified (no issue_id in multiple splits). vsc val=10 due to
a large late-period component cluster — acceptable for the proxy early-stop metric; real test uses 60 pairs.

**T4 — Training**: Direct HuggingFace training loop (sentence-transformers 2.7.0's `model.fit()` had
130s/step overhead due to per-sequence CUDA kernel recompilation with smart batching). Pre-tokenised
TripletDataset, AdamW lr=2e-5, warmup 10%, batch=16, epochs=5, temp=0.05, max_len=128, seed=42.

**T5 — Evaluation**: Full-corpus FAISS rebuild with fine-tuned model. Baseline computed on the same
test-split pairs using the pre-built dup_index_*_bge index. Self-exclusion applied (query issue excluded
from retrieval, matching baseline methodology). Bootstrap 95% paired CI (n=2,000) on delta.

## Consequences

**Results (test split, zero training overlap, re-established on current main 2026-07-04):**

| Repo | n | Baseline R@5 | Fine-tuned R@5 | Delta | CI 95% (2,000 boot) | Script verdict | ADR verdict |
|---|---|---|---|---|---|---|---|
| kubernetes_kubernetes | 152 | 0.5263 | 0.6447 | +11.84 pp | [+0.00, +22.38] | ESCALATE_n300 | REJECTED — CI does not exclude zero |
| microsoft_vscode | 60 | 0.6833 | 0.7833 | +10.00 pp | [−6.67, +25.00] | ESCALATE_n300 | REJECTED — CI crosses zero |

The T5 script's own decision logic (`ESCALATE_n300`) proposes testing at a larger sample before
concluding. That path is itself rejected — see point 3 below. The ADR's final call is REJECTED, not
ESCALATE, because escalation isn't available on equal terms for both repos.

### Verdict: Rejected on current evidence (not "failed")

**1. Does not clear the ADR-0006 bar, applied consistently.** ADR-0006 rejected the cross-encoder
reranker on the rule "CI must exclude zero on BOTH repos to ship." Applying that same rule to our own
fine-tune: k8s's CI lower bound sits exactly *at* zero (0.0000); vscode's lower bound is negative
(−0.0667). Neither excludes zero. A bar that rejected someone else's model and then gets waived for
our own model isn't a bar — it's theater. The fine-tune is rejected by the same standard, applied
consistently.

**2. The stale +13pp was an artifact of a training bug, not a stronger model.** The original
`feat/w3-finetune` run's T4 winner was the *combined* model (both repos trained jointly) only because
per-repo training crashed on a GPU state leak before it could complete and compete fairly — combined
won by elimination, not by being measured against a working alternative. This run's T4 fix (explicit
`del` + `torch.cuda.empty_cache()` between sequential `train_model()` calls) let all three variants
train to completion: combined (val R@5=0.8500), k8s-only (0.8533), vsc-only (0.8000). k8s-only edged
out combined by 0.33pp on val R@5, so T4's `max()` selection correctly picked the **per-repo models**
instead of combined. Per-repo models see roughly 1/3 to 2/3 the training data of the combined model,
which is the direct cause of the wider CIs and smaller point deltas in this honest run. **The
better-looking number came from a defect — the crash silently protected the eval from a weaker,
higher-variance model class it would otherwise have had to compete against.** This belongs in the same
class of finding as the `has_priority` feature leak and the parquet drift bug: a metric that looked
good because something was broken, not because the model was good.

**3. Escalating to n=300 is rejected as a path, not attempted.** vscode has only 411 total gold pairs;
an n=300 vscode test split is not reachable without gutting the training set to near-zero. Running
k8s alone to n=300 while leaving vscode at n=60 would hold the two repos to asymmetric evidentiary
standards and cherry-pick the repo that happens to have enough data to look convincing. If the
protocol can't run identically on both repos, the correct conclusion is **"not demonstrated,"** not
"demonstrated where it was measurable." No re-split, no re-train, no n=300 run was executed for this
ADR.

**What this rejection is and isn't:** Both repos show a positive point estimate (+11.84pp k8s,
+10.00pp vsc) — this is not a null or negative result, and domain fine-tuning of the bi-encoder remains
a promising direction. What isn't established at the available sample size is *significance*: the
gold set (1,435 pairs total, 1,024 k8s / 411 vsc) is too small to produce a test split where both
repos' CIs exclude zero. The limiting factor is data volume, not the modeling approach.

**Path forward — linked to W5:** This is not shippable until either (a) more gold pairs enable a
both-repos CI-excludes-zero result at the existing 70/15/15 split ratio, or (b) a gold-set labeling
expansion grows the pair count enough to support a larger, still-symmetric test split. **W5 (gold set
expansion, currently n=60→150 for the judge eval) is the concrete unblock**: if the W5 labeling
protocol is extended to also grow `gold_related.parquet` (not just `gold_triage_plans.parquet`), the
resulting larger pair count is what would let this fine-tune be honestly revisited — either or both
repos reaching enough test pairs for the CI to genuinely resolve one way or the other. Until that gold
set exists, re-running T2-T5 on the same 1,435 pairs would just reproduce this same ambiguous CI.

**CPU latency**: Not a factor in the rejection — no regression was observed (same architecture, same
FAISS IndexFlatIP, only weights would change). The rejection is entirely about retrieval-quality
significance, not cost or latency.

**What worked**:
- Hard negatives from the existing FAISS top-50 were effective (rank-1 negatives, 15% soft-positive rate tolerated)
- The T4 GPU-state-leak fix now lets per-repo variants train to completion instead of crashing —
  this is a real bug fix, independent of the rejection, and should stay fixed
- Both repos show a positive point-estimate delta after honest re-verification — the hypothesis
  (domain fine-tuning helps) is not refuted, only unproven at this data scale

**What didn't work / open questions**:
- The winning architecture flipped from combined to per-repo between the stale run and this run —
  variant selection is sensitive to the GPU-leak fix, which is itself a sign the val-R@5-based winner
  pick (n=10-60 val pairs) is noisy at this data scale
- Proxy val R@5 peaked early across all three variants (epoch 0-1), declining afterwards — small
  dataset (1,028 triplets) overfits quickly regardless of combined vs. per-repo split

**Alternatives considered**:
- Escalate k8s to n=300 while leaving vscode at n=60: rejected — asymmetric standard, cherry-picks the
  repo with enough data (see point 3 above)
- Ship despite the ambiguous CI, treat as a judgment call: rejected — would abandon the exact bar
  ADR-0006 established and applied against a different model class
- Track B (ms-marco-MiniLM-L-6-v2 CE reranker, 22M): not attempted — Track A was prioritized; moot now
  that Track A itself doesn't clear the bar
- More hard negatives per pair (10 instead of 1): not attempted — would not address the root
  constraint, which is total gold pair count, not negative-mining depth

---

## Correction note (2026-05-31, carried over from the original draft)

**Pre-merge verification caught eval contamination in the original T5 run.**

The initial T5 eval script used `sample_gold(gold, repo, n=100, seed=42)` to draw the evaluation
sample. This function sampled from `data/gold_related.parquet` — the **full gold corpus**, not
filtered to held-out splits. Since the 70/15/15 temporal split assigned ~70% of gold pairs to
training, any random draw from full gold is ~70% training data.

Measured contamination: 66/100 k8s eval pairs and 71/100 vsc eval pairs were in-training, inflating
the reported delta to ~2x the true value (originally reported +25.98 pp k8s / +13.26 pp vsc).

**Fix applied**: `sample_gold` removed from T5. Canonical eval is test-split pairs only
(k8s n=152, vsc n=60, confirmed zero overlap with training data). Baseline computed live on the
same query set using the pre-built `dup_index_*_bge` FAISS indexes. An `assert_eval_disjoint_from_train`
guard prevents silent reintroduction of this bug — kept intact through the rebase.

This correction is kept permanently — it is evidence of the verification discipline applied before
merge, not an embarrassment to hide.

---

## Scripts

- `scripts/w3_t2_mine_negatives.py` — hard-negative mining
- `scripts/w3_t3_split.py` — connected-component temporal split
- `scripts/w3_t4_train.py` — fine-tuning with direct HF loop
- `scripts/w3_t5_eval.py` — evaluation + bootstrap CI (leakage-guarded)

## Data / model artifacts

- `data/w3_hard_negatives.parquet` — 14,350 hard-negative records (ported, gold_related.parquet unchanged)
- `data/w3_split.parquet` — 1,400 pair split assignments (ported, gold_related.parquet unchanged)
- `data/models/bge_finetuned_combined/` — combined-repo SentenceTransformer, trained but NOT the winner this run (val R@5=0.8500) — not committed, gitignored
- `data/models/bge_finetuned_k8s/` — k8s-only SentenceTransformer, T4 winner for k8s (val R@5=0.8533) — not committed, gitignored
- `data/models/bge_finetuned_vsc/` — vsc-only SentenceTransformer, T4 winner for vsc (val R@5=0.8000) — not committed, gitignored
- `data/models/bge_finetuned_k8s_index/` — FAISS index built on the k8s winner model (regenerated)
- `data/models/bge_finetuned_vsc_index/` — FAISS index built on the vsc winner model (regenerated)
- `reports/w3_t4_val_results.json` — T4 variant comparison, committed
- `reports/w3_t5_eval_results.json` — full T5 eval table, committed
