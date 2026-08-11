# Phase B: DeBERTa-v3-base classifier — result summary (negative)

**Question:** does a transformer classifier beat the shipped multi-label TF-IDF+LR on top-3 (the
metric the product actually uses)? Two arms tested architecture and supervision separately.

**Answer: no, on both repos, in every configuration tried, including a diagnosed and partially
corrected version of ARM 2. TF-IDF+LR remains champion. Nothing here changes production.**

## Ship bar

Meaningful top-3 lift over the shipped baseline, with a paired-bootstrap CI clearly excluding
zero. No configuration below reaches that bar — most miss by a wide margin, with non-overlapping
CIs against baseline in every case.

## Results

| Arm | Repo | top-3 | 95% CI | Baseline top-3 | Δ vs baseline | top-1 | macro-F1 | any-valid top-1 |
|---|---|---|---|---|---|---|---|---|
| ARM 2 multi (unweighted BCE) | k8s | 56.3% | [50.5, 61.9] | 87.1% | **−30.8pp** | 31.1% | 0.047 | 44.4% |
| ARM 2 multi (BCE, pos_weight cap=10) | k8s | 66.1% | [60.4, 71.3] | 87.1% | **−21.0pp** | 37.1% | 0.088 | 51.4% |
| ARM 1 single (softmax) | k8s | 66.8% | [61.1, 72.0] | 87.1% | **−20.3pp** | 37.1% | 0.067 | 43.4% |
| ARM 2 multi (BCE, pos_weight cap=10) | vscode | 72.7% | [65.9, 78.6] | 89.8% | **−17.1pp** | 50.3% | 0.185 | 51.9% |
| ARM 1 single (softmax) | vscode | 75.9% | [69.3, 81.5] | 89.8% | **−13.9pp** | 56.7% | 0.228 | 57.8% |

Baseline (shipped, for reference): k8s top-3 87.1% [82.7, 90.5], top-1 60.5%, any-valid top-1
59.4%. vscode top-3 89.8% [84.7, 93.4], top-1 76.5%, any-valid top-1 71.7%.

Tail-class recall (<15 train examples): **0.000 across all 10 k8s tail classes and all 7 vscode
tail classes, in every arm and configuration** — architecture and loss-reweighting changes made
no difference here. These classes have 8–13 training examples; that is a data-volume floor no
model choice tested can clear.

## Pre-training gates (all passed, see session log for detail)

1. GPU free (RTX 3070, 0 processes) — confirmed.
2. Leakage guard — issue-level disjoint train/val/test for both repos, exact match to expected
   counts (vscode 1488/187/187, k8s 2284/286/286).
3. Config: `DebertaV2TokenizerFast` matches `microsoft/deberta-v3-base` by construction;
   `max_seq_length=512` measured truncation 8.49% vscode / 3.40% k8s (closely matches the
   pre-registered 8.33%/3.42%); `problem_type`/loss wiring correct per arm.
4. VRAM fit: the pre-registered 8×2 effective-batch-16 config peaked at 8013/8192 MiB real GPU
   memory (2.2% headroom) — a genuine near-ceiling risk. Switched to 4×4 (same effective batch)
   after a pace check showed it was both safer (6391 MiB, 22% headroom) and faster (1.28s/step vs
   1.93s/step). Used for all six training runs below.

## Runs and mechanism

**ARM 2 / k8s, unweighted BCE (56.3% top-3):** run first per the pre-registered order (k8s has
30.4% multi-true-label collapse vs vscode's 8.0%, so most room for the supervision fix to show an
effect). Training curve was clean — no NaNs, no instability, loss fell monotonically for all 5
epochs (train loss 0.70→0.12, eval loss 0.150→0.122 tracking it down). The failure is not a bug.

**Diagnosis:** unweighted `BCEWithLogitsLoss` over 35 sparse classes (median raw pos_weight
~61×, max ~189×, no rebalancing) lets the model minimize loss by staying uniformly
under-confident rather than learning sharp per-class ranking — eval-loss 0.122 was only modestly
below the ~0.16–0.17 loss of a trivial "always predict base rate" classifier. Added a
`WeightedBCETrainer` (custom `compute_loss` override — HF's `multi_label_classification`
problem_type has no built-in `pos_weight` support) and reran with per-class pos_weight capped at
10×.

**Confirmed, partially:** pos_weight recovered +9.8pp top-3 on k8s (56.3%→66.1%) and nearly
doubled macro-F1 (0.047→0.088). The imbalance mechanism was real. But the corrected arm is still
21.0pp below baseline with CIs not overlapping, and tail-class recall stayed at exactly 0.000 —
pos_weight reweights the loss, it cannot manufacture signal from 8–12 examples. That remainder is
a data-scale ceiling, not a tunable hyperparameter.

**ARM 1 (single-label, matches baseline's own supervision) lost by a similar margin on k8s
(66.8%, −20.3pp)** as the pos_weight-corrected ARM 2 — architecture, not supervision, is the
dominant factor. Both arms did comparably better on vscode (ARM1 75.9%, ARM2-posw10 72.7%),
consistent with vscode's much lower label collapse giving the supervision fix less room to matter
either way.

**Comparison to the prior DistilBERT result** (Phase A/pre-registered): DistilBERT — a smaller
model (66M vs DeBERTa-v3-base's 184M params) — lost by a *narrower* margin on both repos (vscode
88.2% vs 90.4%, −2.2pp; k8s 74.5% vs 82.5%, −8pp) than DeBERTa did here. A larger, architecturally
stronger transformer underperforming a smaller one on the same small dataset (1488–2284 train
issues) is consistent with the data-scale-limited read: more parameters without more data does not
help, and may hurt, at this training-set size.

## Cost

All local GPU, six training runs (k8s unweighted, k8s posw10, vscode posw10, k8s single, vscode
single, plus the posw10 diagnostic already counted): ~51 min total GPU time, $0.

## Recommendation

Do not ship. TF-IDF+LR remains champion for this deployment. If component-classification quality
becomes a renewed priority, the next lever is more labeled data for tail classes (8–13 examples
each is the binding constraint, not model architecture) — not a bigger or better-tuned
transformer. No production change made; models saved locally under `data/models/deberta_*`
(gitignored, not committed) for reference only.
