# vscode eval expansion, 2026-10-12 (v2): 64 new issues, real labels first

Zero Groq. Local Ollama judges only. `eval/eval_set.jsonl` and the gold set are NOT modified: this directory is a
candidate set for the owner to decide on. Produced by `scripts/vscode_gold_expansion.py` (seed 42, resumable).

**Label status, read this first.** 63 of the 64 issues carry their REAL repo component label, which is the ground truth.
Exactly 1 issue (#175206) has no repo label; its label (`settings-sync`) is an **LLM-consensus** label (qwen3:8b and
gemma2:9b agree), NOT human ground truth. The judges' labels on the other 63 are a cross-check only, and the cross-check
says the local judges are not good enough to label unlabelled vscode issues (below).

## How many labelled issues exist (owner question 3a)

| Held-out temporal window (val + test) | issues |
|---|---|
| all | 2,449 |
| after excluding every train set (regenerated temporal + classifier, and the 4,923-row train SHIPPED to production), the gold set, the eval set, earlier expansion candidates | 2,246 |
| with a repo component label | **68** |
| ... label inside the served 47-class space | **63** (used as ground truth) |
| ... label outside the 47 classes | 5 (not usable) |
| without a label | 2,178 |

The sample is all 63 + 1 unlabelled issue (drawn with `10_curate_triage_gold.py`'s own `stratified_sample`). Resolution
buckets are skewed because labels sit on older, faster issues: <7 d 42, 7-30 d 11, >30 d 11. Vocabulary (3b): the SERVED
classifier's 47 classes, read from the production artifact after its sha256 (`5e286b1f...`) matched `data/models/MANIFEST.sha256`
(the local `data/models` copy is the old 28-class classifier). `vocabulary.json`, `vocabulary_provenance.json`.

## Judges and calibration (3c)

Three families, blind (title, body up to 2,000 chars and the allowed list only), temperature 0, seed 42, JSON-schema enum over
the 47 classes. Calibration on LABELLED issues outside the sample: EASY = the component name appears in the title (30 items),
HARD = it does not (40 items). Pre-registered rule (before any run): a judge counts toward the consensus if easy accuracy >= 0.80.

| Judge (family) | EASY accuracy (Wilson 95%) | HARD accuracy | Accuracy vs the 63 real labels | In consensus |
|---|---|---|---|---|
| qwen3:8b (Alibaba Qwen) | 29/30 = 96.7% (83.3-99.4) | 17/40 = 42.5% (28.5-57.8) | 33/63 = 52.4% (40.3-64.2) | yes |
| gemma2:9b (Google Gemma) | 24/30 = 80.0% (62.7-90.5) | 13/40 = 32.5% (20.1-48.0) | 28/63 = 44.4% (32.9-56.7) | yes (exactly at the line) |
| llama3.1:8b (Meta Llama) | 17/30 = 56.7% (39.2-72.6) | 15/40 = 37.5% (24.2-53.0) | 25/63 = 39.7% (28.5-52.0) | **no, failed the easy rule** |

The easy set measures only the easiest cases (96.7% for qwen); on the hard set every judge is below 43%. Because llama failed,
the consensus panel has two judges.

## Agreement (3d)

- Fleiss' kappa, all three judges (reference only; llama is not in the consensus): **0.620** (item-bootstrap 95% CI 0.513-0.716) over 64 items; 0.613 (0.511-0.706) over the 63 real-labelled items.
- Consensus panel (qwen3:8b + gemma2:9b): Cohen's kappa 0.683; both give the same label on 70.3% of items.
- Judges agree with each other far more than with the truth (kappa 0.62 against 40-52% accuracy): they share a reading of the text that
  differs from the repo's labelling convention.
- Where they miss: the real label `error-telemetry` is 14 of the 63; qwen gets 3/14 and gemma 1/14 (they say `debug` or `editor-core`).
  On the other classes they are mostly right (accessibility 6/7 and 6/7, extensions 6/8 and 5/8, keybindings 3/4 and 4/4); `javascript` 0/3 for both.
- Consensus vs real label on the 63: the two consensus judges agree with each other on 44 and with the real label on 26/63 = 41.3% (Wilson 30.0-53.6%); they disagree with each other on 19 labelled items (no consensus).

## Disputed items

`disputed.jsonl` (37 rows): the 63-item cross-check cases where the consensus is missing or differs from the real label, with each judge's answer and the
real label. There are no unlabelled items in dispute (the single unlabelled item has a 2-of-2 consensus).

## What this means

- The 63 real-labelled issues are usable as extra vscode eval ground truth as they are. They are NOT a representative vscode sample (older, faster, 22% `error-telemetry`).
- Local 8B LLM consensus is NOT a substitute for labels on this task: on real labelled issues it is right 41% of the time. Do not label the 2,178 unlabelled window issues this way.
- Not done and not decided: adding these rows to `eval/eval_set.jsonl` (changes the eval set and the frozen retrieval/cassette lineage; needs a recording against Groq, deliberately not started).

## Files

`sample.jsonl`, `sample_counts.json`, `calibration_easy.jsonl`, `calibration_hard.jsonl`, `calibrate_{easy,hard}_<judge>.jsonl`, `label_<judge>.jsonl`
(raw judge outputs; `mode` records whether the GPU model was shared with another session), `final_labels.jsonl`, `disputed.jsonl`, `agreement_report.json`.
The superseded first attempt (36-class vocabulary, easy-only calibration) is kept locally in `_superseded_v1_36class/` and is not committed.
