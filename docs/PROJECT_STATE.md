# TriageIQ — Project State

**Last updated:** 2026-05-31  
**Maintainer:** Gaurav Gandhi  
**Workflow model:** `docs/ORCHESTRATION.md`

---

## 1. What TriageIQ is

TriageIQ turns a raw GitHub issue (repo, title, body) into a structured `TriagePlan` JSON in under 4 seconds. The pipeline has four systems in sequence:

| # | System | Impl | Latency | Key metric |
|---|---|---|---|---|
| 1 | Component classifier | TF-IDF logistic regression, 28–35 classes | ~5 ms | vscode 69% acc, 0.585 macro-F1 |
| 2 | Similar-issue retriever | BGE-base-en-v1.5 + FAISS cosine | ~27 ms | vscode R@5 0.367 baseline → 0.683 W3 fine-tuned (test split) |
| 3 | Resolution predictor | LightGBM quantile regression, 93 feats | ~4 ms | k8s +2.1% vs naive; vscode 0% (naive wins) |
| 4 | LLM synthesis | Groq `llama-3.1-8b-instant`, 3-shot, T=0 | ~3 s | Judge 10.75/15, similar_issues_relevance 2.87/3 |

Trained on ~22K real issues from `microsoft/vscode` and `kubernetes/kubernetes`.

**Repos:**
- `triage-iq` — Python/FastAPI backend, ML models, Cloud Run. This repo.
- `triage-iq-ui` — React 19/Vite SPA. Separate repo. (`gaurav-gandhi-2411/triage-iq-ui`)

**Live endpoints:**
- API: `https://triageiq-api-779563952988.us-central1.run.app`
- UI: Vercel (see triage-iq-ui repo)
- `/health` returns `retrievers: {"microsoft/vscode": "finetuned"|"baseline", ...}`

**Free-tier constraint:** Groq free tier for all LLM calls. The 8b triage LLM and 70b judge LLM share one daily token pool (~100K tokens/day for 70b). Cohere Trial caps at 1,000 judge calls/month — the binding scarce resource. Budget every Cohere call; one bad run wastes ~450 of 1,000.

**Portfolio intent:** Demonstrate a complete production ML lifecycle (eval harness, reproducible builds, Prometheus metrics, Workload Identity Federation CI/CD) for senior IC roles at Microsoft/Google.

---

## 2. Workstream ledger

| ID | Goal | Status | Headline result | ADR | Merge |
|---|---|---|---|---|---|
| W1.1 | Cross-family judge validation | DONE | Cohere Command A ≈ llama-70b; qwen3-32b confirmed as cross-family control | 0002, 0003 | main |
| W1.2 | Component classifier calibration | DONE | Temperature scaling reduces ECE; T<1 underconfidence diagnosed (class imbalance, not defect) | 0004 | main |
| W1.3 | Cross-encoder reranker | REJECTED (clean neg.) | n=100 +6pp k8s collapsed to noise at n=300; CI crossed zero — false positive | 0006 | main |
| W2.A | LLM response cache | DONE | Opt-in SQLite cache; Prometheus hit/miss; 50%+ latency on repeat requests | 0005 | main |
| W3-reframe | Task reframe: dup detection → similar-issue retrieval | DONE | Gold dataset reinterpreted; pipeline unchanged; corrects all downstream metrics | 0008 | main |
| W3-finetune | BGE bi-encoder in-domain fine-tune | **REBASING** | Prior branch result (+13.16 pp k8s / +13.33 pp vsc) unverified on current main — weights never committed; retraining + re-eval in progress on `feat/w3-finetune-rebased` | 0016 (rebased, was 0010 on stale branch) | `feat/w3-finetune-rebased` |
| W4 | Resolution predictor: fix split + remove leakage | DONE | closed_at → created_at; removed triage-time feature leakage. k8s +2.1% vs naive; vscode 0% | 0009 | main |
| W5 | Gold eval set expansion: n=60 → n=150 | **PR #8 OPEN** | 120-candidate pool generated; ingestion + tests ready; awaiting GG labeling | 0011 (branch) | `feat/w5-eval-expansion` |

### PR #7 — `feat/w3-finetune` (superseded by `feat/w3-finetune-rebased`, not yet a PR)

**Status:** PR #7 is 47 commits stale (rooted before CQR/eval-gate/drift-guard/grounding landed) with failing CI from 2026-05-31 — not being merged as-is. `feat/w3-finetune-rebased` ports the four T2–T5 scripts and tracked mining/split data onto current main; the fine-tuned weights (never committed — `data/models/**` gitignored) are being retrained and re-evaluated from scratch. The old +13pp result is provenance only until re-verified.

**Blocked on:** re-established recall@k + bootstrap CI on current main (ship/no-ship gate), then Cohere judge confirming `similar_issues_relevance ≥ 2.87/3` with fine-tuned retriever active.

Will contain (once re-verified): fine-tuned model at `data/models/bge_finetuned_combined/`; loader preference for fine-tuned index; `SimilarIssueRetriever.source` ("finetuned"/"baseline") surfaced in `/health`; `assert_eval_disjoint_from_train()` guard; ADR-0016 with permanent correction note (original +26pp was 66–71% train/eval contaminated; corrected to +13pp on the stale branch — pending re-verification on current main).

### PR #8 — `feat/w5-eval-expansion`

**Blocked on:** GG labeling `data/gold_expansion_candidates.csv` (~2–3h; 120 candidates, target ~90 accepts).

Contains: T1 gold audit (`reports/w5_gold_audit.json`); T2 stratification plan (9 × 5 resolution buckets); T3 candidate pool with TF-IDF + BGE pre-computed offline; T4 labeling protocol (`docs/eval/gold_labeling_protocol.md`); T5 ingestion script (`scripts/w5_ingest_labeled.py`, dry-run by default); 33 ingestion tests (102 tests on that branch); ADR-0011.

---

## 3. Critical path right now

### Pre-condition: GG labels the candidate pool

Open `data/gold_expansion_candidates.csv`. Fill exactly these columns per row:

| Column | Required | Values |
|---|---|---|
| `label_decision` | **yes** | `accept` or `reject` |
| `label_rejection_code` | yes (if reject) | `empty-body` / `bot` / `non-english` / `mislabeled` / `trivial` / `duplicate-theme` / `wontfix` / `other` |
| `corrected_component` | optional | correct value if the `component` column is wrong |
| `labeler_notes` | optional | free text |

Save as `data/gold_expansion_candidates_labeled.csv`. Target: ~9 accepts per resolution bucket per repo (45/repo). The 12-per-bucket pool gives 3 slots of margin. See `docs/eval/gold_labeling_protocol.md` for the rubric.

### T3a gate: confirm fine-tuned model is live (must pass before any Cohere spend)

PR #7 must be checked out or its changes applied. Then:

```bash
curl http://localhost:PORT/health | python -m json.tool
# retrievers must show "finetuned" for BOTH repos
# If either shows "baseline" → STOP, debug loader before spending quota
```

### Runbook (run in order)

```bash
# 1. Ingest labels (dry-run first, then write)
python scripts/w5_ingest_labeled.py \
  --labeled data/gold_expansion_candidates_labeled.csv
# review composition vs strata targets, then:
python scripts/w5_ingest_labeled.py \
  --labeled data/gold_expansion_candidates_labeled.csv \
  --write

# 2. Generate LLM plans for new issues (cache handles repeats)
python scripts/11_evaluate_triage.py \
  --repos microsoft/vscode kubernetes/kubernetes \
  --skip-judge --n-samples 0 --clear-checkpoint

# 3. Run the n=150 Cohere judge — one run, two purposes
python scripts/11_evaluate_triage.py \
  --judge-provider cohere --judge-model command-a-03-2025 \
  --output-file reports/triage_results_w5_n150_cohere.json \
  --judge-delay 6 --skip-reliability --clear-judge-checkpoint
```

**Decision on `similar_issues_relevance` vs baseline 2.87/3:**
- Hold or rise → merge PR #7 (W3 accepted). Record delta in ADR-0016.
- Material drop → surface, investigate before merge.

The same run's full scorecard becomes the n=150 baseline for all future workstreams. Update ADR-0011 before merging PR #8.

**Merge order:** PR #7 first (W3 retriever), then PR #8 (W5 eval infra).

---

## 4. Constraints and recurring gotchas

**Cohere Trial (1,000 calls/month):** The binding constraint on eval cadence. One n=150 judge run = 150 × 3 systems = 450 calls — nearly half the monthly budget. Always verify `/health source=finetuned` before starting. The W3 judge run was wasted because the process launched before the loader change took effect. Never run the judge against a misconfigured model.

**Groq TPD (tokens/day):** The 8b triage LLM and 70b judge share one daily pool. 70b is ~5× more expensive per token. A 60-issue eval with judge ≈ 200K 70b tokens. If TPD is hit mid-run, the JSONL checkpoint resumes where it stopped — re-run the same command the next day. Default delays (1.5s triage, 6s judge) are calibrated for free tier.

**Gemini free tier:** 20 requests-per-day (RPD), not 1,500. Don't schedule any Gemini-backed evals without checking this.

**Checkpoint scoping:** `data/triage_eval_checkpoint.jsonl` and `data/judge_scores_checkpoint_{model}.jsonl` are NOT scoped to the `--output-file`. Changing the output file does NOT reset checkpoints. Use `--clear-checkpoint` and/or `--clear-judge-checkpoint` explicitly when a fresh run is needed.

**n=60 CI width:** At n=30/repo, the 95% CI on any proportion metric is ±18 pp. Deltas smaller than ~7 pp cannot be detected at p=0.05. All current judge scores are directional until n=150 is live. This is why W5 exists.

**Priority dimension:** `gold_priority` is inferred from resolution speed (< 24h → high, < 7d → medium, else → low) because vscode has no explicit priority labels and k8s coverage is sparse. Even at n=150, priority scores are directional only. Do not report priority improvements as headline metrics.

**Baseline-instability lesson:** Never compare a fine-tuned/modified model metric against a baseline measured on a different sample or protocol. The delta is only valid when baseline and model are measured on identical query sets in the same run. Two incidents:
1. W3 original eval: hardcoded full-corpus W1.1 baselines vs contaminated n=100 fine-tuned sample — two different protocols.
2. W1.3: n=100 baseline (0.430/0.470) was a favorable draw vs full-corpus (0.410/0.367) — small-N sampling variance masqueraded as a real signal until n=300 robustness.

**The W3 eval contamination (canonical example):** The original T5 eval script used `sample_gold()` which drew from the full gold corpus. Since ~70% of gold was in the training split, ~70% of the eval sample was training data — inflating the apparent result by ~2×. Caught in pre-merge verification; corrected from +26pp to +13pp. The `assert_eval_disjoint_from_train()` guard in the fixed T5 script makes silent reintroduction impossible.

---

## 5. Open backlog

**W2.B — Migrate triage LLM to `openai/gpt-oss-20b` on Groq:** Server-side prompt caching on Groq means cached tokens do NOT count toward TPD — the structural fix for recurring daily quota walls. Likely a quality upgrade. Requires re-running the full eval suite (one full judge run) to establish a new baseline. ~1 session.

**GitHub #3 — Artifact rename:** Rename `data/models/dup_index_*` directories to `similar_issue_index_*` to match the ADR-0008 vocabulary. Loader has a `TODO(#3)` comment. Low effort, reduces confusion in new sessions.

**GitHub #5 — UI type drift closure:** Surface `resolution_bucket` and `resolution_confidence_pct` in the React UI (returned by API, not yet displayed). Wire `openapi-typescript` codegen to auto-generate TypeScript types from the FastAPI OpenAPI spec — closes the type-drift risk ADR-0001 documented. Scope: 1 session in the UI repo.

**llama-70b W1.2 retrofit (opportunistic):** `data/judge_scores_checkpoint_llama_3_3_70b_versatile.jsonl` has 17/180 scored from a 2026-05-20 session that hit TPD. After W2.B (prompt caching), the remaining 163 calls will largely serve from cache at near-zero token cost. Resume with: `python scripts/11_evaluate_triage.py --judge-model llama-3.3-70b-versatile --output-file reports/triage_results_llama70b_retrofit.json`

---

## 6. File map and ADR index

### Key files

| What | Where |
|---|---|
| Canonical gold eval set | `data/gold_triage_plans.parquet` (60 issues; n=150 post-W5 ingest) |
| W5 labeling worklist | `data/gold_expansion_candidates.csv` |
| Labeling protocol | `docs/eval/gold_labeling_protocol.md` |
| W5 ingestion script | `scripts/w5_ingest_labeled.py` |
| Gold curation script | `scripts/10_curate_triage_gold.py` |
| Full eval + judge script | `scripts/11_evaluate_triage.py` |
| Workflow contract | `docs/ORCHESTRATION.md` |
| Data card (biases) | `reports/01_data_card.md` |
| LLM cache | `data/llm_cache.sqlite` |
| API loader | `src/triage_iq/api/loader.py` (prefers fine-tuned index over baseline) |
| Similar-issue retriever | `src/triage_iq/models/similar_issues.py` (`.source` attribute) |
| Fine-tuned model | `data/models/bge_finetuned_combined/` |
| Fine-tuned FAISS | `data/models/bge_finetuned_k8s_index/`, `bge_finetuned_vsc_index/` |
| Baseline FAISS | `data/models/dup_index_kubernetes_kubernetes_bge/`, `dup_index_microsoft_vscode_bge/` |
| Eval checkpoints | `data/triage_eval_checkpoint.jsonl`, `data/judge_scores_checkpoint_{model}.jsonl` |
| Current judge baseline | `reports/triage_results_w4_cohere.json` — `similar_issues_relevance: 2.87/3` |
| W3 corrected results | `reports/w3_corrected_eval_results.json` |
| W5 gold audit | `reports/w5_gold_audit.json` |
| Tests | `tests/` — 69 on main; 75 on feat/w3-finetune; 102 on feat/w5-eval-expansion |

### ADR index

| ADR | Title | Status | Branch |
|---|---|---|---|
| 0001 | Keep triage-iq and triage-iq-ui as separate repositories | Accepted | main |
| 0002 | Use qwen3-32b as cross-family judge for eval validation | Accepted | main |
| 0003 | Judge default — llama-3.3-70b-versatile retained | Accepted | main |
| 0004 | Temperature scaling for component classifier confidence calibration | Accepted | main |
| 0005 | Opt-in SQLite LLM response cache with Prometheus observability | Accepted | main |
| 0006 | Cross-encoder reranker for similar-issue retrieval | **Rejected** — n=300 robustness check; CI crosses zero | main |
| 0007 | *(number skipped)* | — | — |
| 0008 | Task reframing: "duplicate detection" → "similar-issue retrieval" | Accepted | main |
| 0009 | Resolution-time predictor: split fix + feature leakage removal | Accepted | main |
| 0010 | W3: BGE bi-encoder fine-tune (+13pp R@5, corrected) | Accepted (pending merge) | feat/w3-finetune |
| 0011 | W5: Gold eval set expansion n=60 → n=150 | In progress | feat/w5-eval-expansion |
