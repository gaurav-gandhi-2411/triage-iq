# TriageIQ — Project State & Session Handoff

**Last updated:** 2026-05-31  
**Maintainer:** Gaurav Gandhi  
**Purpose:** Single source of truth for picking up this project in a fresh consultant session. Read top-to-bottom in order; each section is self-contained.

---

## 1. What TriageIQ is

TriageIQ accepts a raw GitHub issue (repo slug, title, body) and returns a structured JSON `TriagePlan` in under 4 seconds. The plan includes: predicted component, component confidence, top-5 similar historical issues, expected resolution window (days p10/p50/p90), priority assessment, suggested assignee class, next steps, and a one-paragraph triage summary. The pipeline has four systems in sequence: (1) TF-IDF logistic-regression component classifier (~5ms), (2) BGE-base-en-v1.5 + FAISS cosine similar-issue retriever (~27ms), (3) LightGBM quantile-regression resolution-time predictor (~4ms), and (4) Groq `llama-3.1-8b-instant` LLM synthesis (~3s p50). It is trained on ~22K real issues from `microsoft/vscode` and `kubernetes/kubernetes`.

**Repos:**
- `triage-iq` — Python/FastAPI backend, ML models, Cloud Run. This repo.
- `triage-iq-ui` — React 19/Vite SPA. Separate repo, hosted on Vercel.

**Live endpoints:**
- API: `https://triageiq-api-779563952988.us-central1.run.app`
- UI: Vercel (separate repo — check `gaurav-gandhi-2411/triage-iq-ui`)
- `/health` returns `{"status":"ok","repos_loaded":[...],"retrievers":{"microsoft/vscode":"finetuned"|"baseline",...}}`

**Rate limits:** 10 req/hr, 30/day per IP on `/triage`. `/`, `/health`, `/metrics` unrated.

**Free-tier constraint:** All LLM calls route through Groq free tier. Triage LLM (`8b-instant`) and judge LLM (`70b-versatile`) share the same Tokens-Per-Day pool (~100K tokens/day for 70b). Cohere Trial key caps at 1,000 judge calls/month — the scarce resource gating all judge evals. Budget every Cohere call deliberately.

**Portfolio intent:** Demonstrate a complete production ML lifecycle (evaluation harness, reproducible builds, Prometheus metrics, Workload Identity Federation CI/CD, CVE-audited deps) to support senior IC job applications at Microsoft/Google/similar Applied ML roles.

---

## 2. The operating model

**Roles:**
- **GG** — data-science consultant (Gaurav Gandhi). Makes final decisions and approvals. Provides domain context. Writes prompts when engaging external consultant. Does not execute shell/git/file operations directly during sessions — everything goes through CC.
- **CC (Claude Code)** — executes all code changes, git operations, file writes, and gcloud commands.
- **External ML consultant** — Principal ML Engineer perspective, relayed to CC via GG. Reviews results, writes session-handoff prompts, diagnoses failure modes.

**Standing operating rules (non-negotiable, never override):**
1. Shell, git, file reads/writes, and gcloud commands all go through CC tools. Never suggest GG run a command in a separate terminal unless it requires interactive auth (in which case use `! <command>` in the CC prompt).
2. `ANTHROPIC_API_KEY` is never set, referenced, or used anywhere in this project.
3. All train/val/test splits for time-series or resolution-time models use `created_at` as the sort key, never `closed_at`. ADR-0009 documents why (closed_at causes systematic distribution shift that makes test metrics meaningless).
4. Verification-before-merge discipline: every model result is verified for eval/train overlap, baseline-protocol consistency, and overfit evidence before a PR is opened. Pre-merge contamination was caught in W3 (+26pp → +13pp correction). This is a feature, not a process tax.
5. Clean negative results are valid and valuable outcomes. W1.3 (reranker rejected) and the W4 vscode resolution predictor (0% improvement) are both documented as accepted decisions, not failures.
6. Never compare a fine-tuned/modified model metric against a baseline measured on a different query set or protocol. The delta is only meaningful if both numbers came from identical sampling.

---

## 3. Workstream ledger

| ID | Goal | Status | Headline result | ADR | PR/branch |
|---|---|---|---|---|---|
| W1.1 | Cross-family judge validation | **DONE, merged** | Cohere Command A ≈ llama-70b on 6 judge dims; qwen3-32b validated as cross-family control | ADR-0002, ADR-0003 | main |
| W1.2 | Component classifier calibration | **DONE, merged** | Temperature scaling reduces ECE; T<1 underconfidence diagnosed (class imbalance, not model defect) | ADR-0004 | main |
| W1.3 | Cross-encoder reranker | **REJECTED (clean negative), merged** | n=100 showed +6pp k8s but CI crossed zero at n=300 robustness check — false positive. No CE candidate improves both repos. | ADR-0006 | main (rejection merged) |
| W2.A | LLM response cache | **DONE, merged** | Opt-in SQLite cache; Prometheus hit/miss counters; 50%+ latency reduction on repeated requests | ADR-0005 | main |
| W3 reframe | Task reframe: dup detection → similar-issue retrieval | **DONE, merged** | Gold dataset reinterpreted; pipeline unchanged; ADR-0008 corrects task framing for all downstream workstreams | ADR-0008 | main |
| W3 fine-tune | BGE bi-encoder in-domain fine-tune | **CODE COMPLETE, PR #7 OPEN** | +13.16pp R@5 k8s [CI +6.58, +19.74]; +13.33pp R@5 vsc [CI +5.00, +23.33] on clean test split. Loader wired. | ADR-0010 (on PR #7 branch only) | `feat/w3-finetune` |
| W4 | Resolution predictor: fix split + remove leakage | **DONE, merged** | Corrected closed_at→created_at split; removed temporal feature leakage. k8s: +2.1% vs naive (not +3.3%). vscode: 0% improvement — naive baseline cannot be beaten with creation-time features on this corpus. | ADR-0009 | main |
| W5 | Gold eval set expansion: n=60 → n=150 | **PREP DONE, PR #8 OPEN** | 120-candidate pool generated; ingestion pipeline + 33 tests ready; awaiting GG labeling (~2–3h) | ADR-0011 (on PR #8 branch only) | `feat/w5-eval-expansion` |

### Open PRs in detail

**PR #7 — `feat/w3-finetune`** ([github.com/gaurav-gandhi-2411/triage-iq/pull/7](https://github.com/gaurav-gandhi-2411/triage-iq/pull/7))

Blocked on: Cohere judge eval confirming `similar_issues_relevance` holds (≥ 2.87/3) with the fine-tuned retriever active.

What's in it:
- Fine-tuned BAAI/bge-base-en-v1.5 model at `data/models/bge_finetuned_combined/`
- `loader.py` prefers `bge_finetuned_{alias}_index` over `dup_index_*_bge` when present
- `SimilarIssueRetriever.source` attribute ("finetuned" | "baseline") visible in `/health` response
- `assert_eval_disjoint_from_train()` guard in `w3_t5_eval.py` — fails loudly on any future eval/train overlap
- 5 loader branch tests (75 tests total on that branch)
- ADR-0010 with correction note: original eval was 66–71% contaminated with training pairs (+26pp → corrected +13pp)

**PR #8 — `feat/w5-eval-expansion`** ([github.com/gaurav-gandhi-2411/triage-iq/pull/8](https://github.com/gaurav-gandhi-2411/triage-iq/pull/8))

Blocked on: GG labeling `data/gold_expansion_candidates.csv` (marks ~90 of 120 candidates accept/reject; ~2–3h).

What's in it:
- T1 audit: current n=60 gold set gaps (era bias 95% from 2014–2016, coarse ">30d" bucket, component concentration)
- T2 stratification plan: +45 per repo, 9 × 5 buckets (hours/days/weeks/months/long)
- T3 candidate pool: `data/gold_expansion_candidates.csv` — 120 candidates with TF-IDF top-3 + BGE top-3 pre-computed (offline, no API)
- T4 labeling protocol: `docs/eval/gold_labeling_protocol.md` — exact column spec, acceptance criteria, rejection codes
- T5 ingestion script: `scripts/w5_ingest_labeled.py` — validates contract, merges into canonical gold, dry-run by default
- 33 ingestion tests (102 tests total on that branch)
- ADR-0011

---

## 4. The critical path right now

GG must label first. Everything else is blocked.

### Step 0 — GG labels (unblocks everything)

Open `data/gold_expansion_candidates.csv`. For each row, fill exactly four columns:
- `label_decision`: `accept` or `reject`  
- `label_rejection_code`: one of `empty-body | bot | non-english | mislabeled | trivial | duplicate-theme | wontfix | other` (required on rejects)
- `corrected_component`: fill if the `component` column is wrong (optional)
- `labeler_notes`: free text (optional)

Save as `data/gold_expansion_candidates_labeled.csv`. Target: ~9 accepts per resolution bucket per repo (45 per repo, 90 total). The 12-per-bucket pool gives a 3-slot margin for rejects. See `docs/eval/gold_labeling_protocol.md` for the full rubric.

### Step 1 — Pre-flight: confirm fine-tuned model is active (W3 gate)

**PR #7 must be checked out or its changes applied** before the judge run. Verify:

```bash
curl http://localhost:PORT/health | python -m json.tool
```

`retrievers` must show `"finetuned"` for both repos. If either shows `"baseline"`, STOP — the loader is not finding the fine-tuned index. Debug with `tests/test_loader.py` before spending any Cohere quota.

### Step 2 — Ingest labels (dry-run first, then write)

```bash
# Dry-run: see composition vs strata targets
python scripts/w5_ingest_labeled.py \
  --labeled data/gold_expansion_candidates_labeled.csv

# Write when composition looks right
python scripts/w5_ingest_labeled.py \
  --labeled data/gold_expansion_candidates_labeled.csv \
  --write
```

### Step 3 — Generate LLM plans for new issues

```bash
python scripts/11_evaluate_triage.py \
  --repos microsoft/vscode kubernetes/kubernetes \
  --skip-judge \
  --n-samples 0 \
  --clear-checkpoint
```

### Step 4 — Run the n=150 Cohere judge (one run, two purposes)

This run simultaneously provides (a) W3's final gate and (b) the new n=150 baseline.

```bash
python scripts/11_evaluate_triage.py \
  --judge-provider cohere \
  --judge-model command-a-03-2025 \
  --output-file reports/triage_results_w5_n150_cohere.json \
  --judge-delay 6 \
  --skip-reliability \
  --clear-judge-checkpoint
```

**Decision criteria:**
- `similar_issues_relevance` vs baseline **2.87/3** (w4_cohere, baseline retriever):
  - Hold (≥ 2.87) or rise → merge PR #7 (W3 accepted)
  - Material drop → surface and investigate before merge
- The full scorecard (all 6 dims, both repos) becomes the new n=150 reference for all future comparisons — update ADR-0009 and ADR-0010/ADR-0011 before merging.

### Step 5 — Merge order

Merge PR #7 first (W3 retriever), then PR #8 (W5 eval infra). Both are independent of each other but the judge run requires PR #7's model to be active.

---

## 5. Known constraints and recurring gotchas

**Cohere Trial key (1000 calls/month):** The single hardest constraint on eval cadence. Every judge run costs 60 issues × 3 systems = 180 calls. At n=150, one run costs 150 × 3 = 450 calls — nearly half the monthly budget. Always check `/health source=finetuned` before starting; never waste quota on a misconfigured run. Quota resets monthly; check the Cohere dashboard if runs fail with 429s.

**Groq TPD (tokens per day):** The triage LLM (8b-instant) and judge LLM (70b-versatile) share one daily pool. 70b is ~5× more expensive per token than 8b. A 60-issue eval run with judge ≈ 200K 70b tokens — can hit the ceiling in one session. Batch with `--triage-delay 1.5` and `--judge-delay 6` (these are the hardcoded defaults). If TPD is hit mid-run, the checkpoint resumes from where it stopped the next day.

**Gemini free tier:** 20 requests-per-day (RPD) — not 1,500. Don't plan any Gemini judge runs without checking this.

**n=60 CI width:** At n=30 per repo, the 95% CI on any proportion metric is ±18 pp. This means you cannot detect deltas smaller than ~7 pp at p=0.05, and most reported improvements are technically not significant at n=30. This is why W5 exists. Until n=150 is live, all judge deltas are directional indicators, not statistical proof.

**Priority dimension unreliability:** `gold_priority` is inferred from resolution speed (< 24h → high, < 7d → medium, else → low), not from explicit GitHub labels (vscode has no priority labels; k8s has sparse priority/critical coverage). Even at n=150, the priority judge dimension is directional at best. Don't report priority improvements as headline metrics.

**Baseline-instability lesson (from W3):** Never compare a post-training model metric against a baseline measured on a different sample. The W3 original eval drew its "baseline" from the full gold corpus (including training pairs) and hardcoded a full-corpus R@5 as the baseline — the delta measured on a contaminated sample against a different-protocol baseline. Rule: baseline and fine-tuned must be measured on identical query sets, in the same run.

**The W3 eval contamination:** The original T5 eval script used `sample_gold()` which sampled from the full gold corpus. Since the training split was ~70% of gold, ~70% of the eval sample was training data. The inflated result (+26pp k8s) was caught in pre-merge verification and corrected to +13pp on the clean test split. The `assert_eval_disjoint_from_train()` guard in the fixed T5 script makes this impossible to reintroduce silently.

**Checkpoint resume:** All eval scripts write to `data/triage_eval_checkpoint.jsonl` and `data/judge_scores_checkpoint_{model}.jsonl`. If a run is interrupted, re-running picks up from the checkpoint. Use `--clear-checkpoint` and/or `--clear-judge-checkpoint` when you need a fresh run. The checkpoint is NOT output-file-scoped — if you change `--output-file`, the old checkpoint is still loaded unless cleared.

---

## 6. Open backlog (not yet started)

**W2.B — Migrate triage LLM:** Replace `llama-3.1-8b-instant` with `openai/gpt-oss-20b` on Groq. Server-side prompt caching on Groq means cached tokens do not count toward TPD — the structural fix for recurring TPD walls on long evals. Likely a quality upgrade as well. Requires re-running the full eval suite to establish a new baseline. Moderate scope (~1 session).

**GitHub issue #3 — Artifact rename:** Rename `dup_index_*` directories to `similar_issue_index_*` to match the ADR-0008 task-reframing vocabulary. The loader currently has a `TODO(#3)` comment pointing at this. Low priority but reduces confusion in new sessions.

**GitHub issue #5 — UI type drift fix:** Surface `resolution_bucket` and `resolution_confidence_pct` fields in the React UI (currently returned by the API but not displayed). Run `openapi-typescript` codegen to regenerate types from the FastAPI OpenAPI spec — closes the type-drift risk ADR-0001 documented. Scope: 1 session in the UI repo.

**llama-70b W1.2 retrofit:** The `data/judge_scores_checkpoint_llama_3_3_70b_versatile.jsonl` checkpoint has 17/180 scores from a 2026-05-20 session that hit TPD. Once Groq TPD resets and the W2.B migration (prompt caching) is in place, re-running this is cheap — cached responses serve at ~0 token cost. Resume with: `python scripts/11_evaluate_triage.py --judge-model llama-3.3-70b-versatile --output-file reports/triage_results_llama70b_retrofit.json`

---

## 7. Key file map

| What | Where |
|---|---|
| Canonical gold eval set | `data/gold_triage_plans.parquet` (60 issues; n=150 after W5 ingestion) |
| W5 labeling worklist | `data/gold_expansion_candidates.csv` (120 candidates; GG fills `label_decision` etc.) |
| Labeling protocol | `docs/eval/gold_labeling_protocol.md` |
| W5 ingestion script | `scripts/w5_ingest_labeled.py` |
| Gold curation script | `scripts/10_curate_triage_gold.py` |
| Full eval + judge script | `scripts/11_evaluate_triage.py` |
| ADRs | `docs/architecture/adr/0001–0011` |
| Data card (biases) | `reports/01_data_card.md` |
| LLM response cache | `data/llm_cache.sqlite` (path from `config.llm_cache_path`) |
| API loader | `src/triage_iq/api/loader.py` — prefers fine-tuned index (`bge_finetuned_{alias}_index/`) over baseline (`dup_index_{slug}_bge/`) |
| Similar-issue retriever | `src/triage_iq/models/similar_issues.py` — `.source` attribute ("finetuned" or "baseline") |
| Fine-tuned model | `data/models/bge_finetuned_combined/` (SentenceTransformer) |
| Fine-tuned FAISS indexes | `data/models/bge_finetuned_k8s_index/`, `data/models/bge_finetuned_vsc_index/` |
| Baseline FAISS indexes | `data/models/dup_index_kubernetes_kubernetes_bge/`, `dup_index_microsoft_vscode_bge/` |
| Eval checkpoints | `data/triage_eval_checkpoint.jsonl`, `data/judge_scores_checkpoint_{model}.jsonl` |
| Current judge baseline | `reports/triage_results_w4_cohere.json` — `similar_issues_relevance: 2.87/3` |
| Verified W3 R@5 results | `reports/w3_corrected_eval_results.json` — corrected test-split results |
| W5 gold audit | `reports/w5_gold_audit.json` |
| Tests | `tests/` — 69 tests on main; 75 on feat/w3-finetune; 102 on feat/w5-eval-expansion |

---

## 8. ADR index

| ADR | Title | Status | Branch |
|---|---|---|---|
| 0001 | Keep triage-iq and triage-iq-ui as separate repositories | Accepted | main |
| 0002 | Use qwen3-32b as cross-family judge for eval validation | Accepted | main |
| 0003 | Judge default — llama-3.3-70b-versatile retained | Accepted | main |
| 0004 | Temperature scaling for component classifier confidence calibration | Accepted | main |
| 0005 | Opt-in SQLite LLM response cache with Prometheus observability | Accepted | main |
| 0006 | Cross-encoder reranker for similar-issue retrieval | **Rejected** (clean negative — n=300 robustness check; CI crosses zero) | main |
| 0007 | *(not used — number skipped)* | — | — |
| 0008 | Task reframing: "duplicate detection" → "similar-issue retrieval" | Accepted | main |
| 0009 | Resolution-time predictor diagnosis: split fix + feature leakage removal | Accepted | main |
| 0010 | W3: Fine-tune BGE bi-encoder (+13pp R@5, corrected from inflated +26pp) | Accepted (on branch; pending merge) | feat/w3-finetune only |
| 0011 | Expand gold eval set n=60 → n=150 (W5 stratification plan) | In progress (labeling phase) | feat/w5-eval-expansion only |

To read any ADR in full: `docs/architecture/adr/NNNN-kebab-title.md`. ADRs 0010 and 0011 are visible on their respective PR branches only until those PRs are merged.
