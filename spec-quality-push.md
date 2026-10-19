# spec-quality-push.md — TriageIQ quality push (v1.0, approved by GG 2026-10-19)

## 1. Goal

Raise every TriageIQ stage to strong, defensible numbers on fresh data, and redesign the
system so the LLM explains verified ML evidence instead of overriding it.

### Pre-registered targets (temporal test split, 2026 issues, 95% CIs)

| Stage | Metric | Now | Target |
|---|---|---|---|
| Component classifier | Top-3 accuracy, per repo | 85.8% vscode / 86.0% k8s | **>= 95%** |
| Component classifier | Conformal set coverage at alpha=0.05 | n/a | **>= 95% guaranteed**, report mean set size (target <= 3) |
| Component classifier | Top-1 accuracy, macro-F1 | 70.0% / 54.8%; F1 0.52 / 0.43 | Maximize; report against the measured label-noise ceiling |
| Retrieval | Duplicate Recall@5, per repo | vscode 53.5% | **>= 80%** |
| Retrieval | Fix-linked Recall@5 (k8s redefined task, see 7.1) | n/a | Maximize; report ceiling |
| System component accuracy | Exact match vs gold, end to end | 68.8% (44/64) | **>= classifier top-1**, never below it |
| Priority | Accuracy / macro-F1 vs real priority labels | LLM guess, judge 0.36/1 | Beat majority class with CI excluding it |
| Resolution | Bucket accuracy vs majority; median AE vs naive median; interval coverage | k8s +6.95pp; median served | Beat naive on median AE; coverage 80% +/- 3pp |
| LLM plan | Judge mean (/15) + objective metrics (sec. 9) | 12.09 | No regression; objective metrics improve |

Top-1 >= 95% is NOT a target: multi-label issues and inconsistent maintainer labeling cap it.
Phase 0 measures that ceiling; all top-1 results are reported against it.

## 2. Hard constraints

- **Zero spend.** Free tiers and already-paid subscriptions only. Never buy credits or Colab units,
  never enable a billable GCP service. BigQuery/GH Archive queries are forbidden (paid billing
  account; overage risk).
- **Compute.**
  - Laptop (RTX 3070 8 GB, 32 GB RAM): the laptop shut down from overheating on 2026-10-19.
    Laptop GPU jobs must be short (< 10 min) and simple. No training on the laptop.
  - Training: Kaggle free GPUs (account gauravgandhi2411, token at C:\Users\gaura\.kaggle\kaggle.json)
    first; Colab (account gauravgandhi429@gmail.com, AI Pro units, max 2 concurrent sessions, never
    2 L4s at once) when Kaggle is insufficient. Stay within existing units.
  - Heavy data on D: (e.g. D:\triageiq-data). C: is short on space.
- **Groq.** Free tier, shared ledger at ~/.groq_budget/, 40K/day production reserve never touched.
  Prompt changes force cassette re-records: batch ALL prompt changes and re-record ONCE (sec. 10).
- **No manual labeling by GG.** Ground truth comes from repo labels, timeline events, duplicate
  markers and linked PRs. LLM-consensus labels only where no label exists, labeled as such.
- **Production.** Every new behavior ships behind a feature flag, default off, until its phase is
  accepted. Production deploys on merge to main; never break the live demo.
- **Serving budget (Cloud Run, 2 vCPU / 4 GiB, CPU only).** End-to-end warm p50 must stay <= 6 s;
  each added model <= 300 ms p50 on CPU (ONNX / int8 allowed); image and artifacts must fit 4 GiB
  memory; report cold-start impact.
- **Process.** Sequential: one task and one subagent at a time, at most one worktree. Gate-sized PRs.
  Remove worktrees after merge. Report disk footprint each session. ADR for every decision.
  VERIFIED/BELIEVED labeling on every claim.

## 3. Phase order (decided)

0 Foundation -> 3a Quick win (no training) -> 1 Classifier -> 2 Retrieval -> 3b System redesign ->
4 Resolution -> 5 Evaluation & final re-record -> 6 README (GG approval).

Each phase ends with a report, an ADR, and a pre-registered acceptance check. If a target is shown
infeasible by evidence (e.g. ceiling below target), stop that target, report the evidence and the
best achievable number, and continue.

## 4. Phase 0 — Foundation

### 4.1 Fresh data
- Scrape `microsoft/vscode` and `kubernetes/kubernetes` issues created **2023-01-01 to now** via the
  GitHub GraphQL/REST API using the existing `gh` authentication (never print the token).
  Respect rate limits; resumable checkpointed scraper; raw JSON to D:\triageiq-data\raw.
- Per issue: title, body, author association, created/closed timestamps, state, labels with the
  **full label timeline** (labeled/unlabeled events with timestamps and actors), milestone,
  comments (count + closing comment text), timeline events including `marked_as_duplicate`,
  `cross-referenced`, `closed` with closing commit/PR, `connected`.
- Keep pull requests separately (titles, bodies, linked issues, merged state) for fix linkage.
- Report counts per repo per year, label coverage, storage size.

### 4.2 Derived ground truth
- **Component labels**: final component labels in the 47-class taxonomies; refresh the taxonomy
  from fresh data (report added/dropped classes); keep a mapping to the old 47.
- **Duplicate pairs**: `marked_as_duplicate` events, vscode bot duplicate comments, k8s
  `/close` + "duplicate of #N", `*duplicate` labels with a referenced target.
- **Fix links**: issue -> merged PR that closed it; issues fixed by the same PR.
- **Priority**: k8s `priority/*` labels; for vscode, report what priority-like signal exists
  (`important`, milestone at triage); if none is reliable, say so and scope priority to k8s.
- **Resolution**: closed-at minus created-at; open issues kept as right-censored.

### 4.3 Splits
Temporal: train <= 2025-06-30, validation 2025-07-01..2025-12-31, test >= 2026-01-01.
Remove near-duplicate leakage across splits (same text hash or duplicate pairs crossing splits).
Freeze split manifests with hashes.

### 4.4 Ceiling analysis (before any modeling)
- Multi-label rate per repo; label churn (labels changed after first triage); disagreement between
  first and final component label. Compute top-1 and "any-valid-label" ceilings.
- Retrieval ceiling: share of duplicate / fix-linked pairs where the target exists in the index
  and shares meaningful text; report per task.
- Write `reports/ceiling_analysis_2026-10.json` and ADR. Targets that the ceiling rules out are
  reported as such.

### 4.5 Eval sets
- Classifier/retrieval/resolution: full temporal test split.
- LLM end-to-end set: stratified ~150 issues (75/repo) from the test split. Size is bounded by the
  Groq re-record budget (~6K tokens/issue; sec. 10).

**Acceptance:** data scraped and reported; ground truth derived with counts; splits frozen;
ceiling report merged; current production models re-scored on the new test split (the new
baseline row for every target).

## 5. Phase 3a — Quick win: classifier decides the component (no training)

- Behind a flag: the served `predicted_component` is the classifier's top-1; the LLM may choose
  another component ONLY from the classifier's top-3 AND only with a cited, verified reason
  (validated-override path); otherwise the classifier wins.
- Measure on recorded data (zero tokens): end-to-end component exact match with the flag on vs off,
  paired McNemar with CI, on the 64 + 63 recorded issues (n=127).
- The LLM summary must not contradict the served component: if the prompt changes, defer the
  prompt change to the single re-record (sec. 10); the flag can ship with post-hoc enforcement.

**Acceptance:** component exact match with flag on >= flag off, CI reported; flag enabled.

## 6. Phase 1 — Component classifier (target top-3 >= 95%, conformal 95%)

Train on fresh train split, select on validation, report on test. Candidates:
1. Retrained TF-IDF + one-vs-rest LR on fresh data (baseline).
2. **Linear/MLP head on the BGE embedding already computed for retrieval** (zero added latency).
3. **kNN neighbor-label vote** from the top-k retrieved issues' components.
4. Fine-tuned encoder (DeBERTa-v3-small/base or ModernBERT-base), multi-label BCE, on Kaggle/Colab;
   export ONNX int8 for CPU serving; must meet the 300 ms budget.
5. Feature engineering: file paths, setting keys, extension IDs, stack-trace frames, command IDs,
   issue-template type.
6. Stacked ensemble (LightGBM or LR) over 1–5.
7. k8s label hierarchy: map near-duplicate labels (e.g. provider/* vs cloudprovider) to a declared
   hierarchy; report both strict and hierarchy-aware accuracy; never silently relax the metric.
8. Calibration (temperature/isotonic), then **conformal prediction sets** (APS/RAPS) calibrated on
   validation at alpha=0.05; report test coverage and mean/median set size.

Report per repo: top-1, top-3, macro-F1, any-valid-label accuracy, ECE, conformal coverage and set
size, CPU latency, artifact size, cold-start impact. Ablations for each candidate.

**Acceptance:** top-3 >= 95% per repo OR the best achievable with evidence against the ceiling;
conformal coverage >= 95% on test; latency within budget. Ship behind a flag, then enable.

## 7. Phase 2 — Retrieval (target duplicate Recall@5 >= 80%)

### 7.1 Task definitions
- **Duplicate retrieval** (primary, both repos): query issue -> its marked duplicate target.
- **Fix-linked retrieval** (k8s replacement for "related"): issues fixed by the same PR.
- The old k8s "related" task is retired as a headline: Phase 0's ceiling analysis documents why.

### Methods
1. **Refresh the index** to include 2023–2026 issues (open and closed). Duplicate targets may be in
   the index, as in real use; queries are test-period issues only, and a query is never its own
   target.
2. Fine-tune BGE-base (CLS pooling, max_seq 512) with MultipleNegativesRanking + hard negatives
   mined from same-component non-duplicates, on train-period pairs only.
3. Hybrid: dense + BM25 with learned fusion weights, re-tested on the new task.
4. **Domain-tuned cross-encoder reranker** over top-30, trained on train pairs; distill/quantize to
   meet <= 300 ms CPU for the rerank stage.
5. Component-aware filtering/boosting using the classifier's conformal set.
6. Index size/memory: fp16 or PQ if needed to fit memory; report recall cost of compression.

Report R@1/5/10, MRR, per repo per task, with CIs; latency; memory.

**Acceptance:** duplicate R@5 >= 80% per repo or best achievable with evidence; fix-linked
reported; latency/memory within budget. Ship behind a flag, then enable.

## 8. Phase 3b — System redesign: "ML decides, LLM explains"

- **Component**: conformal set from Phase 1; LLM chooses within the set only with evidence.
- **Priority model**: trained on real priority labels (k8s; vscode only if Phase 0 finds a
  reliable signal). LLM explains, does not decide.
- **Next steps**: retrieved from how similar issues were actually resolved — closing comments and
  linked fix PR titles/descriptions of the top retrieved issues. Prompt cites them.
- **Resolution estimate**: kNN over similar issues' actual resolution times, combined with Phase 4.
- **Prompt**: structured verified evidence in, narrative and plan out. All prompt changes are
  finalized here and recorded once (sec. 10).
- Degraded mode keeps working with every new field.

**Acceptance:** end-to-end component accuracy >= classifier top-1; priority beats majority with CI;
objective next-step grounding metric (sec. 9) improves; no judge regression beyond noise.

## 9. Phase 4 — Resolution time

- Survival framing: right-censored open issues kept (AFT / discrete-time hazard, LightGBM or similar).
- Creation-time features only: author association/history, component, open-issue load in that
  component at creation, maintainer activity rate, days to next vscode monthly release, template type,
  kNN neighbor resolution times, text embedding.
- Compare against the naive median on median AE and mean AE, bucket accuracy vs majority, C-index,
  conformal interval coverage and width. Serve the learned point only if it beats the median on
  median AE with a CI excluding zero.

**Acceptance:** report with CIs; serve only what beats naive; coverage 80% +/- 3pp.

## 10. Phase 5 — Evaluation and the single re-record

- Objective metrics replace the judge wherever ground truth exists: component exact match,
  priority vs labels, similar-issue relevance vs duplicate/fix links, next-step grounding
  (share of next steps traceable to retrieved resolution evidence), resolution vs actual.
- Judge only for prose quality; add a second local judge from a different family; report agreement.
- **One** cassette re-record of the ~150-issue LLM set after Phase 3b's prompt is final:
  ~150 x 6K = ~900K tokens = ~6 days under the 160K/day ledger cap. Run unattended on Kaggle/Colab
  or the laptop only if thermals allow; the recorder is resumable.
- New baselines and gates are PROPOSED to GG, never written without approval.

## 11. Phase 6 — README and public numbers

Update README and docs with the new numbers through the numbers checker (strict). Queue for GG's
review; GG merges.

## 12. Risks

- Fresh data volume vs API rate limits (mitigation: GraphQL, checkpointed scraper, run over days).
- Fine-tuned models exceeding CPU latency/memory (mitigation: ONNX int8, smaller models, kNN/BGE-head
  alternatives that reuse existing embeddings).
- Taxonomy drift between 2014–16 and 2023–26 (mitigation: refreshed taxonomy with mapping).
- Groq budget for the single re-record (mitigation: bounded eval set, one re-record only).
- Laptop thermals (mitigation: no training on the laptop).
- Cold-start growth from larger artifacts (mitigation: measure; keep-warm already in place).
