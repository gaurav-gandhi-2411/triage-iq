# TriageIQ — Orchestration Contract

**Pattern:** Opus orchestrator + executor/verifier subagents  
**Project state:** `docs/PROJECT_STATE.md`

This document encodes the workflow model and standing discipline for running TriageIQ workstreams. Every hard-won lesson from this project's history is encoded here as a standing check or rule — not as advice, as a mandatory gate.

---

## Roles

### Orchestrator (Opus)

Plans workstreams, writes subagent task briefs, sequences PRs, aggregates results, makes merge recommendations, surfaces decisions to GG. **Never executes shell, git, or file operations directly — delegates everything to executor subagents.**

Responsibilities:
- Decompose each workstream into independent executor tasks
- Write precise task briefs (target file, exact change, acceptance criteria)
- Assign verifier passes after every executor write
- Block merges until verifier sign-off is on record
- Update `docs/PROJECT_STATE.md` after each workstream closes

### Executor subagent

Implements changes: writes code, runs training scripts, generates data artifacts, commits files. Reports exact diffs, outputs, and file paths back to orchestrator. Does not make architectural decisions — flags ambiguities to orchestrator.

### Verifier subagent

Independently checks executor work before any merge. Uses the standing checklist below. Does not see the executor's reasoning — only the artifact. Reports PASS/FAIL with evidence per check. Any FAIL blocks merge until the executor fixes and verifier re-checks.

---

## Verifier standing checklist

These checks are derived from real failures in this project. Each one corresponds to an incident. Run all applicable checks on every PR.

### CHECK 1 — Eval/train overlap (EVAL LEAKAGE)

**What to do:** For any eval sample, enumerate every `(repo, query_number, positive_number)` tuple. Intersect with the training set. Assert zero overlap. Count and report the overlap explicitly.

**Why this exists:** W3 original T5 eval used `sample_gold()` which drew from the full gold corpus. Since ~70% of gold was in the training split, 66% of k8s and 71% of vsc eval pairs were training data. The inflated result (+26pp k8s) was caught only because the verifier enumerated the overlap. Without this check, the inflated number would have been shipped as the headline result.

**Acceptance criterion:** `len(eval_keys ∩ train_keys) == 0`. If non-zero, report the count, give 5 example overlapping pairs, and block merge.

### CHECK 2 — Baseline protocol match

**What to do:** Verify that the baseline R@5 (or other metric) used to compute the delta was measured on the **identical query set and retrieval protocol** as the fine-tuned model. Hardcoded constants are a red flag — trace them to their source and verify the sample they came from.

**Why this exists:** Two incidents:
1. W3: BASELINE_R5 constants were full-corpus W1.1 numbers (n=1024 k8s); fine-tuned was evaluated on a contaminated n=100 sample — two different protocols. Delta was meaningless even before contamination.
2. W1.3: The n=100 reranker screening baseline (0.430/0.470) was a favorable random draw vs the full-corpus baseline (0.410/0.367). Small-N sampling variance made a noise result look like a real gain.

**Acceptance criterion:** Baseline and model must be computed in the same script run, on the same query DataFrame, with the same retrieval depth and self-exclusion handling. No hardcoded baseline constants.

### CHECK 3 — Robustness (extraordinary deltas)

**What to do:** Any point estimate ≥ 2× the expected effect size, or any result with CI lower bound near zero, requires a larger-N re-check before acceptance.

**Why this exists:** W1.3 showed a +6pp k8s improvement at n=100, CI lower bound positive. At n=300, the effect collapsed to +0.67pp with CI spanning zero — a clear false positive from sampling variance at n=100.

**Acceptance criterion:** For any delta with CI lower bound < 2pp, flag for n=300 robustness before merge. For deltas > 15pp on R@5 metrics on this corpus, verify no eval contamination (overlaps with Check 1).

### CHECK 4 — Split methodology

**What to do:** Verify that any time-series or resolution-time model uses `created_at` as the temporal sort key for train/val/test splits. Verify that pair-level splits (retrieval training) use connected-component splitting so no issue appears on both sides of the split boundary.

**Why this exists:** W4 diagnosis revealed the resolution predictor was split by `closed_at`. This caused fast-resolving issues (which close quickly) to cluster in training while slow-resolving issues (which stay open long) clustered in test — a systematic distribution shift that made the evaluation completely non-representative. k8s CI coverage was 0% on the broken split. After the fix, k8s CI coverage became 77%.

**Acceptance criterion:** Any `split` operation on temporal data uses `df.sort_values("created_at")`. Verify in the script, not just the docstring.

### CHECK 5 — Feature leakage

**What to do:** Audit every feature in any model for triage-time availability. A feature is valid only if it would be known at the moment a new issue is filed. Check for any field that is set or updated after initial filing (labels added post-triage, resolution metadata, assignee added by a human, etc.).

**Why this exists:** The W4 resolution model included `has_priority` — a label applied by a human triager after the issue was filed. This feature is unavailable at inference time for new issues, creating an information leak that inflated training accuracy.

**Acceptance criterion:** Each feature in the model must have a documented source available at inference time. List any post-filing fields in the PR and confirm they are excluded.

### CHECK 6 — Which model is live

**What to do:** Before spending any judge quota, verify that the intended model is loaded. Check `/health` response for `retrievers: {"repo": "finetuned"}` (or whatever the target model key is). Do not proceed if the wrong model is active.

**Why this exists:** The W3 judge eval was started before the loader changes took effect (the process launched from an old Python executable). The run consumed Cohere quota to evaluate triage plans generated by the baseline retriever, not the fine-tuned one — invalidating the entire run. We had to abort mid-run after the Cohere trial quota was nearly exhausted.

**Acceptance criterion:** `/health` returns the expected `source` value for all repos before the eval command is issued. Document the health check output in the PR.

### CHECK 7 — Quota pre-flight

**What to do:** Before any eval run using Cohere or Groq judge, estimate the number of API calls and confirm the quota is available. For Cohere: 1,000 calls/month; one n=150 run = 450 calls. For Groq TPD: estimate token cost of the planned run.

**Why this exists:** Multiple partial runs were wasted when quota was hit mid-eval. The Cohere trial was completely exhausted during a run that also turned out to use the wrong model (see Check 6). Both quota and model correctness must be pre-flighted.

**Acceptance criterion:** Explicit quota estimate in the PR or session notes before running. Format: `Run cost: ~450 Cohere calls. Remaining budget: ~550. Proceed: yes/no.`

---

## Standing rules (project invariants)

These rules apply to every session, every workstream, with no exceptions.

1. **Free-tier only.** Never set `ANTHROPIC_API_KEY` anywhere in this repo. Never add paid-tier fallbacks silently.

2. **All execution through subagents; GG approves merges and decisions.** The orchestrator never pushes to main directly (except pure documentation). All production changes go through a PR reviewed by GG.

3. **Clean negative results are first-class outcomes.** W1.3 (reranker rejected) and the W4 vscode resolution predictor (0% improvement over naive) are both documented as accepted results in the ADRs. They are not buried, not hedged, not re-framed as partial successes. The documentation is the evidence that the work was rigorous.

4. **One scarce-resource principle.** Spend Cohere judge quota once, on the most informative version of the question. The decision to expand gold to n=150 before running the W3 judge was made precisely because running the judge at n=60 and then again at n=150 would cost 2× the quota for a less informative first run. Plan the eval, then spend.

5. **ADR for every technical decision.** An ADR is written when a decision is committed to code. ADRs include: Context (why this question arose), Decision (what was chosen), Consequences (results including failures), Alternatives considered. Correction notes are permanent — do not scrub them.

6. **CHANGELOG per substantive PR. Squash-merge, one clean commit per workstream.**

7. **Correction notes are permanent.** The W3 ADR-0010 has a correction note documenting the eval contamination. This note stays in the history permanently. It is evidence of verification discipline, not an embarrassment.

8. **`created_at` temporal splits, always.** See Verifier Check 4.

---

## Decision gates requiring GG approval

The orchestrator flags these and waits:

- Any merge to main
- Any re-scrape of GitHub data (raw data changes)
- Any production model swap (model serving the API changes)
- Any schema change that touches the TriagePlan JSON contract (affects triage-iq-ui)
- Anything irreversible: branch deletion with unmerged commits, production deployments, external service credential rotation
- Any spend of more than ~200 Cohere calls (>20% of monthly budget) in a single run

---

## Current orchestrator task queue

Priority order. Each item is blocked until its predecessor completes.

| Priority | Task | Blocked on | Estimated cost |
|---|---|---|---|
| 1 | **n=150 Cohere judge run** (closes W3 + sets baseline) | GG labels gold_expansion_candidates.csv; /health shows finetuned | ~450 Cohere calls |
| 2 | Merge PR #7 (W3 fine-tuned retriever) | Task 1 result: similar_issues_relevance ≥ 2.87 | — |
| 3 | Merge PR #8 (W5 eval infrastructure) | Task 1 complete; ADR-0011 updated | — |
| 4 | **W2.B: gpt-oss-20b migration** (unlocks Groq server-side caching, fixes TPD) | PRs #7 and #8 merged; new baseline established | 1 full judge run |
| 5 | **UI debt**: GitHub #3 (artifact rename) + #5 (openapi-typescript codegen) | W2.B complete (stable API contract) | ~1 session each |
| 6 | **llama-70b retrofit** (opportunistic, 17/180 done) | W2.B merged (cached tokens free) | ~163 Cohere calls (mostly cached) |
