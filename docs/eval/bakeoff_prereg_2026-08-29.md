# TriageIQ model/prompt bake-off — pre-registration (2026-08-29)

**Status:** Pre-registered before any Part C screen call. This document is committed
*before* the first live call, per the working agreement for this engagement — nothing in
it may be edited after calls start except to add a dated addendum.

Context: ADR-0052 (and its 2026-08-29 update) establish that TriageIQ currently has no
valid LLM-quality baseline in either direction — both committed cassettes are void, one
because its model is retired (`llama-3.1-8b-instant`), one because its recording was
contaminated by the `max_tokens=1024` truncation defect. This bake-off's result becomes
the project's first genuinely valid baseline. Everything below governs how that
measurement is taken.

---

## 1. Arms

| Arm | Model | Prompt |
|---|---|---|
| A | `openai/gpt-oss-20b` | current default: `SYSTEM_PROMPT_PROSE` + 3 few-shot examples (`build_few_shot_examples()`) |
| B | `openai/gpt-oss-20b` | same system prompt, **no few-shot examples** |
| C | `openai/gpt-oss-120b` | same as Arm A (few-shot) |
| D | `qwen/qwen3.6-27b` | same as Arm A (few-shot) |

All arms: native structured output (`use_structured_output=True`), the env-controlled
dynamic token-budget guard (`TRIAGE_MAX_TOKENS=2048`, PR #113), `temperature=0.0`,
`seed=42`.

Arms C and D use the few-shot prompt rather than testing the few-shot axis a second time —
B2/B3 already isolate that axis on gpt-oss-20b (Arms A vs B). Testing model choice and
few-shot-removal as two separate axes on the same 4-arm budget, instead of a full 2×3
factorial (which would need 6 arms), is a deliberate scope cut: if Arm B beats Arm A on
quality, the natural follow-up (not in this screen) is to re-test whichever of C/D wins
without few-shot too.

## 2. Metrics (pre-registered, not chosen after seeing results)

Primary:
1. **Judge mean** (local `qwen3:8b` via Ollama, ADR-0019's existing dimension set,
   zero Groq cost) — the project's standard quality metric.
2. **Grounding rate** (`plan.grounding_status.all_grounded`) — deterministic, not
   judge-dependent.
3. **Fabrication rate** — component or similar-issue citations not present in that
   request's own classifier/retrieval output. Deterministic.

Secondary (diagnostic, not elimination criteria on their own):
4. **Parse-success rate** (`llm_status == "ok"` vs `parse_retry_succeeded` vs any
   `degraded_*`) — structured output should push this near 100%; a regression here is a
   reliability signal independent of prose quality.
5. **Quality split by clamp status** (see §4) — judge mean and grounding rate computed
   separately for calls where `dynamic_max_tokens < 2048` fired vs calls where it didn't.
   This is not optional or exploratory — §4 explains why it is required for Arms A/C/D to
   be comparable to Arm B at all.
6. Latency (P50/P95), `x-ratelimit-remaining-*` trend across the interleaved run.

## 3. Elimination rule (pre-registered)

An arm is eliminated if **any** of:
- Grounding rate more than 10 percentage points below the best arm's grounding rate.
- Fabrication rate is nonzero where another arm's is zero, on the same issues (paired
  comparison — see §5).
- Parse-success rate below 90% (structured output failing to do its job).

Among arms that survive elimination, the primary decision metric is **judge mean**,
compared only between arms that were not eliminated. A single point-estimate difference
with overlapping 95% CIs (n=20 per arm; expect wide intervals, this is a screen not a
final measurement) is not sufficient on its own to declare a winner — report it as
directional and say so.

## 4. Guard against a convenient conclusion (Part C2)

**Few-shot must not be dropped because it is expensive. It has to lose on quality.**
Arm B (no few-shot) is cheaper by construction — that is not evidence it is better. The
only valid reason to prefer Arm B over Arm A is that Arm B's judge mean, grounding rate,
and fabrication rate are equal to or better than Arm A's, measured fairly. Cost is what
determines whether the *winning* arm's cadence is sustainable on the free tier — it does
not get to decide which arm wins on quality.

**What would make me KEEP few-shot, stated before running anything:**
- Arm A's judge mean exceeds Arm B's by a margin that would survive a two-sided test if
  n were large enough to resolve it (i.e., a consistent, non-trivial point-estimate gap in
  Arm A's favor, even with overlapping CIs at n=20) — this is a signal to expand n on that
  specific comparison before deciding, not to already prefer Arm B by default.
- Arm A's grounding or fabrication rate is meaningfully better and that gap does not
  wholly track with clamp status (§5) — i.e., few-shot itself, not just having more
  completion budget, appears to be doing the work.

**If Arm A wins on quality despite costing ~2.7x more per call (§6), the correct
conclusion is that the current prompt does not fit this free tier — a finding about the
tier, not a license to silently drop few-shot to make the numbers more convenient.** That
scenario is reported as: "Arm A is the better prompt; it needs a paid tier or a smaller
few-shot set to run at production cadence" — not folded into "so we removed few-shot."

## 5. Comparability problem: the clamp does not treat both arms the same way (Part C3)

At `max_tokens=2048`, 37/64 issues (57.8%) in the eval set push
`dynamic_max_tokens = min(2048, 8000 - prompt_tokens - 100)` below 2048 under Arm A's
few-shot prompt (prompt tokens 5,662–6,183). **The guard does not reject these requests —
it lets them through at a reduced completion budget**, as low as 1,717 tokens at the
longest prompt. Under Arm B's no-few-shot prompt (prompt tokens 1,406–1,928),
`dynamic_max_tokens` never drops below 2048 for any issue in the eval set — Arm B always
gets the full budget.

**This means Arm A is not measured under one uniform completion budget — it is measured
under a budget that shrinks with prompt length, while Arm B's is flat.** A naive
"Arm A scored worse than Arm B" result would be confounded: it could mean few-shot content
hurts quality, or it could mean Arm A's longer-prompt issues got systematically less room
to write a complete answer (recall observed completions reach up to 1,919 tokens — above
Arm A's worst-case budget of 1,717). These are not the same finding and must not be
reported as if they were.

**Mitigation, decided before running, not after seeing which one looks more convenient:**
1. **Paired design, not independently-sampled arms.** The same 20 issues (§6) run through
   all 4 arms, interleaved. This makes the clamp's effect on Arm A directly observable
   per-issue rather than averaged away.
2. **Capture `dynamic_max_tokens` actually sent, per call, for every arm** (already in the
   capture list). Report Arm A's judge mean and grounding rate split into "clamped"
   (11/20 issues in this sample, see §6) vs "unclamped" (9/20) subsets, separately from
   Arm A's overall mean.
3. **If Arm A's quality gap (if any) is concentrated in the clamped subset and absent in
   the unclamped subset, the finding is "few-shot's current prompt doesn't fit the free
   tier's budget on long issues," not "few-shot is worse."** If the gap is uniform across
   both subsets, that is evidence the content itself (not the budget) is the driver — a
   real, reportable few-shot-vs-no-few-shot finding.
4. This is exactly why §2's metric 5 (quality split by clamp status) is listed as
   required, not exploratory.

## 6. Sample (Part C4) — stratified, paired across arms, seed=42

20 issues drawn from the 64-issue eval set (`eval/eval_set.jsonl`), stratified by
Arm A's (with-few-shot) prompt-token quintile so both tails of the size distribution are
represented, with the single shortest and single longest issue in the eval set forced
into the sample (not left to chance) per the earlier extreme-point validation (A1/A4).
Repo split is not forced to exact 53:11 proportionality — 5 of the eval set's 11 vscode
issues are included (vs. a strict-proportional ~1.7) so vscode isn't reduced to statistical
noise in a 20-issue screen; this is a deliberate, stated deviation, not an oversight.

| repo | number | Arm A (few-shot) prompt tokens | Arm A dynamic_max_tokens @ 2048 | clamped? |
|---|---:|---:|---:|:---:|
| kubernetes/kubernetes | 14054 | 5662 | 2048 | no |
| kubernetes/kubernetes | 12277 | 5708 | 2048 | no |
| kubernetes/kubernetes | 14723 | 5763 | 2048 | no |
| kubernetes/kubernetes | 14557 | 5765 | 2048 | no |
| kubernetes/kubernetes | 12287 | 5781 | 2048 | no |
| microsoft/vscode | 4993 | 5820 | 2048 | no |
| kubernetes/kubernetes | 14835 | 5820 | 2048 | no |
| kubernetes/kubernetes | 12122 | 5833 | 2048 | no |
| kubernetes/kubernetes | 14135 | 5850 | 2048 | no |
| kubernetes/kubernetes | 12254 | 5894 | 2006 | **yes** |
| microsoft/vscode | 4996 | 5896 | 2004 | **yes** |
| microsoft/vscode | 311284 | 5897 | 2003 | **yes** |
| kubernetes/kubernetes | 14363 | 5905 | 1995 | **yes** |
| kubernetes/kubernetes | 12665 | 5938 | 1962 | **yes** |
| kubernetes/kubernetes | 13057 | 5939 | 1961 | **yes** |
| microsoft/vscode | 278113 | 5942 | 1958 | **yes** |
| microsoft/vscode | 312423 | 5961 | 1939 | **yes** |
| kubernetes/kubernetes | 14762 | 6075 | 1825 | **yes** |
| kubernetes/kubernetes | 12784 | 6091 | 1809 | **yes** |
| kubernetes/kubernetes | 13435 | 6183 | 1717 | **yes** |

11/20 (55%) fall in Arm A's clamped range — close to the full eval set's 57.8%, so this
sample preserves the confound §5 needs to be checkable, not a sample that accidentally
avoids it. Repo split: 15 k8s / 5 vscode.

Reproducibility: `random_state=42`, `pandas.qcut` quintiles on Arm A's measured
`estimated_prompt_tokens` (frozen list in `tests/test_token_budget_guard.py`'s
`_EVAL_SET_PROMPT_TOKENS_2026_08_29`), min/max forced in, remaining 18 slots filled by
per-quintile random draw. Script used to generate this table is not committed (ad hoc,
scratch) — the table above is the artifact of record; regenerate by re-running the same
quintile/seed procedure against the frozen token list if verification is needed.

**Call order: round-robin interleaved by issue, not blocked by arm** — issue 1 arm A,
issue 1 arm B, issue 1 arm C, issue 1 arm D, issue 2 arm A, … — so no arm's quota state or
latency measurement is confounded with position in the run (the prior bake-off's
sequential-block mistake, per this session's brief).

## 7. Quota plan (Part B2 recompute + Part C6)

Groq free tier, confirmed directly (console.groq.com/docs/rate-limits, fetched
2026-08-29) — **same limits for all three candidate triage models**: 30 RPM, 1,000 RPD,
8,000 TPM, 200,000 TPD, per model (BELIEVED per-model, not org-pooled across models — the
docs state limits per-model in the rate-limit table but do not explicitly confirm scope;
this will be checked empirically on the first cross-model call via the
`x-ratelimit-remaining-*` headers this screen already captures, before assuming it).

Per-call token cost (prompt tokens VERIFIED this session via tiktoken; completion tokens
BELIEVED, carried from the task brief's cited live-observed range of 1,031–1,919 tokens,
not yet re-verified independently this session):

| Arm | prompt tokens (p50) | completion tokens (range) | tokens/call (range) | calls/day at 200K TPD |
|---|---:|---:|---:|---:|
| A (few-shot) | 5,881 | 1,031–1,919 | 6,912–7,800 | **26–29** (≈27/day midpoint) |
| B (no few-shot) | 1,625 | 1,031–1,919 | 2,656–3,544 | **56–75** (≈65/day midpoint) |
| C (120b, few-shot) | 5,881 | unmeasured for this model | assume ≈ Arm A pending first calls | ≈27/day, unconfirmed |
| D (qwen3.6-27b, few-shot) | 5,881 | unmeasured for this model | assume ≈ Arm A pending first calls | ≈27/day, unconfirmed |

This confirms the ~28/day vs ~66/day expectation stated going into this session, within
the range implied by measured prompt sizes and the previously-cited completion range.
RPD (1,000) and RPM (30) are not binding at any of these call rates — TPD is the only
real constraint. Arms C/D's completion-token behavior is genuinely unmeasured (different
models); their pacing is a placeholder until the first few live calls land, at which point
the actual figures replace the "assume" row rather than being trusted as measured.

**Per-day plan for the 80-call screen (20 issues × 4 arms), reserving diagnostic
headroom:**
- Total screen budget: 80 calls, interleaved round-robin.
- Reserve 20% of each day's 200K TPD per active model for retries/diagnostics (a prior
  session burned 196K/200K on diagnostics before the real measurement ran — this reserve
  exists specifically so that mistake can't repeat).
- Arms A/C/D (few-shot-shaped prompt, ~7,300 tokens/call): 20 calls each = 60 calls total
  across 3 models sharing no quota with each other (separate per-model TPD pools, per the
  belief stated above) — each model's 20 calls cost ~146,000 tokens, comfortably inside
  even the 80%-reserved 160,000 TPD for that model. **All of Arms A, C, D can complete in
  a single day each**, run on separate days or the same day (different models, no shared
  pool) if the per-model-pool belief holds.
- Arm B (no-few-shot, ~3,000 tokens/call): 20 calls ≈ 60,000 tokens — well inside one day.
- **Revised pacing vs the original ~3-day estimate: if per-model pools are confirmed
  independent, the entire 80-call screen fits in a single day of wall-clock quota, one
  day per model in practice for review/monitoring convenience — not the ~3 days planned
  when the screen was scoped as a single shared 28/day pool.** If the pools turn out to be
  shared (org-wide, not per-model), the plan falls back to running arms across separate
  calendar days.

## 8. Eval-set expansion affordability (Part B3)

Prior W5 scoping (`docs/PROJECT_STATE.md`) targets n≈150 (75/repo), blocked throughout on
Groq TPD for generating synthesis plans over the expanded set (separate from, and in
addition to, the still-outstanding human-labeling precondition — GG labeling
`data/gold_expansion_candidates.csv`, ~2-3h, unaffected by anything in this section).

Assuming the expanded set's prompt-size distribution resembles the current 64-issue set
(unverified assumption — the candidate pool is different issues, flagged, not measured):

| | tokens for 150 calls | days of dedicated 200K TPD budget | days at realistic per-day pace |
|---|---:|---:|---:|
| With few-shot (current) | ≈1,095,000 | ≈5.5 | ≈150/27 ≈ 5.6 days |
| No few-shot | ≈450,000–465,000 | ≈2.3 | ≈150/65 ≈ 2.3 days |

**This does meaningfully reduce the Groq-quota cost of the eval-set expansion (roughly
2.4x faster) if Arm B wins the bake-off, but it does not eliminate the blocker — it only
removes the quota half of it.** The human-labeling precondition (GG labeling the
120-candidate pool) is untouched by this and remains the actual gating step; quota was
never the binding constraint on *starting* the expansion, only on *finishing* the
synthesis-generation step once labels exist. Reported here so the affordability claim
isn't overstated as "this unblocks W5" when it only partially does.
