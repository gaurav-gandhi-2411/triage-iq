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

---

## Addendum, 2026-08-30 — amendments made after Part A/B validation calls, before the
## 20-issue screen's first real call. Per this doc's own rule, nothing above is edited;
## everything below is dated and additive.

### A1. Arm D dropped — ELIMINATED ON TOKENIZER FOOTPRINT AGAINST THE FREE-TIER TPM
### CEILING (pre-registered elimination on a real constraint, decided before the screen)

Two live calls against `qwen/qwen3.6-27b` (k8s #14054, the *shortest* prompt in the
64-issue eval set) both hit a hard 413 ("Request too large... Limit 8000, Requested
8073" at margin=100, "Requested 8056" after resizing the margin to 200 — see A2 below).
Root cause, not just a tail case: qwen's real tokenizer counts ~260–350 more tokens than
gpt-oss's for identical content. gpt-oss-20b and gpt-oss-120b were confirmed (this
session) to produce **identical** real `prompt_tokens` for the same input (5750 both) --
same tokenizer family. qwen does not share that family, and the gap is large enough that
even the guard's "capped, no shrink needed" branch (which applies zero margin, since the
cap is `self.max_tokens` not the shrink formula) still overflows the 8,000 TPM ceiling on
the single shortest issue in the set. No margin resize closes this — the margin only
protects the shrink branch; this failure is upstream of that branch's own logic. Fixing
it for real needs a qwen-specific `_CL100K_TO_REAL_RATIO`, which needs its own live
calibration sweep across the length distribution — out of proportion to what a screen
elimination decision requires. **Decision (2026-08-30): do not calibrate qwen. Screen is
3 arms: A (gpt-oss-20b few-shot), B (gpt-oss-20b no few-shot), C (gpt-oss-120b
few-shot).**

### A2. Token-budget margin resized 100→200 (already applied, `src/triage_iq/models/triage.py`)

Two live extreme-point calls (k8s #14054 shortest, #13435 longest) measured real
out-of-sample extrapolation error of +88 tokens (under-predicted) and −74 tokens
(over-predicted) against the guard's gpt-oss-20b calibration — ~21x the 5-sample
interpolation error (≤4.2 tokens) the original 100-token margin was sized against.
200 covers the largest observed error with >2x headroom and independently would have
prevented the qwen 413 above (7973 < 8000 vs the actual 8073 at margin=100). The
"37/64 issues would 413 at a fixed max_tokens=2048" figure quoted in triage.py and its
tests is now 57/64 at margin=200 — expected: a stricter margin makes the *hypothetical
fixed-budget* rejection count worse, which is the whole reason the dynamic guard exists.

### A3. TriagePlan schema/prompt audit — see the companion report for full detail; summary here

A field description read "k8s 76.6% [74.0%, 79.1%]" (percent language) while its own
schema constrained the value to [0,1] (fraction) — the model filled a repo-constant,
non-derivable field with a plausible-looking value that then failed Groq's strict-mode
bounds check, a hard 400. Audited every `TriagePlan`-tree field against its own
constraint: **6 fields are always or conditionally overwritten post-synthesis and were
never told so** (`resolution_interval_conformal`, `resolution_bucket`,
`resolution_confidence_pct`, `grounding`, `grounding_status`, `abstention_status`) — two
of them (`grounding`, `grounding_status`) had *no description text at all*. All 6 now
explicitly instruct the model not to fabricate a value (null where the type permits it,
the exact Pydantic default otherwise, since `resolution_bucket`/`resolution_confidence_pct`
are non-Optional). Added `tests/test_schema_description_consistency.py`: a
percent-vs-fraction check, an explicit-numeric-range-vs-constraint check, and a check
that all 6 non-derivable fields' descriptions instruct against fabrication — a future
field like the original bug cannot land silently. **Checked whether any of these 6 fields
ever fed a published metric: no.** Four are unconditionally overwritten before any
consumer (judge, cassette, published metric) reads `plan` in both the eval harness and
production. The other two (`declared_attribution`, `abstention_status` in the eval path)
are explicitly `exclude`d from what's cached/judge-scored (`eval/run_eval.py`,
`eval/record_cassettes.py`) specifically because they're unconditional-on-`TriagePlan`
fields the harness never populates. README/ADR-0044's 0.0% fabrication-rate claims are
computed from the deterministic post-hoc `grounding_status` overwrite, never from the
model's own raw (possibly fabricated) output for that field.

### B1. TPD/TPM header gap — methodology cannot resolve this as planned (real gap, recorded)

§7's plan to answer per-model-vs-org-pool "empirically off the `x-ratelimit-remaining-*`
headers" cannot work as stated: Groq's response headers (VERIFIED, captured live, several
calls, both models) expose only **TPM/RPM state**
(`x-ratelimit-{limit,remaining,reset}-{requests,tokens}`) — there is no day-scoped header
in the response at all. The original §7 "BELIEVED per-model... will be checked
empirically... via headers" claim cannot be settled that way; downgrading to BELIEVED
with conservative pacing per B2.

### B2. Per-model-pool inference downgraded to BELIEVED; pacing now assumes a SHARED pool

The earlier session's inference (gpt-oss-120b's remaining-tokens, 202, higher than
gpt-oss-20b's immediately-preceding 122) is TPM-window evidence, read at different points
in a rolling per-minute cycle — not TPD state, and not strong enough to plan a multi-day
quota schedule against. **Revised default: assume ONE shared 200,000 TPD pool across all
three models until proven otherwise.** This changes the plan materially — see A5 below.

**Proposed (not run) decisive test:** RPD/RPM are also per-account limits and far cheaper
to exhaust than TPD — 30 tiny (near-zero-content) requests to one model would hit its
30 RPM ceiling in well under a minute at negligible token cost. Immediately following
with one call to a *different* model: if that model's `x-ratelimit-remaining-requests`
is also depleted, the account-level pools are shared (implying token pools likely are
too, same quota architecture); if it shows a fresh ~1000/30, the pools are separate per
model. This tests the *requests* dimension directly and cheaply, and is suggestive (not
proof) for the *tokens* dimension by architectural inference. Not run this session —
proposed per the working agreement's "propose, don't run" instruction.

### A4 (renumbered from the interrupted validation pass). Arm B's actual configuration —
### a harness bug caught before it touched the real 20-issue screen, not shipped in
### production code

v1 of the harness set `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=0` for Arm B, which switches
BOTH the system prompt (to `SYSTEM_PROMPT_LEGACY`, not `SYSTEM_PROMPT_PROSE`) AND the
few-shot set (to the legacy 4-shot set, not zero) — there is no existing "zero few-shot"
code path anywhere in `triage.py`. Caught before any of the 20-issue screen ran: v1's own
validation call for k8s #14054 arm B showed `prompt_tokens=5713`, nearly identical to arm
A's 5750 — nowhere near "no few-shot." That data point is discarded, not counted anywhere
in this document or any result. **Fixed (harness-only, no production code touched):**
keep `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1` (same `SYSTEM_PROMPT_PROSE` system prompt as
Arm A, per §1's own "same system prompt, no few-shot examples") and monkeypatch
`triage_prompt.build_few_shot_examples` to return `[]` only for Arm B's calls. Real
corrected prompt tokens for Arm B: 2,923–2,996 across 3 issues tested (not the ~1,625
p50 §7 originally estimated — that number was computed against the shorter LEGACY system
prompt, not "the same system prompt as Arm A" the arm actually requires; §7's Arm B
row is superseded by A5 below).

**A genuinely important reliability finding surfaced during this fix, not a config
artifact:** on k8s #14054, Arm B's corrected (zero-few-shot, same system prompt) config
produced a completion that stopped generating after `similar_issues` (field 3 of 18),
never emitting the other 15 required fields — Groq's structured-output validator
rejected it as `missing properties: [15 field names]`. Two more issues (k8s #12277,
vscode #4993) completed cleanly (`finish_reason: stop`, full schema). **n=1/3 so far** —
not enough to call systematic, but a real, reproducible-shape failure distinct from a
quality difference: the model can apparently forget to finish the schema at all without
few-shot examples demonstrating the full shape. This is exactly the kind of result §4's
guard exists for — it is evidence about *prompt fit*, not evidence to fold into "few-shot
is worse," and it is now tracked as part of the parse-success-rate metric (§2 metric 4),
not silently absorbed into a judge-mean comparison. Separately: this exact 400 shape
(`missing properties`, no "response_format" substring) is **not** caught by
`_groq_completion`'s existing fallback branch (which only matches a literal
"response_format" substring) — it propagates uncaught, same reliability gap the
`resolution_interval_conformal` bug had. Flagged, not fixed this session (fixing
`_call_llm_verbose`'s degrade path to catch this class generally is a real, separate
change — noted for the defensibility report, not applied here to keep this session's
diff scoped to what the screen actually needs).

### C1-C3. New pre-registered metrics (added before the screen's first real call, per the
### working agreement)

Added to §2's metric list, not informational-only:
7. **Completion-token distribution per arm** (mean/p50/max `completion_tokens`) —
   pre-registered, not descriptive-only. Motivation: this session's own extreme-point and
   validation calls put gpt-oss-20b's few-shot completions at 1,172–1,919 tokens vs.
   `llama-3.1-8b-instant`'s historical ≤~600 (Part C of the prior session, verified by
   direct cassette inspection) — a 2-3x verbosity gap that is the root driver of every
   budget problem in this engagement and should be measured, not accepted as fixed.
8. **Quality-per-completion-token** — judge mean and grounding rate divided by mean
   completion length, computed per arm. A model producing equal judge/grounding scores at
   materially fewer tokens is strictly better on this tier's economics and the screen
   must say so explicitly, not leave it implicit in the raw judge-mean comparison.
9. **Few-shot's effect on verbosity specifically** (Arm A vs Arm B completion-token
   distributions, paired per issue) — few-shot examples may be teaching output *length*
   as much as *format*; this is checked as its own comparison, not folded into the
   overall judge-mean question.

### A5. Revised 3-arm quota plan (supersedes §7's per-model-pool table)

Real per-call costs, this session (n=3 Arm A, n=3 Arm B post-fix, n=1 Arm C — small
samples, will be refined by the screen itself):

| Arm | model | mean tokens/call (n) | max observed |
|---|---|---:|---:|
| A (few-shot) | gpt-oss-20b | 7,269 (n=3) | 7,693 |
| B (no few-shot) | gpt-oss-20b | 4,520 (n=2 completed; 1 failed pre-completion, excluded) | 4,774 |
| C (few-shot) | gpt-oss-120b | 7,288 (n=1) | 7,288 |

Per-issue round-robin triplet (A+B+C, one issue, all 3 arms): ~19,100 tokens.

**Under the now-conservative shared-pool assumption (B2), all three arms draw from ONE
200,000 TPD pool, not three independent 200K pools.** Reserving the same 20% diagnostic
headroom (a prior session burned 196K/200K on diagnostics before its real measurement
ran) leaves 160,000 usable tokens/day → **160,000 / 19,100 ≈ 8 issue-triplets/day.**

**Revised day-split plan (replaces §7's "single day, maybe 2" estimate):**
- Day 1: issues 1–8 of §6's table, all 3 arms each, round-robin interleaved within the
  day (A, B, C per issue, issue 1 through issue 8).
- Day 2: issues 9–16, same structure.
- Day 3: issues 17–20 (4 issues × 3 arms = 12 calls), same structure.
- Per D2's amended instruction (2026-08-30): do **not** reduce n=20 to fit fewer days —
  a smaller n makes the screen unable to discriminate (the exact underpowered-gate
  mistake as the 11-issue grounding arm). Days are free; statistical power is not.
- If a decisive test (B2) is later run and confirms per-model pools are actually
  independent, this collapses back toward §7's faster estimate — but the plan executes
  against the conservative 3-day estimate unless and until that's confirmed, not before.

This plan is reported here, before Day 1's first call, per the working agreement.
