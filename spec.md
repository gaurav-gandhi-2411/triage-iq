# Project Spec: TriageIQ — Structured Generation + Verification Pass

## Goal

Synthesis (Stage 4, Groq llama-3.1-8b) emits free-text JSON that is then parsed and validated
against `TriagePlan`. When the LLM produces malformed JSON or internally-inconsistent output, the
current failure mode is a parse-retry (the ADR-0015 retry-cache path) or a degraded/partial plan.
This iteration hardens synthesis on two fronts:

1. **Structured generation** — constrain the LLM's output to be schema-valid *at generation time*
   (via Groq's structured-output / JSON-schema mode if available, or a tightened prompt +
   validation loop), so malformed JSON stops happening rather than being retried after the fact.
2. **Semantic verification pass** — a deterministic post-generation check (pure Python, no LLM)
   that rejects/flags plans that are internally inconsistent: e.g. a `predicted_component` that
   contradicts the plan's own recommended action, a priority that contradicts the resolution
   estimate, a next-step referencing a component not in the plan. Builds on the existing grounding
   verifier pattern (deterministic, FLAG-not-strip).

Either mechanism is measurable on the existing eval: does structured generation reduce parse-retry
/ malformed rate? Does the verifier catch real inconsistencies? Honest either-outcome framing —
if malformed rate is already ~0 or inconsistencies are ~0, that's a documented finding (synthesis
is already robust), not a failure.

## Current state (existing project)

- Synthesis: Groq llama-3.1-8b, `_call_llm_verbose` in `triage.py`. Output parsed by `_parse_plan`
  → `TriagePlan.model_validate`. ADR-0015 retry-cache path handles parse failures (re-prompt on
  malformed). `_llm_status` reports ok / parse_retry_succeeded / etc.
- Grounding verifier LIVE (`grounding.py`): deterministic, checks claims vs `signals`. FLAG-not-strip.
  This is the pattern to reuse for the semantic verifier.
- Attribution feature shipped (flag-gated off): `declared_attribution` field,
  `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION`.
- Eval: clean n=65, local qwen3:8b judge, mean-band gate, live rev 00061-4xk. Baseline
  cassette_hash c9966414, vscode 8.3636 / k8s 10.5185.
- Existing flag pattern: `TRIAGE_PROMPT_INCLUDE_BUCKET`, `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION`,
  `TRIAGE_ENABLE_ABSTENTION_GATE` — all additive, default-off.

## Scope

### In scope

**Measure the CURRENT failure rate FIRST (before building anything):**
- Over the n=65 eval set (and ideally a broader sample if cheaply available), measure: how often
  does synthesis currently produce malformed JSON / trigger a parse-retry / return a
  semantically-inconsistent plan? This is the baseline the feature must improve on. If the malformed
  rate is already near-zero, structured generation has little to fix — report that and it changes
  the build (verifier-only, or documented "already robust"). MEASURE before building.

**Structured generation (if the measured rate justifies it):**
- Check whether Groq's API supports structured output / JSON-schema-constrained generation for the
  llama-3.1-8b model. If yes: constrain synthesis output to the `TriagePlan` schema at generation.
  If no: a tightened prompt + a bounded validation-retry loop (already partly exists) — improve it,
  don't rebuild it.
- Flag-gated (`TRIAGE_STRUCTURED_SYNTHESIS`, default off) so it can be A/B'd against current and
  doesn't change live behavior until enabled.

**Semantic verification pass (deterministic, pure Python — the reusable core):**
- `src/triage_iq/models/plan_verify.py`: `verify_plan_consistency(plan) -> ConsistencyReport`.
  Checks internal consistency, e.g.: component named in next-steps matches predicted_component;
  priority is consistent with resolution bucket (a "critical" priority with a "months" resolution
  estimate is flagged as inconsistent for review); no next-step references a nonexistent field.
  Define the concrete consistency rules in the ADR — keep them conservative (flag clear
  contradictions, not stylistic judgments).
- FLAG-not-strip (same as grounding): return the plan + a `consistency_status` marking any
  inconsistency, don't mutate the plan. Additive optional field on TriagePlan.
- Deterministic and unit-tested on synthetic consistent/inconsistent plans.

**Eval-gate invariant:**
- Add a structural invariant: over the eval set, `consistency_status` is computed correctly for
  every plan (catches verifier regressions). If you gate on a max-inconsistency-rate, tie it to
  eval_set_hash like the grounding ratchet. TEST THE TEST: inject an inconsistent plan, confirm the
  verifier flags it, confirm the invariant behaves, revert.

### Out of scope

- No change to synthesis MODEL (stays llama-3.1-8b — model change invalidates the cassette).
- No re-record UNLESS structured generation changes the output enough to change cassette keys —
  and if it does, that's the deliberate re-baseline path (report means, human-approve). The
  verifier alone is scoring-only, no re-record.
- No LLM in the verifier (deterministic pure Python — it runs in the gate).
- No mutation of LLM output beyond FLAG (no auto-fixing inconsistent plans in v1).
- No reopening closed eval-integrity work.

## Tech stack

- Existing Python. Groq structured-output API if available (check first). Local ollama judge for
  any re-record. No new heavy deps.

## Architecture

```
triage-iq/
  src/triage_iq/models/plan_verify.py   # NEW — verify_plan_consistency (pure, deterministic)
  src/triage_iq/models/triage.py        # structured-generation path (flag-gated) + call verifier
  src/triage_iq/api/app.py              # additive consistency_status field (additive)
  eval/test_invariants.py               # consistency invariant + test-the-test
  docs/architecture/adr/ADR-0022-*.md   # measured current failure rate, rules, result, decision
```

## Autonomy & escalation (CC runs autonomously — escalate ONLY these)

CC decides + executes: whether Groq structured output is available and how to use it, the exact
consistency rules (propose them in the ADR), field shapes, all verification, the ADR.
Escalate ONLY:
1. **The measured current failure rate** (the measure-first result) — report before building the
   structured-generation part, because it decides whether that part is worth building or whether
   this is verifier-only. If malformed rate is ~0, say so and we scope down.
2. **Baseline means** if a re-record is triggered (structured gen changing cassette keys) — before
   --update-baseline.
3. **Any prod deploy** — if structured gen or the verifier surfaces on live /triage or /eval.

## Hard rules

- Zero-cost: local ollama for judge/re-record. No paid tiers. (Groq structured output uses the
  existing 8B synthesis budget — fine, within limits.)
- Additive-only schema; flag-gated features default-off (structured gen + verifier gate off until
  proven and enabled).
- Verifier is deterministic pure Python; no LLM in it.
- FLAG-not-strip — never mutate/auto-fix LLM output in v1.
- TEST THE TEST for the consistency invariant (inject inconsistent plan → flagged → revert).
- Branch only (`feat/structured-verification`); I merge. Claude Max — never ANTHROPIC_API_KEY.
  Don't touch aetherart-497918.

## Verification commands

```yaml
- name: eval-gate
  cmd: pytest eval/ -v
  required: true
- name: api-tests
  cmd: pytest -v
  required: true
```

## Success criteria (CC verifies before reporting)

- Current malformed/inconsistency rate MEASURED and reported (the measure-first gate).
- Semantic verifier `plan_verify.py` deterministic, unit-tested, FLAG-not-strip, additive field.
- If built: structured generation flag-gated, measured against current (does it reduce malformed
  rate), honest either-outcome result.
- Consistency invariant added; TEST THE TEST demonstrated (inject → flag → revert).
- eval_baseline.json unchanged unless a deliberate re-record is escalated + approved.
- ADR-0022: measured current rate, the consistency rules, structured-gen availability + result,
  the ship/flag/reject decision framed as a finding.
- Staged on branch; nothing prod-facing without escalation.

## Build order (CC autonomous)

1. MEASURE FIRST: current malformed-JSON / parse-retry / semantic-inconsistency rate over n=65.
   Report it — this gates whether structured generation is worth building. ESCALATE the number.
2. Build the deterministic semantic verifier (plan_verify.py) + unit tests — valuable regardless of
   the measured rate. Propose the consistency rules.
3. If the rate justifies it: structured-generation path (Groq schema mode or tightened loop),
   flag-gated, measured vs current.
4. Consistency invariant + TEST THE TEST.
5. ADR-0022 with the honest result. Stage on branch.
```

