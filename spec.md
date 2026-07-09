# Project Spec: TriageIQ — W6 Attribution Fidelity (single-condition measurement)

## Goal

Change the synthesis prompt ONCE to make the LLM attribute its claims — declare where
`predicted_component` came from (classifier top-3 or its own override, with a reason) and cite
which retrieved issues support its summary/next-steps — then measure, with the existing grounding
verifier as the instrument, the FIDELITY of that attribution: of the citations the LLM produces,
what fraction are verifiably grounded (cited issue actually retrieved; component actually in
classifier_top3) vs FABRICATED. Per repo, exact counts.

This replaces the two-arm prompt-swap A/B, which died at the step-0 determinism gate for a
design reason, not an infra reason: local qwen is byte-deterministic on FIXED input (probe 2/2)
but PROMPT-SENSITIVE (near-tie tokens flip on trivial prompt changes), and Groq has proven
replica-jitter. Since the A/B's treatment IS a prompt change, no stack can separate "attribution
effect" from "the prompt bytes changed" — the cross-arm comparison is confounded at the root. Do
NOT attempt to force determinism to rescue an A/B; the causal "attribution reduces ungrounded
rate" claim is DROPPED.

The single-condition design keeps the more useful product question — "when asked to attribute,
does the LLM cite REAL sources or invent them?" — and the probe result actively supports it:
fixed-input determinism means one prompt + one recording is EXACTLY reproducible. No second
condition, no cross-condition confound.

CAN'T-LOSE framing (the ADR captures either outcome as a finding):
- High grounded-citation rate → attribution is a trustworthy feature; ship it with the verifier
  as the safety net (FLAG-not-strip, ratchet extended to citations).
- Low rate → documented finding that ATTRIBUTION INDUCES FABRICATED REFERENCES in issue triage;
  it stays FLAG-gated or doesn't ship — a real negative result, ADR-0006/ADR-0016 style.

ADR-0015 lineage: the existing `GroundingAttribution` field is a *reconstruction* ("not new
attribution elicited from a prompt change"); this iteration is the elicitation it anticipated.

## Known validity threats (designed-in mitigations — these are in scope, not optional)

1. **Measurement stack = ship stack.** Fabrication rate is a property of model+prompt. This
   iteration measures LOCAL qwen3:8b synthesis (zero-cost, deterministic, reproducible without a
   key) — so a SHIP decision includes the synthesis-provider change from Groq-8B to local qwen.
   That coupling is explicit; confirm it at the escalation checkpoint before recording. Measuring
   qwen and shipping Groq would be measuring the wrong system.
2. **Validity ≠ aptness (blanket-citation gaming).** "Cited number ∈ retrieved set" measures
   fabrication, not whether the citation supports the claim. A model citing all 5 retrieved
   issues everywhere scores 100% grounded while attributing nothing. MANDATORY companion
   metrics: citations-per-claim distribution, % of plans citing all retrieved issues
   (blanket rate), distinct-citation overlap between summary and next-steps. OPTIONAL secondary
   (labeled subjective): local-judge aptness rating on a 10-plan sample.
3. **Rate is conditional on compliance.** Plans omitting the attribution block contribute no
   citations. Compliance rate (attribution present + well-formed) is CO-PRIMARY, per repo.
4. **Result pinned to this exact prompt+model.** Prompt-sensitivity means any future prompt edit
   invalidates the measured fidelity — re-measure on change (same discipline as the grounding
   pins). Stated in the ADR.
5. **No cross-recording comparisons as claims.** New-cassette ungrounded/judge numbers vs the
   old Groq-era baseline (2/65, means 10.5185/8.3636) are condition-confounded — report only as
   a labeled side observation (model-change effect), never as the finding.

## Current state (existing project)

- Clean n=65 eval set (54 k8s / 11 vscode), frozen CPU similar_issues, local qwen3:8b judge,
  committed Groq-era cassette + baseline live as the CI gate (ADR-0019 bands 0.22/0.45).
- Determinism probe (this arc, recorded result): local qwen3:8b synthesis byte-identical 2/2 on
  fixed input under warm-up + keep_alive=-1 + serial + fixed order; prompt-sensitive on trivial
  prompt edits; Groq replica-jitter confirmed. Consequence: single-condition local recording is
  exactly reproducible; cross-prompt comparisons are not clean on any stack.
- Grounding verifier (`src/triage_iq/models/grounding.py:verify_plan_grounding`): pure Python,
  deterministic given a fixed plan, FLAG-not-strip. Semantics FROZEN — it keeps measuring what
  it measures (predicted_component vs top-3; similar_issues numbers vs retrieved set). New
  citation fields get a NEW additive checker.
- Synthesis today: Groq llama-3.1-8b-instant in `TriageAssistant` (temp 0, seed 42, 3 few-shots,
  parse-retry + fallback). An ollama synthesis path may partially exist from the aborted W6 A/B
  build — reuse/finish it (profile-switched, e.g. SYNTHESIS_PROVIDER=ollama; Groq path intact).
- The committed CI gate stays untouched during measurement: the new recording is an experiment
  cassette. Promotion to committed baseline happens ONLY on the human ship decision (full
  ceremony: means, ratchet, pins re-derived, human-approved).

## Scope

### In scope (this iteration)

**The prompt change — ONE change, applied once (ESCALATE exact diff before recording):**
- SYSTEM_PROMPT: add an `"attribution"` object to the embedded JSON schema plus instruction
  lines: (a) "predicted_component should come from the classifier top-3; if you deviate, you
  MUST set component_source='model_override' and give component_override_reason"; (b) "every
  issue number cited anywhere in the plan MUST be one listed in SYSTEM 2"; (c) cite which
  SYSTEM-2 issues support the summary and next steps — cite ONLY issues that actually support
  the specific claim; empty lists are allowed and honest.
- Update all 3 few-shot assistant outputs with correct `attribution` blocks that model
  SELECTIVE citation (not blanket) — the exemplars are the main lever against threat #2.
- No other prompt edits. After recording, the prompt is PINNED: edits invalidate the measurement.

**The additive schema fields (ESCALATE exact shapes with the prompt diff):**

  ```python
  class DeclaredAttribution(BaseModel):
      """LLM-emitted source attribution (W6, elicited by the prompt — contrast
      GroundingAttribution, a post-hoc reconstruction; ADR-0015/ADR-0020)."""
      component_source: Literal["classifier_top3", "model_override"]
      component_override_reason: str = ""   # required by prompt iff model_override
      summary_cited_issues: list[int] = []     # SYSTEM-2 numbers supporting triage_summary
      next_steps_cited_issues: list[int] = []  # SYSTEM-2 numbers informing suggested_next_steps

  # TriagePlan (additive only):
  declared_attribution: DeclaredAttribution | None = Field(default=None)
  ```
- `/triage` gains the field automatically. No existing field changes shape. Missing/malformed
  attribution → None (tolerant parse), COUNTED as a compliance failure, never a request failure.

**The fidelity checker (additive; `verify_plan_grounding` UNTOUCHED):**
- New `verify_declared_attribution(plan, classifier_top3, retrieved_numbers)` returning, per
  plan: fabricated citations (cited ∉ retrieved, deduped, sorted — mirrors existing verifier
  style), grounded-citation count/total, component declaration class
  (GROUNDED_DECLARED = said top-3 and is | HONEST_OVERRIDE = declared override |
  MISATTRIBUTED = said top-3 but ISN'T — the worst class), compliance flag, and the
  selectivity stats feeding threat-#2 metrics.

**The ONE recording (all local, zero-cost — wall-clock is the budget):**
- n=65, attributed prompt, local qwen3:8b synthesis + local qwen3:8b judge, warm-up +
  keep_alive=-1, temp 0, seed 42, serial, fixed issue order, checkpointed/resume-safe. New
  experiment cassette (`eval/cassettes/w6_attribution.json`); committed cassette untouched.
- Post-recording stability spot-check: re-run 10 issues, byte-compare against the cassette
  (confirms determinism held across the recording window).
- Self-judging caveat (qwen judges qwen): document in the ADR; judge means from this cassette
  are a new-baseline candidate, not comparable to the Groq-era means (ADR-0019 precedent
  language: "a new baseline, not a corrected old one").

**The measurement (primary deliverable — `reports/w6_attribution_fidelity.json` + script):**
- CO-PRIMARY, per repo, exact counts with denominators stated (vscode n=11 — counts, not
  percentage theater): (1) compliance rate; (2) grounded-citation rate = grounded/total cited,
  with the fabricated-citation list; (3) component declaration classes, especially
  MISATTRIBUTED count.
- Companion (mandatory): selectivity metrics (citations-per-claim distribution, blanket rate);
  parse-failure/fallback counts; existing-verifier outputs on the new plans (component axis +
  similar_issues axis) as the continuity view.
- Secondary (optional, labeled subjective): judge aptness spot-check on 10 plans.
- Side observation (labeled, not a claim): new-cassette judge means + ungrounded profile vs the
  Groq-era committed baseline.
- PRE-REGISTER the decision rubric at the escalation checkpoint (proposal, human may adjust):
  trustworthy-feature bar = compliance ≥ 90% overall AND zero MISATTRIBUTED components AND
  fabricated citations = 0 on k8s (n=54) with any vscode fabrications individually explained
  (n=11 is too small for a rate bar). Below bar → the negative-result framing applies.

**Docs:**
- ADR-0020: the two design pivots and why (one-arm: condition-drift confound; two-arm: prompt-
  sensitivity + replica-jitter make ANY cross-prompt A/B unclean — the determinism probe as
  evidence), the single-condition design, the dropped causal claim (explicitly: this does NOT
  show attribution reduces ungrounded rate vs no-attribution), validity threats + mitigations,
  exact prompt diff, field shapes, the fidelity tables, decision (ship / FLAG-gated / no-ship)
  recorded as human, ADR-0015 lineage.
- If SHIP: separate approved step — promote the cassette to committed baseline (new per-repo
  means human-approved, ratchet + pins re-derived, citation-fabrication ratchet ADDED alongside
  the existing grounding ratchet, `eval_summary.json`/`/eval` updated, synthesis-provider change
  Groq→local documented, deploy its own approval).

### Out of scope (do not build)

- Any second recording condition, any cross-prompt comparison, any attempt to force cross-prompt
  determinism. The causal A/B claim is dropped — do not resurrect it in the report or ADR.
- Any change to `verify_plan_grounding` semantics, the existing ratchet definition, or any band.
- Changing eval_set.jsonl, the gold set, the judge model/params, or issue order.
- Retrieval/classifier/resolution changes; structured-output migration; self-consistency (other
  roadmap items). One variable enters the codebase: the attribution prompt + its additive fields.
- Promoting the experiment cassette, changing prod synthesis, or deploying — human decisions
  after the measurement is read.

## Tech stack

- Local ollama qwen3:8b (synthesis AND judge), existing cassette recorder/player + LLM cache
  (provider="ollama" keys), pandas/pytest. No new deps, zero paid calls.

## Architecture

```
triage-iq/
  src/triage_iq/models/triage.py           # ollama synthesis path (profile-switched);
                                           #   DeclaredAttribution; additive TriagePlan field;
                                           #   tolerant attribution parse
  src/triage_iq/prompts/triage_prompt.py   # attribution schema block + instructions + 3
                                           #   few-shots modeling SELECTIVE citation
  src/triage_iq/models/grounding.py        # UNCHANGED verify_plan_grounding; NEW additive
                                           #   verify_declared_attribution
  scripts/w6_record.py                     # checkpointed single-condition recorder
  scripts/w6_fidelity_report.py            # compliance + grounded-vs-fabricated + selectivity
  eval/cassettes/w6_attribution.json       # NEW experiment cassette (the one recording)
  reports/w6_attribution_fidelity.json     # the measurement artifact
  docs/architecture/adr/ADR-0020-*.md      # either-outcome finding, both pivots documented
  # committed eval_cassette.json / eval_baseline.json / ratchet / pins: UNTOUCHED until ship
```

## Verification commands

```yaml
- name: eval-gate
  cmd: pytest eval/ -v
  required: true
- name: api-tests
  cmd: pytest -v
  required: true
```

## Subagent usage rules

- `executor` writes/edits; `verifier` runs checks. Orchestrator delegates, never codes.

## Escalation rules (orchestrator must ask before doing)

- **ESCALATE before recording, as one checkpoint:** the exact prompt diff, the
  DeclaredAttribution shape, confirmation of the measurement-stack-=-ship-stack coupling
  (local qwen synthesis is what a ship decision ships), and the pre-registered decision rubric
  numbers. WAIT for approval.
- Report the recording plan + wall-clock estimate before starting (single arm, n=65,
  synthesis + judge, checkpointed — scope as a real multi-hour operation).
- If the post-recording 10-issue stability spot-check finds ANY byte mismatch, STOP and report —
  the exact-reproducibility premise of the design is then wrong and that is itself a finding.
- Report the full fidelity tables BEFORE writing the ADR decision section — ship / FLAG-gated /
  no-ship is human.
- Cassette promotion, baseline ceremony, prod synthesis change, deploy: each ONLY on explicit
  instruction.

## Hard rules (existing project)

- Committed cassette, baseline, ratchet, pins, bands: FROZEN during measurement. Suite green
  throughout.
- `verify_plan_grounding` semantics frozen. Additive-only on `/triage`.
- One recording condition; the prompt is pinned after recording; exact counts with n stated
  everywhere; the dropped causal claim stays dropped.
- Branch only; nothing deploys. Claude Max — never set ANTHROPIC_API_KEY. Zero-cost: everything
  local. Don't touch aetherart-497918.
- Run full suite after every executor pass; escalate on any pre-existing failure.

## Budget

- Soft target: 2 CC sessions (session 1: build + escalation checkpoint; session 2: recording
  [multi-hour, checkpointed] + fidelity report + ADR). Hard cap: escalate after 20 executor
  invocations.
- `/cost` at midpoint.

## Success criteria (orchestrator verifies ALL before declaring done)

- Prompt diff + field shapes + stack coupling + decision rubric human-approved BEFORE recording.
- `DeclaredAttribution` additive; tolerant parse; ollama synthesis path profile-switched with
  Groq path intact; `verify_declared_attribution` additive; full suite green; committed gate
  untouched (verified, not assumed).
- One cassette recorded (n=65, local qwen synthesis + judge, solved discipline), post-recording
  10-issue byte-stability spot-check PASSED.
- `reports/w6_attribution_fidelity.json` complete: per-repo compliance, grounded-vs-fabricated
  counts + fabricated-citation list, component declaration classes incl. MISATTRIBUTED,
  selectivity/blanket metrics, parse counts, continuity view, labeled side observation.
- ADR-0020 written: both design pivots with the determinism-probe evidence, dropped causal claim
  stated, validity threats + mitigations, fidelity results, human decision recorded
  (ship / FLAG-gated / no-ship).
- All staged on a branch; nothing deployed; no baseline ceremony performed (unless separately
  instructed after the decision).

## Build order (recommended)

1. Fresh branch off current main (`git add spec.md` so the spec is versioned with the work).
   Reuse/finish the ollama synthesis path from the aborted A/B build if present.
2. Draft the prompt diff + DeclaredAttribution + rubric → ESCALATE (one checkpoint). WAIT.
3. Implement: schema field + tolerant parse → prompt + few-shots (selective-citation exemplars)
   → verify_declared_attribution → w6_record.py / w6_fidelity_report.py. Full suite green.
4. Report recording plan + estimate → record (checkpointed) → 10-issue stability spot-check
   (STOP on any mismatch).
5. w6_fidelity_report → REPORT full tables against the pre-registered rubric. WAIT for human
   decision.
6. ADR-0020 (either-outcome finding; decision recorded as human). Stage on branch, report.
   Promotion/ship steps only on explicit follow-up instruction.
