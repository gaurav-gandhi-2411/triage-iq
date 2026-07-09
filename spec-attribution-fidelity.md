# Project Spec: TriageIQ — Attribution Fidelity (single-condition measurement)

## Goal

Measure whether asking the synthesis LLM to CITE ITS SOURCES produces citations that are
**verifiably real** or **fabricated**. This is NOT a two-arm A/B (that design is confounded:
Groq synthesis has replica-jitter, local qwen synthesis is prompt-sensitive, so a prompt-swap
comparison can't isolate the attribution variable). Instead: change the synthesis prompt ONCE to
request attribution, record ONE clean cassette, and measure — via the existing deterministic
grounding verifier — the fraction of attributed citations that are grounded vs fabricated, per repo.

The measurement is confound-free because the ground truth (retrieved similar-issue numbers,
classifier top-3) is captured in `signals` BEFORE synthesis runs, so it's fixed regardless of what
synthesis generates. The verifier checks the LLM's citations against that fixed upstream — a single
generation, no cross-condition comparison.

**Either outcome is a finding:** high grounded-citation rate → attribution is a trustworthy feature,
ship it with the verifier as safety net. Low rate → documented negative (attribution induces
hallucinated references), stays FLAG-gated or doesn't ship — reranker/W3-rejection style.

## Current state (existing project)

- Synthesis: Groq llama-3.1-8b (current). Judge: local qwen3:8b (ollama, deterministic on replay).
- Grounding verifier LIVE (`src/triage_iq/models/grounding.py`): deterministic pure-Python, checks
  plan claims against `signals` retrieval/classifier output. FLAG-not-strip. Already wired into
  `triage_with_metadata`; `grounding`/`grounding_status` are additive fields on `TriagePlan`.
- Eval: clean n=65 disjoint gold set, local qwen3:8b judge, two-tier gate (byte-exact REPLAY
  zero-tolerance; RE-RECORD comparison per-repo-mean within 2×SEM bands). Live on rev 00059-855.
- `signals` (retrieval results + classifier top-3) captured pre-synthesis — the fixed anchor.

## Scope

### In scope

**Synthesis model decision (CC decides + verifies):**
- The attribution prompt change makes local-qwen synthesis prompt-sensitive (near-tie flips), and
  Groq has replica-jitter. For a SINGLE-CONDITION measurement neither nondeterminism confounds the
  result (there's no second arm to hold identical) — the verifier scores whatever single plan is
  generated against the fixed upstream anchor. So synthesis can stay on Groq-8B (cheaper, no local
  determinism burden) OR move local. CC decides based on: does the measurement need the cassette to
  be byte-reproducible on re-record? For a one-time fidelity measurement it does not — the mean-band
  gate already handles synthesis nondeterminism. CC picks, states the choice + rationale in the ADR.

**Attribution prompt change:**
- Modify the synthesis prompt to request, for each component claim and each similar-issue reference,
  the specific source it rests on (which retrieved issue number, which classifier output). Keep the
  existing plan structure; attribution is additive to the output, not a restructure.
- This changes cassette keys → a re-record is required (see below).

**Measurement (via existing verifier, deterministic):**
- For each of the n=65 issues, run the attributed synthesis, then `verify_plan_grounding` against
  the fixed `signals`. Compute per repo: total attributed citations, grounded count, fabricated
  (flagged) count, grounded-citation rate. This is the headline result.
- Also report judge means both for the attributed run (does attribution change quality vs the
  current baseline — noting this is descriptive, the mean-band gate applies, NOT a clean causal A/B).

**Re-record (local judge, solved discipline):**
- Re-record the n=65 cassette under the attribution prompt: chosen synthesis stack + local qwen3:8b
  judge, warm-up call + keep_alive=-1, GPU-only held throughout. All local for the judge = zero-cost,
  no rate limit. If synthesis stays Groq-8B, that's the only external call (separate 500K/day bucket,
  fine for 65 issues).

**Docs:**
- ADR-0020: the confounded-A/B reasoning (why single-condition, not two-arm), the synthesis-model
  choice + rationale, the per-repo grounded-citation-rate result, and the ship/flag/reject decision
  based on the measured fidelity. Frame either outcome as a finding.

### Out of scope

- No two-arm prompt-swap A/B (confounded — that's the whole reframe).
- No causal "attribution reduces ungrounded rate vs no-attribution" claim (needs the confounded
  counterfactual). The claim is descriptive: measured fidelity of attributed citations.
- No change to the grounding verifier logic (it's deterministic and correct — reuse as-is).
- No change to retrieval/classifier/resolution models. No schema field meaning changes (additive only).
- No reopening the closed eval-integrity work.

## Tech stack

- Existing Python. Existing grounding verifier. Local ollama (qwen3:8b judge; qwen synthesis if
  chosen). No new deps.

## Autonomy & escalation (CC runs autonomously — escalate ONLY these)

CC decides and executes everything else without gating, including: the synthesis-model choice, the
exact attribution prompt wording, the additive field shapes, the re-record execution, and all
verification. Escalate to the human ONLY for:
1. **The measured per-repo grounded-citation rate + judge means** — report before `--update-baseline`
   and before the ship/flag/reject decision. Human approves the baseline and the ship decision.
2. **The prod deploy** (updating `/eval` or `eval_summary.json`, which `app.py` serves live) — rollback
   anchor, drift guard, live verification. Human confirms the deploy.

Everything else: CC proceeds, applying the discipline established this arc (verify-don't-assume,
determinism checks before recording, warm-up for local inference, additive-only schema, honest
either-outcome framing).

## Hard rules

- Zero-cost: local ollama for the judge; if synthesis goes local it's local too. No paid tiers.
- Additive-only on `TriagePlan`; no existing field meaning/type changes.
- The grounding verifier logic stays untouched (deterministic; reuse).
- Re-record uses the solved local-judge discipline (warm-up, keep_alive=-1, GPU-only verified).
- Baseline is human-approved (escalation 1). The mean-band gate applies to judge means; the
  grounded-citation RATE is the primary result and is deterministic (verifier is pure).
- Branch only (`feat/attribution-fidelity`); human merges. Claude Max — never ANTHROPIC_API_KEY.
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

## Success criteria (CC verifies all before reporting the result)

- Attribution prompt change made; `/triage` returns attributed citations as additive fields; all
  existing fields unchanged (diff confirms).
- n=65 re-recorded under the single attribution condition; local judge; determinism discipline
  applied (synthesis determinism verified if local; judge replay byte-exact).
- Per-repo grounded-citation rate computed via the existing verifier (deterministic); fabricated
  citations flagged, counted.
- Judge means reported (descriptive, mean-band gate applied).
- ADR-0020 written with the single-condition rationale, synthesis choice, result, and the
  ship/flag/reject decision framed as a finding either way.
- Staged on branch; result + means reported to human before baseline/deploy (the 2 escalations).

## Build order (CC executes autonomously)

1. Decide + state the synthesis-model choice (Groq-8B stay vs local qwen). If local, verify
   synthesis determinism under warm-up first (single-condition doesn't strictly need it, but state
   whether the recording is reproducible either way).
2. Make the attribution prompt change + additive field shapes. Confirm existing fields unchanged.
3. Re-record n=65 under the single attribution condition (local judge, solved discipline).
4. Run the grounding verifier over the attributed plans; compute per-repo grounded-citation rate +
   fabricated count. Compute judge means.
5. ESCALATE: report the rate + means to the human before --update-baseline and before the ship
   decision.
6. On approval: baseline, ADR-0020, and (if shipping) the /eval update as a deliberate prod deploy
   (ESCALATE the deploy). Stage on branch for human merge.
```

