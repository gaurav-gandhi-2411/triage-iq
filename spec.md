# Project Spec: TriageIQ — Elevate Fabrication Metric to a Hard Gate

## Goal

The eval audit (ADR-0028) found the synthesis quality metric was structurally blind to fabrication:
the LLM judge never sees the upstream signals (classifier_top3, retrieved-issue set), so it cannot
detect when synthesis cites a component or similar-issue that upstream never produced — a known
hallucination (#311836) scored ABOVE its repo mean. B3 added a deterministic fabrication check
(reusing the grounding verifier) but wired it INFORMATIONAL-ONLY: it's surfaced but gates nothing,
so a fabricating plan still passes.

This iteration elevates fabrication from informational to a HARD GATE — a plan that cites a
component not in classifier_top3, or a similar-issue not in the retrieved set, is a FAIL, not a
judge-averaged soft miss. Fabrication is the failure mode that most misleads a human triager (a
confidently-cited nonexistent source actively harms someone who trusts the tool), so it should
block, not just inform.

**But gate elevation is gated on MEASUREMENT** (the reason it was informational-only first): confirm
the current fabrication rate is low enough that a hard gate won't false-fail legitimate plans. If the
rate is near-zero, elevate safely. If it's higher than expected, understand why before elevating.

## Current state

- Grounding verifier (`grounding.py`): deterministic, checks plan claims vs `signals`
  (classifier_top3, retrieved similar-issue numbers). Already live, FLAG-not-strip.
- B3 (ADR-0028): fabrication_rate computed and surfaced, INFORMATIONAL-ONLY — gates nothing.
  Floor-rate metric also added (report-only). Mean-band judge gate unchanged (regression detector).
- Eval: clean n=64 (post-B1 quarantine), local qwen3:8b judge, per-repo.
- Known: grounding fires ~1.9% k8s / 9.1% vscode ungrounded (from the synthesis audit / attribution
  work) — but that's the ungrounded RATE, which the gate design needs to reason about precisely.

## Scope

### In scope

**1. MEASURE the fabrication rate precisely (before elevating — the gate on the gate):**
- Over the clean eval set, per repo: what fraction of plans contain a fabricated claim (component
  not in classifier_top3, OR similar-issue ref not in retrieved set)? Break down by fabrication
  TYPE (component vs similar-issue) and severity.
- This is the number that decides whether a hard gate is safe. Report it.

**2. Decide the gate POLICY based on the measured rate (escalate the decision):**
- If fabrication rate is near-zero (e.g. the 1-2 known cases): a hard gate is safe — it'll pass all
  legitimate plans and fail only genuine fabrications. Elevate to hard fail.
- If the rate is meaningfully non-zero: understand WHY first. Are the "fabrications" real (the LLM
  genuinely invents sources) or artifacts (e.g. the LLM cites a valid component that's in top-5 but
  not top-3 — arguably not fabrication)? The gate threshold (top-3 vs top-5 vs label-space) matters
  here. Propose the precise fabrication DEFINITION for the gate, escalate it.
- Decide: does the gate BLOCK (hard fail in CI) or does it fail the plan at SERVE time (reject/flag
  the fabricated claim in the live response)? These are different — CI-gate catches regressions;
  serve-time catches live fabrications. Propose which (or both).

**3. Wire the hard gate (once policy is set):**
- CI eval-gate: a fabricated claim in the eval set → the gate FAILS (not continue-on-error
  informational). Tie to eval_set_hash like the other ratchets. TEST THE TEST: inject a fabricated
  plan → gate fails → revert → passes.
- Consider serve-time: should live /triage reject/flag a fabricated claim before returning? (The
  grounding verifier already computes this; elevating means acting on it, not just reporting.)
  Escalate whether to make this live (prod-facing) or keep it CI-only this iteration.

**4. Baseline + disclosure:**
- If the gate changes what passes, re-establish the relevant baseline (human-approved).
- ADR-0029: the measured fabrication rate, the gate definition, block-vs-serve-time decision, and
  the elevation from informational → hard gate, framed as closing the ADR-0028 finding.

### Out of scope

- No change to the mean-band judge gate (stays a regression detector — this ADDS a fabrication gate).
- No retraining any model (this gates on existing outputs).
- No change to the grounding verifier LOGIC (it's correct; this elevates how its output is USED).

## Autonomy & escalation

CC measures + wires autonomously. Escalate ONLY:
1. **The measured fabrication rate + the gate-policy decision** (fabrication definition, block-vs-
   serve-time, hard-fail-vs-flag) — this is a policy call, human-decided from the measured rate.
2. Any baseline re-establishment (if the gate changes what passes).
3. Any prod deploy (if serve-time rejection goes live).

## Hard rules

- MEASURE before elevating — don't hard-gate a rate you haven't confirmed is low enough to not
  false-fail legitimate plans (that's the reason it was informational-first).
- The fabrication DEFINITION must be precise and defensible (top-3 membership is the current product
  definition per grounding.py — if the gate uses a different threshold, justify it).
- TEST THE TEST for the hard gate (inject fabrication → fail → revert → pass).
- Human-approve any baseline change + any prod deploy. Branch only (`feat/fabrication-gate`);
  I merge. Zero-cost, local judge. Claude Max — never ANTHROPIC_API_KEY. Don't touch aetherart-497918.

## Success criteria

- Fabrication rate measured per repo, by type — reported.
- Gate policy decided (definition, block-vs-serve, hard-vs-flag) from the measured rate — escalated.
- Hard gate wired; TEST THE TEST demonstrated (inject → fail → revert → pass).
- Baseline re-established if needed (approved); ADR-0029 documents the elevation.
- Staged on branch; prod-facing only if serve-time rejection is approved.

## Build order (CC autonomous)

1. Measure the fabrication rate per repo, by type. ESCALATE it + the gate-policy proposal.
2. On approval: wire the hard gate per the decided policy (CI-gate ± serve-time).
3. TEST THE TEST. Baseline re-establish if needed (escalate).
4. ADR-0029. Stage on branch.
```

