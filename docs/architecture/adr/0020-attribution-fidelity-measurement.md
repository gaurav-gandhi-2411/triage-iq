# ADR-0020 — Attribution Fidelity Measurement

**Status:** Proposed — measurement in progress; results section pending
**Date:** 2026-07-07
**Decider:** Gaurav Gandhi

## Context

ADR-0015 added `verify_plan_grounding` as a **post-hoc** check: the synthesis prompt was left
unchanged, and the verifier reconstructed what the LLM already said and diffed it against
`signals["classifier_top3"]` / `signals["similar_raw"]`. That ADR explicitly deferred the more
invasive change — asking the LLM to *declare* its sources in the prompt itself — as an unmeasured
hypothesis not worth an expensive cassette re-record without evidence it would help. This ADR is
that deferred elicitation, now undertaken and measured.

**Goal:** measure the fidelity of LLM-declared source attribution in `TriagePlan` synthesis — when
asked to attribute, does the model cite REAL sources (retrieved similar issues, classifier top-3)
or fabricate them?

### Why this is not an A/B

Two designs were considered and rejected before landing on the single-condition measurement below.
This is the key decision history and is recorded precisely because both rejections cost real
investigation time.

**(1) One-arm design — current committed baseline as "before," a fresh attributed re-record as
"after" — rejected.** The committed baseline (ADR-0019) was recorded under different conditions
(different judge, different gold set, different synthesis-prompt history). Comparing a fresh
recording against it would conflate the attribution change with synthesis jitter and condition
drift already documented in ADR-0019 — there is no way to attribute a delta to the prompt change
alone.

**(2) Two-arm fresh A/B — same gold set, unattributed prompt vs. attributed prompt, both freshly
recorded — rejected at its own determinism gate.** ADR-0019 established, by direct experiment, that
local qwen3:8b synthesis is byte-deterministic only on a FIXED input (5/5 and 10/10 identical
back-to-back) but is PROMPT-SENSITIVE — near-tie tokens flip on trivial prompt changes, which is
exactly what an attribution-prompt addition is. Groq `llama-3.1-8b-instant` (the shipped synthesis
model) has replica-level nondeterminism even with an explicit `seed`, also established in ADR-0019
(three identical live calls, three different outputs, diverging within 100 characters). Since the
A/B's *treatment* is itself a prompt change, no available inference stack isolates the attribution
variable from ordinary generation jitter — the cross-prompt comparison is confounded at the root.
This is a measurement-design conclusion, not an infra bug to chase further.

**Therefore: single-condition measurement.** The upstream anchor — `classifier_top3` and the
retrieved similar-issue numbers in `signals` — is captured *before* synthesis runs and is fixed
regardless of what synthesis subsequently generates. The deterministic, pure-Python verifier scores
the generated plan against that fixed anchor. One generation, no cross-condition comparison, no
confound to control for.

The causal claim "attribution reduces the ungrounded rate vs. no attribution" is **explicitly
dropped** — it would require the confounded counterfactual described above. The claim this ADR
measures is descriptive only: the fidelity of attributed citations *as shipped*.

## Decision

**Synthesis stays Groq `llama-3.1-8b-instant`.** Three reasons:

1. **Measurement stack = ship stack.** Production (Cloud Run) serves Groq synthesis and cannot host
   Ollama. Measuring local-qwen synthesis instead would measure a system that is not the one
   shipped — a wrong-system measurement, not just a less-clean one.
2. **Single-condition design needs no byte-reproducible generation.** The verifier scores whatever
   plan is actually generated against the fixed upstream anchor; there is no cross-run comparison
   to protect with determinism. Cassette REPLAY stays byte-exact once recorded (ADR-0019's replay
   invariant is unaffected and unchanged).
3. **Cost is trivial.** 65 calls fit comfortably in the 8B model's free-tier daily bucket.

**Honest caveat carried forward from ADR-0019:** the recording is one draw from Groq's jittery
output distribution. The fidelity numbers below carry that sampling caveat, and the ADR-0019
mean-band gate (SEM-derived per-repo tolerance) is exactly the instrument already designed to
absorb re-record jitter on judge means — it is reused here rather than invented fresh.

**Judge stays local qwen3:8b (Ollama)**, unchanged from ADR-0019 — warm-up + `keep_alive=-1`
discipline carried forward as-is; this ADR does not touch judge selection.

**Prompt change (the one variable under test):** an ATTRIBUTION RULES section added to the
synthesis prompt, a `declared_attribution` object added to the embedded output schema, and all 3
few-shot exemplar outputs extended with selective-citation examples. An 8B model reliably omits
fields its exemplars omit, so the exemplars must *demonstrate* selective citation — citing some
claims and explicitly not others — to counter a default failure mode of blanket-citing everything
regardless of whether it was actually used.

**Additive schema:** `DeclaredAttribution` —

- `component_source: classifier_top3 | model_override`
- `component_override_reason` (present iff `model_override`)
- `summary_cited_issues`
- `next_steps_cited_issues`

— lands as `TriagePlan.declared_attribution`, tolerant-parsed (malformed input → `None`, counted as
a compliance failure, never a request failure). `/triage` is additive-only; no existing field
changes shape or meaning.

**New additive checker: `verify_declared_attribution`.** `verify_plan_grounding` (ADR-0015) is left
untouched — this is a new, separate check, not a modification of the existing one. Failure
taxonomy:

- `grounded_declared` — declares `classifier_top3` sourcing and the citation is real.
- `honest_override` — declares `model_override` with a stated reason; not measured against the
  anchor because the model isn't claiming provenance from it.
- `misattributed` — declares `classifier_top3` sourcing but the citation isn't real. **The worst
  class**: this is fabricated attribution, not just an unattributed guess.
- `missing` — `declared_attribution` absent or malformed.

### Measurement validity threats and their designed mitigations

1. **Validity ≠ aptness.** "Cited ∈ retrieved" measures whether a citation is fabricated, not
   whether it's a *good* citation (relevant, on-topic). Blanket-citation rate and
   citations-per-plan (selectivity) are mandatory companion metrics, not optional color —
   reporting only the fabrication rate without a selectivity check would let a model that cites
   everything every time score perfectly on fidelity while providing zero signal.
2. **Rate is conditional on compliance.** A model that mostly emits `missing` and occasionally
   emits a real citation should not read as "high fidelity" — compliance rate is reported as
   co-primary with the fidelity rate, not as a footnote.
3. **Result is pinned to this exact prompt + model pair.** Any future prompt edit invalidates the
   measured numbers below; a prompt change requires re-measurement, not an assumption that fidelity
   carries forward.
4. **No cross-recording comparisons as claims.** The new cassette's judge per-repo means and
   ungrounded-claim profile will be reported as side observations against the Groq-era baseline
   (ADR-0019), explicitly labeled as such — the gold set is unchanged but the prompt is, so any
   delta is not attributable to one cause. This mirrors the ADR-0019 discipline of not decomposing
   a two-cause delta.

### Either-outcome framing

This measurement is designed so both outcomes are actionable, not just the favorable one:

- **High fidelity** → attribution ships, with `verify_declared_attribution` as the safety net
  (FLAG, not strip — same policy ADR-0015 established for grounding).
- **Low fidelity** → this becomes a documented finding that eliciting attribution induces fabricated
  references, and the feature is FLAG-gated or rejected outright, following the negative-result
  pattern already used in this project (ADR-0006's cross-encoder rejection, ADR-0016's fine-tuned
  retriever rejection).

## Results

<!-- RESULTS PENDING: filled after recording -->

### Per-repo compliance

<!-- RESULTS PENDING: filled after recording -->

### Per-repo grounded vs. fabricated citation counts and rate

<!-- RESULTS PENDING: filled after recording -->

### Component declaration classes (including misattributed)

<!-- RESULTS PENDING: filled after recording -->

### Selectivity / blanket-citation metrics

<!-- RESULTS PENDING: filled after recording -->

### Parse / fallback counts

<!-- RESULTS PENDING: filled after recording -->

### Judge per-repo means (side observation vs. Groq-era baseline, labeled)

<!-- RESULTS PENDING: filled after recording -->

### Decision: ship / flag-gate / reject

<!-- RESULTS PENDING: human decision, pending gate-1 review -->

## Consequences

- The additive `declared_attribution` field lands in `/triage` responses regardless of the ship
  decision above — it is `None`-safe and does not change existing response shape.
- The eval cassette is re-recorded under the attributed prompt. This is a deliberate re-baseline,
  pending human approval: per-repo judge means, the grounding ratchet, and the known-case pins
  (ADR-0015, ADR-0019) all get re-derived against the new cassette, not silently compared to the
  old one.
- n=65 with vscode n=11 (ADR-0019's clean gold set) — all counts below are exact, not statistical
  estimates; no significance claims are made on subsets this small.
- Generalization caveat: these numbers describe Groq `llama-3.1-8b-instant` on this exact prompt
  and this exact gold set. They do not transfer to a different synthesis model, a different prompt
  revision, or repos outside kubernetes/kubernetes and microsoft/vscode.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| One-arm before/after (current committed baseline vs. fresh attributed re-record) | Baseline recorded under different conditions (judge, gold set, prompt history) — any delta conflates the attribution change with condition drift, not isolatable. |
| Two-arm fresh A/B (unattributed vs. attributed, both freshly recorded) | Fails at the determinism gate established in ADR-0019: local qwen3:8b is prompt-sensitive, Groq synthesis has replica-level nondeterminism even with `seed` — the treatment (a prompt change) is itself confounded with ordinary generation jitter on every stack available to this project. |
| Local-qwen synthesis for this measurement (to get determinism) | Unshippable — production serves Groq synthesis via Cloud Run, which cannot host Ollama. Measuring local-qwen synthesis would measure a system that is not the one shipped. |
| Force determinism via `llama.cpp` pinning, strict decoding, or similar stack-level tricks | ADR-0019 already tried strict decoding (`top_k=1, top_p=0`) at full scale and still saw 9.2% divergence; and none of these techniques transfer to the shipped Groq path regardless, so pursuing them here would not make the eventual A/B any less confounded. |
