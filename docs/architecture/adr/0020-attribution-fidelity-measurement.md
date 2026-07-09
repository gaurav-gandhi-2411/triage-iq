# ADR-0020 — Attribution Fidelity Measurement

**Status:** Accepted — measured, gate-1 human decision: ship
**Date:** 2026-07-07 (measured); decided 2026-07-09
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

n=65 (vscode 11, kubernetes/kubernetes 54), Groq `llama-3.1-8b-instant` synthesis, local
qwen3:8b judge. Source: `scripts/measure_attribution_fidelity.py`, full output at
`reports/attribution_fidelity.json`.

### Per-repo compliance

`declared_attribution` present and well-formed on every response — no `missing`, no
`unparseable_raw`, no tolerant-parse fallback triggered.

| Repo | n | compliant | absent | malformed | unparseable_raw |
|---|---|---|---|---|---|
| microsoft/vscode | 11 | 11/11 | 0 | 0 | 0 |
| kubernetes/kubernetes | 54 | 54/54 | 0 | 0 | 0 |
| **overall** | 65 | **65/65** | 0 | 0 | 0 |

### Per-repo grounded vs. fabricated citation counts and rate

| Repo | total citations | grounded | fabricated | grounded rate |
|---|---|---|---|---|
| microsoft/vscode | 33 | 33 | 0 | 100% |
| kubernetes/kubernetes | 216 | 214 | 2 | 99.07% |
| **overall** | 249 | 247 | 2 | **99.20%** |

Both fabrications are on kubernetes/kubernetes:

| Issue | Fabricated ref | Retrieved (real) set |
|---|---|---|
| #14557 | 14263 | 3270, 7743, 10057, 11091, 12694 |
| #12277 | 11631 | 3733, 12545, 12929, 13642, 14748 |

Both fabricated numbers are plausible near-misses (same order of magnitude as the real
retrieved set, not wildly out of range) — consistent with the model attempting to cite and
missing, not inventing an unrelated number.

### Component declaration classes (including misattributed)

| Repo | grounded_declared | honest_override | misattributed | missing |
|---|---|---|---|---|
| microsoft/vscode | 10 | 0 | 1 | 0 |
| kubernetes/kubernetes | 53 | 0 | 1 | 0 |
| **overall** | 63 | 0 | **2** | 0 |

The two `misattributed` cases are **#13057** (kubernetes/kubernetes) and **#311836**
(microsoft/vscode) — the identical two issues that were component-ungrounded in the prior,
pre-attribution Groq-era recording under `verify_plan_grounding` (ADR-0015). Same underlying
classifier miss, now additionally mis-declared rather than silently ungrounded. No new
component-fabrication case was introduced by asking the model to declare attribution.

### Selectivity / blanket-citation metrics

- Plans with ≥1 citation: 64/65 (one vscode plan cited nothing).
- Blanket citation (cites every retrieved issue with no selectivity): 36/65 overall
  (k8s 32/54, vscode 4/11).
- Citations-per-plan distribution: overall min 0 / median 5 / max 5 (retrieval always
  surfaces 5 candidates; median-5 means most plans cite the full retrieved set).

**Aptness caveat (validity threat #1, restated as a result, not just a risk):** a 55%
blanket-citation rate means fidelity (99.2% grounded) is measuring "doesn't fabricate,"
not "cites selectively and only what it used." The 8B model's default behavior leans toward
citing everything offered rather than picking the subset it actually reasoned from. This is
an honest limitation of the shipped feature, not hidden by the headline fidelity number.

### Parse / fallback counts

0/65 — no fallback path was exercised. Every response parsed as well-formed
`DeclaredAttribution` on the first attempt.

### Judge per-repo means (side observation vs. Groq-era baseline, labeled)

Computed via `python eval/run_eval.py` (dry run, no `--update-baseline`) against the
committed n=65 attribution cassette; compared against the currently-committed
`reports/eval_baseline.json` (ADR-0019, pre-attribution prompt):

| Repo | ADR-0019 baseline mean | this recording's mean | delta | one-directional band | tripped? |
|---|---|---|---|---|---|
| microsoft/vscode | 8.3636 | 8.7273 | +0.3636 | 0.45 | No (improvement) |
| kubernetes/kubernetes | 10.5185 | 10.6852 | +0.1667 | 0.22 | No (improvement) |
| overall | 10.1538 | 10.3538 | +0.2000 | n/a (per-repo gate only) | — |

**Labeled explicitly, per validity threat #4:** this is a side observation, not a causal
claim. The prompt changed (attribution rules + schema + exemplars added) between the two
recordings, so a positive delta cannot be attributed to attribution improving quality — it
is equally consistent with ordinary Groq replica jitter (ADR-0019 measured std=0.748/issue).
Stated flat, in both directions: attribution did **not** improve judge means (no such claim
is made), and it did **not** regress them either — both deltas land inside the tolerance
band the gate itself defines as noise. Neither is hidden behind the other.

### Baseline decision: do NOT re-baseline

**Decision: `reports/eval_baseline.json` stays unchanged.** Three reasons:

1. **The delta is inside the gate's own noise band.** +0.1667 (k8s, band 0.22) and +0.3636
   (vscode, band 0.45) are both within the 2×SEM tolerance ADR-0019 derived specifically to
   absorb re-record jitter. By the gate's own definition this is not a regression — there is
   no quality justification to re-baseline on it.
2. **Keep the eval reference separate from the shipped feature** — additive features should
   not force a re-baseline every time, or every future additive feature does the same.
3. **Separation of concerns**: the quality-regression gate keeps testing the pre-attribution
   synthesis path; attribution fidelity is measured on its own terms (this ADR) via its own
   cassette.

**Finding, discovered while verifying this decision, then resolved (not worked around):**
`ATTRIBUTION RULES` (prompt section, schema field, few-shot exemplars) had been added
**unconditionally** to `src/triage_iq/prompts/triage_prompt.py` — no flag, every synthesis call
sent the attribution prompt regardless. Verified directly: restoring
`eval/cassettes/eval_cassette.json` to its pre-`ced5252` byte-exact state (hash `c9966414...`,
equal to `reports/eval_baseline.json`'s `cassette_hash`) and re-running the suite produced
`CassetteMissError` on every synthesis call — current code's request never matched the clean
cassette, because that cassette was recorded under the old prompt.

**Resolution: `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION` feature flag**, same env-var-gated pattern as
the existing `TRIAGE_PROMPT_INCLUDE_BUCKET` toggle in `triage.py`. Off by default.
`SYSTEM_PROMPT_LEGACY` / `build_few_shot_examples_legacy()` are a frozen, byte-exact snapshot of
the pre-attribution prompt (verified `c9966414...`); `SYSTEM_PROMPT` / `build_few_shot_examples()`
(unchanged names, so `tests/test_attribution.py` and `tests/test_api.py` needed no edits) remain
the attribution-augmented versions, used only when the flag is `"1"`.
`eval/cassettes/eval_cassette.json` is restored to the clean recording;
`eval/cassettes/eval_cassette_attribution.json` is a new file holding the attribution recording,
read only by `scripts/measure_attribution_fidelity.py` (which sets the flag on and points at it).

**Second-order finding, also resolved:** the prompt flag alone did not fully restore
replayability — `TriagePlan` gained the `declared_attribution` field unconditionally too
(independent of the prompt flag), so `plan.model_dump()` always emits it, changing the judge's
input text vs. what the clean cassette's judge calls were recorded against.
`eval/run_eval.py`'s `plan_json` construction now excludes it
(`model_dump(exclude={"declared_attribution"})`) — the exact same workaround pattern the file
already documents for ADR-0015's `grounding`/`grounding_status` fields.

**Verified end-to-end:** `python eval/run_eval.py` (default, flag off) now reproduces
`reports/eval_baseline.json` exactly — means 8.3636 / 10.5185 / 10.1538, cassette hash
`c9966414...` matching the baseline's recorded hash. `python scripts/measure_attribution_fidelity.py`
(flag on, dedicated cassette) reproduces `reports/attribution_fidelity.json` byte-for-byte. Full
suite (`eval/test_quality_regression.py`, `eval/test_invariants.py`, `tests/test_attribution.py`,
`tests/test_api.py`) — 63/63 pass, zero regressions, zero skips.

### Decision: ship / flag-gate / reject

**Ship.** 100% of citations grounded-or-honestly-flagged as fabricated is 247/249 (99.2%),
0 compliance failures, and the 2 misattributed cases are pre-existing classifier misses
(ADR-0015) rather than new fabrication induced by the attribution prompt. `verify_declared_attribution`
ships as an additive safety net (FLAG, not strip — same policy as ADR-0015) to catch the
~0.8% fabrication rate at read time. The blanket-citation caveat is documented above and
tracked as a follow-up (selectivity, not fidelity, is the next thing to measure) rather than
a blocker — the ADR's stated goal was fabrication resistance, not citation aptness.

## Consequences

- The additive `declared_attribution` field lands in `/triage` responses regardless of the ship
  decision above — it is `None`-safe and does not change existing response shape.
- **Decided:** `reports/eval_baseline.json` is NOT re-derived (see "Baseline decision" above).
  The grounding ratchet and known-case pins (`eval/test_invariants.py`) needed no change —
  re-run against the attribution cassette, both pass unmodified (same two pre-existing cases,
  #13057 k8s / #311836 vscode).
- **Resolved:** the attribution prompt addition is now gated behind `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION`
  (off by default), so production behavior and the eval gate are unaffected until this env var is
  explicitly set. `eval/cassettes/eval_cassette.json` is the clean recording again (matches
  `reports/eval_baseline.json`'s `cassette_hash` exactly); `eval/cassettes/eval_cassette_attribution.json`
  is the new dedicated attribution recording. `eval/run_eval.py`'s `plan_json` construction
  excludes `declared_attribution` for the same reason ADR-0015 excluded `grounding`/
  `grounding_status` before its own re-record. Full suite green (63/63) — see "Baseline decision"
  above for the complete before/after.
- **What reaches production:** nothing, until `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1` is explicitly
  set on the Cloud Run service (a separate, deliberate deploy decision, not part of this ADR).
  Merging this branch to `main` changes no live behavior by itself — the flag defaults off in
  every environment, including production, until that env var is set.
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
