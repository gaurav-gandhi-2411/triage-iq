# Phase 3b design ("ML decides, LLM explains"): scope, flags and pre-registered checks

Status: pre-registered by the orchestrator in the unattended run (2026-10-23), before any Phase 3b code is written.
Sources: `spec-quality-push.md` section 8 (Phase 3b) and section 10 (Phase 5); `spec-autonomous-run.md` Q7/Q8.
Spec targets and acceptance checks are NOT changed here; this file says how they are checked and what gates the
single re-record.

## What exists (measured, merged)

| Piece | Evidence | Usable in serving? |
|---|---|---|
| Component (k8s) | TF-IDF+LR `sig/*`, top-3 95.4% [94.3, 96.5] (strict = lenient, #282); RAPS conformal (ADR-0069, exploratory) | yes, as a new served model behind a flag |
| Component (vscode) | family-only TF-IDF+LR, strict top-3 36.4% vs a 41.8% coverage bound, in-family 87.0%; RAPS (ADR-0071, exploratory) | yes, behind a flag; taxonomy drift needs GG's retraining decision |
| Priority (k8s) | P3 0.522 vs majority 0.409 (#264) uses creation-time labels the API does not receive; text-only P4 is being evaluated | only P4 (API inputs are repo, title, body, number, created_at) |
| Priority (vscode) | no priority labels | no model; LLM as today |
| Resolution | no learned point clears the serving rule on the censored population (#260); kNN negative (#265) | no change; the 3.04 d constant is GG's decision |
| Next-step evidence | audited fix links only (closer + merged, #280/#281): k8s 1,957, vscode 11,927 links | needs an evidence artifact (below) |
| Retrieval index | served index holds the 2014-2016 k8s corpus and 13,315 vscode issues; 0 duplicate/fix targets of the 150-issue set are in it | a refreshed index is a model cutover |

## Flags (all default OFF; read like the Phase 3a flag: env var in `loader.load_all`, constructor argument, class default False)

1. `TRIAGE_PRIORITY_MODEL=1` (k8s only): `priority_guess` comes from the P4 model; the LLM writes `priority_rationale`
   for that value. Shipped only if P4 meets its pre-registered criterion.
2. `TRIAGE_RESOLUTION_EVIDENCE=1`: the prompt gets, for the top-3 retrieved neighbours, the title of their audited fix
   PR (closer + merged) and the last non-bot closing comment (bots filtered: `k8s-ci-robot`, `kubernetes-prow`, `vscode-triage-bot`
   and any `[bot]` login), each truncated; the LLM must cite neighbours by number for next steps. Evidence comes from an
   offline artifact keyed by issue number, built from derived_v2 for the issues in the served index.
3. `TRIAGE_COMPONENT_FROM_CONFORMAL=1`: generalises the Phase 3a rule from "classifier top-3" to "the classifier's
   conformal set" (k8s ADR-0069, vscode ADR-0071); the classifier's top-1 is served unless the LLM picks another label
   inside the set with a grounded reason.
4. The new Phase 1 classifiers and a refreshed retrieval index are model artifacts; serving them is a cutover and stays GG's decision.

All prompt changes for flags 1-3 land together as prompt v3 behind the flags, because the single re-record can only
happen once.

## Pre-registered Q7 acceptance checks (all zero-token; together they gate Q8)

- **C1 No change when off.** With every new flag off, the 127 recorded issues (64 + 63) replay in strict cassette mode
  with responses identical to main (byte-identical JSON after removing timing fields). Any difference fails.
- **C2 Priority.** P4's test accuracy minus the majority's has a paired-bootstrap lower bound > 0 (same protocol as
  #263). If not, flag 1 is not shipped and priority stays with the LLM.
- **C3 Evidence coverage.** On the 150-issue set (design 2.1) with the retriever that will be served, at least 50% of
  k8s rows have >= 1 of their top-3 neighbours with evidence (fix-PR title or non-bot closing comment). Below that,
  flag 2 is not shipped (the block would cost tokens for little grounding). vscode is reported, not gated, until its
  refreshed index exists.
- **C4 Token budget.** Mean prompt tokens of prompt v3 on the 150-issue set (all flags on, tokenizer count, not chars/4)
  <= 1.15 x the v2 prompt's mean on the same set, and the projected re-record total fits the plan (~908K tokens at the
  160K/day ledger cap). Above that, the evidence block is truncated further before the re-record.
- **C5 Degraded mode.** Every new field is present and well-formed in every degraded path (`_make_fallback_plan`,
  provider error, schema invalid), proven by unit tests.
- **C6 CI.** Full suite, structural invariants and the cassette-replayed quality job green on the PRs.

Q8 (the single re-record of the 150-issue set) starts only after C1-C6 pass AND the retriever to be evaluated is frozen
into the set (design 2.1 records it as UNSCORABLE on the served index). The spec's Phase 3b acceptance (end-to-end
component accuracy >= classifier top-1; priority beats majority with CI; next-step grounding improves; no judge
regression beyond noise) needs the recorded outputs and is measured in Q8, not here.

## Known blockers

- vscode embeddings (refreshed vscode index, Q5a) need the Kaggle GPU quota (weekly reset).
- A refreshed index and the new classifiers are cutovers (GG). The re-record must evaluate what will be served, so the
  order is: GG approves the cutover candidates -> freeze them into the set -> re-record. If GG has not decided by the
  stop time, Q8 does not start in this run, and the return brief says so.
