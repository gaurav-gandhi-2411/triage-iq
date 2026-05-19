# ADR-0003: Judge default — llama-3.3-70b-versatile retained

Status: Accepted  
Date: 2026-05-19

## Context

The baseline eval (ADR-0002, `reports/triage_results_judge_llama70b.json`) uses
`llama-3.3-70b-versatile` (Groq) as the LLM-as-judge. The production triage model is
`llama-3.1-8b-instant` — same Llama family (Meta). The concern: a larger Llama model
may score a smaller Llama model's outputs more charitably than a cross-family judge
would, inflating the reported 73% Full System score.

To test this, a cross-family judge comparison was run. After two failed provider
attempts (Groq qwen3-32b hit TPD; Gemini 2.5 Flash had a 20 RPD free-tier wall — see
ADR-0002 amendments), a complete 180-call run was completed using **Cohere Command A
(`command-a-03-2025`)** via the Cohere Trial API.

Full comparison data: `reports/judge_comparison.md`.

## Findings

| Metric | Value |
|---|---|
| Cohere Full System score | 10.40/15 (69.3%) |
| llama-70b Full System score | 10.93/15 (72.9%) |
| Gap (Cohere − llama) | −0.53 pts (−3.6pp) |
| Pearson r (per-issue totals, n=60) | **0.729** |
| Within-1-pt agreement (5 of 6 dims) | 98–100% |
| Primary divergence | `next_steps_actionability`: −0.67 pts |

The Pearson correlation of **r = 0.729** exceeds the 0.70 threshold established in
planning. The −3.6pp gap is below the 5pp threshold for flagging systematic inflation.
The judges agree on issue ranking; the disagreement is concentrated in one dimension
(`next_steps_actionability`) where Cohere applies a stricter bar for "precise,
ordered, repo-appropriate" steps.

## Decision

**Retain `llama-3.3-70b-versatile` as the default judge.** Do not switch.

Rationale: r ≥ 0.70 and gap < 5pp. The family-bias hypothesis is not supported at a
material effect size. The 73% baseline is a mild overestimate at most (true value
likely 69–73%). Switching defaults for a 3.6pp gap on an eval set of 60 issues is not
justified — it would reduce comparability with historical reports without a proportionate
accuracy benefit.

**Recommended action for GG:** No action required on the judge default. If the gold set
grows beyond n=100 and a more rigorous cross-family verification is needed, re-run with
Cohere Command A (same trial key, same checkpoint infrastructure). If Cohere trial quota
is exhausted, a paid Cohere key or an upgraded Google AI Studio project (1,500 RPD for
`gemini-2.5-flash` on paid tiers) are the next options.

## Consequences

- **What changes:** Nothing changes in the production eval pipeline. Default judge remains
  `llama-3.3-70b-versatile` via Groq.
- **What is documented:** Cohere Command A is the validated cross-family sanity-check
  option. Run it with `--judge-provider cohere --judge-model command-a-03-2025`. The
  checkpoint infrastructure (model-slug-scoped JSONL files) means future runs resume
  cleanly.
- **Historical reports:** `reports/triage_results_judge_llama70b.json` and derived
  reports are valid. They may overstate quality by up to ~4pp relative to a strictly
  cross-family judge. This is within acceptable range for a portfolio eval.
- **What stays open:** If a future eval shows the gap widening (e.g., after model
  updates change llama-70b's scoring behavior), re-run the cross-family check before
  publishing results.

## Alternatives considered

- **Switch default to Cohere Command A:** Rejected. Gap is below threshold; would break
  comparability with historical scores without proportionate benefit.
- **Require both judges and report both scores:** Over-engineering for a 60-issue eval.
  Reserve for when gold set exceeds n=200.
- **Add a third judge to break the tie:** Not needed — r=0.73 is not a tie. The two
  judges agree on rankings; only the absolute scale differs by 3.6pp.
