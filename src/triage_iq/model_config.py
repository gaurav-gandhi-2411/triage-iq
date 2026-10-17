from __future__ import annotations

"""Single source of truth for Groq model IDs.

Every consumer of a Groq model ID -- TriageAssistant, app.py's cost
estimator, record-cassette.yml's quota probe, and health-monitor.yml's
model-availability check -- imports from here instead of hardcoding the string
literal. Groq deprecated the previous triage/judge models on 2026-08-16; the fix
had to touch six separate files (see PR #101) specifically because there was no
single place to change it. This module exists so the next deprecation is a
one-line change.
"""

TRIAGE_MODEL: str = "openai/gpt-oss-120b"
# 2026-08-30, corrected 2026-09-03: selected over gpt-oss-20b (few-shot) per ADR-0054.
# The ADR's original "44/44 vs 29/31" parse-success figures were never traceable to a
# committed artifact and did not match the raw per-call records recovered afterward
# (verified: 20/20 vs 19/20 on the pre-registered 20-issue sample) -- see ADR-0054's
# 2026-09-03 correction for the full accounting and the revised basis for this
# selection (completion-token distribution and ceiling headroom, not parse-success,
# which does not cleanly resolve between the two models on the corrected numbers
# either). gpt-oss-20b (no few-shot) and qwen/qwen3.6-27b were eliminated per
# ADR-0053 and the bake-off pre-registration respectively -- see
# docs/eval/bakeoff_prereg_2026-08-29.md.
# JUDGE_MODEL (Groq "llama-3.3-70b-versatile") was removed 2026-09-24. Groq retired that
# model on 2026-08-16, and nothing that runs used it: the eval judge is local qwen3:8b via
# Ollama (ADR-0019), set in eval/run_eval.py and eval/record_cassettes.py. Its only live
# reader was health-monitor.yml's model-availability check, which failed on every run for a
# model production never calls. That check now covers TRIAGE_MODEL only, and TriageJudge
# requires an explicit model instead of defaulting to a retired one.

# USD per million tokens, separate input/output rates -- Groq's published rate for the
# current TRIAGE_MODEL varies enough between input and output (4x) that a single blended
# constant hides real error. Cost-estimation only; not billing-authoritative.
# 2026-09-03: corrected from a stale blended 0.27 (the retired llama-3.1-8b-instant
# rate, itself never actually wired to the cost formula below -- see the same date's
# fix in models/triage.py:triage_with_metadata, which had its own hardcoded 0.27
# duplicate that ignored this constant entirely). Source: Groq's own published pricing,
# https://console.groq.com/docs/model/openai/gpt-oss-120b (fetched 2026-09-03) --
# $0.15/M input tokens, $0.60/M output tokens. Re-verify if TRIAGE_MODEL changes; these
# rates are model-specific, not a platform-wide constant.
TRIAGE_PRICE_PROMPT_PER_MTOK: float = 0.15
TRIAGE_PRICE_COMPLETION_PER_MTOK: float = 0.60
