from __future__ import annotations

"""Single source of truth for Groq model IDs.

Every consumer of a Groq model ID -- TriageAssistant, TriageJudge, app.py's cost
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
JUDGE_MODEL: str = "llama-3.3-70b-versatile"
# NOTE (2026-08-30, found while updating TRIAGE_MODEL, not fixed here -- out of this
# change's scope): this constant is retired (Groq deprecated it 2026-08-16) and is read
# live by .github/workflows/health-monitor.yml for a model-availability check, but every
# actual judge call site (eval/run_eval.py, eval/record_cassettes.py) already shadows it
# with a local "qwen3:8b" (ADR-0019) and never imports this constant. health-monitor.yml
# is therefore checking availability of a model nothing in this codebase actually calls
# -- a separate, pre-existing staleness, not part of the triage-model selection this
# session made. Flagged for its own fix, not bundled in here.

# USD per million tokens, blended in/out. Cost-estimation only; not billing-authoritative.
# NOTE: still the llama-3.1-8b-instant rate -- gpt-oss-120b's actual Groq pricing has not
# been verified against this constant. Cost estimates (TriageAssistant.triage_with_metadata's
# estimated_cost_usd) will be wrong until this is checked and updated.
TRIAGE_PRICE_PER_MTOK: float = 0.27
