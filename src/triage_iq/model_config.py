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

TRIAGE_MODEL: str = "llama-3.1-8b-instant"
JUDGE_MODEL: str = "llama-3.3-70b-versatile"

# USD per million tokens, blended in/out. Cost-estimation only; not billing-authoritative.
TRIAGE_PRICE_PER_MTOK: float = 0.27
