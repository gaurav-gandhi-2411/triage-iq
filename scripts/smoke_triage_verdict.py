"""Verdict for the post-deploy /triage smoke check (deploy.yml and post-deploy-smoke.yml).

Reads the /triage response body on stdin and the HTTP status code as argv[1]. Exit codes:

  0  PASS        real LLM answer: predicted_component set, _degraded false, summary non-empty.
  3  QUOTA_PASS  the ONLY problem is a Groq daily-quota 429 (llm_status degraded_provider_error,
                 reason exactly rate_limited_tpd) and the signals-only plan is otherwise sound.
                 This is a budget state, not a defect in the build, so the deploy may promote;
                 the caller must warn and schedule a full smoke test once the pool recovers.
  1  FAIL        anything else: non-200, empty component or summary, any other degradation
                 (401 key rejected, model 404, parse failure, per-minute 429 ...).

The reason match is EXACT on purpose. A looser rule ("any degraded_provider_error") would let a
rejected key or a retired model promote a build whose LLM never answers, which is the failure the
smoke test exists to stop (ADR-0055 P1b, ADR-0063).
"""

from __future__ import annotations

import json
import sys

QUOTA_STATUS = "degraded_provider_error"
QUOTA_REASON = "rate_limited_tpd"

PASS, FAIL, QUOTA_PASS = 0, 1, 3


def verdict(http_code: str, body: str) -> tuple[int, str]:
    """Return (exit_code, message) for one /triage response."""
    if http_code != "200":
        return FAIL, f"FAILED: /triage returned HTTP {http_code} -- body: {body[:500]}"
    try:
        d = json.loads(body)
    except ValueError:
        return FAIL, "FAILED: /triage body is not JSON"
    if not isinstance(d, dict):
        return FAIL, "FAILED: /triage body is not a JSON object"

    component = d.get("predicted_component")
    degraded = d.get("_degraded")
    summary = d.get("triage_summary")
    llm_status = d.get("_llm_status")
    llm_reason = d.get("_llm_status_reason")

    failures = []
    if not component:
        failures.append("predicted_component is empty")
    if not summary or not str(summary).strip():
        failures.append("triage_summary is empty")

    if degraded is False:
        if failures:
            return FAIL, "FAILED: " + "; ".join(failures)
        return (
            PASS,
            f"Triage check passed -- predicted_component={component} degraded=False llm_status={llm_status}",
        )

    if degraded is True and llm_status == QUOTA_STATUS and llm_reason == QUOTA_REASON:
        if failures:
            return FAIL, "FAILED: " + "; ".join(failures)
        return QUOTA_PASS, (
            f"QUOTA PASS -- /triage degraded only because of {QUOTA_REASON} "
            f"(Groq daily token pool exhausted); predicted_component={component}"
        )

    failures.append(
        f"_degraded is {degraded!r}, expected False -- llm_status={llm_status!r} llm_status_reason={llm_reason!r}"
    )
    return FAIL, "FAILED: " + "; ".join(failures)


def main(argv: list[str]) -> int:
    code, msg = verdict(argv[1] if len(argv) > 1 else "", sys.stdin.read())
    print(msg)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
