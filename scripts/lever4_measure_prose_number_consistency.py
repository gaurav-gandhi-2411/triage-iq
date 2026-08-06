"""LEVER 4 measurement: how often does the LLM's free-text expected_resolution_summary
contradict the numeric interval (expected_resolution_lower_days/upper_days) it was given?

Uses src/triage_iq/models/resolution_consistency.py::verify_resolution_consistency() directly
-- this script is a thin runner over the current cassette, not a second copy of the check logic.

Reads:  eval/cassettes/eval_cassette.json (offline, no live LLM calls)
Writes: reports/lever4_prose_number_consistency.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.resolution_consistency import verify_resolution_consistency  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CASSETTE_PATH = Path("eval/cassettes/eval_cassette.json")
REPORTS = Path("reports")


def main() -> None:
    entries = json.loads(CASSETTE_PATH.read_text(encoding="utf-8"))["entries"]
    results = []
    for key, entry in entries.items():
        try:
            plan = json.loads(entry["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not isinstance(plan, dict) or "expected_resolution_summary" not in plan:
            continue
        lower = plan.get("expected_resolution_lower_days")
        upper = plan.get("expected_resolution_upper_days")
        summary = plan.get("expected_resolution_summary", "")
        if lower is None or upper is None:
            continue

        report = verify_resolution_consistency(summary, lower, upper)
        results.append(
            {
                "cassette_key": key,
                "has_time_claim": report.has_time_claim,
                "implied_range_days": list(report.implied_range_days)
                if report.implied_range_days
                else None,
                "actual_range_days": list(report.actual_range_days),
                "contradiction": report.contradicts,
                "summary": summary,
            }
        )

    n_total = len(results)
    n_with_claim = sum(1 for r in results if r["has_time_claim"])
    n_contradictions = sum(1 for r in results if r["contradiction"])
    contradictions = [r for r in results if r["contradiction"]]

    out = {
        "n_synthesis_plans_checked": n_total,
        "n_with_extractable_time_claim": n_with_claim,
        "n_contradictions": n_contradictions,
        "contradiction_rate_overall": round(n_contradictions / n_total, 4) if n_total else 0.0,
        "contradiction_rate_among_plans_with_a_claim": (
            round(n_contradictions / n_with_claim, 4) if n_with_claim else 0.0
        ),
        "contradictions": contradictions,
    }

    log.info("Checked %d synthesis plans (%d had an extractable time claim)", n_total, n_with_claim)
    log.info(
        "Contradictions: %d (%.1f%% of all plans, %.1f%% of plans with a time claim)",
        n_contradictions,
        out["contradiction_rate_overall"] * 100,
        out["contradiction_rate_among_plans_with_a_claim"] * 100,
    )
    for c in contradictions:
        log.info(
            "  [%s] implied=%s actual=%s -- %r",
            c["cassette_key"][:12],
            c["implied_range_days"],
            c["actual_range_days"],
            c["summary"],
        )

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "lever4_prose_number_consistency.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    log.info("Wrote reports/lever4_prose_number_consistency.json")


if __name__ == "__main__":
    main()
