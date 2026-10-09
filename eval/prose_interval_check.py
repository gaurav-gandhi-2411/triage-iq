from __future__ import annotations

"""Does a recorded synthesis agree with the interval the API serves? (ADR-0059 addendum 2026-10-09)

The API returns a conformal interval (CQR) next to the plan. Until 2026-10-09 the prompt showed the
raw model interval, so the model wrote prose about numbers a reader never saw. The assistant now
applies the CQR adjustment before synthesis and labels the prompt interval "(coverage-calibrated)".
This module checks a recorded cassette entry for exactly that, with no LLM and no model load:

* the entry's prompt interval carries the calibrated marker (absent on every pre-change entry);
* the prose time claim overlaps that interval (the existing LEVER 4 rule, applied to the interval
  the prompt, the plan fields and the API now all share).
"""

import json
import re

from triage_iq.models.resolution_consistency import verify_resolution_consistency

CALIBRATED_MARKER = "80% prediction interval (coverage-calibrated):"

_PROMPT_RE = re.compile(
    r"Repository: (?P<repo>\S+).*?Point estimate: (?P<pt>[\d.]+) days\s*\n"
    r"80% prediction interval(?P<cal> \(coverage-calibrated\))?: "
    r"\[(?P<lo>[\d.]+)d, (?P<hi>[\d.]+)d\]",
    re.S,
)


def parse_entry(entry: dict) -> dict | None:
    """(repo, point, lo, hi, calibrated, plan) of a synthesis entry; None for any other entry.

    The real request is the LAST user turn: earlier user turns are few-shot examples.
    """
    msgs = entry.get("request_messages") or []
    user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
    m = _PROMPT_RE.search(user)
    if not m:
        return None
    try:
        plan = json.loads(entry["response"]["content"])
    except (KeyError, TypeError, ValueError):
        return None
    if "expected_resolution_summary" not in plan:
        return None
    return {
        "repo": m["repo"],
        "point": float(m["pt"]),
        "lo": float(m["lo"]),
        "hi": float(m["hi"]),
        "calibrated": m["cal"] is not None,
        "plan": plan,
    }


def violations(entry: dict) -> list[str]:
    """Reasons this synthesis entry disagrees with the served conformal interval ([] if none)."""
    parsed = parse_entry(entry)
    if parsed is None:
        return ["not a synthesis entry"]
    out = []
    if not parsed["calibrated"]:
        out.append("prompt interval is the raw model interval, not the served conformal one")
    rep = verify_resolution_consistency(
        parsed["plan"]["expected_resolution_summary"], parsed["lo"], parsed["hi"]
    )
    if rep.contradicts:
        out.append(
            f"prose range {rep.implied_range_days} has no overlap with the prompt interval "
            f"[{parsed['lo']}, {parsed['hi']}] d"
        )
    return out
