"""Deterministic verification that a TriagePlan's free-text resolution summary doesn't
contradict the numeric resolution interval it was generated alongside (LEVER 4, ADR-0042).

Motivating case (ADR-0037, k8s-14756): expected_resolution_summary said "typically 1 day or
less for a straightforward configuration tweak" against a numeric interval of [2.8d, 21.6d] --
the prose's claimed range has ZERO overlap with the actual interval it was supposed to be
summarizing. A real correctness defect (the model contradicts numbers it was directly given in
its own prompt), separate from ADR-0037's hedging-tone investigation into the same synthesis
call. Measured over the current cassette (reports/lever4_prose_number_consistency.json):
0/64 plans contradict -- not currently material, but built as a standing, zero-cost check
(same "measure, keep watching" discipline as grounding.py's fabrication_rate) rather than
skipped because today's rate happens to be zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNIT_TO_DAYS: dict[str, float] = {
    "hour": 1 / 24,
    "hours": 1 / 24,
    "hr": 1 / 24,
    "hrs": 1 / 24,
    "day": 1.0,
    "days": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "month": 30.0,
    "months": 30.0,
}
_UNIT_PATTERN = "|".join(sorted(UNIT_TO_DAYS, key=len, reverse=True))

# "1-3 days", "1 to 3 days" -- an explicit range.
_RANGE_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b", re.IGNORECASE
)
# "1 day or less", "within 1 day", "less than 1 day", "under 1 day" -- an upper-bound-only claim.
_UPPER_BOUND_RE = re.compile(
    rf"(?:within|less than|under|no more than)\s+(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b"
    rf"|(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\s+or less\b",
    re.IGNORECASE,
)
# "more than 1 day", "at least 1 day", "1 day or more" -- a lower-bound-only claim.
_LOWER_BOUND_RE = re.compile(
    rf"(?:more than|at least|over)\s+(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b"
    rf"|(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\s+or more\b",
    re.IGNORECASE,
)
# A bare "N <unit>" claim not already consumed above -- treated as a point estimate with a
# +/-25% tolerance window (so "3 days" doesn't get flagged against a [2.9, 3.1] interval over
# what is otherwise a reasonable rounding of the same number).
_POINT_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})\b", re.IGNORECASE)


@dataclass(frozen=True)
class ResolutionConsistencyReport:
    """Result of checking a plan's prose resolution summary against its own numeric interval.

    has_time_claim: whether any parseable time expression was found in the summary at all.
    implied_range_days: the (lo, hi) day range every extracted time expression implies,
        unioned together. None if has_time_claim is False.
    actual_range_days: the plan's own (expected_resolution_lower_days, expected_resolution_upper_days).
    contradicts: True only when implied_range_days has ZERO overlap with actual_range_days --
        a narrower prose estimate that sits INSIDE the true interval is NOT a contradiction,
        that's an expected, useful narrowing of a wide bound.
    """

    has_time_claim: bool
    implied_range_days: tuple[float, float] | None
    actual_range_days: tuple[float, float]
    contradicts: bool


def _extract_implied_range_days(text: str) -> tuple[float, float] | None:
    spans_consumed: list[tuple[int, int]] = []
    los: list[float] = []
    his: list[float] = []

    for m in _RANGE_RE.finditer(text):
        a, b, unit = float(m.group(1)), float(m.group(2)), m.group(3).lower()
        factor = UNIT_TO_DAYS[unit]
        los.append(min(a, b) * factor)
        his.append(max(a, b) * factor)
        spans_consumed.append(m.span())

    for m in _UPPER_BOUND_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans_consumed):
            continue
        val, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        his.append(float(val) * UNIT_TO_DAYS[unit.lower()])
        los.append(0.0)
        spans_consumed.append(m.span())

    for m in _LOWER_BOUND_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans_consumed):
            continue
        val, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        los.append(float(val) * UNIT_TO_DAYS[unit.lower()])
        his.append(float("inf"))
        spans_consumed.append(m.span())

    for m in _POINT_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans_consumed):
            continue
        val, unit = float(m.group(1)), m.group(2).lower()
        days = val * UNIT_TO_DAYS[unit]
        los.append(days * 0.75)
        his.append(days * 1.25)

    if not los:
        return None
    return min(los), max(his)


def verify_resolution_consistency(
    expected_resolution_summary: str,
    expected_resolution_lower_days: float,
    expected_resolution_upper_days: float,
) -> ResolutionConsistencyReport:
    """Check whether the free-text summary's implied time range overlaps the numeric interval
    it was generated alongside. See module docstring for the motivating case."""
    actual = (expected_resolution_lower_days, expected_resolution_upper_days)
    implied = _extract_implied_range_days(expected_resolution_summary)
    if implied is None:
        return ResolutionConsistencyReport(
            has_time_claim=False,
            implied_range_days=None,
            actual_range_days=actual,
            contradicts=False,
        )
    implied_lo, implied_hi = implied
    overlaps = not (implied_hi < actual[0] or implied_lo > actual[1])
    return ResolutionConsistencyReport(
        has_time_claim=True,
        implied_range_days=implied,
        actual_range_days=actual,
        contradicts=not overlaps,
    )
