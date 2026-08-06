"""Unit tests for src/triage_iq/models/resolution_consistency.py (LEVER 4, ADR-0042).

No model load, no I/O -- pure regex/logic tests of verify_resolution_consistency().
"""

from __future__ import annotations

from triage_iq.models.resolution_consistency import verify_resolution_consistency


def test_no_time_claim_never_contradicts() -> None:
    r = verify_resolution_consistency("Quick fix expected once triaged.", 1.0, 7.0)
    assert r.has_time_claim is False
    assert r.contradicts is False


def test_motivating_case_k8s_14756_is_a_contradiction() -> None:
    """ADR-0037's actual example: "1 day or less" claimed against a [2.8, 21.6] interval --
    the claim's entire implied range sits below the interval's own lower bound."""
    r = verify_resolution_consistency(
        "typically 1 day or less for a straightforward configuration tweak", 2.8, 21.6
    )
    assert r.has_time_claim is True
    assert r.contradicts is True


def test_narrower_estimate_inside_wide_interval_is_not_a_contradiction() -> None:
    """A tighter prose guess inside a wide numeric interval is a useful narrowing, not a
    contradiction -- this is the common, expected case (verified against 63/64 real cassette
    entries, reports/lever4_prose_number_consistency.json)."""
    r = verify_resolution_consistency("typically resolved within 1-3 days", 0.2, 53.7)
    assert r.has_time_claim is True
    assert r.contradicts is False


def test_range_touching_interval_boundary_overlaps() -> None:
    r = verify_resolution_consistency("usually 3-5 days", 5.0, 10.0)
    assert r.contradicts is False


def test_lower_bound_only_claim_outside_interval_contradicts() -> None:
    r = verify_resolution_consistency("more than 2 weeks to resolve", 1.0, 5.0)
    assert r.has_time_claim is True
    assert r.contradicts is True


def test_lower_bound_only_claim_inside_interval_does_not_contradict() -> None:
    r = verify_resolution_consistency("at least 2 days needed", 1.0, 30.0)
    assert r.contradicts is False


def test_point_estimate_within_tolerance_does_not_contradict() -> None:
    r = verify_resolution_consistency("about 3 days", 2.9, 3.1)
    assert r.contradicts is False


def test_point_estimate_genuinely_outside_interval_contradicts() -> None:
    r = verify_resolution_consistency("roughly 10 days", 0.5, 2.0)
    assert r.has_time_claim is True
    assert r.contradicts is True


def test_hours_unit_converted_to_days() -> None:
    r = verify_resolution_consistency("resolved within 6 hours", 0.0, 1.0)
    assert r.contradicts is False


def test_weeks_unit_converted_to_days() -> None:
    r = verify_resolution_consistency("typically 1-2 weeks", 5.0, 20.0)
    assert r.contradicts is False


def test_months_unit_outside_interval_contradicts() -> None:
    r = verify_resolution_consistency("could take 2-3 months", 0.1, 10.0)
    assert r.has_time_claim is True
    assert r.contradicts is True
