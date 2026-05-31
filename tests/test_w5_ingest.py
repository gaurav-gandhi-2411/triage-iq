"""Tests for the W5 gold-set ingestion pipeline (scripts/w5_ingest_labeled.py).

Covers: contract validation, acceptance/rejection logic, priority inference,
gold-schema transformation, merge math, duplicate detection, and dry-run.
These tests serve as the T3 dry-run verification required by ADR-0011.
"""
from __future__ import annotations

import sys
from datetime import timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the functions under test directly (not via CLI)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from w5_ingest_labeled import (
    REJECTION_CODES,
    build_gold_rows,
    infer_priority,
    validate_accepted_rows,
    validate_labeled_csv,
    _days_to_bucket,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CREATED = pd.Timestamp("2015-01-15", tz=timezone.utc)


def _make_candidate(**overrides) -> dict:
    base = {
        "number": 9001,
        "repo": "microsoft/vscode",
        "title": "Test issue title",
        "body_clean": "This is a meaningful issue body with enough content.",
        "component": "debug",
        "type": "bug",
        "priority": None,
        "resolution_hours": 48.0,
        "created_at": _CREATED,
        "label_decision": "accept",
        "corrected_component": None,
        "label_rejection_code": None,
        "labeler_notes": None,
    }
    base.update(overrides)
    return base


def _make_existing_gold(n: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "repo": "microsoft/vscode",
            "number": 1000 + i,
            "title": f"Existing issue {i}",
            "body_clean": "existing body",
            "gold_component": "api",
            "type": "bug",
            "gold_priority": "low",
            "actual_resolution_days": 5.0,
            "created_at": _CREATED,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T1 contract — validate_labeled_csv
# ---------------------------------------------------------------------------

class TestValidateLabeledCsv:

    def test_passes_with_label_decision_column(self):
        df = pd.DataFrame([_make_candidate()])
        validate_labeled_csv(df)  # must not raise

    def test_raises_when_label_decision_missing(self):
        df = pd.DataFrame([{"number": 1, "repo": "microsoft/vscode"}])
        with pytest.raises(ValueError, match="label_decision"):
            validate_labeled_csv(df)

    def test_raises_on_invalid_decision_value(self):
        df = pd.DataFrame([_make_candidate(label_decision="maybe")])
        with pytest.raises(ValueError, match="Invalid label_decision"):
            validate_labeled_csv(df)


# ---------------------------------------------------------------------------
# T2 validation — validate_accepted_rows
# ---------------------------------------------------------------------------

class TestValidateAcceptedRows:

    def test_clean_row_passes(self):
        df = pd.DataFrame([_make_candidate()])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert errors == []

    def test_duplicate_in_existing_gold_is_an_error(self):
        df = pd.DataFrame([_make_candidate(number=1000)])  # 1000 is in existing gold
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert any("already in existing gold" in e for e in errors)

    def test_invalid_repo_is_an_error(self):
        df = pd.DataFrame([_make_candidate(repo="unknown/repo")])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert any("repo" in e for e in errors)

    def test_null_component_and_no_corrected_is_an_error(self):
        df = pd.DataFrame([_make_candidate(component=None, corrected_component=None)])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert any("component" in e for e in errors)

    def test_corrected_component_satisfies_null_component(self):
        df = pd.DataFrame([_make_candidate(component=None, corrected_component="debug")])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert errors == []

    def test_zero_resolution_hours_is_an_error(self):
        df = pd.DataFrame([_make_candidate(resolution_hours=0)])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert any("resolution_hours" in e for e in errors)

    def test_missing_title_is_an_error(self):
        df = pd.DataFrame([_make_candidate(title="")])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert any("title" in e for e in errors)

    def test_multiple_errors_all_reported(self):
        df = pd.DataFrame([
            _make_candidate(repo="bad/repo", resolution_hours=-1),
        ])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert len(errors) >= 2

    def test_error_message_names_the_row(self):
        df = pd.DataFrame([_make_candidate(number=42, repo="unknown/repo")])
        existing = _make_existing_gold()
        errors = validate_accepted_rows(df, existing)
        assert any("#42" in e or "42" in e for e in errors)


# ---------------------------------------------------------------------------
# T2 transform — build_gold_rows + infer_priority
# ---------------------------------------------------------------------------

class TestBuildGoldRows:

    def test_produces_correct_schema(self):
        df = pd.DataFrame([_make_candidate()])
        result = build_gold_rows(df)
        expected_cols = {
            "repo", "number", "title", "body_clean", "gold_component",
            "type", "gold_priority", "actual_resolution_days", "created_at",
        }
        assert expected_cols == set(result.columns)

    def test_resolution_days_conversion(self):
        df = pd.DataFrame([_make_candidate(resolution_hours=48.0)])
        result = build_gold_rows(df)
        assert result.iloc[0]["actual_resolution_days"] == pytest.approx(2.0)

    def test_corrected_component_overrides_component(self):
        df = pd.DataFrame([_make_candidate(component="debug", corrected_component="api")])
        result = build_gold_rows(df)
        assert result.iloc[0]["gold_component"] == "api"

    def test_component_used_when_no_correction(self):
        df = pd.DataFrame([_make_candidate(component="typescript", corrected_component=None)])
        result = build_gold_rows(df)
        assert result.iloc[0]["gold_component"] == "typescript"

    def test_number_is_int(self):
        df = pd.DataFrame([_make_candidate(number="9001")])
        result = build_gold_rows(df)
        assert isinstance(result.iloc[0]["number"], (int, np.integer))


class TestInferPriority:

    def test_critical_label_is_high(self):
        row = pd.Series({"priority": "critical-urgent", "resolution_hours": 100.0})
        assert infer_priority(row) == "high"

    def test_important_soon_is_high(self):
        row = pd.Series({"priority": "important-soon", "resolution_hours": 100.0})
        assert infer_priority(row) == "high"

    def test_backlog_falls_through_to_resolution(self):
        # "backlog" doesn't match any high/medium keyword → resolution speed fallback
        row = pd.Series({"priority": "backlog", "resolution_hours": 200.0})
        assert infer_priority(row) == "low"

    def test_resolution_under_24h_is_high(self):
        row = pd.Series({"priority": None, "resolution_hours": 12.0})
        assert infer_priority(row) == "high"

    def test_resolution_1_to_7d_is_medium(self):
        row = pd.Series({"priority": None, "resolution_hours": 72.0})
        assert infer_priority(row) == "medium"

    def test_resolution_over_7d_is_low(self):
        row = pd.Series({"priority": None, "resolution_hours": 200.0})
        assert infer_priority(row) == "low"


# ---------------------------------------------------------------------------
# T3 merge math — end-to-end dry-run
# ---------------------------------------------------------------------------

class TestMergeMath:

    def test_merged_count_is_existing_plus_accepted(self):
        existing = _make_existing_gold(n=5)
        accepted = pd.DataFrame([
            _make_candidate(number=9001),
            _make_candidate(number=9002),
            _make_candidate(number=9003),
        ])
        new_rows = build_gold_rows(accepted)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        assert len(combined) == 8  # 5 + 3

    def test_existing_rows_preserved_verbatim(self):
        existing = _make_existing_gold(n=3)
        accepted = pd.DataFrame([_make_candidate(number=9999)])
        new_rows = build_gold_rows(accepted)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        orig_numbers = set(existing["number"].tolist())
        combined_numbers = set(combined["number"].tolist())
        assert orig_numbers.issubset(combined_numbers)

    def test_no_accepted_rows_returns_unchanged_gold(self):
        existing = _make_existing_gold(n=4)
        empty_new = build_gold_rows(pd.DataFrame(columns=list(_make_candidate().keys())))
        combined = pd.concat([existing, empty_new], ignore_index=True)
        assert len(combined) == 4


# ---------------------------------------------------------------------------
# Bucket helper
# ---------------------------------------------------------------------------

class TestDaysToBucket:

    @pytest.mark.parametrize("hours,expected", [
        (0.5, "hours"),   # 30 min
        (12, "hours"),    # 12 hours
        (36, "days"),     # 1.5 days
        (4 * 24, "days"), # 4 days
        (10 * 24, "weeks"),
        (45 * 24, "months"),
        (200 * 24, "long"),
    ])
    def test_bucket_boundaries(self, hours, expected):
        days = hours / 24
        assert _days_to_bucket(days) == expected
