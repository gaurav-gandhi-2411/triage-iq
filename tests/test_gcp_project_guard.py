"""Unit tests for the pre-flight GCP project guard (scripts/publish_models.py,
scripts/verify_model_manifest.py) -- exercises the actual failure path (wrong active project
exits non-zero), not just the EXPECTED_PROJECT constant value. Written per GG's instruction
during the 2026-08-12 gaurav.gandhi1129@gmail.com project migration: no prior test existed for
this hard gate at all, so this covers both scripts' independent copies of the same check.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "scripts")
import publish_models  # noqa: E402
import verify_model_manifest  # noqa: E402

MODULES = [publish_models, verify_model_manifest]


def _fake_gcloud_result(project: str) -> MagicMock:
    result = MagicMock()
    result.stdout = f"{project}\n"
    return result


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_wrong_active_project_hard_stops(module, capsys):
    with patch.object(module.subprocess, "run", return_value=_fake_gcloud_result("some-other-project")):
        with pytest.raises(SystemExit) as exc_info:
            module._assert_correct_project()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "HARD STOP" in out
    assert module.EXPECTED_PROJECT in out


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_correct_active_project_does_not_exit(module):
    with patch.object(module.subprocess, "run", return_value=_fake_gcloud_result(module.EXPECTED_PROJECT)):
        module._assert_correct_project()  # must not raise


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_empty_active_project_hard_stops(module):
    """No active project configured at all (`gcloud config get-value project` prints empty) --
    must fail closed, not pass by accident because '' happens to compare unequal harmlessly."""
    with patch.object(module.subprocess, "run", return_value=_fake_gcloud_result("")):
        with pytest.raises(SystemExit) as exc_info:
            module._assert_correct_project()
    assert exc_info.value.code == 1


def test_expected_project_matches_current_migration_target():
    """Regression guard for the constant itself -- both scripts' EXPECTED_PROJECT must stay in
    sync with each other (they're independent copies of the same check)."""
    assert publish_models.EXPECTED_PROJECT == verify_model_manifest.EXPECTED_PROJECT == "triageiq-prod-260812"
