"""Smoke-test quota refinement (deploy.yml + post-deploy-smoke.yml, 2026-10-12).

A deploy may promote ONLY when /health?deps=1 is healthy AND /triage degraded with exactly
rate_limited_tpd. Every other degradation, a 401 or a model 404 still blocks. One test per branch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from scripts.smoke_triage_verdict import FAIL, PASS, QUOTA_PASS, verdict

ROOT = Path(__file__).resolve().parents[1]


def _body(**over) -> str:
    d = {
        "predicted_component": "extensions",
        "triage_summary": "Extension host crashes on startup.",
        "_degraded": False,
        "_llm_status": "ok",
        "_llm_status_reason": None,
    }
    d.update(over)
    return json.dumps(d)


def _quota(**over) -> str:
    return _body(
        _degraded=True,
        _llm_status="degraded_provider_error",
        _llm_status_reason="rate_limited_tpd",
        **over,
    )


def test_real_llm_answer_passes():
    assert verdict("200", _body())[0] == PASS


def test_daily_quota_429_is_a_quota_pass():
    code, msg = verdict("200", _quota())
    assert code == QUOTA_PASS
    assert "rate_limited_tpd" in msg


def test_per_minute_rate_limit_still_blocks():
    body = _body(
        _degraded=True, _llm_status="degraded_provider_error", _llm_status_reason="rate_limited_tpm"
    )
    assert verdict("200", body)[0] == FAIL


def test_time_budget_exhaustion_still_blocks():
    body = _body(
        _degraded=True,
        _llm_status="degraded_provider_error",
        _llm_status_reason="llm_time_budget_exceeded",
    )
    assert verdict("200", body)[0] == FAIL


def test_rejected_key_401_still_blocks():
    body = _body(
        _degraded=True, _llm_status="degraded_provider_error", _llm_status_reason="auth_error"
    )
    assert verdict("200", body)[0] == FAIL
    assert verdict("401", '{"detail":"invalid_api_key"}')[0] == FAIL


def test_model_404_still_blocks():
    assert verdict("404", '{"error":"model_not_found"}')[0] == FAIL
    body = _body(
        _degraded=True, _llm_status="degraded_provider_error", _llm_status_reason="model_not_found"
    )
    assert verdict("200", body)[0] == FAIL


def test_parse_failure_still_blocks():
    body = _body(_degraded=True, _llm_status="parse_failure", _llm_status_reason=None)
    assert verdict("200", body)[0] == FAIL


def test_reason_must_be_paired_with_the_provider_error_status():
    body = _body(_degraded=True, _llm_status="parse_failure", _llm_status_reason="rate_limited_tpd")
    assert verdict("200", body)[0] == FAIL


def test_quota_pass_still_needs_a_sound_signals_plan():
    assert verdict("200", _quota(predicted_component=""))[0] == FAIL
    assert verdict("200", _quota(triage_summary="  "))[0] == FAIL


def test_non_200_and_garbage_block():
    assert verdict("500", "boom")[0] == FAIL
    assert verdict("200", "not json")[0] == FAIL
    assert verdict("200", "[1, 2]")[0] == FAIL
    assert verdict("200", _body(_degraded=None))[0] == FAIL  # missing flag is not "false"


def test_cli_exit_codes_match_the_workflow_contract():
    def run(body: str) -> int:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "smoke_triage_verdict.py"), "200"],
            input=body,
            text=True,
            capture_output=True,
            cwd=ROOT,
        ).returncode

    assert run(_body()) == 0
    assert run(_quota()) == 3
    assert run(_body(_degraded=True, _llm_status="parse_failure")) == 1


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text("utf-8"))


def test_deploy_workflow_wires_the_quota_branch_and_only_that_branch():
    wf = _workflow("deploy.yml")
    steps = wf["jobs"]["deploy"]["steps"]
    smoke = next(s for s in steps if s.get("id") == "smoke")
    assert "scripts/smoke_triage_verdict.py" in smoke["run"]
    assert "quota_pass=true" in smoke["run"]
    # Only exit code 3 maps to the quota pass; every other non-zero exits the step.
    assert '"$RC" = "3"' in smoke["run"] and "exit 1" in smoke["run"]
    queue = next(s for s in steps if "Queue the full post-deploy smoke" in s.get("name", ""))
    assert "quota_pass == 'true'" in queue["if"]
    assert wf["jobs"]["deploy"]["permissions"]["issues"] == "write"
    assert "smoke-pending-quota" in queue["with"]["script"]


def test_post_deploy_workflow_closes_on_pass_and_alerts_on_other_failures():
    wf = _workflow("post-deploy-smoke.yml")
    assert wf[True]["schedule"][0]["cron"] == "*/30 * * * *"
    job = wf["jobs"]["recheck"]
    assert job["permissions"]["issues"] == "write"
    text = (ROOT / ".github" / "workflows" / "post-deploy-smoke.yml").read_text("utf-8")
    # no pending issue -> the Groq-spending steps are skipped
    for step in job["steps"][1:]:
        assert "pending == 'true'" in step["if"]
    assert "smoke-pending-quota" in text and "post-deploy-smoke-alert" in text
    assert "state: 'closed'" in text
    assert "rate_limited_tpd" in text  # documented: the quota result leaves the issue open
