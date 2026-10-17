"""Packaging guards for the Cloud Run recording job: weekly cadence, no retries, secret from Secret Manager."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_and_dockerfile_keep_the_cost_and_safety_guards() -> None:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "recording-job.yml").read_text(encoding="utf-8"))
    sched = wf[True]["schedule"][0]["cron"]
    assert sched.split()[4] == "1" and sched.split()[2:4] == ["*", "*"]  # weekly, never daily by accident
    text = (ROOT / ".github" / "workflows" / "recording-job.yml").read_text(encoding="utf-8")
    for guard in ("--max-retries 0", "--tasks 1", "--parallelism 1", "--task-timeout 3300s", "--set-secrets GROQ_API_KEY=groq-api-key:latest"):
        assert guard in text, guard
    assert wf["concurrency"]["cancel-in-progress"] is False
    docker = (ROOT / "docker" / "Dockerfile.recorder").read_text(encoding="utf-8")
    assert "FROM ${API_IMAGE}" in docker and "job_entrypoint.py" in docker and "USER appuser" in docker
    req = (ROOT / "requirements-recorder.txt").read_text(encoding="utf-8")
    assert "google-cloud-storage==" in req  # pinned
