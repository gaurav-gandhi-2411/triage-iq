"""deploy.yml triggers only on paths that reach the image or the deploy job.

Why: #160 (analysis scripts, no serving change) triggered two deploys whose smoke tests
failed for lack of Groq tokens. The filter is an allowlist; this test pins it from both sides:
every local COPY source of docker/Dockerfile.prod must match it (no served change can skip a
deploy), and the known non-serving trees must not (no analysis merge can trigger one).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _push_paths() -> list[str]:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text("utf-8"))
    push = wf[True]["push"]  # PyYAML parses the bare key `on` as boolean True
    assert "paths-ignore" not in push, "paths and paths-ignore are mutually exclusive"
    return push["paths"]


def _matches(pattern: str, path: str) -> bool:
    """GitHub path glob: `**` crosses directories, `*` does not."""
    rx = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            rx += ".*"
            i += 2
        elif pattern[i] == "*":
            rx += "[^/]*"
            i += 1
        else:
            rx += re.escape(pattern[i])
            i += 1
    return re.fullmatch(rx, path) is not None


def _triggers(path: str) -> bool:
    return any(_matches(p, path) for p in _push_paths())


def _dockerfile_copy_sources() -> list[str]:
    """Build-context sources of every COPY (multi-stage `--from=` copies excluded)."""
    sources = []
    for line in (ROOT / "docker" / "Dockerfile.prod").read_text("utf-8").splitlines():
        line = line.strip()
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        sources.extend(line.split()[1:-1])
    return sources


def test_every_tracked_copy_source_triggers_a_deploy():
    """Sources that are git-tracked files/dirs must match. data/models/*.pkl, the bge index
    dirs and data/processed parquet come from GCS at deploy time, not from git; they are pinned
    through data/models/MANIFEST.sha256 (asserted below)."""
    for src in _dockerfile_copy_sources():
        if src.startswith(("data/models/", "data/processed/")):
            continue
        probe = src.rstrip("/") + "/x.py" if (ROOT / src).is_dir() else src
        assert _triggers(probe), f"Dockerfile.prod COPYs {src} but deploy.yml would not run"


def test_artifact_changes_trigger_via_manifest():
    assert _triggers("data/models/MANIFEST.sha256")
    assert "COPY data/models/MANIFEST.sha256" in re.sub(
        r"\s+", " ", (ROOT / "docker" / "Dockerfile.prod").read_text("utf-8")
    )


def test_job_inputs_trigger_a_deploy():
    for p in (
        "src/triage_iq/api/app.py",
        "src/triage_iq/prompts/triage_prompt.py",
        "requirements.lock",
        "pyproject.toml",
        "docker/Dockerfile.prod",
        "reports/eval_summary.json",
        "reports/eval_baseline.json",
        "scripts/verify_model_manifest.py",
        ".github/workflows/deploy.yml",
    ):
        assert _triggers(p), p


def test_non_serving_changes_do_not_trigger_a_deploy():
    for p in (
        "scripts/run_recording_unattended.py",  # the #160 shape
        "scripts/build_resolution_served_block.py",
        "eval/cassettes/eval_cassette.json",
        "eval/record_cassettes.py",
        "tests/test_api.py",
        "docs/DECISION_LOG_2026-10.md",
        "README.md",
        "notebooks/x.ipynb",
        "reports/served_k8s_metrics.json",
        "reports/d7_resolution_before_after.json",
        "data/d2_hard_negatives_k8s_related.parquet",
        "docker/Dockerfile",
        ".github/workflows/ci.yml",
    ):
        assert not _triggers(p), p
