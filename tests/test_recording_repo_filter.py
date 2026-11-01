"""TRIAGEIQ_RECORDING_REPOS: with a repo filter the recorder never prompts, calls the LLM for, or
counts an issue outside it; unset, nothing changes. No LLM is called anywhere here."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import record_cassettes as rc  # noqa: E402
import recording_config as rcfg  # noqa: E402

K8S, VSC = "kubernetes/kubernetes", "microsoft/vscode"
MODEL, PH, AH = "m", "p", "a"


def _issue(i: int, repo: str) -> dict:
    slug = "k8s" if repo == K8S else "vscode"
    return {"id": f"{slug}-{i}", "repo": repo, "number": i, "title": "t", "body": "b"}


ISSUES = [_issue(1, K8S), _issue(2, K8S), _issue(3, VSC), _issue(4, VSC)]


@pytest.fixture
def eval_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "eval_set.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in ISSUES) + "\n", encoding="utf-8")
    monkeypatch.setattr(rc, "EVAL_SET", path)
    monkeypatch.delenv(rcfg.REPOS_ENV, raising=False)
    return path


def test_filter_parsing_and_fail_closed() -> None:
    assert rcfg.recording_repo_filter({}) is None
    for blank in ("", " , "):  # set but naming no repo must not silently record every repo
        with pytest.raises(rcfg.RecordingConfigError):
            rcfg.recording_repo_filter({rcfg.REPOS_ENV: blank})
    assert rcfg.recording_repo_filter({rcfg.REPOS_ENV: f"{K8S}, {VSC}"}) == {K8S, VSC}
    assert rcfg.filter_issues(ISSUES, None) is ISSUES
    with pytest.raises(rcfg.RecordingConfigError):  # a typo must not silently record nothing
        rcfg.filter_issues(ISSUES, frozenset({"kubernetes/kubernets"}))


def test_load_eval_set_unfiltered_is_unchanged_and_filtered_drops_vscode(
    eval_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert [i["id"] for i in rc.load_eval_set()] == [i["id"] for i in ISSUES]
    monkeypatch.setenv(rcfg.REPOS_ENV, K8S)
    assert [i["id"] for i in rc.load_eval_set()] == ["k8s-1", "k8s-2"]


def test_synthesis_never_builds_a_prompt_or_calls_the_llm_for_a_vscode_issue(
    eval_set: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(rcfg.REPOS_ENV, K8S)
    plan = MagicMock()
    plan.model_dump.return_value = {"predicted_component": "x"}
    plan.predicted_component = "x"
    meta = {"llm_status": "ok", "synthesis_cache_key": "key", "llm_cache_hit": True}
    k8s_llm, vscode_llm = MagicMock(return_value=(plan, meta)), MagicMock(side_effect=AssertionError("vscode LLM call"))
    models = {
        K8S: {"assistant": SimpleNamespace(triage_with_metadata=k8s_llm)},
        VSC: {"assistant": SimpleNamespace(triage_with_metadata=vscode_llm)},
    }
    rows_built: list[int] = []
    real_to_row = rc.issue_to_row
    monkeypatch.setattr(rc, "issue_to_row", lambda issue: rows_built.append(issue["number"]) or real_to_row(issue))
    monkeypatch.setattr(rc, "_load_models", lambda *a, **k: models)
    monkeypatch.setattr(rc, "_enforce_budget_reserve", lambda key: None)
    monkeypatch.setattr(rc.time, "sleep", lambda s: None)
    monkeypatch.setattr(rc, "CHECKPOINT_PATH", tmp_path / "ckpt.json")
    monkeypatch.setattr(rc, "STATUS_PATH", tmp_path / "status.txt")
    cassette = MagicMock()
    cassette.stats.return_value = {"entries": 0}

    checkpoint = {"done": {}}
    rc.run_synthesis("k", rc.load_eval_set(), cassette, MODEL, PH, AH, checkpoint, {})

    assert k8s_llm.call_count == 2
    vscode_llm.assert_not_called()
    assert rows_built == [1, 2]  # no vscode prompt row was ever built
    assert {r["issue_id"] for r in checkpoint["done"].values()} == {"k8s-1", "k8s-2"}


def test_load_models_skips_the_filtered_out_repo(eval_set: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rcfg.REPOS_ENV, K8S)
    built: list[str] = []
    monkeypatch.setattr(rc, "build_frozen_retrievers", lambda path: {K8S: object(), VSC: object()})
    monkeypatch.setattr(rc, "load_recording_config", lambda *a, **k: object())
    monkeypatch.setattr(rc.artifact_fingerprint, "compute_artifact_hashes", lambda *a, **k: {})
    monkeypatch.setattr(rc, "_load_conformal_adjustments", lambda d: {})
    monkeypatch.setattr(rc, "build_assistant", lambda repo, *a, **k: built.append(repo) or {"assistant": repo})
    assert set(rc._load_models("k", MagicMock())) == {K8S}
    assert built == [K8S]


def test_live_status_and_done_filter_count_only_filtered_issues(
    eval_set: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    done = {
        f"{i['id']}::{MODEL}::{PH}::{AH}": {
            "issue_id": i["id"], "model": MODEL, "prompt_hash": PH, "artifact_hash": AH, "plan": {"x": 1},
        }
        for i in ISSUES
    }
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text(json.dumps({"done": done}), encoding="utf-8")
    monkeypatch.setattr(rc, "CHECKPOINT_PATH", ckpt)
    monkeypatch.setattr(rc, "STATUS_PATH", tmp_path / "status.txt")
    rc.write_live_status("synthesis", MODEL, PH, AH, 4)
    assert "Synthesis-done: 4/4" in (tmp_path / "status.txt").read_text(encoding="utf-8")
    monkeypatch.setenv(rcfg.REPOS_ENV, K8S)
    rc.write_live_status("synthesis", MODEL, PH, AH, 2)
    assert "Synthesis-done: 2/2" in (tmp_path / "status.txt").read_text(encoding="utf-8")

    by_issue = {i["id"]: {"issue_id": i["id"]} for i in ISSUES}
    assert set(rc._filter_done_to_issues(by_issue, rc.load_eval_set())) == {"k8s-1", "k8s-2"}
    monkeypatch.delenv(rcfg.REPOS_ENV)
    assert rc._filter_done_to_issues(by_issue, ISSUES) is by_issue


# ---- launcher -------------------------------------------------------------------------------------

import run_recording_unattended as ru  # noqa: E402


@pytest.fixture
def launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    ckpt = tmp_path / "recording_checkpoint.json"
    for name, val in {
        "CHECKPOINT_PATH": ckpt, "STATUS_PATH": tmp_path / "status.txt", "LOG_DIR": tmp_path / "logs",
        "PID_PATH": tmp_path / "pid",
    }.items():
        monkeypatch.setattr(ru, name, val)
    monkeypatch.setattr(ru, "_REPOS", frozenset({K8S}))
    monkeypatch.setattr(ru, "_ISSUE_IDS", frozenset({"k8s-1", "k8s-2"}))
    monkeypatch.setattr(ru, "TOTAL_ISSUES", 2)
    monkeypatch.setattr(ru, "_current_model_and_hash", lambda: (MODEL, PH, AH))
    monkeypatch.setattr(ru, "stale_issue_ids", lambda *a: set())
    monkeypatch.setattr(ru, "_load_groq_key", lambda: "dummy")
    monkeypatch.setattr(ru, "parse_args", lambda: SimpleNamespace(mode="synthesis"))
    monkeypatch.setattr(ru.tpd_budget_gate, "start_recording_heartbeat", lambda: MagicMock())
    monkeypatch.setattr(ru.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(ru.time, "sleep", lambda s: pytest.fail("launcher slept (budget-wait loop)"))
    # The launcher is Windows-only (BELOW_NORMAL_PRIORITY_CLASS); CI runs these tests on Linux.
    monkeypatch.setattr(ru.subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x4000, raising=False)

    def write(ids: list[str]) -> None:
        ckpt.write_text(json.dumps({"done": {
            f"{i}::{MODEL}::{PH}::{AH}": {
                "issue_id": i, "model": MODEL, "prompt_hash": PH, "artifact_hash": AH, "plan": {"x": 1},
            } for i in ids
        }}), encoding="utf-8")

    return SimpleNamespace(write=write, status=tmp_path / "status.txt")


def test_launcher_counts_only_filtered_issues(launcher: SimpleNamespace) -> None:
    launcher.write(["k8s-1", "vscode-3", "vscode-4"])
    assert ru._progress(MODEL, PH, AH)["synthesized"] == 1
    assert len(ru._checkpoint_entries(MODEL, PH, AH)) == 1


def test_launcher_reports_filtered_done_and_exits_without_calling_the_recorder(
    launcher: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    launcher.write(["k8s-1", "k8s-2"])  # vscode rows are not done, and must not matter
    monkeypatch.setattr(ru.subprocess, "run", lambda *a, **k: pytest.fail("child started although all filtered issues are done"))
    ru.main()
    out = capsys.readouterr().out
    assert f"DONE (synthesis) (filtered: {K8S} 2/2)" in out
    status = launcher.status.read_text(encoding="utf-8")
    assert "STOPPED (terminal)" in status and "BLOCKED" not in status
    assert f"Repo filter: {K8S} (2 issues in scope)" in status and "Synthesis-done: 2/2" in status


def test_launcher_is_not_done_while_a_filtered_issue_is_missing_and_passes_the_filter_to_the_child(
    launcher: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher.write(["k8s-1", "vscode-3", "vscode-4"])  # 3 raw entries >= TOTAL 2, but only 1 is in scope
    calls: list[dict] = []

    def fake_run(cmd, **kw):  # noqa: ANN001, ANN003, ANN202
        calls.append({"cmd": cmd, "env": kw["env"]})
        return subprocess.CompletedProcess(cmd, 0, stdout="=== SYNTHESIS COMPLETE ===", stderr="")

    monkeypatch.setattr(ru.subprocess, "run", fake_run)
    ru.main()
    assert len(calls) == 1 and calls[0]["env"][rcfg.REPOS_ENV] == K8S
