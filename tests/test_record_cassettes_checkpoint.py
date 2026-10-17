"""ADR-0060 Phase 2c: the checkpoint must distinguish synthesis-done from fully-done.

A judge pass must never silently skip an unjudged entry, and a completion summary must
never report N/64 while some of those N are synthesis-only. These are exactly the two ways
splitting synthesis from judging could silently regress back to the single-pass behavior it
replaced -- this test would fail loudly if either regressed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import record_cassettes as rc  # noqa: E402

from triage_iq.evaluation.triage_eval import compute_judge_prompt_hash  # noqa: E402

_CURRENT = object()  # sentinel: stamp the entry with the currently configured judge config


def _entry(
    issue_id: str, *, plan: dict | None, judge_score: dict | None,
    judge_model_used: object = _CURRENT, judge_prompt_hash_used: object = _CURRENT,
) -> dict:
    """A judged entry is stamped with the CURRENT judge config by default (what run_judge
    writes today). Pass explicit values to simulate a score recorded under an older judge
    model/rubric, or None to simulate a pre-B4 entry that carries no judge stamp at all."""
    rec = {
        "issue_id": issue_id,
        "model": "openai/gpt-oss-120b",
        "prompt_hash": "abc123",
        "artifact_hash": "def456",
        "plan": plan,
        "judge_score": judge_score,
    }
    if judge_score is not None:
        rec["judge_model_used"] = rc.JUDGE_MODEL if judge_model_used is _CURRENT else judge_model_used
        rec["judge_prompt_hash_used"] = (
            compute_judge_prompt_hash() if judge_prompt_hash_used is _CURRENT else judge_prompt_hash_used
        )
    return rec


def test_pending_judge_entries_excludes_fully_done() -> None:
    """A judge pass reading _pending_judge_entries() must see the synthesis-only entry and
    must NOT see the fully-done one -- reprocessing an already-judged entry would waste an
    Ollama call and (if scores aren't perfectly deterministic) could silently overwrite a
    valid recorded score."""
    current_done = {
        "k1": _entry("k1", plan={"predicted_component": "api"}, judge_score=None),
        "k2": _entry("k2", plan={"predicted_component": "ui"}, judge_score={"overall_quality": 3}),
    }
    pending = rc._pending_judge_entries(current_done)
    assert set(pending) == {"k1"}, "fully-judged entry k2 must not appear in the judge pass's work queue"


def test_pending_judge_entries_excludes_dead_issues() -> None:
    """An issue with no plan at all (permanent synthesis failure) is neither pending judge
    nor fully done -- it must not appear in the judge queue (there's nothing to judge)."""
    current_done = {"k1": _entry("k1", plan=None, judge_score=None)}
    assert rc._pending_judge_entries(current_done) == {}


def test_fully_judged_count_never_counts_synthesis_only() -> None:
    """The exact bug this whole checkpoint split could reintroduce: a completion summary
    computed from len(current_done) or from a synthesis-only count would report 64/64 while
    some entries are still awaiting judgment. _fully_judged_count must only count entries
    with a real judge_score."""
    current_done = {
        "k1": _entry("k1", plan={"predicted_component": "api"}, judge_score=None),  # synthesis-only
        "k2": _entry("k2", plan={"predicted_component": "ui"}, judge_score={"overall_quality": 3}),
        "k3": _entry("k3", plan={"predicted_component": "cli"}, judge_score={"overall_quality": 2}),
    }
    assert rc._fully_judged_count(current_done) == 2
    assert len(current_done) == 3, "sanity: len(current_done) alone would wrongly report 3, not 2"


def test_run_judge_mode_reports_incomplete_when_entries_are_synthesis_only(monkeypatch, tmp_path, capsys) -> None:
    """End-to-end: run_judge() must exit non-zero (and print PARTIAL, not COMPLETE) if any
    eval-set issue has no plan yet -- it must never silently declare success while entries
    haven't even been synthesized, let alone judged."""
    issues = [
        {"id": "k1", "repo": "kubernetes/kubernetes", "title": "t1", "body": "b1",
         "gold_component": "api", "gold_priority": "P1", "actual_resolution_days": 1.0},
        {"id": "k2", "repo": "kubernetes/kubernetes", "title": "t2", "body": "b2",
         "gold_component": "ui", "gold_priority": "P2", "actual_resolution_days": 2.0},
    ]
    # k1 already fully judged; k2 not synthesized at all yet.
    current_done = {
        rc._checkpoint_key("k1", "openai/gpt-oss-120b", "abc123", "def456"): _entry(
            "k1", plan={"predicted_component": "api"}, judge_score={"overall_quality": 3},
        ),
    }
    checkpoint = {"done": dict(current_done)}

    monkeypatch.setattr(rc, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")

    try:
        rc.run_judge(
            issues, cassette=None, current_model="openai/gpt-oss-120b",
            current_prompt_hash="abc123", current_artifact_hash="def456",
            checkpoint=checkpoint, current_done=current_done,
        )
        raised = False
    except SystemExit as exc:
        raised = True
        assert exc.code != 0

    out = capsys.readouterr().out
    assert raised, "run_judge must exit non-zero when an eval-set issue hasn't even been synthesized yet"
    assert "NOTHING TO JUDGE" in out or "JUDGE PARTIAL" in out
    assert "JUDGE COMPLETE" not in out, "must never declare COMPLETE while an issue is unsynthesized"


# --- B4: judge-config invalidation must fail closed -------------------------------------
# A judge_score recorded under a different judge model or rubric is NOT a judged entry under
# the current config. These tests pin that it is (a) re-queued for judging, (b) never counted
# toward "judged" anywhere a completion decision is made, and (c) actually re-judged and
# re-stamped by run_judge. An entry with no stamp at all (pre-B4) is treated the same way --
# "couldn't verify the judge config" is a re-judge, never a silent pass (rule 98a).

_STALE_CASES = {
    "stale_model": {"judge_model_used": "llama3.1:8b"},
    "stale_rubric": {"judge_prompt_hash_used": "0000000000000000"},
    "unstamped_model": {"judge_model_used": None},
    "unstamped_rubric": {"judge_prompt_hash_used": None},
}


def _stale_entry(issue_id: str, case: str) -> dict:
    return _entry(issue_id, plan={"predicted_component": "api"},
                  judge_score={"overall_quality": 3}, **_STALE_CASES[case])


def test_stale_judge_config_is_requeued_and_never_counted() -> None:
    for case in _STALE_CASES:
        current_done = {
            "fresh": _entry("fresh", plan={"predicted_component": "ui"}, judge_score={"overall_quality": 2}),
            "stale": _stale_entry("stale", case),
        }
        assert set(rc._pending_judge_entries(current_done)) == {"stale"}, case
        assert rc._fully_judged_count(current_done) == 1, case
        assert not rc._judge_config_current(current_done["stale"]), case


def test_status_writers_do_not_count_stale_judge_scores(monkeypatch, tmp_path) -> None:
    """RECORDING_STATUS.txt (record_cassettes.write_live_status) and the unattended driver's
    terminal check (run_recording_unattended._progress) are the other two places "judged"
    is counted -- both must agree with _fully_judged_count, not with judge_score presence."""
    import json

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import run_recording_unattended as ru

    ckpt = tmp_path / "checkpoint.json"
    done = {
        rc._checkpoint_key("fresh", "openai/gpt-oss-120b", "abc123", "def456"): _entry(
            "fresh", plan={"predicted_component": "ui"}, judge_score={"overall_quality": 2}),
        rc._checkpoint_key("stale", "openai/gpt-oss-120b", "abc123", "def456"): _stale_entry(
            "stale", "stale_rubric"),
    }
    ckpt.write_text(json.dumps({"done": done}), encoding="utf-8")
    status = tmp_path / "status.txt"
    monkeypatch.setattr(rc, "CHECKPOINT_PATH", ckpt)
    monkeypatch.setattr(rc, "STATUS_PATH", status)
    monkeypatch.setattr(ru, "CHECKPOINT_PATH", ckpt)

    rc.write_live_status("judge", "openai/gpt-oss-120b", "abc123", "def456", 2)
    assert "Judged: 1/2" in status.read_text(encoding="utf-8")
    assert ru._progress("openai/gpt-oss-120b", "abc123", "def456")["judged"] == 1


def test_run_judge_rejudges_stale_entry_and_restamps(monkeypatch, tmp_path, capsys) -> None:
    """End-to-end through run_judge with the Ollama judge stubbed out: the stale entry must
    be sent to the judge again, and its rewritten record must carry the CURRENT judge
    config -- after which (and only after which) it counts as judged."""
    issues = [
        {"id": f"k{i}", "repo": "kubernetes/kubernetes", "title": f"t{i}", "body": f"b{i}",
         "gold_component": "api", "gold_priority": "P1", "actual_resolution_days": 1.0}
        for i in (1, 2)
    ]
    current_done = {
        "k1": _entry("k1", plan={"predicted_component": "ui"}, judge_score={"overall_quality": 2}),
        "k2": _stale_entry("k2", "stale_model"),
    }
    checkpoint = {"done": {
        rc._checkpoint_key(k, "openai/gpt-oss-120b", "abc123", "def456"): v for k, v in current_done.items()
    }}

    judged_ids: list[str] = []

    def fake_judge_one(judge, issue, plan_dict, n, cassette, parent_key):
        judged_ids.append(issue["id"])
        return {"overall_quality": 1}, n + 1, False

    monkeypatch.setattr(rc, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(rc, "STATUS_PATH", tmp_path / "status.txt")
    monkeypatch.setattr(rc, "_build_judge", lambda cassette: object())
    monkeypatch.setattr(rc, "_unload_judge_model", lambda: None)
    monkeypatch.setattr(rc, "_judge_one", fake_judge_one)

    rc.run_judge(issues, cassette=None, current_model="openai/gpt-oss-120b",
                 current_prompt_hash="abc123", current_artifact_hash="def456",
                 checkpoint=checkpoint, current_done=current_done)

    assert judged_ids == ["k2"], "only the stale entry is re-judged; the current one is not re-sent"
    rewritten = checkpoint["done"][rc._checkpoint_key("k2", "openai/gpt-oss-120b", "abc123", "def456")]
    assert rewritten["judge_model_used"] == rc.JUDGE_MODEL
    assert rewritten["judge_prompt_hash_used"] == compute_judge_prompt_hash()
    assert rewritten["judge_score"] == {"overall_quality": 1}
    assert "JUDGE COMPLETE" in capsys.readouterr().out
