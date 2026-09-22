from __future__ import annotations

"""ADR-0060 Phase 2c: the checkpoint must distinguish synthesis-done from fully-done.

A judge pass must never silently skip an unjudged entry, and a completion summary must
never report N/64 while some of those N are synthesis-only. These are exactly the two ways
splitting synthesis from judging could silently regress back to the single-pass behavior it
replaced -- this test would fail loudly if either regressed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import record_cassettes as rc  # noqa: E402


def _entry(issue_id: str, *, plan: dict | None, judge_score: dict | None) -> dict:
    return {
        "issue_id": issue_id,
        "model": "openai/gpt-oss-120b",
        "prompt_hash": "abc123",
        "artifact_hash": "def456",
        "plan": plan,
        "judge_score": judge_score,
    }


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
