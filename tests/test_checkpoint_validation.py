"""eval/checkpoint_validation.py: a 'done' checkpoint entry is trusted only if its request,
recomputed through the CURRENT code, still hits the cassette (ADR-0062 addendum).

These tests use tiny synthetic cassettes and stub assistants/judges that derive their keys with
the real CassettePlayer.compute_key -- the validator's logic (hit / miss / key drift / judge
dependence on the plan / fail-closed) is what is under test, not the real pipeline. The real
pipeline is exercised by `record_cassettes.py --validate-checkpoint` (proof in the PR body).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import checkpoint_validation as cv  # noqa: E402
from cassette import CassetteMissError, CassettePlayer  # noqa: E402

ISSUES = [
    {"id": "k8s-1", "repo": "kubernetes/kubernetes", "title": "t1", "body": "b1", "number": 1,
     "gold_component": "c", "gold_priority": "p", "actual_resolution_days": 1.0},
    {"id": "vs-2", "repo": "microsoft/vscode", "title": "t2", "body": "b2", "number": 2,
     "gold_component": "c", "gold_priority": "p", "actual_resolution_days": 2.0},
]


class _Plan:
    def __init__(self, text: str) -> None:
        self._text = text

    def model_dump(self, exclude: set[str] | None = None) -> dict:
        return {"summary": self._text}


class _StubAssistant:
    """Request text = f(issue, numbers_version): `version` plays the role of the resolution
    numbers embedded in the real prompt, which is what changed under #150."""

    def __init__(self, cassette: CassettePlayer, version: str, status: str = "ok") -> None:
        self._cassette, self._version, self._status = cassette, version, status

    def triage_with_metadata(self, row):  # noqa: ANN001, ANN201
        msgs = [{"role": "user", "content": f"{row['title']}|{row['body_clean']}|{self._version}"}]
        key = CassettePlayer.compute_key("groq", "m", msgs, 0.0, 100)
        resp = self._cassette.get(key)  # strict: raises CassetteMissError on a miss
        return _Plan(resp["content"]), {"llm_status": self._status, "synthesis_cache_key": key}


class _StubJudge:
    def __init__(self, cassette: CassettePlayer) -> None:
        self._cassette = cassette

    def score(self, *, issue_title: str, issue_body: str, triage_plan_json: str, gold: dict):  # noqa: ANN201
        key = CassettePlayer.compute_key(
            "ollama", "j", [{"role": "user", "content": f"{issue_title}|{triage_plan_json}"}], 0.0
        )
        self._cassette.get(key)
        return object()


def _key(issue: dict, version: str) -> str:
    msgs = [{"role": "user", "content": f"{issue['title']}|{issue['body']}|{version}"}]
    return CassettePlayer.compute_key("groq", "m", msgs, 0.0, 100)


def _record(path: Path, synth_versions: list[str], judged_plans: list[tuple[dict, str]]) -> None:
    rec = CassettePlayer(path, strict=False, allow_record=True)
    for issue in ISSUES:
        for v in synth_versions:
            msgs = [{"role": "user", "content": f"{issue['title']}|{issue['body']}|{v}"}]
            rec.set(_key(issue, v), "groq", "m", msgs, {"content": f"plan-{v}"})
    for issue, text in judged_plans:
        plan_json = json.dumps({"summary": text}, ensure_ascii=False)
        msgs = [{"role": "user", "content": f"{issue['title']}|{plan_json}"}]
        k = CassettePlayer.compute_key("ollama", "j", msgs, 0.0)
        rec.set(k, "ollama", "j", msgs, {"content": "{}"})


def _entry(issue: dict, version: str, *, judged: bool = True, plan_text: str | None = None) -> dict:
    return {
        "issue_id": issue["id"],
        "plan": {"summary": plan_text or f"plan-{version}"},
        "judge_score": {"overall_quality": 3} if judged else None,
        "synthesis_cache_key": _key(issue, version),
    }


def _validate(tmp_path: Path, checkpoint: dict, version: str, *, judge: bool = True,
              status: str = "ok") -> dict[str, str]:
    cassette = CassettePlayer(tmp_path / "c.json", strict=True)
    assistants = {i["repo"]: _StubAssistant(cassette, version, status) for i in ISSUES}
    return cv.validate_checkpoint(
        ISSUES, cassette, checkpoint,
        assistants=assistants, judge=_StubJudge(cassette) if judge else None,
    )


def test_valid_when_synthesis_and_judge_hit(tmp_path: Path) -> None:
    _record(tmp_path / "c.json", ["v2"], [(i, "plan-v2") for i in ISSUES])
    cp = {i["id"]: _entry(i, "v2") for i in ISSUES}
    assert _validate(tmp_path, cp, "v2") == {"k8s-1": "valid", "vs-2": "valid"}


def test_stale_synthesis_when_request_changed(tmp_path: Path) -> None:
    """The k8s incident shape: the code now builds a different request (v2) than the one the
    checkpoint entry was recorded under (v1), and the cassette has no v2 entry."""
    _record(tmp_path / "c.json", ["v1"], [(i, "plan-v1") for i in ISSUES])
    cp = {i["id"]: _entry(i, "v1") for i in ISSUES}
    assert _validate(tmp_path, cp, "v2") == {"k8s-1": "stale_synthesis", "vs-2": "stale_synthesis"}


def test_stale_synthesis_even_when_old_cassette_entry_still_hits(tmp_path: Path) -> None:
    """A cassette is append-only, so the OLD entry can still hit. A bare 'did it hit?' check
    would call that valid; the checkpoint's recorded synthesis_cache_key exposes the drift."""
    _record(tmp_path / "c.json", ["v1", "v2"], [(i, "plan-v2") for i in ISSUES])
    cp = {i["id"]: _entry(i, "v1") for i in ISSUES}  # recorded under v1; code now says v2
    res = _validate(tmp_path, cp, "v2")
    assert set(res.values()) == {"stale_synthesis"}


def test_stale_judge_when_plan_changed_but_synthesis_hits(tmp_path: Path) -> None:
    """Synthesis (the LLM call) still hits, but the plan built from it differs from the one
    the judge was recorded on, so the judge request misses: only the judge is stale."""
    _record(tmp_path / "c.json", ["v2"], [])  # no judge entries for the current plan
    cp = {i["id"]: _entry(i, "v2") for i in ISSUES}
    res = _validate(tmp_path, cp, "v2")
    assert res == {"k8s-1": "stale_judge", "vs-2": "stale_judge"}
    detailed = cv.validate_checkpoint_detailed(
        ISSUES, CassettePlayer(tmp_path / "c.json", strict=True), cp,
        assistants={i["repo"]: _StubAssistant(CassettePlayer(tmp_path / "c.json"), "v2")
                    for i in ISSUES},
        judge=_StubJudge(CassettePlayer(tmp_path / "c.json")),
    )
    assert detailed["k8s-1"].replayed_plan == {"summary": "plan-v2"}


def test_judge_pending_entry_is_valid_without_a_judge_hit(tmp_path: Path) -> None:
    _record(tmp_path / "c.json", ["v2"], [])
    cp = {i["id"]: _entry(i, "v2", judged=False) for i in ISSUES}
    assert set(_validate(tmp_path, cp, "v2").values()) == {"valid"}


def test_judge_not_checked_when_judge_is_none(tmp_path: Path) -> None:
    """--mode synthesis validates the synthesis stage only (never imports TriageJudge)."""
    _record(tmp_path / "c.json", ["v2"], [])
    cp = {i["id"]: _entry(i, "v2") for i in ISSUES}
    assert set(_validate(tmp_path, cp, "v2", judge=False).values()) == {"valid"}


def test_degraded_replay_is_stale_synthesis(tmp_path: Path) -> None:
    _record(tmp_path / "c.json", ["v2"], [(i, "plan-v2") for i in ISSUES])
    cp = {i["id"]: _entry(i, "v2") for i in ISSUES}
    res = _validate(tmp_path, cp, "v2", status="degraded_schema_invalid")
    assert set(res.values()) == {"stale_synthesis"}


def test_entries_without_a_plan_are_not_validated(tmp_path: Path) -> None:
    _record(tmp_path / "c.json", ["v2"], [])
    cp = {"k8s-1": {"issue_id": "k8s-1", "plan": None, "judge_score": None, "error": "boom"}}
    assert _validate(tmp_path, cp, "v2") == {}


def test_raw_checkpoint_file_shape_is_accepted(tmp_path: Path) -> None:
    _record(tmp_path / "c.json", ["v2"], [(i, "plan-v2") for i in ISSUES])
    raw = {"done": {f"{i['id']}::m::h::a": _entry(i, "v2") for i in ISSUES}}
    assert set(_validate(tmp_path, raw, "v2").values()) == {"valid"}


def test_validation_cannot_run_is_fail_closed(tmp_path: Path) -> None:
    """Missing cassette/eval set => CheckpointValidationUnavailable, never a silent pass."""
    with pytest.raises(cv.CheckpointValidationUnavailable, match="required input missing"):
        cv.build_replay_context(tmp_path / "nope.json", tmp_path / "nope.jsonl", need_judge=False)


def test_replay_crash_other_than_a_miss_is_fail_closed(tmp_path: Path) -> None:
    class _Boom:
        def triage_with_metadata(self, row):  # noqa: ANN001, ANN201
            raise ValueError("artifact unreadable")

    _record(tmp_path / "c.json", ["v2"], [])
    cassette = CassettePlayer(tmp_path / "c.json", strict=True)
    cp = {"k8s-1": _entry(ISSUES[0], "v2")}
    with pytest.raises(cv.CheckpointValidationUnavailable, match="artifact unreadable"):
        cv.validate_checkpoint(ISSUES, cassette, cp, assistants={"kubernetes/kubernetes": _Boom()})


def test_validation_never_writes_to_the_cassette(tmp_path: Path) -> None:
    _record(tmp_path / "c.json", ["v1"], [])
    before = (tmp_path / "c.json").read_bytes()
    cp = {i["id"]: _entry(i, "v1") for i in ISSUES}
    _validate(tmp_path, cp, "v2")  # all misses
    assert (tmp_path / "c.json").read_bytes() == before
    with pytest.raises(RuntimeError):
        CassettePlayer(tmp_path / "c.json", strict=True).set("k", "p", "m", [], {})


def test_cassette_miss_error_is_what_a_strict_miss_raises(tmp_path: Path) -> None:
    _record(tmp_path / "c.json", ["v1"], [])
    with pytest.raises(CassetteMissError):
        CassettePlayer(tmp_path / "c.json", strict=True).get("0" * 64)


def test_reconcile_per_mode() -> None:
    done = {
        "a": {"issue_id": "a", "plan": {"x": 1}, "judge_score": {"s": 1}},
        "b": {"issue_id": "b", "plan": {"x": 2}, "judge_score": {"s": 2}},
        "c": {"issue_id": "c", "plan": {"x": 3}, "judge_score": {"s": 3}},
    }
    res = {
        "a": cv.ValidationResult(cv.VALID),
        "b": cv.ValidationResult(cv.STALE_SYNTHESIS),
        "c": cv.ValidationResult(cv.STALE_JUDGE, replayed_plan={"x": 99}),
    }
    judge = cv.reconcile_current_done(done, res, "judge")
    assert set(judge) == {"a", "c"}  # stale_synthesis dropped
    assert judge["c"]["judge_score"] is None and judge["c"]["plan"] == {"x": 99}
    assert judge["c"]["judge_pending"] is True and judge["a"]["judge_score"] == {"s": 1}
    assert set(cv.reconcile_current_done(done, res, "full")) == {"a"}
    assert set(cv.reconcile_current_done(done, res, "synthesis")) == {"a", "c"}
    assert set(done) == {"a", "b", "c"}  # input not mutated


# --- recorder wiring: every mode validates at start and fails closed -------------------------


def _wired(monkeypatch: pytest.MonkeyPatch, results: dict | Exception):  # noqa: ANN202
    import record_cassettes as rc

    def fake(entries, *, check_judge):  # noqa: ANN001, ANN202
        fake.check_judge = check_judge  # type: ignore[attr-defined]
        if isinstance(results, Exception):
            raise results
        return results

    monkeypatch.setattr(rc.cv, "validate_from_disk", fake)
    return rc, fake


def test_recorder_treats_stale_entries_as_pending_in_every_mode(monkeypatch) -> None:  # noqa: ANN001
    res = {
        "a": cv.ValidationResult(cv.VALID),
        "b": cv.ValidationResult(cv.STALE_SYNTHESIS),
        "c": cv.ValidationResult(cv.STALE_JUDGE, replayed_plan={"x": 9}),
    }
    rc, fake = _wired(monkeypatch, res)
    for mode, expected_ids, expect_check_judge in (
        ("synthesis", {"a", "c"}, False),
        ("judge", {"a", "c"}, True),
        ("full", {"a"}, True),
    ):
        done = {k: {"issue_id": k, "plan": {"x": 1}, "judge_score": {"s": 1}} for k in "abc"}
        ckpt = {"done": {rc._checkpoint_key(k, "m", "p", "a"): v for k, v in done.items()}}
        out = rc._validated_current_done(mode, ckpt, done, "m", "p", "a")
        assert set(out) == expected_ids, mode
        assert fake.check_judge is expect_check_judge, mode  # type: ignore[attr-defined]
        # raw checkpoint mirrors the view, so run_judge's final count cannot count stale 'b'
        assert ("b::m::p::a" in ckpt["done"]) is False, mode


def test_recorder_stops_when_validation_cannot_run(monkeypatch) -> None:  # noqa: ANN001
    rc, _ = _wired(monkeypatch, cv.CheckpointValidationUnavailable("missing artifacts"))
    done = {"a": {"issue_id": "a", "plan": {"x": 1}, "judge_score": None}}
    with pytest.raises(SystemExit) as exc:
        rc._validated_current_done("synthesis", {"done": {}}, done, "m", "p", "a")
    assert exc.value.code == 1


def test_recorder_skips_validation_for_an_empty_checkpoint(monkeypatch) -> None:  # noqa: ANN001
    rc, _ = _wired(monkeypatch, RuntimeError("must not be called"))
    assert rc._validated_current_done("full", {"done": {}}, {}, "m", "p", "a") == {}


def test_unattended_driver_does_not_declare_done_on_stale_entries(monkeypatch) -> None:  # noqa: ANN001
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import run_recording_unattended as ru

    monkeypatch.setattr(
        ru, "_checkpoint_entries",
        lambda *a: [{"issue_id": "a", "plan": {}}, {"issue_id": "b", "plan": {}}],
    )
    seen = {}

    def fake(entries, *, check_judge):  # noqa: ANN001, ANN202
        seen["check_judge"] = check_judge
        return {"a": cv.ValidationResult(cv.VALID), "b": cv.ValidationResult(cv.STALE_SYNTHESIS)}

    monkeypatch.setattr(ru.cv, "validate_from_disk", fake)
    assert ru.stale_issue_ids("synthesis", "m", "p", "a") == {"b"}
    assert seen["check_judge"] is False

    def unavailable(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise cv.CheckpointValidationUnavailable("x")

    monkeypatch.setattr(ru.cv, "validate_from_disk", unavailable)
    with pytest.raises(cv.CheckpointValidationUnavailable):
        ru.stale_issue_ids("judge", "m", "p", "a")
