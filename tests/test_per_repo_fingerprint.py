"""A checkpoint entry is keyed by the hash of ITS repo's artifacts (plus the shared CQR store), so a
vscode-only artifact change leaves k8s entries valid; the old combined-hash entries migrate only
when proven. Hermetic: every root is tmp_path; no LLM, no real artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import artifact_fingerprint as af  # noqa: E402
import checkpoint_validation as cv  # noqa: E402
import record_cassettes as rc  # noqa: E402
import repo_fingerprint as rf  # noqa: E402

K8S, VSC = "kubernetes/kubernetes", "microsoft/vscode"
MODEL, PH = "m", "p"
VSC_INDEX = "data/models/similar_issue_index_microsoft_vscode_bge/index.faiss"
K8S_INDEX = "data/models/similar_issue_index_kubernetes_kubernetes_bge/index.faiss"
CQR = "data/models/cqr_conformal_adjustments_v2.json"
ISSUES = [
    {"id": "k8s-1", "repo": K8S}, {"id": "k8s-2", "repo": K8S}, {"id": "k8s-5", "repo": K8S},
    {"id": "vscode-3", "repo": VSC}, {"id": "vscode-4", "repo": VSC},
]
REPO_OF = {i["id"]: i["repo"] for i in ISSUES}


def _make_root(root: Path) -> None:
    for repo in af.REPO_SLUGS:
        for rel in af.artifact_paths_for_repo(repo):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(f"{rel} v1", encoding="utf-8")


def _hashes(root: Path) -> tuple[dict[str, str], rf.RepoHashes]:
    full = af.compute_artifact_hashes(root)
    return full, rf.per_repo_hashes(full)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    root = tmp_path / "root"
    _make_root(root)
    eval_set = tmp_path / "eval_set.jsonl"
    eval_set.write_text("\n".join(json.dumps(i) for i in ISSUES) + "\n", encoding="utf-8")
    ckpt = tmp_path / "recording_checkpoint.json"
    monkeypatch.setattr(rc, "EVAL_SET", eval_set)
    monkeypatch.setattr(rc, "CHECKPOINT_PATH", ckpt)
    return SimpleNamespace(root=root, ckpt=ckpt, tmp=tmp_path)


def _record_all(checkpoint: dict, hashes: rf.RepoHashes) -> None:
    for i in ISSUES:
        rc._record_done(checkpoint, i["id"], {"plan": {"x": 1}, "judge_score": None}, MODEL, PH, hashes)


def _current_ids(hashes: rf.RepoHashes, data: dict) -> set[str]:
    ckpt = rc.CHECKPOINT_PATH
    ckpt.write_text(json.dumps(data), encoding="utf-8")
    return set(rc.load_checkpoint(MODEL, PH, hashes)[1])


def test_vscode_only_change_keeps_k8s_entries_valid_and_invalidates_vscode(env: SimpleNamespace) -> None:
    _, before = _hashes(env.root)
    checkpoint: dict = {"done": {}}
    _record_all(checkpoint, before)
    assert _current_ids(before, checkpoint) == {"k8s-1", "k8s-2", "k8s-5", "vscode-3", "vscode-4"}

    (env.root / VSC_INDEX).write_text("a different vscode index", encoding="utf-8")
    _, after = _hashes(env.root)
    assert after[K8S] == before[K8S] and after[VSC] != before[VSC]
    assert _current_ids(after, checkpoint) == {"k8s-1", "k8s-2", "k8s-5"}


def test_shared_cqr_store_change_invalidates_both_repos(env: SimpleNamespace) -> None:
    _, before = _hashes(env.root)
    checkpoint: dict = {"done": {}}
    _record_all(checkpoint, before)
    (env.root / CQR).write_text("a different CQR store", encoding="utf-8")
    _, after = _hashes(env.root)
    assert after[K8S] != before[K8S] and after[VSC] != before[VSC]
    assert _current_ids(after, checkpoint) == set()


def test_k8s_only_change_invalidates_k8s_not_vscode(env: SimpleNamespace) -> None:
    _, before = _hashes(env.root)
    checkpoint: dict = {"done": {}}
    _record_all(checkpoint, before)
    (env.root / K8S_INDEX).write_text("a different k8s index", encoding="utf-8")
    _, after = _hashes(env.root)
    assert _current_ids(after, checkpoint) == {"vscode-3", "vscode-4"}


def test_plain_str_artifact_hash_keeps_the_previous_behaviour(env: SimpleNamespace) -> None:
    checkpoint: dict = {"done": {}}
    for i in ISSUES:
        rc._record_done(checkpoint, i["id"], {"plan": None}, MODEL, PH, "legacy")
    assert all(k.endswith("::legacy") for k in checkpoint["done"])
    env.ckpt.write_text(json.dumps(checkpoint), encoding="utf-8")
    assert len(rc.load_checkpoint(MODEL, PH, "legacy")[1]) == 5
    assert rc.load_checkpoint(MODEL, PH, "other")[1] == {}


# ---- migration ------------------------------------------------------------------------------------


class _Cassette:
    """Stands in for the read-only CassettePlayer: {synthesis_cache_key: stamped artifact hashes}."""

    table: dict[str, dict[str, str]] = {}

    def __init__(self, *_a: object, **_k: object) -> None:
        pass

    def get_provenance(self, key: str) -> dict[str, str] | None:
        return self.table.get(key)


def _old_checkpoint(root: Path) -> tuple[dict, str, dict[str, dict[str, str]]]:
    """A checkpoint recorded under the old combined hash, plus the cassette stamps that go with it."""
    full, _ = _hashes(root)
    legacy = af.combined_hash(full)
    stamps = {r: af.compute_artifact_hashes(root, repo=r) for r in af.REPO_SLUGS}
    done = {}
    for i in (i for i in ISSUES if i["id"] != "k8s-5"):
        rec = {"issue_id": i["id"], "model": MODEL, "prompt_hash": PH, "artifact_hash": legacy,
               "plan": {"x": 1}, "judge_score": None, "synthesis_cache_key": f"key-{i['id']}"}
        done[rc._checkpoint_key(i["id"], MODEL, PH, legacy)] = rec
    # k8s-5 is a permanent failure: no plan, so no cassette entry and no stamp to prove anything with.
    dead = {"issue_id": "k8s-5", "model": MODEL, "prompt_hash": PH, "artifact_hash": legacy,
            "plan": None, "judge_score": None, "error": "boom"}
    done[rc._checkpoint_key("k8s-5", MODEL, PH, legacy)] = dead
    return {"done": done}, legacy, {f"key-{i['id']}": stamps[i["repo"]] for i in ISSUES}


def _wire_migration(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, stamps: dict, valid: set[str]) -> None:
    _Cassette.table = stamps
    monkeypatch.setattr(rc, "ROOT", env.root)
    monkeypatch.setattr(rc, "CASSETTE_PATH", env.tmp / "cassette.json")
    monkeypatch.setattr(rc, "_CASSETTE_DIR", env.tmp)
    monkeypatch.setattr(rc, "CassettePlayer", _Cassette)
    monkeypatch.setattr(rc, "_compute_prompt_hash", lambda: PH)
    monkeypatch.setattr(rc, "TRIAGE_MODEL", MODEL)
    monkeypatch.setattr(rc, "_resolve_and_verify_artifacts", lambda: af.compute_artifact_hashes(env.root))
    monkeypatch.setattr(
        cv, "validate_from_disk",
        lambda entries, check_judge: {
            i: cv.ValidationResult(cv.VALID if i in valid else cv.STALE_SYNTHESIS) for i, r in entries.items()
            if r.get("plan") is not None
        },
    )


def test_migration_rekeys_only_proven_valid_entries_with_a_backup(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old, legacy, stamps = _old_checkpoint(env.root)
    env.ckpt.write_text(json.dumps(old), encoding="utf-8")
    # k8s-1, k8s-2 and vscode-3 replay VALID; vscode-4 does not; k8s-5 is plan-less (proven by the
    # unchanged combined hash).
    _wire_migration(env, monkeypatch, stamps, valid={"k8s-1", "k8s-2", "vscode-3"})
    assert rc.run_migrate_checkpoint(fix=False) == 2  # report only: nothing written
    assert json.loads(env.ckpt.read_text(encoding="utf-8")) == old and not list(env.tmp.glob("*.bak-*"))

    assert rc.run_migrate_checkpoint(fix=True) == 0
    new_hashes = _hashes(env.root)[1]
    done = json.loads(env.ckpt.read_text(encoding="utf-8"))["done"]
    assert len(list(env.tmp.glob("recording_checkpoint.json.bak-*"))) == 1
    migrated = {r["issue_id"] for r in done.values() if r["artifact_hash"] in new_hashes.values()}
    assert migrated == {"k8s-1", "k8s-2", "k8s-5", "vscode-3"}
    for k, r in done.items():  # every migrated entry sits under its repo's key
        if r["artifact_hash"] != legacy:
            assert k == rc._checkpoint_key(r["issue_id"], MODEL, PH, new_hashes[REPO_OF[r["issue_id"]]])
    assert any(r["issue_id"] == "vscode-4" and r["artifact_hash"] == legacy for r in done.values())
    assert "stale  vscode-4" in capsys.readouterr().out
    assert set(rc.load_checkpoint(MODEL, PH, new_hashes)[1]) == {"k8s-1", "k8s-2", "k8s-5", "vscode-3"}


def test_migration_refuses_entries_whose_stamped_artifacts_changed(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, legacy, stamps = _old_checkpoint(env.root)
    env.ckpt.write_text(json.dumps(old), encoding="utf-8")
    (env.root / VSC_INDEX).write_text("a different vscode index", encoding="utf-8")  # after recording
    _wire_migration(env, monkeypatch, stamps, valid={i["id"] for i in ISSUES})
    assert rc.run_migrate_checkpoint(fix=True) == 0
    done = json.loads(env.ckpt.read_text(encoding="utf-8"))["done"]
    new_hashes = _hashes(env.root)[1]
    kept_old = {r["issue_id"] for r in done.values() if r["artifact_hash"] == legacy}
    # k8s plan entries proven by their stamps; both vscode entries differ from the stamp; the plan-less
    # k8s entry cannot be proven because the combined hash moved.
    assert {r["issue_id"] for r in done.values() if r["artifact_hash"] == new_hashes[K8S]} == {"k8s-1", "k8s-2"}
    assert kept_old == {"k8s-5", "vscode-3", "vscode-4"}


def test_unstamped_and_unknown_entries_are_never_candidates() -> None:
    cur = rf.RepoHashes({K8S: "kk", VSC: "vv"})
    done = {
        "a": {"issue_id": "k8s-1", "model": MODEL, "prompt_hash": PH, "artifact_hash": "old", "plan": {}, "synthesis_cache_key": "x"},
        "b": {"issue_id": "ghost", "model": MODEL, "prompt_hash": PH, "artifact_hash": "old", "plan": {}},
        "c": {"issue_id": "k8s-2", "model": "other", "prompt_hash": PH, "artifact_hash": "old", "plan": {}},
        "d": {"issue_id": "vscode-3", "model": MODEL, "prompt_hash": PH, "artifact_hash": "vv", "plan": {}},
    }
    cands, rejected = rf.plan_migration(
        done, model=MODEL, prompt_hash=PH, current=cur, legacy_combined="old",
        repos=REPO_OF, current_provenance={K8S: {}, VSC: {}},
        provenance_of=lambda key: None,
    )
    assert cands == {} and set(rejected) == {"k8s-1", "ghost"}  # other-model and already-migrated: ignored


# ---- launcher -------------------------------------------------------------------------------------


def test_launcher_counts_each_entry_under_its_own_repo_hash(env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_recording_unattended as ru

    _, before = _hashes(env.root)
    checkpoint: dict = {"done": {}}
    _record_all(checkpoint, before)
    env.ckpt.write_text(json.dumps(checkpoint), encoding="utf-8")
    monkeypatch.setattr(ru, "CHECKPOINT_PATH", env.ckpt)
    monkeypatch.setattr(ru, "_REPO_BY_ISSUE", REPO_OF)
    assert ru._progress(MODEL, PH, before)["synthesized"] == 5

    (env.root / VSC_INDEX).write_text("a different vscode index", encoding="utf-8")
    _, after = _hashes(env.root)
    p = ru._progress(MODEL, PH, after)
    assert p["synthesized"] == 3  # the k8s entries; the vscode ones are stale and will be re-recorded
    assert {r["issue_id"] for r in ru._checkpoint_entries(MODEL, PH, after)} == {"k8s-1", "k8s-2", "k8s-5"}


def test_launcher_counts_respect_both_the_repo_filter_and_the_per_repo_key(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Combined: with TRIAGEIQ_RECORDING_REPOS=k8s only k8s entries count; a vscode artifact change
    must not move them; a k8s artifact change must; vscode entries never count under the filter."""
    import run_recording_unattended as ru

    _, before = _hashes(env.root)
    checkpoint: dict = {"done": {}}
    _record_all(checkpoint, before)
    env.ckpt.write_text(json.dumps(checkpoint), encoding="utf-8")
    monkeypatch.setattr(ru, "CHECKPOINT_PATH", env.ckpt)
    monkeypatch.setattr(ru, "_REPO_BY_ISSUE", REPO_OF)
    monkeypatch.setattr(ru, "_ISSUE_IDS", frozenset(i for i, r in REPO_OF.items() if r == K8S))

    def ids(h: rf.RepoHashes) -> set[str]:
        return {r["issue_id"] for r in ru._checkpoint_entries(MODEL, PH, h)}

    assert ids(before) == {"k8s-1", "k8s-2", "k8s-5"}  # filter drops vscode although its key is current
    (env.root / VSC_INDEX).write_text("a different vscode index", encoding="utf-8")
    assert ids(_hashes(env.root)[1]) == {"k8s-1", "k8s-2", "k8s-5"}  # vscode change: k8s unaffected
    (env.root / K8S_INDEX).write_text("a different k8s index", encoding="utf-8")
    assert ids(_hashes(env.root)[1]) == set()  # k8s change: k8s entries stale
