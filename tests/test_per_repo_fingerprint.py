"""A checkpoint entry is keyed by the hash of ITS repo's artifacts (plus the shared CQR store), so a
vscode-only artifact change leaves k8s entries valid. Hermetic: every root is tmp_path; no LLM,
no real artifacts."""

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
