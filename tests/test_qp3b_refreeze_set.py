from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "qp3b_refreeze", ROOT / "scripts" / "qp3b_refreeze_set.py"
)
rf = importlib.util.module_from_spec(_SPEC)
sys.modules["qp3b_refreeze"] = rf
_SPEC.loader.exec_module(rf)
q = rf.q

K8S, VSC = "kubernetes/kubernetes", "microsoft/vscode"


def _row(repo: str, number: int, body: str = "hello world") -> dict[str, Any]:
    return {
        "id": f"{repo[0]}-{number}", "repo": repo, "number": number, "title": "t", "body": body,
        "created_at": "2026-02-01T00:00:00+00:00",
        "meta": {"dup_targets": [7], "fix_peers": [], "fix_prs": []},
        "similar_issues": [{"number": 1, "score": 0.1, "text": "old"}],
    }  # fmt: skip


class _FakeRetriever:
    """Stands in for SimilarIssueRetriever: records the directory it was loaded from."""

    loaded: list[str] = []

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.issue_numbers = np.array([7, 8, 9])

    @classmethod
    def load(cls, out_dir: str) -> _FakeRetriever:
        cls.loaded.append(out_dir)
        return cls(out_dir)

    def retrieve_with_embedding(self, text: str, k: int, exclude_number: int) -> tuple[Any, Any]:
        hits = [{"number": n, "score": 0.5, "text": f"from {Path(self.directory).name}"}
                for n in (8, 9) if n != exclude_number]  # fmt: skip
        return hits, np.ones(768, dtype=np.float32)


@pytest.fixture
def fake_freeze_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from triage_iq.models import similar_issues

    _FakeRetriever.loaded = []
    monkeypatch.setattr(similar_issues, "SimilarIssueRetriever", _FakeRetriever)
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    monkeypatch.setattr(q, "created_lookup", lambda repo: {7: ts, 8: ts, 9: ts})
    monkeypatch.setattr(
        q, "_read", lambda *a, **k: pd.DataFrame({"number": [7], "created_at": [ts]})
    )
    monkeypatch.setattr(q, "sha256_file", lambda p: "sha")


def test_freeze_default_uses_the_served_index_dir(fake_freeze_env: None) -> None:
    rows = [_row(K8S, 100)]
    q.freeze(rows, embeddings_out=None)
    assert _FakeRetriever.loaded == [str(q.served_index_dir(K8S))]
    assert q.served_index_dir(K8S).name == "similar_issue_index_kubernetes_kubernetes_bge"


def test_freeze_index_dirs_override_and_only_repos(fake_freeze_env: None, tmp_path: Path) -> None:
    rows = [_row(K8S, 100), _row(VSC, 200)]
    untouched = json.dumps(rows[1])
    cand = tmp_path / "similar_issue_index_kubernetes_kubernetes_bge_v2"
    info = q.freeze(rows, index_dirs={K8S: cand}, only_repos={K8S}, embeddings_out=None)
    assert _FakeRetriever.loaded == [str(cand)]  # vscode index never opened
    assert [h["text"] for h in rows[0]["similar_issues"]] == [f"from {cand.name}"] * 2
    assert rows[0]["meta"]["dup_targets_in_index"] == [7]  # recomputed from the new index
    assert json.dumps(rows[1]) == untouched  # other repo's row untouched
    assert list(info["neighbour_filter"]) == [K8S]
    assert not list(tmp_path.glob("*.npz"))  # embeddings_out=None writes nothing


def test_freeze_writes_embeddings_only_where_asked(fake_freeze_env: None, tmp_path: Path) -> None:
    out = tmp_path / "emb.npz"
    q.freeze([_row(K8S, 100)], embeddings_out=out)
    z = np.load(out)
    assert z["numbers"].tolist() == [100] and z["embeddings"].shape == (1, 768)


def test_add_body_raw_places_key_after_body_and_counts_changes() -> None:
    from triage_iq.data.preprocess import clean_text

    raw_md = "see `code` here\n\n```\nblock\n```"
    assert clean_text(raw_md)[0] != raw_md  # the case under test: cleaning changes the text
    rows = [_row(K8S, 1, "plain"), _row(K8S, 2, clean_text(raw_md)[0])]
    raw = {K8S: {1: "plain", 2: raw_md}}
    st = rf.add_body_raw(rows, raw)
    keys = list(rows[0])
    assert keys[keys.index("body") + 1] == "body_raw"
    assert rows[1]["body_raw"] == raw_md and rows[0]["body_raw"] == "plain"
    assert st[K8S] == {"rows": 2, "text_changed": 1, "identical": 1}


def test_add_body_raw_fails_loudly() -> None:
    with pytest.raises(KeyError):
        rf.add_body_raw([_row(K8S, 1)], {K8S: {}})
    with pytest.raises(ValueError, match="clean_text"):
        rf.add_body_raw([_row(K8S, 1, "stored")], {K8S: {1: "something else entirely"}})


def test_verify_manifest_detects_a_modified_file(tmp_path: Path) -> None:
    (tmp_path / "index.faiss").write_bytes(b"abc")
    (tmp_path / "meta.pkl").write_bytes(b"def")
    files = {
        n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
        for n in ("index.faiss", "meta.pkl")
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}))
    assert rf.verify_manifest(tmp_path) == files
    (tmp_path / "meta.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        rf.verify_manifest(tmp_path)


def test_relevance_counts() -> None:
    a, b = _row(K8S, 1), _row(K8S, 2)
    a["meta"] |= {"dup_targets_in_index": [7], "fix_peers_in_index": [], "indexable_targets": [7]}
    b["meta"] |= {"dup_targets_in_index": [], "fix_peers_in_index": [], "indexable_targets": []}
    c = rf.relevance_counts([a, b], [K8S])[K8S]
    assert c == {
        "with_dup_or_peer": 2,
        "target_in_served_index": 1,
        "indexable_in_derived_v2_before_query": 1,
    }


def test_embeddings_match_tolerates_float_noise_only() -> None:
    base = {"repos": np.array([K8S, K8S]), "numbers": np.array([1, 2]),
            "embeddings": np.eye(2, 768, dtype=np.float32)}  # fmt: skip
    noisy = {**base, "embeddings": base["embeddings"] + np.float32(3e-7)}
    ok, diff = rf.embeddings_match(base, noisy)
    assert ok and 0 < diff < rf.EMB_ATOL
    moved = {**base, "embeddings": base["embeddings"] + np.float32(1e-3)}
    assert rf.embeddings_match(base, moved)[0] is False
    reordered = {**base, "numbers": np.array([2, 1])}
    assert rf.embeddings_match(base, reordered) == (False, float("inf"))
