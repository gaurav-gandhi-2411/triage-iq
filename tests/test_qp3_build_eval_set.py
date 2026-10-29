from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))  # the replay harness imports its siblings top-level
_SPEC = importlib.util.spec_from_file_location("qp3b", ROOT / "scripts" / "qp3_build_eval_set.py")
q = importlib.util.module_from_spec(_SPEC)
sys.modules["qp3b"] = q
_SPEC.loader.exec_module(q)

COMMITTED_KEYS = set(
    json.loads((ROOT / "eval" / "eval_set.jsonl").read_text("utf-8").splitlines()[0])
)


def _row(**kw: object) -> pd.Series:
    base = {
        "number": 5, "title": "t", "body": "b", "body_raw": "B", "cls": "sig/node", "stratum": "s", "primary": "p",
        "parent": "p", "in_family": True, "state": "CLOSED", "labels_final": ["kind/bug"],
        "dup_targets": [], "fix_peers": [], "fix_prs": [], "prio_label": None, "days": 2.0,
        "created_at": pd.Timestamp("2026-02-01T00:00:00Z"),
        "scraped_at_max": pd.Timestamp("2026-10-01T00:00:00Z"),
    }  # fmt: skip
    return pd.Series({**base, **kw})


def test_to_row_schema_and_priority_sources() -> None:
    r = q.to_row(_row(), "kubernetes/kubernetes", 2.0)
    assert COMMITTED_KEYS - {"similar_issues"} <= set(r)  # same keys as the committed set
    assert r["id"] == "k8s-5" and r["gold_priority"] == "medium"
    assert r["body"] == "b" and r["body_raw"] == "B"
    assert r["meta"]["gold_priority_source"] == "resolution_inferred"
    lab = q.to_row(_row(prio_label="high"), "kubernetes/kubernetes", 1.0)
    assert lab["gold_priority"] == "high" and lab["meta"]["gold_priority_source"] == "label"
    op = q.to_row(_row(state="OPEN", days=float("nan")), "microsoft/vscode", 1.0)
    assert op["actual_resolution_days"] is None and op["gold_priority"] is None
    assert op["meta"]["censored"] is True and op["meta"]["gold_priority_source"] == "none"


def test_replay_loader_consumes_the_schema(tmp_path: Path) -> None:
    """The harness's own loaders accept rows in this schema (with frozen similar_issues)."""
    import run_eval
    from frozen_retriever import build_frozen_retrievers

    rows = []
    for i, repo in enumerate(("kubernetes/kubernetes", "microsoft/vscode")):
        r = q.to_row(_row(number=100 + i), repo, 1.0)
        r["similar_issues"] = [{"number": 1, "score": 0.5, "text": "x"}]
        rows.append(r)
    path = tmp_path / "eval_set.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    assert [i["id"] for i in run_eval._load_eval_set(path)] == ["k8s-100", "vscode-101"]
    fr = build_frozen_retrievers(path)
    assert fr["kubernetes/kubernetes"].retrieve("", k=5, exclude_number=100)[0]["number"] == 1


# ---- the committed artifact -------------------------------------------------------------------
SET_DIR = ROOT / "eval" / "sets" / "quality_push_2026-10"
REPORT = ROOT / "reports" / "qp3_eval_set_150_2026-10.json"
REPOS = ("kubernetes/kubernetes", "microsoft/vscode")


def _rows() -> list[dict]:
    # split on newline only (as the harness's file iteration does): bodies may hold U+2028
    text = (SET_DIR / "eval_set.jsonl").read_text("utf-8")
    return [json.loads(x) for x in text.split("\n") if x]


def test_committed_set_matches_report_and_quotas() -> None:
    rows, rep = _rows(), json.loads(REPORT.read_text("utf-8"))
    assert len(rows) == 150 and len({r["id"] for r in rows}) == 150
    for repo in REPOS:
        mine = [r for r in rows if r["repo"] == repo]
        assert len(mine) == rep[repo]["n"] == q.N_PER_REPO
        got = pd.Series([r["meta"]["stratum"] for r in mine]).value_counts().to_dict()
        for h, v in rep[repo]["strata"].items():
            assert got.get(h, 0) == v["sample"]
            assert v["sample"] >= min(v["population"], rep[repo]["minimum"])  # minimums respected
    assert rep["files"]["eval_set.jsonl"] == q.sha256_file(SET_DIR / "eval_set.jsonl")
    assert (
        rep["design_version"] == "2.1" and rep["supersedes_v1"]["eval_set.jsonl"] == q.V1_SET_SHA256
    )
    assert rep["tokens"]["planning_total_tokens"] == 900_000
    assert rep["relevance_scoring"]["status"].startswith("UNSCORABLE")
    for repo in REPOS:
        assert rep[repo]["weights"]["kish_n_eff"] <= rep[repo]["weights"]["n"]


def test_committed_set_rows_are_clean_and_replay_loadable() -> None:
    import run_eval
    from frozen_retriever import build_frozen_retrievers

    rows = _rows()
    assert all(len(r["body"]) > 0 and r["title"] for r in rows)
    assert max(len(r["body"]) for r in rows) > q.PROMPT_BODY_CHARS  # long bodies are kept (v2)
    assert all(len(r["similar_issues"]) <= q.K for r in rows)
    # issue numbers follow creation order, so no hit may be numbered above its query
    assert all(h["number"] < r["number"] for r in rows for h in r["similar_issues"])
    assert all(set(r) >= COMMITTED_KEYS for r in rows)
    assert all(r["created_at"] >= "2026-01-01" for r in rows)  # test period only
    loaded = run_eval._load_eval_set(SET_DIR / "eval_set.jsonl")
    assert [r["id"] for r in loaded] == [r["id"] for r in rows]
    frozen = build_frozen_retrievers(SET_DIR / "eval_set.jsonl")  # also loads the npz
    assert set(frozen) == set(REPOS)
    k8s = [r for r in rows if r["repo"] == REPOS[0]]
    for r in k8s:
        hits, emb = frozen[REPOS[0]].retrieve_with_embedding("", k=5, exclude_number=r["number"])
        assert [h["number"] for h in hits] == [s["number"] for s in r["similar_issues"]]
        assert emb is not None and emb.shape == (768,)


@pytest.mark.skipif(not (q.DATA / "splits").exists(), reason="D: split manifests not available")
def test_committed_set_only_eligible_test_issues() -> None:
    for repo, cfg in q.REPOS.items():
        sp = {
            s: pd.read_parquet(q.DATA / "splits" / cfg["slug"] / f"{s}.parquet")
            for s in ("train", "validation", "test")
        }
        elig = set(sp["test"].loc[sp["test"]["classifier_eligible"], "number"])
        nums = {r["number"] for r in _rows() if r["repo"] == repo}
        assert nums <= elig
        assert not nums & (set(sp["train"]["number"]) | set(sp["validation"]["number"]))


def test_committed_set_is_splitlines_safe() -> None:
    text = (SET_DIR / "eval_set.jsonl").read_text("utf-8")
    assert len(text.splitlines()) == 150
