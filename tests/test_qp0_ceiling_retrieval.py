from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _S / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _load("qp0cr", "qp0_ceiling_retrieval.py")

T0 = pd.Timestamp("2024-03-01T00:00:00Z")


def ts(h: float, base: pd.Timestamp = T0) -> pd.Timestamp:
    return base + pd.Timedelta(hours=h)


def _retrieval_issues() -> pd.DataFrame:
    texts = [
        "kubelet crash loop pod restart",
        "scheduler plugin config yaml parse error",
        "kubelet crash loop pod restart again",
        "apiserver audit log rotate disk full",
        "scheduler plugin config yaml error",
        "apiserver audit log rotate disk",
    ]
    return pd.DataFrame(
        {
            "number": [10, 11, 12, 13, 14, 15], "created_at": [ts(i * 24) for i in range(6)],
            "title": texts, "body": "", "text_hash": [f"h{i}" for i in range(6)],
        }
    )  # fmt: skip


def test_pair_section_buckets_rank_and_recall() -> None:
    idx = cr.Index(_retrieval_issues())
    pairs = pd.DataFrame(
        {"source": [12, 14, 15, 10, 12, 15], "canonical": [10, 11, 13, 12, 7, 99]}
    )  # indexed, indexed, indexed, newer, older-than-window, PR
    sec = cr.pair_section(idx, pairs, is_pr={99})
    b = sec["canonical_breakdown"]
    assert [b[k]["k"] for k in ("indexed", "newer_or_equal", "older_than_scrape_window", "pr")] == [
        3,
        1,
        1,
        1,
    ]
    assert sec["tfidf_baseline_recall_indexed_queries_only"]["recall@1"]["k"] == 3  # all are top-1
    allp = sec["tfidf_baseline_recall_all_pairs_unindexed_are_misses"]["recall@1"]
    assert allp["n"] == 6 and allp["k"] == 3  # denominators differ
    assert sec["cosine_query_target"]["share_ge_0.1"] == 1.0


def test_fix_groups_use_only_audited_merged_links_and_fix_section() -> None:
    links = pd.DataFrame(
        {
            "issue": [10, 12, 11, 14, 13, 15], "pr": [1, 1, 2, 2, 3, 3],
            "method": [cr.METHOD, cr.METHOD, cr.METHOD, "pr_closing_refs", cr.METHOD, cr.METHOD],
            "pr_merged": [True, True, True, True, True, False],
        }
    )  # fmt: skip
    g = cr.fix_groups(links, {10, 11, 12, 13, 14, 15})
    assert g == {1: [10, 12]}  # pr 2 loses its non-audited link, pr 3 has an unmerged link
    idx = cr.Index(_retrieval_issues())
    split = pd.Series({10: "train", 11: "train", 12: "test", 13: "test", 14: "test", 15: "test"})
    sec = cr.fix_section(idx, g, split)
    assert sec["n_groups"] == 1 and sec["test"]["n_answerable_queries"] == 1  # 12 has earlier 10
    assert sec["train"]["n_answerable_queries"] == 0  # 10 has no earlier member
    assert sec["test"]["tfidf_baseline_recall"]["recall@1"]["k"] == 1
    assert sec["test"]["control_random_same_year"]["recall"]["recall@1"]["n"] == 1


def test_fix_linked_withholding_rule(tmp_path: Path) -> None:
    missing = tmp_path / "none.json"
    assert "missing" in cr.audit_gate(missing)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"acceptance": {"result": "fail"}}))
    assert "not 'pass'" in cr.audit_gate(bad)
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"acceptance": {"result": "pass"}}))
    assert cr.audit_gate(ok) is None


def test_retrieval_run_deterministic_and_withholds(tmp_path: Path) -> None:
    d, sd = tmp_path / "d", tmp_path / "s"
    d.mkdir()
    sd.mkdir()
    iss = _retrieval_issues().assign(labels_final=[[]] * 6)
    iss.to_parquet(d / "issues.parquet")
    pd.DataFrame({"source": [12], "canonical": [10], "canonical_is_pr": [False]}).to_parquet(
        d / "duplicate_pairs.parquet"
    )
    pd.DataFrame({"source": [12], "canonical": [10]}).to_parquet(sd / "dup_task_pairs.parquet")
    for s in cr.SPLITS:
        pd.DataFrame({"number": [10], "split": [s]}).to_parquet(sd / f"{s}.parquet")
    r1 = cr.run(d, sd, tmp_path / "nope.json")
    r2 = cr.run(d, sd, tmp_path / "nope.json")
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert "withheld" in r1["fix_linked"] and "n_groups" not in r1["fix_linked"]


def test_script_imports_no_classifier() -> None:
    allowed = {"sklearn.feature_extraction.text"}
    for f in ("qp0_ceiling_retrieval.py",):
        tree = ast.parse((_S / f).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sklearn"):
                assert n.module in allowed and [a.name for a in n.names] == ["TfidfVectorizer"], f
            if isinstance(n, ast.Import):
                assert not any(
                    a.name.split(".")[0] in {"sklearn", "torch", "xgboost"} for a in n.names
                )
        src = (_S / f).read_text(encoding="utf-8")
        assert not any(
            w in src for w in (".fit(", "LogisticRegression", "RandomForest", "predict(")
        )
