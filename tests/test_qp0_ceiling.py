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


cl = _load("qp0cl", "qp0_ceiling_labels.py")
cr = _load("qp0cr", "qp0_ceiling_retrieval.py")
T0 = pd.Timestamp("2024-03-01T00:00:00Z")
SIG = lambda x: x.startswith("sig/")  # noqa: E731


def ts(h: float, base: pd.Timestamp = T0) -> pd.Timestamp:
    return base + pd.Timedelta(hours=h)


def issue(n: int, final: list[str], t72: list[str], text: str = "", created: pd.Timestamp = T0):
    return {
        "number": n, "created_at": created, "labels_final": final, "labels_t72h": t72,
        "text_hash": text or f"h{n}", "title": text or f"title {n}", "body": text,
    }  # fmt: skip


def events(rows: list[tuple[int, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [r[0] for r in rows], "ts": pd.to_datetime([ts(r[1]) for r in rows], utc=True),
            "action": [r[3] for r in rows], "label": [r[2] for r in rows],
        }
    )  # fmt: skip


def test_primary_is_earliest_labeled_among_final_ties_alphabetical() -> None:
    iss = pd.DataFrame(
        [
            issue(1, ["sig/b", "sig/a"], ["sig/a"]),
            issue(2, ["sig/z", "sig/y"], []),
            issue(3, ["sig/b"], ["sig/b"]),
        ]  # fmt: skip
    )
    ev = events(
        [(1, 5, "sig/b", "labeled"), (1, 1, "sig/removed", "labeled"), (1, 9, "sig/a", "labeled"),
         (2, 4, "sig/z", "labeled"), (2, 4, "sig/y", "labeled"), (3, 2, "sig/b", "labeled")]
    )  # fmt: skip
    d = cl.per_issue(iss, ev, SIG).set_index("number")
    assert d.at[1, "primary"] == "sig/b"  # removed label is not final; b (5h) beats a (9h)
    assert d.at[2, "primary"] == "sig/y"  # tie -> alphabetical
    assert d.at[1, "earliest72"] == "sig/a" and d.at[1, "primary_hours"] == 5.0


def test_label_metrics_timing_and_first_vs_final() -> None:
    iss = pd.DataFrame(
        [
            issue(1, ["sig/a"], ["sig/a"]),  # present at 72h
            issue(2, ["sig/b"], []),  # primary appears after 72h (100h)
            issue(3, ["sig/a", "sig/b"], ["sig/a"]),  # multi-label, 72h set != final
            issue(4, [], []),  # unlabelled
        ]
    )
    ev = events([(1, 1, "sig/a", "labeled"), (2, 100, "sig/b", "labeled"),
                 (3, 2, "sig/a", "labeled"), (3, 200, "sig/b", "labeled")])  # fmt: skip
    d = cl.per_issue(iss, ev, SIG)
    m = cl.label_metrics(d)
    assert m["n_labelled"] == 3 and m["multi_label"]["k"] == 1
    assert m["set_size_dist"] == {"1": 2, "2": 1}
    assert m["set72_equals_final"]["k"] == 1 and m["primary_present_at_72h"]["k"] == 2
    assert m["median_hours_until_primary_if_absent_at_72h"] == 100.0
    assert m["earliest72_equals_final_primary"] == cl.rate(2, 2)  # issues 1 and 3 have a 72h set
    assert m["final_primary_in_set72"]["k"] == 2


def _split_fixture():
    rows, ev, sp = [], [], {}
    for i in range(1, 41):  # 40 train issues: 30 sig/a (primary), 10 sig/b
        rows.append(issue(i, ["sig/a" if i <= 30 else "sig/b"], []))
        ev.append((i, 1, "sig/a" if i <= 30 else "sig/b", "labeled"))
        sp[i] = "train"
    for i in range(41, 51):  # 10 test issues, 4 sig/a / 6 sig/b
        lab = "sig/a" if i <= 44 else "sig/b"
        rows.append(issue(i, [lab], []))
        ev.append((i, 1, lab, "labeled"))
        sp[i] = "test"
    return pd.DataFrame(rows), events(ev), pd.Series(sp)


def test_baselines_use_train_priors_only() -> None:
    iss, ev, sp = _split_fixture()
    d = cl.per_issue(iss, ev, SIG)
    b = cl.baselines(d, sp)
    assert b["modelled_classes"] == ["sig/a"]
    assert b["majority_class"] == "sig/a" and b["other_share"] == 0.25  # sig/b has 10 < 30
    assert b["test"]["majority_top1"]["k"] == 4 and b["test"]["majority_top1"]["n"] == 10
    # flipping every TEST label must not move the majority class or the train table
    d2 = d.copy()
    d2.loc[d2["number"] > 40, ["final", "primary"]] = [frozenset({"sig/zzz"}), "sig/zzz"]
    b2 = cl.baselines(d2, sp)
    assert b2["majority_class"] == b["majority_class"]
    assert (
        b2["train_class_table"] == b["train_class_table"]
        and b2["modelled_classes"] == b["modelled_classes"]
    )
    assert b2["test"]["majority_top1"]["k"] == 0  # only the evaluation side changed


def test_pair_stats_differs_disjoint_and_split_of_later_issue() -> None:
    iss = pd.DataFrame(
        [
            issue(1, ["sig/a"], [], created=ts(0)), issue(2, ["sig/b"], [], created=ts(5)),
            issue(3, ["sig/a", "sig/b"], [], created=ts(9)),
        ]
    )  # fmt: skip
    d = cl.per_issue(iss, events([(1, 1, "sig/a", "labeled"), (2, 1, "sig/b", "labeled"),
                                  (3, 1, "sig/a", "labeled"), (3, 2, "sig/b", "labeled")]), SIG)  # fmt: skip
    split = pd.Series({0: "train", 1: "validation", 2: "test"})  # keyed by position
    st = cl.pair_stats([(0, 1), (0, 2)], d, split)
    assert st["all"]["primary_differs"]["k"] == 1 and st["all"]["sets_disjoint"]["k"] == 1
    assert st["validation"]["n_pairs"] == 1 and st["test"]["n_pairs"] == 1  # later issue's split
    assert st["validation"]["sets_disjoint"]["k"] == 1 and st["test"]["sets_disjoint"]["k"] == 0


def test_nn_pairs_finds_near_identical_and_masks_identical_hash() -> None:
    base = "kubelet crash loop pod restart failure after node reboot"
    texts = pd.Series(
        [
            base,
            base + " today",
            "scheduler plugin config yaml parse",
            "scheduler plugin config yaml parse x",
            base,
        ]
    )
    hashes = pd.Series(
        ["a", "b", "c", "d", "a"]
    )  # 0 and 4 are identical-hash: masked from each other
    pairs = cl.nn_pairs(texts, hashes, 0.8)
    assert (0, 1) in pairs and (2, 3) in pairs and (0, 4) not in pairs


def test_label_report_deterministic_and_complete() -> None:
    iss, ev, sp = _split_fixture()
    dup = pd.DataFrame({"source": [41], "canonical": [1]})
    r1 = cl.run(iss, ev, dup, sp, SIG)
    r2 = cl.run(iss, ev, dup, sp, SIG)
    assert json.dumps(r1, sort_keys=True, default=str) == json.dumps(
        r2, sort_keys=True, default=str
    )
    assert set(r1["labels"]) == {"all", "train", "validation", "test"}
    assert r1["inconsistency"]["duplicate_pairs"]["all"]["n_pairs"] == 1
    assert "PROXIES" in r1["ceiling_proxies"]["note"]


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


def test_scripts_import_no_classifier() -> None:
    allowed = {"sklearn.feature_extraction.text"}
    for f in ("qp0_ceiling_labels.py", "qp0_ceiling_retrieval.py"):
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
