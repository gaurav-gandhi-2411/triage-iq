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


def test_script_imports_no_classifier() -> None:
    allowed = {"sklearn.feature_extraction.text"}
    for f in ("qp0_ceiling_labels.py",):
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
