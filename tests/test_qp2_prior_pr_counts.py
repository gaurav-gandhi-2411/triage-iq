from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp2c", _S / "qp2_prior_pr_counts.py")
qc = importlib.util.module_from_spec(_SPEC)
sys.modules["qp2c"] = qc
_SPEC.loader.exec_module(qc)

TQ = pd.Timestamp("2024-06-01T12:00:00Z")


def pr(
    n: int, merged_at: pd.Timestamp | None, files: list[str], base: str = "master", **kw: Any
) -> dict[str, Any]:
    return {
        "number": n, "merged": merged_at is not None, "base": base, "merged_at": merged_at,
        "created_at": kw.get("created_at", pd.Timestamp("2024-01-01T00:00:00Z")),
        "files": tuple(files), "truncated": kw.get("truncated", False),
    }  # fmt: skip


def tables(fix: int = 900, issue: int = 5, split: str = "test"):  # noqa: ANN201
    issues = pd.DataFrame({"number": [issue], "created_at": [TQ]})
    links = pd.DataFrame({"issue": [issue], "pr": [fix], "method": ["closed_event_closer_pr"], "pr_merged": [True], "pr_merged_at": [TQ + pd.Timedelta(days=1)]})  # fmt: skip
    return issues, links, pd.Series({issue: split})


BACKGROUND = [  # keeps any file touched by a handful of test PRs well under the 1% frequency rule
    pr(10_000 + i, pd.Timestamp("2023-02-01T00:00:00Z") + pd.Timedelta(hours=i), [f"bg/{i}.go"])
    for i in range(500)
]


def run_one(prs: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return qc.run([*BACKGROUND, *prs], *tables(**kw))


FIX = pr(900, TQ + pd.Timedelta(days=1), ["pkg/kubelet/a.go", "go.mod"])


def test_pattern_exclusions() -> None:
    for path in ["go.mod", "staging/src/k8s.io/api/go.sum", "vendor/x/y.go", "staging/src/k8s.io/client-go/vendor/a.go",
                 "pkg/apis/core/zz_generated.deepcopy.go", "staging/x/generated.pb.go", "a/b.pb.go", "x/generated.proto",
                 "api/openapi-spec/swagger.json", "CHANGELOG/CHANGELOG-1.30.md", "pkg/OWNERS", "OWNERS_ALIASES"]:  # fmt: skip
        assert qc.pattern_excluded(path), path
    for path in [
        "pkg/kubelet/a.go",
        "test/testdata/x.yaml",
        "docs/vendor-guide.md",
        "pkg/owners.go",
        "go.mod.sample",
    ]:
        assert not qc.pattern_excluded(path), path


def test_index_cutoff_is_strict_and_fix_pr_never_a_positive() -> None:
    one_sec_before = pr(1, TQ - pd.Timedelta(seconds=1), ["pkg/kubelet/a.go"])
    at_tq = pr(2, TQ, ["pkg/kubelet/a.go"])
    after = pr(3, TQ + pd.Timedelta(seconds=1), ["pkg/kubelet/a.go"])
    s = run_one([one_sec_before, at_tq, after, FIX])["per_split"]["test"]
    assert (
        s["answerable"] == 1 and s["positives_per_answerable"]["max"] == 1
    )  # only the PR 1 s before
    idx = qc.Index([one_sec_before, at_tq, after, FIX], "master")
    assert idx.size_before(TQ) == 1 and idx.positives({"pkg/kubelet/a.go"}, TQ, 900) == {1}
    # a fix PR that (wrongly) merged before the query is still excluded and counted
    early_fix = {**FIX, "merged_at": TQ - pd.Timedelta(days=2)}
    idx2 = qc.Index([early_fix], "master")
    assert idx2.positives({"pkg/kubelet/a.go"}, TQ, 900) == set()
    res = run_one([one_sec_before, early_fix])["per_split"]["test"]
    assert res["fix_not_after_query"] == 1 and res["positives_per_answerable"]["max"] == 1


def test_base_branch_window_and_cherry_pick_sensitivity() -> None:
    master = pr(1, TQ - pd.Timedelta(days=1), ["pkg/kubelet/a.go"])
    cherry = pr(2, TQ - pd.Timedelta(days=1), ["pkg/kubelet/a.go"], base="release-1.30")
    old = pr(
        3,
        TQ - pd.Timedelta(days=1),
        ["pkg/kubelet/a.go"],
        created_at=pd.Timestamp("2022-12-31T23:59:59Z"),
    )
    unmerged = pr(4, None, ["pkg/kubelet/a.go"])
    rep = run_one([master, cherry, old, unmerged, FIX])
    assert (
        rep["per_split"]["test"]["positives_per_answerable"]["max"] == 1
    )  # only the master PR in the window
    assert (
        rep["sensitivity"]["cherry_picks_included"]["test"]["positives_per_answerable"]["max"] == 2
    )
    assert (
        rep["pr_population"]["merged_other_base"] == 1
        and rep["pr_population"]["merged_created_before_window"] == 1
    )


def test_answerable_logic() -> None:
    only_excluded = pr(900, TQ + pd.Timedelta(days=1), ["go.mod", "vendor/a.go"])
    p = pr(1, TQ - pd.Timedelta(days=1), ["go.mod", "vendor/a.go"])
    s = run_one([p, only_excluded])["per_split"]["test"]
    assert (
        s["fix_has_informative_files"] == 0 and s["answerable"] == 0
    )  # no informative file -> not answerable
    no_overlap = pr(1, TQ - pd.Timedelta(days=1), ["pkg/other/b.go"])
    assert (
        run_one([no_overlap, FIX])["per_split"]["test"]["answerable"] == 0
    )  # informative but no positive
    assert (
        run_one([no_overlap])["per_split"]["test"]["fix_pr_files_known"] == 0
    )  # fix PR absent from the table


def test_frequency_rule_is_strictly_more_than_threshold_and_train_only() -> None:
    early = pd.Timestamp("2024-01-01T00:00:00Z")
    prs = [pr(i, early + pd.Timedelta(hours=i), ["common.go", f"u{i}.go"]) for i in range(100)]
    prs += [
        pr(1000, pd.Timestamp("2025-09-01T00:00:00Z"), ["late.go"] * 1)
    ]  # after TRAIN_END: not counted
    common, n = qc.common_files(prs, "master", 0.01)
    assert n == 100 and set(common) == {
        "common.go"
    }  # each u{i} is exactly 1.0%: not MORE than 1.0%
    assert qc.common_files(prs, "master", 0.0099)[0].keys() >= {"common.go", "u1.go"}
    rep = qc.common_report(prs, "master")
    assert rep["files"] == ["common.go"] and len(rep["sha256_sorted_newline_joined"]) == 64


def test_frequency_exclusion_removes_popular_overlap() -> None:
    early = pd.Timestamp("2024-01-01T00:00:00Z")
    popular = [pr(i, early + pd.Timedelta(hours=i), ["common.go"]) for i in range(1, 60)]
    fix = pr(900, TQ + pd.Timedelta(days=1), ["common.go"])
    p = pr(1000, TQ - pd.Timedelta(days=1), ["common.go"])
    # common.go is in >1% of train PRs -> informative set is empty -> not answerable
    assert run_one([*popular, p, fix])["per_split"]["test"]["answerable"] == 0
    sens = run_one([*popular, p, fix])["sensitivity"]
    assert sens["threshold_2pct"]["test"]["answerable"] == 0  # still above 2%


def test_summarise_ci_clusters_and_trivial_popularity() -> None:
    res = [
        {"split": "test", "issue": i, "fix": 900 + (i % 2), "answerable": True, "positives": 2, "trivial_pairs": 1, "fix_known": True,
         "informative": 1, "index_size": 10, "fix_after_query": True, "fix_in_window": True}
        for i in range(4)
    ]  # fmt: skip
    s = qc.summarise(res, "test")
    assert (
        s["answerable"] == 4
        and s["distinct_fix_prs_among_answerable"] == 2
        and s["effective_n_kish"] == pytest.approx(2.0)
    )
    assert s["wilson_half_width_at_n"]["p0.5"] == pytest.approx(1.96 * 0.5 / 2)
    assert s["trivially_popular_positive_share"] == pytest.approx(0.5)
    assert s["answerable_with_a_nontrivial_positive"] == 4


def test_deterministic_and_json_serialisable() -> None:
    prs = [pr(1, TQ - pd.Timedelta(days=1), ["pkg/kubelet/a.go"]), FIX]
    a, b = run_one(prs), run_one(prs)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
