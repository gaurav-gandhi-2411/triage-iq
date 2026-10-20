"""Label coverage and routing-semantics evidence for the component taxonomy decision.

PRE-REGISTERED (GG decision D79 item 4, 2026-10-20): the taxonomy is decided on label coverage
and routing semantics ONLY, before any model is scored. This script imports and trains no
classifier and reports no accuracy and no comparison with any earlier class count.
Candidate families. k8s: A = `area/*`, S = `sig/*`, SA = an issue with either. vscode: V_old =
the component list in preprocess.LABEL_FACET_PATTERNS['microsoft/vscode']['component']
(imported, case-insensitive); V_new = labels not in V_old and not in TYPE_WORKFLOW (a heuristic
exclusion set) with >= 100 issues in the data; V_all = V_old union V_new. Label-frequency
tables of every other prefix family are listed so non-components are visible.
Metrics per family, overall and per created year 2023-2026, on ISSUES (not PRs):
 (1) coverage_t72h / coverage_final: share with >= 1 family label at 72 h / at the end;
 (2) share with exactly 1 and with >= 2 family labels (final);
 (3) churn: among issues with >= 1 final family label, share whose t72h family set != final set;
 (4) median / p90 hours from creation to the first family `labeled` event (issues never
     receiving one are excluded and counted);
 (5) who executed that first event: prow_bot (login ends 'bot' / '[bot]', or k8s-ci-robot, or
     kubernetes-prow, which replaced k8s-ci-robot on 2026-06-20) / other / unknown (no actor).
     CAVEAT: Prow applies sig/* and area/* on behalf of humans' `/sig x` and `/area x`
     comments, so this says "Prow executed it", NOT "automatic labelling"; bot-vs-other is
     no evidence about routing intent;
 (6) class table: per label the final count and per-year counts; number of classes with >= 30
     and >= 100 issues; share of family-labelled issues having >= 1 label in a class >= 30;
 (7) stability: classes with >= 5 issues in every year vs only some years;
 (8) k8s: co-occurrence of S and A (share of S issues with an A label and vice versa) and, per
     sig/*, its top 3 co-occurring area/* labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from triage_iq.data.preprocess import LABEL_FACET_PATTERNS  # noqa: E402

YEARS = (2023, 2024, 2025, 2026)
DESIGN = __doc__.split("PRE-REGISTERED", 1)[1].strip()
# Heuristic (not a ground truth): issue type / workflow-state labels that are not components.
TYPE_WORKFLOW = {
    "bug", "feature-request", "enhancement", "question", "documentation", "info-needed",
    "needs more info", "verified", "verification-needed", "verification-found",
    "insiders-released", "unreleased", "confirmed", "important", "upstream",
    "upstream-issue-linked", "duplicate", "invalid", "wontfix", "help wanted", "good first issue",
    # added 2026-10-20 after reading the real vscode label list (process / triage / platform / test-
    # infra labels that are not components); vscode's "*" labels are triage resolutions
    "triage-needed", "spam", "stale", "candidate", "confirmation-pending", "polish", "debt", "wont-fix",
    "regression", "recent-regression", "papercut :drop_of_blood:", "author-verification-requested",
    "testplan-item", "on-testplan", "plan-item", "under-discussion", "on-release-notes", "new release",
    "stable-anomaly", "endgame-plan", "engineering", "modernization", "windows", "linux", "macos", "wsl",
    "ai-translated", "ai-generated", "ai-analyzed", "ai-triaged", "smoke-test-failure",
    "integration-test-failure", "unit-test-failure", "freeze-slow-crash-leak",
}  # fmt: skip
Pred = Callable[[str], bool]
BOT_RE = re.compile(r"(bot|\[bot\])$|^(k8s-ci-robot|kubernetes-prow)$", re.I)


def is_type_workflow(label: str) -> bool:
    low = label.lower()
    return (
        low in TYPE_WORKFLOW
        or low.endswith(("-wontfix", "wontfix"))
        or low.startswith(("*", "translation-required"))
    )


def v_new_labels(issues: pd.DataFrame, v_old: set[str], min_issues: int = 100) -> list[str]:
    """Labels outside V_old and TYPE_WORKFLOW with >= min_issues final issues."""
    cnt = Counter(lbl for ls in issues["labels_final"] for lbl in set(ls))
    return sorted(
        lbl
        for lbl, n in cnt.items()
        if n >= min_issues and lbl.lower() not in v_old and not is_type_workflow(lbl)
    )


def families(repo: str, issues: pd.DataFrame) -> dict[str, Pred]:
    """Candidate family predicates for the repo (label name -> in family)."""
    if repo.lower() == "kubernetes/kubernetes":
        a, s = (lambda x: x.startswith("area/")), (lambda x: x.startswith("sig/"))
        return {"A": a, "S": s, "SA": lambda x: a(x) or s(x)}
    if repo.lower() == "microsoft/vscode":
        old = {x.lower() for x in LABEL_FACET_PATTERNS["microsoft/vscode"]["component"]}
        new = set(v_new_labels(issues, old))
        return {
            "V_old": lambda x: x.lower() in old,
            "V_new": lambda x: x in new,
            "V_all": lambda x: x.lower() in old or x in new,
        }
    raise SystemExit(f"no candidate families defined for {repo}")


def first_events(events: pd.DataFrame, pred: Pred, issues: pd.DataFrame) -> pd.DataFrame:
    """First family `labeled` event per issue with hours since creation and actor kind."""
    e = events[(events["action"] == "labeled") & events["label"].map(pred)]
    e = e.sort_values(["number", "ts", "label"], kind="stable").drop_duplicates("number")
    e = e.merge(issues[["number", "created_at"]], on="number")
    e["hours"] = (e["ts"] - e["created_at"]).dt.total_seconds() / 3600
    e["kind"] = [
        "unknown" if a is None or a != a else "prow_bot" if BOT_RE.search(a) else "other"
        for a in e["actor"]
    ]
    return e


def pct(k: int | float, n: int) -> float | None:
    return k / n if n else None


def class_metrics(sets_final: pd.Series) -> dict[str, Any]:
    """Class table, thresholds and the share of labelled issues inside classes >= 30."""
    cnt = Counter(lbl for s in sets_final for lbl in s)
    big30 = {k for k, n in cnt.items() if n >= 30}
    labelled = [s for s in sets_final if s]
    return {
        "n_classes": len(cnt),
        "classes_ge30": len(big30),
        "classes_ge100": sum(1 for n in cnt.values() if n >= 100),
        "share_labelled_in_ge30": pct(sum(1 for s in labelled if s & big30), len(labelled)),
    }


def metrics(df: pd.DataFrame, pred: Pred, fe: pd.DataFrame) -> dict[str, Any]:
    """Metrics (1)-(5) and the class thresholds for one issue subset."""
    n = len(df)
    fin = df["labels_final"].map(lambda ls: frozenset(x for x in ls if pred(x)))
    t72 = df["labels_t72h"].map(lambda ls: frozenset(x for x in ls if pred(x)))
    has = fin.map(bool)
    f = fe[fe["number"].isin(df["number"])]
    h = f["hours"]
    return {
        "n_issues": n,
        "coverage_t72h": pct(int(t72.map(bool).sum()), n),
        "coverage_final": pct(int(has.sum()), n),
        "exactly_1": pct(int((fin.map(len) == 1).sum()), n),
        "ge_2": pct(int((fin.map(len) >= 2).sum()), n),
        "churn": pct(int(((t72 != fin) & has).sum()), int(has.sum())),
        "hours_to_first_label": {
            "median": float(h.median()) if len(h) else None,
            "p90": float(h.quantile(0.9)) if len(h) else None,
            "n_with_label": len(f),
            "n_never_labelled": n - len(f),
        },  # fmt: skip
        "first_label_actor_prow_bot_vs_other": {
            k: int((f["kind"] == k).sum()) for k in ("prow_bot", "other", "unknown")
        },
        "classes": class_metrics(fin),
    }


def class_table(issues: pd.DataFrame, pred: Pred) -> dict[str, Any]:
    """(6) per-label counts overall/per year and (7) stability across 2023-2026."""
    rows: dict[str, dict[str, int]] = {}
    for yr, ls in zip(issues["created_year"], issues["labels_final"], strict=True):
        for lbl in {x for x in ls if pred(x)}:
            r = rows.setdefault(lbl, {"total": 0, **{str(y): 0 for y in YEARS}})
            r["total"] += 1
            r[str(yr)] = r.get(str(yr), 0) + 1
    present = {k: [v[str(y)] >= 5 for y in YEARS] for k, v in rows.items()}
    return {
        "classes": dict(sorted(rows.items(), key=lambda kv: -kv[1]["total"])),
        "stability": {
            "present_ge5_every_year": sorted(k for k, p in present.items() if all(p)),
            "present_ge5_some_years": sorted(
                k for k, p in present.items() if any(p) and not all(p)
            ),
            "never_ge5_in_a_year": sorted(k for k, p in present.items() if not any(p)),
        },
    }


def cooccurrence(issues: pd.DataFrame) -> dict[str, Any]:
    """(8) k8s: how sig/* and area/* labels overlap, and each sig's top 3 areas."""
    s = issues["labels_final"].map(lambda ls: {x for x in ls if x.startswith("sig/")})
    a = issues["labels_final"].map(lambda ls: {x for x in ls if x.startswith("area/")})
    s_any, a_any = s.map(bool), a.map(bool)
    per_sig: dict[str, Counter[str]] = {}
    for ss, aa in zip(s, a, strict=True):
        for sig in ss:
            per_sig.setdefault(sig, Counter()).update(aa)
    n_sig = {sig: sum(sig in x for x in s) for sig in per_sig}
    top = {
        sig: {
            "n_issues": n_sig[sig],
            "top3_area": sorted(per_sig[sig].items(), key=lambda kv: (-kv[1], kv[0]))[:3],
        }
        for sig in sorted(per_sig, key=lambda k: -n_sig[k])
    }
    return {
        "share_S_issues_with_A": pct(int((s_any & a_any).sum()), int(s_any.sum())),
        "share_A_issues_with_S": pct(int((s_any & a_any).sum()), int(a_any.sum())),
        "n_S_issues": int(s_any.sum()), "n_A_issues": int(a_any.sum()),
        "per_sig_top3_areas": top,
    }  # fmt: skip


def prefix_of(label: str) -> str:
    if "/" in label:
        return label.split("/", 1)[0] + "/"
    if label.startswith("needs-"):
        return "needs-*"
    return "(no prefix)"


def other_families(freq: pd.DataFrame) -> dict[str, Any]:
    """Every prefix family from label_frequency.csv, with its top 10 labels by final count."""
    freq = freq.assign(prefix=freq["label"].map(prefix_of))
    out = {}
    for p, g in freq.groupby("prefix"):
        g = g.sort_values("n_issues_final", ascending=False)
        top = g.head(10)[["label", "n_issues_final"]].to_dict("records")
        out[p] = {
            "n_labels": len(g), "sum_issue_labels_final": int(g["n_issues_final"].sum()),
            "top10": top,
        }  # fmt: skip
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["sum_issue_labels_final"]))


def build_report(repo: str, derived: Path) -> dict[str, Any]:
    issues = pd.read_parquet(derived / "issues.parquet")
    issues = issues[issues["created_year"].isin(YEARS)]
    events = pd.read_parquet(derived / "label_events.parquet")
    freq = pd.read_csv(derived / "label_frequency.csv")
    fams = families(repo, issues)
    report: dict[str, Any] = {
        "repo": repo, "design": DESIGN, "generated_with": "scripts/qp0_taxonomy_coverage.py",
        "n_issues": len(issues), "families": {},
    }  # fmt: skip
    for name, pred in fams.items():
        fe = first_events(events, pred, issues)
        fam = {"overall": metrics(issues, pred, fe)}
        fam["per_year"] = {
            str(y): metrics(issues[issues["created_year"] == y], pred, fe) for y in YEARS
        }
        fam["class_table"] = class_table(issues, pred)
        report["families"][name] = fam
    if repo.lower() == "microsoft/vscode":
        old = {x.lower() for x in LABEL_FACET_PATTERNS["microsoft/vscode"]["component"]}
        report["v_new_labels"] = v_new_labels(issues, old)
    if repo.lower() == "kubernetes/kubernetes":
        report["cooccurrence_S_A"] = cooccurrence(issues)
    report["other_label_families"] = other_families(freq)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Label coverage evidence for the taxonomy decision")
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--derived-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    derived = a.derived_dir or Path(r"D:\triageiq-data") / "derived" / slug
    out = a.out or Path(f"reports/qp0_taxonomy_coverage_{slug}_2026-10.json")
    rep = build_report(a.repo, derived)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
