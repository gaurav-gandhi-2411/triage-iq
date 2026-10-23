"""vscode component-taxonomy evidence for ADR-0070 (label-only: no model, no text).

PRE-REGISTERED DESIGN (2026-10-23; do not change after seeing results).
HARD RULE: every number comes from issues created < 2026-01-01 (train + validation periods of
qp0_make_splits). Rows created >= 2026-01-01 are filtered out at read time (parquet predicate) and
`assert_no_test_rows` raises if any survive; no label statistic is ever computed on them.
 * Exclusion (non-component labels), by semantics: issue type, workflow / resolution state, test
   and plan bookkeeping, platform / OS, upstream dependency, cross-cutting attribute, translation
   and bot / process markers (EXCLUDE below). A label starting with `*` or `~` is a process marker.
 * Primary label = ADR-0067 rule via qp0_ceiling_labels.per_issue over the non-excluded labels:
   labels applied at creation (no event) rank first, otherwise the earliest `labeled` event, ties
   alphabetical. Population = issues with >= 1 non-excluded label at their final state.
 * (a) For a training window [start, 2025-07-01), class family = non-excluded labels carried by
   >= 30 window issues (labels_final, issue level). Reported on VALIDATION (2025-07-01..2025-12-31):
   coverage = share of labelled issues whose primary is in the family, split of the remainder into
   labels never seen in the window vs seen but sparse; window size cost; train-prior top-1/top-3
   accuracy of the family (label-only baseline, abstentions count as misses).
 * (b) Open-set: abstention rate = 1 - coverage, per validation month. Refit evidence: validation
   Q4 coverage with the family from the train window vs from train + validation Q3.
 * (c) Parent areas (mechanical prefix rule): parent(L) = P if L == P or L starts with `P-` for P in
   PARENT_PREFIXES, else L itself; parent classes = parents carried by >= 30 window issues; a
   label unseen in the window maps by the same rule. Coverage and class count on validation.
 * Frozen artefacts: sha256 of the exclusion spec and of the class list of the recommended window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp0_ceiling_labels as cl  # noqa: E402
from qp0_make_splits import TEST_START, VAL_START  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
MIN_ISSUES = 30  # ADR-0067: smallest class with enough train issues to learn and to score
WINDOWS = ("2023-01-01", "2024-01-01", "2024-07-01", "2025-01-01")
Pred = Callable[[str], bool]
# Parent-area prefixes: families of >= 5 window labels that share a product-area prefix (chosen from
# label NAMES seen in train + validation; a first-token rule wrongly merged tree-sitter / code-cli /
# multi-monitor / simple-browser, so the list is explicit). `extension` and `extensions` unify.
PARENT_PREFIXES = ("chat", "terminal", "notebook", "editor", "workbench", "authentication", "debug",
                   "settings", "remote", "search", "keybindings", "extension", "extensions", "file")  # fmt: skip
EXCLUDE: dict[str, list[str]] = {
    "issue_type": ["bug", "feature-request", "enhancement", "question", "documentation", "debt",
                   "polish", "papercut :drop_of_blood:", "engineering", "regression", "api-proposal"],
    "workflow_resolution": ["info-needed", "triage-needed", "verified", "verification-needed",
        "verification-found", "verification-steps-needed", "author-verification-requested",
        "insiders-released", "unreleased", "confirmed", "confirmation-pending", "important",
        "candidate", "under-discussion", "stale", "invalid", "spam", "wont-fix", "duplicate",
        "mitigated", "deferred", "blocked", "on-hold", "keep", "help wanted", "good first issue",
        "repro-steps-needed", "team-low-hanging", "postponed-during-endgame", "question-discussion",
        "api-finalization", "accessibility-sla", "msal-logs-requested", "product-feedback"],
    "test_plan_bookkeeping": ["testplan-item", "on-testplan", "invalid-testplan-item", "plan-item",
        "iteration-plan", "endgame-plan", "on-release-notes", "release-notes",
        "integration-test-failure", "smoke-test-failure", "unit-test-failure", "flaky-test",
        "smoketest-ai-triage", "ai-analyzed", "ai-translated", "vsc-eval-candidate", "plg",
        "/tests", "a11ymas", "issue-bot", "bot-proposal", "exploration"],
    "platform_os": ["windows", "linux", "macos", "windows 11", "windows 7", "android", "ios-ipados",
                    "safari", "firefox", "snap", "deb", "rpm", "winget", "wayland", "macos-catalina",
                    "macos-big-sur", "virtual-machine", "remote-desktop"],
    "upstream_dependency": ["upstream", "upstream-issue-linked", "upstream-issue-fixed",
                            "upstream-issue-pending", "electron", "chromium", "electron-blocker"],
    "cross_cutting_attribute": ["accessibility", "ux", "perf", "perf-startup", "perf-bloat",
                                "perf-profile", "freeze-slow-crash-leak", "security",
                                "error-telemetry", "chat-ext-issue"],
    "language_without_owning_team": ["python", "ruby", "go", "c++", "c#", "c", "lua", "latex",
                                     "nodejs", "npm"],
    "model_vendor": ["gemini", "claude", "claude-agent", "gpt", "xai", "codex-agent"],
}  # fmt: skip
EXCLUDE_RE = re.compile(
    r"^[*~]|^translation-required-|^fixed-in-electron-|^electron-\d+-update$|^help translation$"
)
_EXACT = {x.lower(): c for c, xs in EXCLUDE.items() for x in xs}
SPEC_SHA256 = hashlib.sha256(
    json.dumps([EXCLUDE, EXCLUDE_RE.pattern], sort_keys=True).encode()
).hexdigest()


def is_excluded(label: str) -> bool:
    """True for labels that are not product components (see EXCLUDE)."""
    return label.lower() in _EXACT or bool(EXCLUDE_RE.search(label))


def not_excluded(label: str) -> bool:
    return not is_excluded(label)


def assert_no_test_rows(df: pd.DataFrame, col: str = "created_at") -> None:
    """Hard guard: no row created at or after the test boundary may be analysed."""
    late = int((df[col] >= TEST_START).sum())
    if late:
        raise AssertionError(f"{late} rows created >= {TEST_START.date()} (test period): refused")


def load_train_val(derived: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Issues and label events of the train + validation periods only (predicate pushdown)."""
    cols = ["number", "created_at", "labels_final", "labels_t72h", "text_hash"]
    iss = pd.read_parquet(derived / "issues.parquet", columns=cols,
                          filters=[("created_at", "<", TEST_START)])  # fmt: skip
    assert_no_test_rows(iss)
    ev = pd.read_parquet(derived / "label_events.parquet",
                         filters=[("number", "in", iss["number"].tolist())])  # fmt: skip
    return iss.sort_values("number").reset_index(drop=True), ev


def window_counts(
    window: pd.DataFrame, to_class: Callable[[str], str] = lambda x: x
) -> Counter[str]:
    """Issue-level label counts of non-excluded labels (optionally mapped to a parent)."""
    return Counter(c for ls in window["labels_final"]
                   for c in {to_class(x) for x in ls if not_excluded(x)})  # fmt: skip


def family(counts: Counter[str], min_issues: int = MIN_ISSUES) -> list[str]:
    return sorted(k for k, n in counts.items() if n >= min_issues)


def parent_of(label: str) -> str:
    """Mechanical parent area: a label equal to, or starting with, a PARENT_PREFIXES entry + '-'."""
    for p in PARENT_PREFIXES:
        if label == p or label.startswith(p + "-"):
            return "extensions" if p == "extension" else p
    return label


def cov(k: int, n: int) -> dict[str, Any]:
    return cl.rate(k, n)


def assess(prim: pd.DataFrame, window: pd.DataFrame, classes: list[str],
           to_class: Callable[[str], str] = lambda x: x) -> dict[str, Any]:  # fmt: skip
    """Validation coverage of a class list, with abstention split and train-prior baselines."""
    seen = set(window_counts(window, to_class))
    val = prim[prim["primary"].notna()].copy()
    val["cls"] = val["primary"].map(to_class)
    inf = val["cls"].isin(classes)
    out = val[~inf]
    prior = window_counts(window, to_class)
    top = [c for c, _ in sorted(((c, prior[c]) for c in classes), key=lambda t: (-t[1], t[0]))]
    return {
        "n_classes": len(classes), "classes_sha256": hashlib.sha256("\n".join(classes).encode()).hexdigest(),
        "val_labelled": len(val), "coverage_primary_in_family": cov(int(inf.sum()), len(val)),
        "abstention_rate": 1 - inf.mean(),
        "top_outside_classes": dict(Counter(out["cls"]).most_common(10)),
        "outside_never_seen_in_window": int((~out["cls"].isin(seen)).sum()),
        "outside_seen_but_sparse": int(out["cls"].isin(seen).sum()),
        "naive_top1_acc_prior_majority": float((val["cls"] == top[0]).mean()),
        "naive_top3_acc_prior": float(val["cls"].isin(top[:3]).mean()),
    }  # fmt: skip


def run(issues: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    assert_no_test_rows(issues)
    d = cl.per_issue(issues, events, not_excluded)
    d["split"] = ["train" if t < VAL_START else "validation" for t in d["created_at"]]
    val_iss = issues[issues["created_at"] >= VAL_START]
    val = d[d["split"] == "validation"]
    lab_share = lambda x: float((x["final"].map(len) > 0).mean())  # noqa: E731
    rep: dict[str, Any] = {"n_train_val_issues": len(issues), "n_validation_issues": len(val_iss),
                           "validation_share_with_component_label": lab_share(val),
                           "train_share_with_component_label": lab_share(d[d["split"] == "train"]),
                           "exclusion_spec_sha256": SPEC_SHA256,
                           "exclusion": EXCLUDE | {"regex": EXCLUDE_RE.pattern}}  # fmt: skip
    a: dict[str, Any] = {}
    for w in WINDOWS:
        win = issues[
            (issues["created_at"] >= pd.Timestamp(w, tz="UTC")) & (issues["created_at"] < VAL_START)
        ]
        classes = family(window_counts(win))
        wl = d[(d["number"].isin(win["number"]))]
        r = assess(val, win, classes)
        r |= {"window": f"{w}..2025-06-30", "window_issues": len(win), "window_labelled": int((wl["final"].map(len) > 0).sum()),
              "window_primaries_in_family": int(wl["primary"].isin(classes).sum()), "classes": classes}  # fmt: skip
        a[w] = r
    rep["a_windows"] = a
    # (b) monthly abstention for the earliest window and the most recent window (train-only family)
    mon = val[val["primary"].notna()].assign(m=lambda x: x["created_at"].dt.strftime("%Y-%m"))
    for w in (WINDOWS[0], WINDOWS[-1]):
        cls = set(a[w]["classes"])
        rep.setdefault("b_monthly_abstention", {})[w] = {
            m: {"n": len(g), "abstention": float((~g["primary"].isin(cls)).mean())} for m, g in mon.groupby("m")}  # fmt: skip
    # refit: Q4 coverage, family from window vs window + validation Q3 (a periodic retrain would do this)
    q3_end, win = (
        pd.Timestamp("2025-10-01", tz="UTC"),
        issues[issues["created_at"] >= pd.Timestamp(WINDOWS[0], tz="UTC")],
    )
    q4 = val[(val["created_at"] >= q3_end) & val["primary"].notna()]
    base = family(window_counts(win[win["created_at"] < VAL_START]))
    refit = family(window_counts(win[win["created_at"] < q3_end]))
    rep["b_refit_q4"] = {"n_q4_labelled": len(q4),
                         "coverage_train_only_family": float(q4["primary"].isin(base).mean()),
                         "coverage_refit_with_q3": float(q4["primary"].isin(refit).mean()),
                         "n_classes_train_only": len(base), "n_classes_refit": len(refit)}  # fmt: skip
    pb, pr = (
        family(window_counts(win[win["created_at"] < t], parent_of)) for t in (VAL_START, q3_end)
    )
    rep["b_refit_q4"] |= {"parent_coverage_train_only_family": float(q4["primary"].map(parent_of).isin(pb).mean()),
                          "parent_coverage_refit_with_q3": float(q4["primary"].map(parent_of).isin(pr).mean())}  # fmt: skip
    # (c) parent areas, per window
    c: dict[str, Any] = {}
    for w in WINDOWS:
        win = issues[
            (issues["created_at"] >= pd.Timestamp(w, tz="UTC")) & (issues["created_at"] < VAL_START)
        ]
        par = parent_of
        pc = family(window_counts(win, par))
        r = assess(val, win, pc, par)
        r["groups_with_children"] = sorted(
            p for p in pc if any(par(x) == p and x != p for x in window_counts(win))
        )
        r["parents"] = pc
        c[w] = r
    rep["c_parent_areas"] = c
    rep["b_gap_decay"] = gap_decay(issues, d)
    rep["a_min_count_sensitivity"] = min_count_sensitivity(issues, val)
    return rep


def gap_decay(issues: pd.DataFrame, d: pd.DataFrame) -> dict[str, Any]:
    """Coverage of a family frozen at a cutoff on later 2025 quarters (window starts 2023-01-01).

    Shows how fast a frozen family goes stale; the cutoff 2025-07-01 row is the real train family.
    """
    t0, out = pd.Timestamp(WINDOWS[0], tz="UTC"), {}
    for cut in ("2025-01-01", "2025-04-01", "2025-07-01"):
        c = pd.Timestamp(cut, tz="UTC")
        win = issues[(issues["created_at"] >= t0) & (issues["created_at"] < c)]
        par = parent_of
        fams = {"exact": (family(window_counts(win)), lambda x: x),
                "parent": (family(window_counts(win, par)), par)}  # fmt: skip
        for kind, (cls, fn) in fams.items():
            row = {}
            for q in ("2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"):
                lo = pd.Timestamp(q, tz="UTC")
                if lo < c:
                    continue
                sub = d[
                    (d["created_at"] >= lo)
                    & (d["created_at"] < lo + pd.DateOffset(months=3))
                    & d["primary"].notna()
                ]
                row[f"eval_from_{q}"] = {"n": len(sub), "coverage": float(sub["primary"].map(fn).isin(cls).mean())}  # fmt: skip
            out[f"{kind}_family_frozen_{cut}"] = {"n_classes": len(cls), "quarters": row}
    return out


def min_count_sensitivity(issues: pd.DataFrame, val: pd.DataFrame) -> dict[str, Any]:
    """Validation coverage of the 2023-01-01 window family at other minimum counts (ADR keeps 30)."""
    win = issues[
        (issues["created_at"] >= pd.Timestamp(WINDOWS[0], tz="UTC"))
        & (issues["created_at"] < VAL_START)
    ]
    par, v = parent_of, val[val["primary"].notna()]
    return {str(m): {"exact": {"n_classes": len(family(window_counts(win), m)),
                              "coverage": float(v["primary"].isin(family(window_counts(win), m)).mean())},
                     "parent": {"n_classes": len(family(window_counts(win, par), m)),
                                "coverage": float(v["primary"].map(par).isin(family(window_counts(win, par), m)).mean())}}
            for m in (10, 20, 30, 50)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vscode taxonomy evidence (train + validation only)")
    ap.add_argument("--derived-dir", type=Path, default=Path(r"D:\triageiq-data\derived_v2\microsoft__vscode"))  # fmt: skip
    ap.add_argument("--out", type=Path, default=Path("reports/qp2_vscode_taxonomy_2026-10.json"))
    a = ap.parse_args(argv)
    issues, events = load_train_val(a.derived_dir)
    rep = run(issues, events) | {"design": DESIGN, "generated_with": "scripts/qp2_vscode_taxonomy.py",
                                 "repo": "microsoft/vscode", "max_created_at": str(issues["created_at"].max())}  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({"n": len(issues), "max_created_at": rep["max_created_at"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
