"""Classification ceiling analysis (spec 4.4, part 1): what do the labels themselves allow?

PRE-REGISTERED DESIGN (2026-10-20; do not change after seeing results). No model is trained or
scored. Taxonomy per ADR-0067: the label family (default `sig/*`); primary label = the family
label with the earliest `labeled` event among the labels still present at the end (ties:
alphabetical; literal earliest event, even if later removed and re-added); modelled classes =
family labels carried by >= 30 TRAIN-split issues, the rest `other`. Population: issues with
>= 1 family label at the end ("labelled"). Everything per split (train / validation / test) and
overall, Wilson 95% intervals on every proportion.
 (a) multi-label rate (>= 2 labels) and the label-set size distribution.
 (b) timing / churn: P(set at 72 h == final set), P(primary present at 72 h), and for issues whose
     primary was NOT present at 72 h the median hours until it first appears.
 (c) first-vs-final: among issues with >= 1 family label at 72 h, P(earliest-applied label of
     the 72 h set == final primary) and P(final primary in the 72 h set).
 (d) INCONSISTENCY on near-identical content, the direct proxy for irreducible top-1 error, on
     pairs where both issues are labelled: (i) duplicate pairs, (ii) exact text_hash groups
     (adjacent issue numbers within a group, to avoid quadratic cluster blow-up), (iii) top-1
     nearest neighbours at TF-IDF cosine >= 0.9 among labelled issues (title + first 1000 chars
     of body; neighbours with an equal text_hash are masked so (ii) and (iii) are disjoint).
     Per group: n pairs, P(primary labels differ), P(label sets disjoint). A pair belongs to the
     split of its LATER issue. top1_ceiling_estimate = 1 - P(differ) and any_valid_label_ceiling
     = 1 - P(disjoint) from the largest-n group, with interval and n: PROXIES, not a bound;
     small n limits the claim.
 (e) train class table (counts, shares, `other` mass) and the baselines for top-1 accuracy on
     validation and test from TRAIN priors only: majority class, and the expected accuracy of
     sampling a class from the train prior. Also any-valid accuracy of the majority class.
AMENDMENT (2026-10-20, after the original k8s run): a label with no event at all, or whose first
event is an unlabel, was applied at creation (issue templates / forms add labels without a
LabeledEvent), so its applied time is created_at and it ranks first for the primary label (ties
alphabetical). The first version treated such a label as applied last. Reports computed before the
amendment are marked by their git commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # similarity measurement only

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qp0_fixlink_audit import wilson  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
SPLITS = ("train", "validation", "test")
Family = Callable[[str], bool]
H72 = pd.Timedelta(hours=72)


def rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "wilson95": ci and list(ci)}


def per_issue(issues: pd.DataFrame, events: pd.DataFrame, family: Family) -> pd.DataFrame:
    """One row per issue with its family label sets and primary label (see module docstring).

    Applied time of a label = its earliest `labeled` event, EXCEPT that a label with no event at all,
    or whose first event is an unlabel, was present from creation (issue templates apply labels
    without a LabeledEvent): its applied time is created_at.
    """
    ev = events[events["label"].map(family)].sort_values(["number", "ts"], kind="stable")
    first = ev[ev["action"] == "labeled"].groupby(["number", "label"])["ts"].min().to_dict()
    first_action = ev.groupby(["number", "label"])["action"].first().to_dict()
    created = dict(zip(issues["number"], issues["created_at"], strict=True))

    def applied(n: int, x: str) -> Any:
        if first_action.get((n, x)) == "unlabeled" or (n, x) not in first:
            return created[n]
        return first[(n, x)]

    rows = []
    for r in issues.itertuples():
        fin = frozenset(x for x in r.labels_final if family(x))
        t72 = frozenset(x for x in r.labels_t72h if family(x))
        prim = min(fin, key=lambda x, n=r.number: (applied(n, x), x)) if fin else None
        cand = [
            (applied(r.number, x), x) for x in t72 if applied(r.number, x) <= r.created_at + H72
        ]
        early = min(cand)[1] if cand else None
        hrs = (applied(r.number, prim) - r.created_at).total_seconds() / 3600 if prim else None
        rows.append(
            {"number": r.number, "final": fin, "t72": t72, "primary": prim, "earliest72": early,
             "primary_hours": hrs, "text_hash": r.text_hash, "created_at": r.created_at}
        )  # fmt: skip
    return pd.DataFrame(rows)


def label_metrics(d: pd.DataFrame) -> dict[str, Any]:
    """(a)-(c) for a set of issues (rows of per_issue with a split already selected)."""
    lab = d[d["final"].map(bool).astype(bool)]
    n = len(lab)
    size = lab["final"].map(len)
    in72 = pd.Series(
        [p in t for p, t in zip(lab["primary"], lab["t72"], strict=True)],
        index=lab.index,
        dtype=bool,
    )
    late = lab.loc[~in72, "primary_hours"]
    l72 = lab[lab["t72"].map(bool).astype(bool)]
    return {
        "n_issues": len(d), "n_labelled": n,
        "share_labelled": rate(n, len(d)),
        "multi_label": rate(int((size >= 2).sum()), n),
        "set_size_dist": {str(k): int(v) for k, v in size.clip(upper=4).value_counts().sort_index().items()},
        "set72_equals_final": rate(int((lab["t72"] == lab["final"]).sum()), n),
        "primary_present_at_72h": rate(int(in72.sum()), n),
        "median_hours_until_primary_if_absent_at_72h": float(late.median()) if len(late) else None,
        "n_primary_absent_at_72h": int(len(late)),
        "earliest72_equals_final_primary": rate(int((l72["earliest72"] == l72["primary"]).sum()), len(l72)),
        "final_primary_in_set72": rate(sum(p in t for p, t in zip(l72["primary"], l72["t72"], strict=True)), len(l72)),
    }  # fmt: skip


def nn_pairs(texts: pd.Series, hashes: pd.Series, thr: float = 0.9) -> list[tuple[int, int]]:
    """Positions (i, j), i < j, of mutual-or-not top-1 neighbours with cosine >= thr."""
    x = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, dtype=np.float32)
    m = x.fit_transform(texts)
    h = hashes.to_numpy()
    found: set[tuple[int, int]] = set()
    for s in range(0, m.shape[0], 500):
        sim = (m[s : s + 500] @ m.T).toarray()
        for r in range(sim.shape[0]):
            sim[r, h == h[s + r]] = 0.0  # masks itself and identical-text issues
        j = sim.argmax(axis=1)
        for r, c in enumerate(j):
            if sim[r, c] >= thr:
                found.add((min(s + r, int(c)), max(s + r, int(c))))
    return sorted(found)


def pair_stats(pairs: list[tuple[int, int]], idx: pd.DataFrame, split: pd.Series) -> dict[str, Any]:
    """P(primary differs) and P(sets disjoint) overall and per split of the later issue."""
    out: dict[str, Any] = {}
    rows = []
    for a, b in pairs:
        ra, rb = idx.loc[a], idx.loc[b]
        later = a if ra["created_at"] >= rb["created_at"] else b
        rows.append(
            (split.get(later), ra["primary"] != rb["primary"], not (ra["final"] & rb["final"]))
        )
    for name in ("all", *SPLITS):
        sel = [r for r in rows if name == "all" or r[0] == name]
        out[name] = {
            "n_pairs": len(sel),
            "primary_differs": rate(sum(r[1] for r in sel), len(sel)),
            "sets_disjoint": rate(sum(r[2] for r in sel), len(sel)),
        }
    return out


def baselines(d: pd.DataFrame, split: pd.Series) -> dict[str, Any]:
    """Class table and baselines. Priors/classes are fitted on TRAIN rows only."""
    lab = d[d["final"].map(bool).astype(bool)].assign(split=lambda x: x["number"].map(split))
    train = lab[lab["split"] == "train"]
    carried = Counter(lbl for s in train["final"] for lbl in s)
    classes = sorted(k for k, v in carried.items() if v >= 30)
    mp = lambda p: p if p in classes else "other"  # noqa: E731
    prior = Counter(train["primary"].map(mp))
    major = sorted(prior, key=lambda c: (-prior[c], c))[0]
    total = sum(prior.values())
    out: dict[str, Any] = {
        "modelled_classes": classes, "majority_class": major,
        "train_class_table": {c: {"n": prior[c], "share": prior[c] / total} for c in sorted(prior)},
        "other_share": prior.get("other", 0) / total, "n_train_labelled": total,
    }  # fmt: skip
    for s in ("validation", "test"):
        ev = lab[lab["split"] == s]
        got = Counter(ev["primary"].map(mp))
        n = len(ev)
        valid = sum(1 for f in ev["final"] if major in {mp(x) for x in f})
        out[s] = {
            "n": n,
            "majority_top1": rate(got.get(major, 0), n),
            "majority_any_valid": rate(valid, n),
            "prior_sampling_expected_top1": sum(prior[c] / total * got.get(c, 0) / n for c in prior) if n else None,
        }  # fmt: skip
    return out


def coverage_bound(d: pd.DataFrame, classes: list[str], split: pd.Series,
                   eligible: pd.Series | None = None) -> dict[str, Any]:  # fmt: skip
    """Share of labelled issues whose truth is inside the modelled class list, per split.

    Under a STRICT headline (abstention = miss, ADR-0070) a top-1 hit needs the primary in the
    list, so coverage is the upper bound on top-1 (and top-3); any-valid needs >= 1 label in it.
    `eligible` (classifier_eligible flag) adds the same figures for the leak-free evaluation set.
    """
    lab = d[d["final"].map(bool).astype(bool)].assign(split=lambda x: x["number"].map(split))
    cls = set(classes)
    lab = lab.assign(
        prim_in=lab["primary"].isin(cls), any_in=lab["final"].map(lambda f: bool(f & cls))
    )
    out: dict[str, Any] = {"n_classes": len(classes)}
    subsets = {"labelled": lab}
    if eligible is not None:
        subsets["labelled_and_classifier_eligible"] = lab[lab["number"].map(eligible).astype(bool)]
    for name, sub in subsets.items():
        out[name] = {}
        for sp in SPLITS:
            g = sub[sub["split"] == sp]
            out[name][sp] = {
                "n": len(g),
                "primary_in_family_top1_bound": rate(int(g["prim_in"].sum()), len(g)),
                "any_label_in_family_any_valid_bound": rate(int(g["any_in"].sum()), len(g)),
            }
    return out


def run(issues: pd.DataFrame, events: pd.DataFrame, dup: pd.DataFrame, splits: pd.Series,
        family: Family, to_class: Callable[[str], str] | None = None,
        eligible: pd.Series | None = None) -> dict[str, Any]:  # fmt: skip
    """`to_class` (vscode ADR-0070 parent areas): label sets, primary and 72 h sets are mapped to
    classes AFTER the primary is chosen on raw labels, so every ceiling is on the parent classes."""
    issues = issues[issues["number"].isin(splits.index)]  # splits may start after the data (none for k8s)
    d = per_issue(issues.sort_values("number").reset_index(drop=True), events, family)
    if to_class is not None:
        for c in ("final", "t72"):
            d[c] = d[c].map(lambda s: frozenset(to_class(x) for x in s))
        d["primary"] = d["primary"].map(lambda x: None if x is None else to_class(x))
        d["earliest72"] = d["earliest72"].map(lambda x: None if x is None else to_class(x))
    d["split"] = d["number"].map(splits)
    rep: dict[str, Any] = {"design": DESIGN, "generated_with": "scripts/qp0_ceiling_labels.py"}
    rep["labels"] = {
        n: label_metrics(d if n == "all" else d[d["split"] == n]) for n in ("all", *SPLITS)
    }
    lab = d[d["final"].map(bool).astype(bool)].reset_index(drop=True)
    pos = {n: i for i, n in enumerate(lab["number"])}
    groups: dict[str, list[tuple[int, int]]] = {}
    groups["duplicate_pairs"] = sorted(
        {
            (min(pos[s], pos[c]), max(pos[s], pos[c]))
            for s, c in zip(dup["source"], dup["canonical"], strict=True)
            if s in pos and c in pos and s != c
        }  # fmt: skip
    )
    chains = []
    for _, g in lab.groupby("text_hash"):
        ix = sorted(pos[n] for n in g["number"])
        chains += list(zip(ix, ix[1:], strict=False))
    groups["exact_text_hash_adjacent"] = chains
    src = issues.set_index("number").loc[lab["number"]]
    text = (src["title"].fillna("") + " " + src["body"].fillna("").str.slice(0, 1000)).reset_index(
        drop=True
    )
    groups["tfidf_top1_nn_cos_ge_0.9"] = nn_pairs(text, lab["text_hash"].reset_index(drop=True))
    sp = splits
    stats = {
        k: pair_stats(v, lab, sp.reindex(lab["number"]).set_axis(lab.index))
        for k, v in groups.items()
    }
    rep["inconsistency"] = stats
    best = max(stats, key=lambda k: stats[k]["all"]["n_pairs"])
    a = stats[best]["all"]
    inv = lambda r: None if r["wilson95"] is None else [1 - r["wilson95"][1], 1 - r["wilson95"][0]]  # noqa: E731
    rep["ceiling_proxies"] = {
        "note": "PROXIES from label inconsistency on near-identical content, not a bound; small n limits the claim",
        "group_used": best, "n_pairs": a["n_pairs"],
        "top1_ceiling_estimate": 1 - a["primary_differs"]["rate"] if a["n_pairs"] else None,
        "top1_ceiling_interval95": inv(a["primary_differs"]),
        "any_valid_label_ceiling": 1 - a["sets_disjoint"]["rate"] if a["n_pairs"] else None,
        "any_valid_label_ceiling_interval95": inv(a["sets_disjoint"]),
    }  # fmt: skip
    rep["baselines"] = baselines(d, splits)
    if to_class is not None:
        rep["coverage_bounded_ceiling"] = {
            "note": "strict ADR-0070 headline: abstention (primary outside the train-only class "
            "list) is a miss, so coverage upper-bounds top-1/top-3; the proxy ceilings above are "
            "conditional on being in the family, so product = coverage x proxy is an estimate",
            **coverage_bound(d, rep["baselines"]["modelled_classes"], splits, eligible),
        }
    return rep


def load_splits(splits_dir: Path) -> pd.Series:
    parts = [
        pd.read_parquet(splits_dir / f"{s}.parquet", columns=["number", "split"]) for s in SPLITS
    ]
    df = pd.concat(parts)
    return df.set_index("number")["split"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Label-side ceiling analysis")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--derived-dir", type=Path, default=None)
    ap.add_argument("--splits-dir", type=Path, default=None)
    ap.add_argument("--label-prefix", default="sig/")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--taxonomy", choices=("prefix", "vscode"), default="prefix",
                    help="vscode: ADR-0070 non-excluded labels mapped to parent areas")  # fmt: skip
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    derived = a.derived_dir or Path(r"D:\triageiq-data") / "derived" / slug
    splits_dir = a.splits_dir or Path(r"D:\triageiq-data") / "splits" / slug
    out = a.out or Path(f"reports/qp0_ceiling_labels_{slug}_2026-10.json")
    issues = pd.read_parquet(derived / "issues.parquet")
    family: Family = lambda x: x.startswith(a.label_prefix)  # noqa: E731
    to_class, eligible = None, None
    if a.taxonomy == "vscode":
        import qp2_vscode_taxonomy as tx  # noqa: PLC0415  (only needed for this taxonomy)

        family, to_class = tx.not_excluded, tx.parent_of
        eligible = pd.concat(
            pd.read_parquet(splits_dir / f"{s}.parquet", columns=["number", "classifier_eligible"])
            for s in SPLITS
        ).set_index("number")["classifier_eligible"]
    rep = run(issues, pd.read_parquet(derived / "label_events.parquet"),
              pd.read_parquet(derived / "duplicate_pairs.parquet"), load_splits(splits_dir),
              family, to_class, eligible)  # fmt: skip
    rep["repo"], rep["label_prefix"] = a.repo, a.label_prefix
    if a.taxonomy == "vscode":
        rep["taxonomy"] = "vscode ADR-0070: non-excluded labels -> parent_of; label_prefix unused"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("ceiling_proxies",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
