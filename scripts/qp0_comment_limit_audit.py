"""Audit of the scraper's last-5-comments limit: what does it miss, and is the miss systematic?

PRE-REGISTERED DESIGN (GG decision D79 item 5, 2026-10-20; do not change after seeing results)
 1. Every sampled issue's COMPLETE comment list is fetched via GraphQL (comments(first:100) with
    pagination), plus its ClosedEvent time(s). "Saved" = the last 5 of that list, which is exactly
    what the scraper kept (lastComments = comments(last:5)). Timeline events (MarkedAsDuplicate,
    ClosedEvent) were scraped completely, so they cannot be lost; the marker audit therefore
    isolates comment-derived pairs (timelineItems left empty, labels from the derived table).
 2. Sample A (markers): CLOSED issues whose SAVED data has any duplicate hint (a label containing
    'duplicate', stateReason DUPLICATE, a marked_as_duplicate_event pair, or 'duplicate'/'dup of'
    in a saved comment); all of them if <= 150, else 150 seeded. Sample B (closing explanations +
    general): CLOSED issues with n_comments > 0, --per-stratum per created year 2023-2026, seeded.
    Samplers see only (issue number, year). Companion count: share of CLOSED issues with
    n_comments <= 5 (these cannot lose anything). Scope = CLOSED issues with n_comments > 5.
 3. (a) Markers: qp0_derive_events.extract_duplicates on the full list vs the last 5; miss rate =
    pairs found only with the full list / pairs found with the full list (pair level); also
    issue level: issues with a target in the full list but none in the last 5. (b) Closing
    explanation = the last comment by an OWNER/MEMBER/COLLABORATOR human (author __typename ==
    'User') created at or before (last ClosedEvent time + 5 min); miss = its id is not among the
    last 5. Denominator = issues having such a comment. Wilson 95% intervals.
 4. Systematic-miss analysis (issue level) by total-comment bucket (<=5, 6-10, 11-20, >20),
    created year, stateReason, author kind (markers: pair method; explanations: association) and
    kind/* label. A PATTERN is flagged when a bucket with n >= 20 has miss rate > 0 and
    (>= 2x the overall issue-level rate or >= 10%). Verdict block: material_markers /
    material_explanations (pair-level / explanation point estimate >= 0.05), pattern_found,
    rescrape_comments_recommended (any of the three). Rates are also given for in-scope issues
    only (> 5 comments) so the headline reads both ways.
Limits: hint detection uses saved data, so a marker lost entirely cannot be hinted (sample B
catches those); the token is read in-process from `gh auth token` and never printed or written.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts, as qp0_derive_*.py
import qp0_derive_events as de  # noqa: E402
import qp0_scrape_github as sg  # noqa: E402

YEARS = (2023, 2024, 2025, 2026)
DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].split("Limits:")[0].strip()
STAFF = {"OWNER", "MEMBER", "COLLABORATOR"}
Q = """query($id:ID!, $c:String){ rateLimit{cost remaining resetAt}
node(id:$id){... on Issue{ comments(first:100, after:$c){totalCount pageInfo{hasNextPage endCursor}
nodes{id databaseId body createdAt authorAssociation author{login __typename}}}
closedEvents: timelineItems(last:10, itemTypes:[CLOSED_EVENT]){nodes{... on ClosedEvent{createdAt}}}
}}}"""


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval (rounded) for k of n; None when n == 0."""
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def has_hint(row: Any, marker_issues: set[int]) -> bool:
    """Duplicate hint in the SAVED data of one derived issue row."""
    if any("duplicate" in x.lower() for x in row.labels_final):
        return True
    if row.state_reason == "DUPLICATE" or row.closed_as_duplicate or row.number in marker_issues:
        return True
    return any(
        "duplicate" in (c["body"] or "").lower() or "dup of" in (c["body"] or "").lower()
        for c in row.closing_comments
    )


def sample_a(numbers: list[int], cap: int, seed: int) -> list[int]:
    """All hinted issues if <= cap, else a seeded sample; sees only issue numbers."""
    pool = sorted(numbers)
    return pool if len(pool) <= cap else sorted(random.Random(seed).sample(pool, cap))


def sample_b(items: list[tuple[int, int]], per_stratum: int, seed: int) -> list[int]:
    """min(per_stratum, available) per created year; sees only (number, year)."""
    rng, out = random.Random(seed), []
    for y in YEARS:
        pool = sorted(n for n, yr in items if yr == y)
        out += rng.sample(pool, min(per_stratum, len(pool)))
    return sorted(out)


def fetch_issue(client: sg.Client, node_id: str) -> tuple[list[dict[str, Any]], list[datetime]]:
    """(all comments in order, ClosedEvent times) for one issue."""
    comments, cursor, closed = [], None, []
    while True:
        node = client.query(Q, {"id": node_id, "c": cursor})["node"]
        comments += [c for c in node["comments"]["nodes"] if c]
        closed = [ts(n["createdAt"]) for n in node["closedEvents"]["nodes"] if n]
        page = node["comments"]["pageInfo"]
        if not page["hasNextPage"]:
            return comments, sorted(closed)
        cursor = page["endCursor"]


def pairs(number: int, labels: list[str], comments: list[dict[str, Any]]) -> dict[int, str]:
    """canonical -> method for comment-derived pairs found in ``comments``."""
    rec = {
        "number": number, "state": "CLOSED", "_schema": 2,
        "labels": {"nodes": [{"name": x} for x in labels]},
        "timelineItems": {"nodes": []},
        "lastComments": {"nodes": comments},
    }  # fmt: skip
    return {p["canonical"]: p["method"] for p in de.extract_duplicates(rec, "o/r")}


def explanation(comments: list[dict[str, Any]], closed: list[datetime]) -> dict[str, Any] | None:
    """Last staff-human comment at or before the last close + 5 min."""
    if not closed:
        return None
    limit = closed[-1] + timedelta(minutes=5)
    ok = [
        c for c in comments
        if c["authorAssociation"] in STAFF
        and (c.get("author") or {}).get("__typename") == "User"
        and ts(c["createdAt"]) <= limit
    ]  # fmt: skip
    return ok[-1] if ok else None


def analyse(meta: dict[str, Any], comments: list[dict[str, Any]], closed: list[datetime]) -> dict:
    """One issue's row: marker and explanation outcomes with their stratifying attributes."""
    last5 = comments[-5:]
    wire = lambda cs: [  # noqa: E731
        {"body": c["body"], "createdAt": c["createdAt"], "author": c.get("author")} for c in cs
    ]
    full = pairs(meta["number"], meta["labels"], wire(comments))
    kept = pairs(meta["number"], meta["labels"], wire(last5))
    only = {k: v for k, v in full.items() if k not in kept}
    ex = explanation(comments, closed)
    kinds = sorted(x for x in meta["labels"] if x.startswith("kind/"))
    total = len(comments)
    return {
        "issue": meta["number"], "year": meta["year"], "state_reason": meta["state_reason"],
        "total_comments": total, "n_comments_derived": meta["n_comments"],
        "bucket": "<=5" if total <= 5 else "6-10" if total <= 10 else "11-20" if total <= 20 else ">20",
        "kind": kinds[0] if kinds else "none", "samples": meta["samples"],
        "pairs_full": sorted(full), "pairs_saved": sorted(kept), "pairs_only_full": sorted(only),
        "marker_method": next(iter(only.values()), None) or next(iter(full.values()), None),
        "explanation_id": ex["id"] if ex else None,
        "explanation_assoc": ex["authorAssociation"] if ex else None,
        "explanation_missed": (ex["id"] not in {c["id"] for c in last5}) if ex else None,
        "url": f"https://github.com/{meta['repo']}/issues/{meta['number']}",
    }  # fmt: skip


def flag_buckets(
    rows: list[dict[str, Any]], dims: dict[str, str], has: str, miss: str
) -> tuple[dict, list[dict]]:
    """Per-dimension breakdown of the issue-level miss rate and the PATTERN flags."""
    base = [r for r in rows if r[has]]
    overall = sum(1 for r in base if r[miss]) / len(base) if base else 0.0
    table, flags = {}, []
    for name, field in dims.items():
        groups = defaultdict(list)
        for r in base:
            groups[str(r[field])].append(r)
        table[name] = {}
        for val, g in sorted(groups.items()):
            k = sum(1 for r in g if r[miss])
            table[name][val] = rate(k, len(g))
            if len(g) >= 20 and k and (k / len(g) >= 2 * overall or k / len(g) >= 0.10):
                flags.append({"dimension": name, "value": val, "n": len(g), "rate": k / len(g)})
    return table, flags


def summarise(rows: list[dict[str, Any]], tag: str) -> dict[str, Any]:
    """Marker and explanation statistics for the rows carrying sample ``tag``."""
    rs = [dict(r) for r in rows if tag in r["samples"]]
    for r in rs:
        r["has_pair"] = bool(r["pairs_full"])
        r["pair_missed"] = bool(r["pairs_only_full"])
        r["has_expl"] = r["explanation_id"] is not None
        r["hint_lost"] = bool(r["pairs_only_full"]) and not r["pairs_saved"]
        r["expl_assoc"] = r["explanation_assoc"]
    dims = {"bucket": "bucket", "year": "year", "state_reason": "state_reason", "kind": "kind"}
    mt, mf = flag_buckets(rs, {**dims, "author_kind": "marker_method"}, "has_pair", "pair_missed")
    et, ef = flag_buckets(
        rs, {**dims, "author_kind": "expl_assoc"}, "has_expl", "explanation_missed"
    )
    nf = sum(len(r["pairs_full"]) for r in rs)
    no = sum(len(r["pairs_only_full"]) for r in rs)
    scoped = [r for r in rs if r["total_comments"] > 5]
    sc = lambda f: sum(1 for r in scoped if f(r))  # noqa: E731
    return {
        "n_issues": len(rs),
        "markers_pair_level": rate(no, nf),
        "markers_issue_level": rate(sum(r["pair_missed"] for r in rs), sum(r["has_pair"] for r in rs)),
        "markers_hint_issues_target_only_in_full": rate(
            sum(r["hint_lost"] for r in rs), len(rs)
        ),
        "markers_pair_level_in_scope_only": rate(
            sum(len(r["pairs_only_full"]) for r in scoped), sum(len(r["pairs_full"]) for r in scoped)
        ),
        "explanations": rate(
            sum(bool(r["explanation_missed"]) for r in rs), sum(r["has_expl"] for r in rs)
        ),
        "explanations_in_scope_only": rate(
            sc(lambda r: r["explanation_missed"]), sc(lambda r: r["has_expl"])
        ),
        "marker_breakdown": mt, "explanation_breakdown": et,
        "pattern_flags": {"markers": mf, "explanations": ef},
    }  # fmt: skip


def run(client: sg.Client, repo: str, derived: Path, per_stratum: int, seed: int) -> dict:
    iss = pd.read_parquet(derived / "issues.parquet")
    dup = pd.read_parquet(derived / "duplicate_pairs.parquet")
    closed = iss[iss["state"] == "CLOSED"]
    marker_issues = set(dup.loc[dup["method"] == "marked_as_duplicate_event", "source"])
    hinted = [int(r.number) for r in closed.itertuples() if has_hint(r, marker_issues)]
    a = sample_a(hinted, 150, seed)
    cand = closed[closed["n_comments"] > 0]
    b = sample_b(list(zip(cand["number"], cand["created_year"], strict=True)), per_stratum, seed)
    tags = defaultdict(list)
    for tag, nums in (("A", a), ("B", b)):
        for n in nums:
            tags[int(n)].append(tag)
    by_num = closed.set_index("number")
    rows = []
    for n in sorted(tags):
        r = by_num.loc[n]
        meta = {
            "number": n, "year": int(r["created_year"]), "state_reason": r["state_reason"],
            "labels": list(r["labels_final"]), "n_comments": int(r["n_comments"]),
            "samples": tags[n], "repo": repo,
        }  # fmt: skip
        comments, closed_at = fetch_issue(client, r["node_id"])
        rows.append(analyse(meta, comments, closed_at))
        print(f"#{n} {tags[n]} total={len(comments)}", file=sys.stderr)
    scope = sorted(int(n) for n in closed.loc[closed["n_comments"] > 5, "number"])
    out = {tag: summarise(rows, tag) for tag in ("A", "B")}
    m, e = out["A"]["markers_pair_level"]["rate"], out["B"]["explanations"]["rate"]
    pattern = any(out[t]["pattern_flags"][k] for t in out for k in ("markers", "explanations"))
    mat_m, mat_e = (m or 0) >= 0.05, (e or 0) >= 0.05
    return {
        "repo": repo, "seed": seed, "design": DESIGN, "generated_with": "scripts/qp0_comment_limit_audit.py",
        "population": {
            "closed_issues": len(closed), "closed_n_comments_le_5": int((closed["n_comments"] <= 5).sum()),
            "share_le_5": float((closed["n_comments"] <= 5).mean()), "in_scope_gt_5": len(scope),
            "hinted_issues": len(hinted), "sample_a": len(a), "sample_b": len(b),
        },
        "sample_A_markers": out["A"], "sample_B_general": out["B"],
        "verdict": {
            "material_markers": mat_m, "material_explanations": mat_e, "pattern_found": pattern,
            "rescrape_comments_recommended": mat_m or mat_e or pattern,
            "issues_in_scope": len(scope), "materiality_threshold": 0.05,
            "markers_rate_used": m, "explanations_rate_used": e,
        },
        "rescrape_scope": scope, "issues": rows,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit the last-5-comments scrape limit")
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--derived-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-stratum", type=int, default=75)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    derived = a.derived_dir or Path(r"D:\triageiq-data") / "derived" / slug
    out = a.out or Path(f"reports/qp0_comment_limit_audit_{slug}_2026-10.json")
    rep = run(
        sg.Client(sg.requests_transport(sg.gh_token())), a.repo, derived, a.per_stratum, a.seed
    )
    scope = rep.pop("rescrape_scope")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    side = derived / f"{out.name}.rescrape_scope.txt"
    side.write_text("\n".join(map(str, scope)) + "\n", encoding="utf-8")
    keep = {k: rep[k] for k in ("population", "verdict")}
    print(json.dumps(keep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
