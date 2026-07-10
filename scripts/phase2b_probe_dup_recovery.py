"""Phase 2b PROBE (not the full scrape): per-era dup-target recovery rate for microsoft/vscode.

ADR-0026's vscode GO case rests on a 49% comment-channel recovery rate measured on 238
dup-labeled issues that skew 2015-16. Dup-linking conventions changed over a decade, so this
probe samples ~150 `*duplicate`-labeled issues spread across era buckets and measures, per era:

1. STRICT comment recovery — the same regex the feasibility analysis used (comparable to 49%).
2. LOOSE comment recovery — any issue ref (#N or issue URL) inside a comment that mentions
   "duplicate" (upper bound; messier to parse but still automatable).
3. TIMELINE recovery — `marked_as_duplicate` events, and whether the event payload actually
   exposes the canonical target (the structured channel ADR-0026 hoped for).

Budget: ~150 issues x 2 API calls + 12 search calls ~= 320 requests. Analysis only —
raw responses are not written into the corpus.

Output: reports/dup_recovery_probe.json
Reproduce: python scripts/phase2b_probe_dup_recovery.py  (needs GITHUB_TOKEN in .env)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

API = "https://api.github.com"
REPO = "microsoft/vscode"
OUTPUT_PATH = Path("reports/dup_recovery_probe.json")

# Era buckets: 2016 doubles as a calibration point against the in-corpus 49% measurement.
ERA_BUCKETS = ["2016", "2018", "2020", "2022", "2024", "2026"]
PER_BUCKET = 26  # 13 sampled from each end of the year to spread within-year

# STRICT: identical to scripts/phase2a_corpus_feasibility.py's dup-comment pattern, so the
# per-era rates are directly comparable with the 49% in-corpus measurement.
STRICT_PAT = re.compile(r"[Dd]up(?:licate|e)?\s*(?:of|to|:)?\s*#?(\d{2,})|/duplicate\s+#?(\d+)")
# LOOSE: any issue reference inside a comment that talks about duplicates.
REF_PAT = re.compile(r"#(\d{2,})|github\.com/microsoft/vscode/issues/(\d+)")


def _session() -> requests.Session:
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.critical("GITHUB_TOKEN not set")
        sys.exit(1)
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return s


def _get(s: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    for attempt in range(4):
        r = s.get(url, params=params, timeout=30)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            wait = 30 * (attempt + 1)
            log.warning("rate limited, sleeping %ss", wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"rate-limited out on {url}")


def sample_bucket(s: requests.Session, year: str) -> list[dict]:
    """~26 dup-labeled issues per era, half from each end of the year (deterministic)."""
    end = "2026-07-10" if year == "2026" else f"{year}-12-31"
    q = f"repo:{REPO} type:issue label:*duplicate created:{year}-01-01..{end}"
    issues: dict[int, dict] = {}
    for order in ("asc", "desc"):
        r = _get(
            s,
            f"{API}/search/issues",
            {"q": q, "sort": "created", "order": order, "per_page": PER_BUCKET // 2},
        )
        for it in r.json().get("items", []):
            issues[it["number"]] = it
        time.sleep(2)  # search API is 30 req/min
    log.info("[%s] sampled %d dup-labeled issues", year, len(issues))
    return list(issues.values())


def probe_issue(s: requests.Session, number: int) -> dict:
    comments = _get(
        s, f"{API}/repos/{REPO}/issues/{number}/comments", {"per_page": 100}
    ).json()
    timeline = _get(
        s, f"{API}/repos/{REPO}/issues/{number}/timeline", {"per_page": 100}
    ).json()
    time.sleep(0.2)

    strict_target = None
    loose_target = None
    for c in comments:
        body = str(c.get("body", ""))
        m = STRICT_PAT.search(body)
        if m and strict_target is None:
            strict_target = int(next(g for g in m.groups() if g))
        if loose_target is None and "duplicate" in body.lower():
            m2 = REF_PAT.search(body)
            if m2:
                loose_target = int(next(g for g in m2.groups() if g))

    marked_dup_events = [e for e in timeline if e.get("event") == "marked_as_duplicate"]
    # does the structured event expose the canonical issue? (empirical question)
    timeline_target = None
    for e in marked_dup_events:
        for key in ("canonical", "duplicate_of", "issue", "source"):
            v = e.get(key)
            if isinstance(v, dict) and v.get("number"):
                timeline_target = int(v["number"])
                break
        if timeline_target:
            break
    cross_refs = sum(1 for e in timeline if e.get("event") == "cross-referenced")

    return {
        "number": number,
        "n_comments": len(comments),
        "strict_comment_target": strict_target,
        "loose_comment_target": loose_target,
        "n_marked_as_duplicate_events": len(marked_dup_events),
        "timeline_event_keys": sorted(marked_dup_events[0].keys()) if marked_dup_events else [],
        "timeline_target": timeline_target,
        "n_cross_referenced_events": cross_refs,
    }


def main() -> None:
    s = _session()
    buckets: dict[str, dict] = {}
    for year in ERA_BUCKETS:
        sampled = sample_bucket(s, year)
        rows = []
        for it in sampled:
            try:
                rows.append(probe_issue(s, it["number"]))
            except requests.HTTPError as exc:
                log.warning("#%s failed: %s", it["number"], exc)
        n = len(rows)
        strict = sum(1 for r in rows if r["strict_comment_target"])
        loose = sum(1 for r in rows if r["loose_comment_target"])
        marked = sum(1 for r in rows if r["n_marked_as_duplicate_events"] > 0)
        tl_target = sum(1 for r in rows if r["timeline_target"])
        either = sum(1 for r in rows if r["strict_comment_target"] or r["timeline_target"])
        buckets[year] = {
            "n": n,
            "strict_comment_rate": round(strict / n, 3) if n else None,
            "loose_comment_rate": round(loose / n, 3) if n else None,
            "marked_as_duplicate_event_rate": round(marked / n, 3) if n else None,
            "timeline_target_recovery_rate": round(tl_target / n, 3) if n else None,
            "strict_or_timeline_rate": round(either / n, 3) if n else None,
            "issues": rows,
        }
        log.info(
            "[%s] n=%d strict=%.0f%% loose=%.0f%% marked_dup_evt=%.0f%% tl_target=%.0f%%",
            year, n, 100 * strict / n, 100 * loose / n, 100 * marked / n, 100 * tl_target / n,
        )

    report = {
        "generated_by": "scripts/phase2b_probe_dup_recovery.py",
        "probed_at": "2026-07-11",
        "repo": REPO,
        "sampling": "per era bucket: 13 earliest + 13 latest dup-labeled issues by created_at "
                    "(deterministic; within-year edges, not uniform — fine for an era trend)",
        "gate": "ADR-0026 vscode GO holds if strict/timeline recovery >= ~40% in modern eras",
        "buckets": {
            y: {k: v for k, v in b.items() if k != "issues"} for y, b in buckets.items()
        },
        "per_issue": {y: b["issues"] for y, b in buckets.items()},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", OUTPUT_PATH)
    print(json.dumps(report["buckets"], indent=2))


if __name__ == "__main__":
    main()
