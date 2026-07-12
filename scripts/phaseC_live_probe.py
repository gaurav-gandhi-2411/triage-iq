"""Phase C: small LIVE-API probe for channels that require data not present locally.

Local corpus gap: neither repo has GitHub timeline data (native "linked issue" connected/
disconnected events, cross-referenced events) scraped; k8s additionally has ZERO comments_data
at all (neither the original #1-15,002 scrape nor the #15,003-30,000 forward-scrape fetched
comments -- see scripts/01_scrape_issues.py, scripts/phase2b_scrape_k8s_forward.py). This probe
follows the same budget discipline as scripts/phase2b_probe_dup_recovery.py: a small, fixed-seed,
deterministic-stride sample (not a scrape) to measure whether a future scrape investment is
justified, before committing to one.

For each repo, a deterministic stride sample of issue numbers spanning the LIVE index's number
space, fetching (one call each) the issue timeline. k8s additionally gets a comments probe.

Channels measured:
  C. Native "linked issues" -- GitHub's manual link feature (timeline events
     "connected"/"disconnected"), distinct from closing-keyword references.
  D. Timeline cross-referenced events -- any mention of this issue elsewhere; measured overall
     and split issue-sourced vs PR-sourced (only issue-sourced counts toward product-task).
  B (k8s only). Comments "related to / see also / similar to #N" -- feasibility rate for a
     future k8s comment-scrape investment.

Budget: ~25 issues/repo x 1 timeline call, +25 x 1 comments call for k8s = ~75 API calls.
Output: reports/phaseC_live_probe.json
Reproduce: python scripts/phaseC_live_probe.py  (needs GITHUB_TOKEN in .env)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

API = "https://api.github.com"
OUTPUT_PATH = Path("reports/phaseC_live_probe.json")
PROCESSED_DIR = Path("data/processed")
SAMPLE_N = 25
SEED = 42

COMMENT_RELATED_PATTERN = re.compile(
    r"[Rr]elated(?: to)? #(\d+)|[Ss]ee(?: also)? #(\d+)|[Ss]imilar to #(\d+)"
)

REPOS = {
    "kubernetes_kubernetes": "kubernetes/kubernetes",
    "microsoft_vscode": "microsoft/vscode",
}


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


def _get(s: requests.Session, url: str, params: dict | None = None) -> list | dict:
    for attempt in range(4):
        r = s.get(url, params=params, timeout=30)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            wait = 30 * (attempt + 1)
            log.warning("rate limited, sleeping %ss", wait)
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"retries exhausted on {url}")


def stride_sample(numbers: np.ndarray, n: int) -> list[int]:
    """Deterministic, evenly-spaced sample across the sorted number range (not random)."""
    numbers = np.sort(numbers)
    if len(numbers) <= n:
        return numbers.tolist()
    idx = np.linspace(0, len(numbers) - 1, n).round().astype(int)
    return sorted(set(numbers[idx].tolist()))


def probe_timeline(s: requests.Session, slug: str, number: int) -> dict:
    events = _get(s, f"{API}/repos/{slug}/issues/{number}/timeline", {"per_page": 100})
    if not isinstance(events, list):
        events = []
    connected = [e for e in events if e.get("event") in ("connected", "disconnected")]
    cross_refs = [e for e in events if e.get("event") == "cross-referenced"]

    cross_ref_detail = []
    for e in cross_refs:
        src = (e.get("source") or {}).get("issue") or {}
        cross_ref_detail.append(
            {
                "source_number": src.get("number"),
                "source_title": src.get("title"),
                "source_is_pr": "pull_request" in src,
                "source_body_excerpt": str(src.get("body") or "")[:200],
            }
        )
    return {
        "number": number,
        "n_timeline_events": len(events),
        "n_connected_or_disconnected": len(connected),
        "connected_detail": connected,
        "n_cross_referenced": len(cross_refs),
        "cross_ref_issue_sourced": sum(1 for c in cross_ref_detail if not c["source_is_pr"]),
        "cross_ref_pr_sourced": sum(1 for c in cross_ref_detail if c["source_is_pr"]),
        "cross_ref_detail": cross_ref_detail,
    }


def probe_comments_related(s: requests.Session, slug: str, number: int) -> dict:
    comments = _get(s, f"{API}/repos/{slug}/issues/{number}/comments", {"per_page": 100})
    if not isinstance(comments, list):
        comments = []
    hits = []
    for c in comments:
        body = str(c.get("body", ""))
        m = COMMENT_RELATED_PATTERN.search(body)
        if m:
            hits.append({"target": int(next(g for g in m.groups() if g)), "excerpt": body[:200]})
    return {"number": number, "n_comments": len(comments), "related_hits": hits}


def main() -> None:
    s = _session()
    report: dict = {"generated_by": "scripts/phaseC_live_probe.py", "sample_n": SAMPLE_N, "repos": {}}

    for repo, slug in REPOS.items():
        df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
        numbers = df["number"].astype(int).to_numpy()
        sample = stride_sample(numbers, SAMPLE_N)
        log.info("[%s] timeline probe on %d issues: %s", repo, len(sample), sample[:5])

        timeline_rows = []
        for n in sample:
            try:
                timeline_rows.append(probe_timeline(s, slug, n))
            except requests.HTTPError as exc:
                log.warning("#%s timeline failed: %s", n, exc)
            time.sleep(0.3)

        n = len(timeline_rows)
        native_link_rate = sum(1 for r in timeline_rows if r["n_connected_or_disconnected"] > 0) / n if n else 0
        any_cross_ref_rate = sum(1 for r in timeline_rows if r["n_cross_referenced"] > 0) / n if n else 0
        issue_sourced_cross_ref_rate = (
            sum(1 for r in timeline_rows if r["cross_ref_issue_sourced"] > 0) / n if n else 0
        )
        total_cross_refs = sum(r["n_cross_referenced"] for r in timeline_rows)
        total_issue_sourced = sum(r["cross_ref_issue_sourced"] for r in timeline_rows)

        entry = {
            "n_sampled": n,
            "channel_C_native_linked_issues": {
                "issues_with_connected_event": sum(1 for r in timeline_rows if r["n_connected_or_disconnected"] > 0),
                "rate": round(native_link_rate, 3),
            },
            "channel_D_cross_referenced": {
                "issues_with_any_cross_ref": sum(1 for r in timeline_rows if r["n_cross_referenced"] > 0),
                "rate_any": round(any_cross_ref_rate, 3),
                "issues_with_issue_sourced_cross_ref": sum(1 for r in timeline_rows if r["cross_ref_issue_sourced"] > 0),
                "rate_issue_sourced": round(issue_sourced_cross_ref_rate, 3),
                "total_cross_ref_events": total_cross_refs,
                "total_issue_sourced_events": total_issue_sourced,
                "pct_cross_refs_that_are_issue_sourced": round(100 * total_issue_sourced / total_cross_refs, 1) if total_cross_refs else None,
            },
            "timeline_rows": timeline_rows,
        }

        if repo == "kubernetes_kubernetes":
            log.info("[%s] comments-related probe on %d issues (k8s has 0 local comments)", repo, len(sample))
            comment_rows = []
            for num in sample:
                try:
                    comment_rows.append(probe_comments_related(s, slug, num))
                except requests.HTTPError as exc:
                    log.warning("#%s comments failed: %s", num, exc)
                time.sleep(0.3)
            nc = len(comment_rows)
            hit_rate = sum(1 for r in comment_rows if r["related_hits"]) / nc if nc else 0
            entry["channel_B_comments_k8s_feasibility"] = {
                "n_sampled": nc,
                "issues_with_related_comment_hit": sum(1 for r in comment_rows if r["related_hits"]),
                "rate": round(hit_rate, 3),
                "comment_rows": comment_rows,
            }

        report["repos"][repo] = entry
        log.info(
            "[%s] native_link_rate=%.0f%% any_cross_ref=%.0f%% issue_sourced_cross_ref=%.0f%%",
            repo, 100 * native_link_rate, 100 * any_cross_ref_rate, 100 * issue_sourced_cross_ref_rate,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
