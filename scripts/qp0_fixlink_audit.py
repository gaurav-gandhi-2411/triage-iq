"""Precision audit of derived fix links (issue -> same-repo merged PR, ClosedEvent closer).

A stratified random sample (strata = issue created year) of ``closed_event_closer_pr`` links
from fix_links.parquet is checked against the GitHub REST API, which is independent of the
scraped GraphQL data. Fix links may serve as ground truth only at >= 95% precision.

VERDICT RULES (pre-registered 2026-10-20, GG decision D79 item 2; do not change after results)
  a. GET /repos/{repo}/pulls/{pr}: 404 => WRONG(pr_not_in_repo); merged != true =>
     WRONG(pr_not_merged).
  b. Closed events come from /issues/{n}/events (plus /timeline when fetched). CONFIRMED if
     (1) a closed event's commit_id equals the PR's merge_commit_sha [commit_match], OR (2) the
     timeline has a cross-referenced event whose source.issue is that PR AND a closed event is
     within 120 s of the PR's merged_at [cross_ref_timing], OR (3) the PR title/body has a
     closing keyword + the issue number AND a closed event is within 120 s of merged_at
     [keyword_timing]. The first kind in that order is recorded as the evidence.
  c. WRONG(closed_before_pr_merged) if the LAST closed event precedes merged_at by > 1 h and
     the issue was not reopened after it; WRONG(no_relation) if the PR is merged in the repo,
     there is no cross-reference, no commit match and no keyword, and the last closed event
     differs from merged_at by > 1 h.
  d. Everything else => UNVERIFIABLE with a reason (api_error, issue_gone, issue_transferred,
     no_closed_event, reopened_after_close, ambiguous_timing, structural_evidence_but_timing).
Primary precision = CONFIRMED / n_sampled (UNVERIFIABLE counts as NOT confirmed). Precision
among decided = CONFIRMED / (CONFIRMED + WRONG). Both with Wilson 95% intervals. Acceptance:
"pass" iff the primary point estimate >= 0.95; the Wilson lower bound is always printed beside.
Post-audit hardening (keyword word boundary, repo-scoped references, no-timestamp guard) can
only turn a keyword_timing CONFIRMED into UNVERIFIABLE; the committed k8s report's 118
confirmations were all cross_ref_timing, so it is unaffected.
The token comes from `gh auth token` inside the process and is never printed or written.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API = "https://api.github.com"
YEARS = (2023, 2024, 2025, 2026)
NEAR_S, FAR_S = 120, 3600  # rule thresholds, seconds
MAX_PAGES = 40  # 4000 events; a longer thread is reported as unverifiable, not truncated
METHOD = "closed_event_closer_pr"
RULES = __doc__.split("VERDICT RULES", 1)[1].split("Primary precision")[0].strip()
Transport = Callable[[str, dict[str, Any]], tuple[int, dict[str, str], Any]]
KEYWORD = r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+"  # \b: not "prefix"/"disclose"


class ApiError(RuntimeError):
    """An API call failed after retries; the message never contains request details."""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials (None when n == 0)."""
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def gh_token() -> str:
    """Token from the gh CLI; output is never echoed (not even on failure)."""
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)  # noqa: S607
    except (subprocess.CalledProcessError, OSError):
        raise ApiError("`gh auth token` failed; run `gh auth login`") from None
    return out.stdout.strip()


def requests_transport(token: str) -> Transport:
    """Real transport; exceptions are re-raised without request details (headers hold the token)."""
    sess = requests.Session()
    sess.headers.update(
        {"Authorization": f"bearer {token}", "Accept": "application/vnd.github+json"}
    )

    def get(url: str, params: dict[str, Any]) -> tuple[int, dict[str, str], Any]:
        try:
            r = sess.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            raise ConnectionError(type(e).__name__) from None
        try:
            body = r.json()
        except ValueError:
            body = None
        return r.status_code, dict(r.headers), body

    return get


class Api:
    """Sequential REST client: retries, rate-limit sleeps; core limit is separate from GraphQL."""

    def __init__(
        self,
        transport: Transport,
        min_remaining: int = 200,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.transport, self.min_remaining, self.sleep, self.now = (
            transport,
            min_remaining,
            sleep,
            now,
        )

    def _wait_reset(self, headers: dict[str, str]) -> None:
        reset = float(headers.get("X-RateLimit-Reset", self.now() + 60))
        self.sleep(max(1.0, reset - self.now()) + 2)

    def get(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
        """(status, json) for 2xx/404/410; other statuses retry 3x then raise ApiError."""
        for attempt in range(4):
            try:
                status, hdr, body = self.transport(API + path, params or {})
            except (ConnectionError, TimeoutError, OSError):
                status, hdr, body = 0, {}, None
            if status == 0 or status >= 500:
                self.sleep(2.0 * 2**attempt)
                continue
            if status in (403, 429):  # primary or secondary rate limit
                if "Retry-After" in hdr:
                    self.sleep(float(hdr["Retry-After"]) + 1)
                else:
                    self._wait_reset(hdr)
                continue
            if int(hdr.get("X-RateLimit-Remaining", self.min_remaining + 1)) < self.min_remaining:
                self._wait_reset(hdr)
            if status in (404, 410) or 200 <= status < 300:
                return status, body
            raise ApiError(f"http_{status}")
        raise ApiError("retries_exhausted")

    def paged(self, path: str) -> tuple[int, list[dict[str, Any]]]:
        out: list[dict[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            status, body = self.get(path, {"per_page": 100, "page": page})
            if status != 200:
                return status, []
            out += body
            if len(body) < 100:
                return 200, out
        raise ApiError("too_many_pages")


def ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def has_keyword(text: str, issue: int, repo: str) -> bool:
    """Closing keyword directly followed by a reference to ``issue`` of ``repo`` (not #12 inside
    #123; a cross-repo ``other/x#7`` or URL is a different issue and must not match)."""
    r = re.escape(repo)
    ref = rf"(?:(?:{r})?#|https?://github\.com/{r}/issues/){issue}(?!\d)"
    return re.search(KEYWORD + ref, text, re.I) is not None


def has_cross_ref(timeline: list[dict[str, Any]], pr: int) -> bool:
    for e in timeline:
        src = (e.get("source") or {}).get("issue") or {}
        if (
            e.get("event") == "cross-referenced"
            and src.get("number") == pr
            and "pull_request" in src
        ):
            return True
    return False


def judge(api: Api, repo: str, issue: int, pr: int) -> dict[str, Any]:
    """Verdict dict {verdict, reason, evidence, ...} for one link, per the module-docstring rules."""
    v = lambda verdict, reason, **kw: {"verdict": verdict, "reason": reason, **kw}  # noqa: E731
    try:
        st, p = api.get(f"/repos/{repo}/pulls/{pr}")
        if st in (404, 410):
            return v("WRONG", "pr_not_in_repo")
        if p.get("merged") is not True:
            return v("WRONG", "pr_not_merged")
        merged_at, sha = ts(p.get("merged_at")), p.get("merge_commit_sha")
        st, events = api.paged(f"/repos/{repo}/issues/{issue}/events")
        if st != 200:
            return v("UNVERIFIABLE", "issue_gone")
        if any(e.get("event") == "transferred" for e in events):
            return v("UNVERIFIABLE", "issue_transferred")
        closed = [e for e in events if e.get("event") == "closed"]
        if not closed or merged_at is None:
            return v("UNVERIFIABLE", "no_closed_event")
        if sha and any(e.get("commit_id") == sha for e in closed):
            return v("CONFIRMED", "commit_match", evidence="commit_match")
        st, tl = api.paged(f"/repos/{repo}/issues/{issue}/timeline")
        if st != 200:
            return v("UNVERIFIABLE", "issue_gone")
    except ApiError as e:
        return v("UNVERIFIABLE", f"api_error:{e}")
    except (AttributeError, TypeError, ValueError) as e:  # malformed payload: never a verdict
        return v("UNVERIFIABLE", f"api_error:malformed_{type(e).__name__}")
    times = sorted(
        t
        for e in closed + [x for x in tl if x.get("event") == "closed"]
        if (t := ts(e.get("created_at")))
    )
    if not times:  # closed events without created_at: no usable timing, nothing to index
        return v("UNVERIFIABLE", "no_closed_event_time")
    gaps = [abs((t - merged_at).total_seconds()) for t in times]
    near = any(g <= NEAR_S for g in gaps)
    xref = has_cross_ref(tl, pr)
    kw = has_keyword(f"{p.get('title') or ''}\n{p.get('body') or ''}", issue, repo)
    if xref and near:
        return v("CONFIRMED", "cross_ref_timing", evidence="cross_ref_timing")
    if kw and near:
        return v("CONFIRMED", "keyword_timing", evidence="keyword_timing")
    last = times[-1]
    last_gap = (last - merged_at).total_seconds()  # negative: closed before the merge
    reopened = any(
        e.get("event") == "reopened" and (t := ts(e.get("created_at"))) and t > last for e in events
    )
    if last_gap < -FAR_S:
        if reopened:
            return v("UNVERIFIABLE", "reopened_after_close")
        return v("WRONG", "closed_before_pr_merged")
    if not (xref or kw) and abs(last_gap) > FAR_S:
        return v("WRONG", "no_relation")
    if xref or kw:
        return v("UNVERIFIABLE", "structural_evidence_but_timing")
    return v("UNVERIFIABLE", "ambiguous_timing")


def load_links(derived: Path) -> list[tuple[int, int, int]]:
    """(issue, pr, created_year) for every closer link in a strata year; sorted, no verdict data."""
    fl = pd.read_parquet(derived / "fix_links.parquet", columns=["issue", "pr", "method"])
    iss = pd.read_parquet(derived / "issues.parquet", columns=["number", "created_year"])
    m = fl[fl["method"] == METHOD].merge(iss, left_on="issue", right_on="number")
    m = m[m["created_year"].isin(YEARS)].drop_duplicates(["issue", "pr"])
    return sorted(
        zip(m["issue"].astype(int), m["pr"].astype(int), m["created_year"].astype(int), strict=True)
    )


def stratified_sample(
    links: list[tuple[int, int, int]], per_stratum: int, seed: int = 42
) -> list[tuple[int, int, int]]:
    """min(per_stratum, available) links per year via one seeded RNG; sees only (issue, pr, year)."""
    rng, out = random.Random(seed), []
    for year in YEARS:
        pool = sorted(x for x in links if x[2] == year)
        out += sorted(rng.sample(pool, min(per_stratum, len(pool))))
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts, precisions and Wilson intervals for a set of judged links."""
    c = Counter(r["verdict"] for r in rows)
    n, k, w = len(rows), c["CONFIRMED"], c["WRONG"]
    ci = lambda a, b: None if (x := wilson(a, b)) is None else [round(x[0], 4), round(x[1], 4)]  # noqa: E731
    return {
        "n_sampled": n,
        "verdicts": dict(c),
        "reasons": dict(Counter(f"{r['verdict']}:{r['reason']}" for r in rows)),
        "primary_precision": k / n if n else None,
        "primary_wilson95": ci(k, n),
        "decided_precision": k / (k + w) if k + w else None,
        "decided_wilson95": ci(k, k + w),
    }


def run_audit(api: Api, repo: str, derived: Path, per_stratum: int, seed: int) -> dict[str, Any]:
    links = load_links(derived)
    rows = []
    for issue, pr, year in stratified_sample(links, per_stratum, seed):
        res = judge(api, repo, issue, pr)
        rows.append(
            {"issue": issue, "pr": pr, "year": year, **res}
            | {"issue_url": f"https://github.com/{repo}/issues/{issue}"}
            | {"pr_url": f"https://github.com/{repo}/pull/{pr}"}
        )
        print(f"{year} #{issue} -> PR {pr}: {res['verdict']} {res['reason']}", file=sys.stderr)
    overall = summarise(rows)
    lo = (overall["primary_wilson95"] or [None])[0]
    return {
        "repo": repo, "seed": seed, "per_stratum": per_stratum, "method": METHOD,
        "population": {str(y): sum(1 for x in links if x[2] == y) for y in YEARS},
        "rules": RULES, "generated_with": "scripts/qp0_fixlink_audit.py",
        "overall": overall,
        "per_stratum_summary": {str(y): summarise([r for r in rows if r["year"] == y]) for y in YEARS},
        "acceptance": {
            "threshold": 0.95,
            "primary_precision": overall["primary_precision"],
            "wilson_lower_bound": lo,
            "result": "pass" if (overall["primary_precision"] or 0) >= 0.95 else "fail",
        },
        "links": rows,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit fix-link precision against the REST API")
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--per-stratum", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--derived-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sleep-min-remaining", type=int, default=200)
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    derived = a.derived_dir or Path(r"D:\triageiq-data") / "derived" / slug
    out = a.out or Path(f"reports/qp0_fixlink_audit_{slug}_2026-10.json")
    api = Api(requests_transport(gh_token()), a.sleep_min_remaining)
    rep = run_audit(api, a.repo, derived, a.per_stratum, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    acc = rep["acceptance"]
    print(json.dumps({"overall": rep["overall"], "acceptance": acc}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
