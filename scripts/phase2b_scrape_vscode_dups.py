"""Phase 2b: scrape vscode `*duplicate`-labeled issues + comments and their dup targets.

Approved scope (ADR-0026 + probe addendum + GG go): ~4,500-5,000 dup-labeled issues spread
across 2016-2026, parse the dup TARGET from comments (the `/duplicate <URL or #N>` triage
command and 'duplicate of' text forms), then fetch each unique target issue and record whether
it is a PR (shared number space — PR targets are dropped downstream, guard #3).

Collection only — this script does NOT touch gold_related.parquet or any split. Pair building,
PR filtering, disjointness asserts, and the spot-check sample live in
scripts/phase2b_build_candidates.py.

Outputs (resumable checkpoints, gitignored):
  data/raw/phase2b/vscode_dup_issues.jsonl   one line per dup-labeled issue (+ parsed refs)
  data/raw/phase2b/vscode_targets.jsonl      one line per fetched target issue
  data/raw/microsoft_vscode/{n}.json         raw issue JSONs, pipeline convention (skip-if-exists)

Budget: ~55 search + ~4,700 comment + ~2,500-3,000 target calls ~= 7,500-8,000 requests,
paced under the authed 5,000/hr core limit -> ~1.5-2h. Reproduce/resume:
python scripts/phase2b_scrape_vscode_dups.py
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
CKPT_DIR = Path("data/raw/phase2b")
RAW_DIR = Path("data/raw/microsoft_vscode")
DUPS_CKPT = CKPT_DIR / "vscode_dup_issues.jsonl"
TARGETS_CKPT = CKPT_DIR / "vscode_targets.jsonl"

YEARS = [str(y) for y in range(2016, 2027)]
PER_YEAR_MAX = 500  # 5 search pages/year; search API caps any query at 1,000 results
TARGET_DUP_ISSUES = 4700  # global stop (probe: 62% recovery -> ~2,900 parsed pairs)

# Dup-target extraction, priority order. The /duplicate (or \duplicate) triage command is the
# high-precision modern convention (probe addendum, ADR-0026); text forms cover older eras.
DUP_PATTERNS = [
    (
        "dup_cmd",
        re.compile(
            r"[\\/]duplicate\s+(?:of\s+)?(?:https?://github\.com/microsoft/vscode/issues/(\d+)|#?(\d+))",
            re.IGNORECASE,
        ),
    ),
    (
        "dup_text",
        re.compile(
            r"dup(?:licate|e)?\s*(?:of|to|:)\s*(?:https?://github\.com/microsoft/vscode/issues/(\d+)|#?(\d+))",
            re.IGNORECASE,
        ),
    ),
]
# Bonus related-channel: body refs, same patterns as scripts/07_extract_related_pairs.py plus
# the extended set from scripts/phase2a_corpus_feasibility.py (candidate-grade).
BODY_REF_PATTERNS = [
    r"[Dd]uplicate[sd]?(?: of)? #?(\d+)",
    r"[Dd]up(?:licate)? of #?(\d+)",
    r"[Ss]ame as #(\d+)",
    r"[Cc]losing as dup(?:licate)? of #?(\d+)",
    r"[Ss]ee(?: also)? #(\d+)",
    r"[Cc]loses? #(\d+)",
    r"[Ff]ixes? #(\d+)",
]
BODY_REF_EXTENDED = [
    r"[Rr]elated(?: to)? #(\d+)",
    r"[Ss]imilar to #(\d+)",
    r"[Rr]efs? #(\d+)",
    r"github\.com/microsoft/vscode/issues/(\d+)",
]


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


def _get(s: requests.Session, url: str, params: dict | None = None) -> requests.Response | None:
    """GET with rate-limit backoff. Returns None on 404/410 (deleted/transferred issues)."""
    for attempt in range(5):
        try:
            r = s.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            log.warning("network error %s — retry %d", exc, attempt)
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code in (404, 410):
            return None
        if r.status_code == 403 and int(r.headers.get("X-RateLimit-Remaining", "1")) == 0:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 300))
            wait = max(30, min(3600, reset - int(time.time()) + 5))
            log.warning("rate limit exhausted, sleeping %ss", wait)
            time.sleep(wait)
            continue
        if r.status_code in (403, 429, 502, 503):
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"retries exhausted on {url}")


def _load_ckpt(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rows[rec["number"]] = rec
    return rows


def _append_ckpt(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _save_raw(number: int, payload: dict) -> None:
    dest = RAW_DIR / f"{number}.json"
    if not dest.exists():  # never clobber the existing corpus files (they carry comments_data)
        dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def parse_dup_target(comments: list[dict]) -> tuple[int | None, str | None, str | None]:
    """First dup-target match across comments, with pattern name + snippet for the spot-check."""
    for c in comments:
        body = str(c.get("body", ""))
        for name, pat in DUP_PATTERNS:
            m = pat.search(body)
            if m:
                target = int(next(g for g in m.groups() if g))
                start = max(0, m.start() - 60)
                return target, name, body[start : m.end() + 60].replace("\n", " ")
    return None, None, None


def parse_body_refs(text: str) -> tuple[set[int], set[int]]:
    current, extended = set(), set()
    for pats, out in ((BODY_REF_PATTERNS, current), (BODY_REF_EXTENDED, extended)):
        for pat in pats:
            for m in re.finditer(pat, text, re.IGNORECASE):
                out.add(int(m.group(1)))
    return current, extended - current


def scrape_dup_issues(s: requests.Session) -> dict[int, dict]:
    done = _load_ckpt(DUPS_CKPT)
    log.info("checkpoint: %d dup issues already scraped", len(done))
    for year in YEARS:
        if len(done) >= TARGET_DUP_ISSUES:
            break
        end = "2026-07-10" if year == "2026" else f"{year}-12-31"
        q = f"repo:{REPO} type:issue label:*duplicate created:{year}-01-01..{end}"
        fetched_this_year = 0
        for page in range(1, PER_YEAR_MAX // 100 + 1):
            r = _get(
                s,
                f"{API}/search/issues",
                {"q": q, "sort": "created", "order": "asc", "per_page": 100, "page": page},
            )
            time.sleep(2.1)  # search API: 30 req/min
            items = r.json().get("items", []) if r else []
            if not items:
                break
            for it in items:
                n = int(it["number"])
                if n in done:
                    continue
                cr = _get(s, f"{API}/repos/{REPO}/issues/{n}/comments", {"per_page": 100})
                time.sleep(0.55)
                comments = cr.json() if cr else []
                target, pattern, snippet = parse_dup_target(comments)
                body = str(it.get("body") or "")
                refs_cur, refs_ext = parse_body_refs(str(it.get("title") or "") + " " + body)
                rec = {
                    "number": n,
                    "title": it.get("title"),
                    "body": body,
                    "created_at": it.get("created_at"),
                    "state": it.get("state"),
                    "labels": [lb["name"] for lb in it.get("labels", [])],
                    "dup_target": target,
                    "dup_pattern": pattern,
                    "dup_snippet": snippet,
                    "body_refs_current": sorted(refs_cur - {n}),
                    "body_refs_extended": sorted(refs_ext - {n}),
                    "n_comments_fetched": len(comments),
                }
                _append_ckpt(DUPS_CKPT, rec)
                it["comments_data"] = comments
                _save_raw(n, it)
                done[n] = rec
                fetched_this_year += 1
                if len(done) % 200 == 0:
                    log.info("dup issues scraped: %d (year %s)", len(done), year)
                if len(done) >= TARGET_DUP_ISSUES:
                    break
            if len(done) >= TARGET_DUP_ISSUES:
                break
        log.info("[%s] +%d dup issues (total %d)", year, fetched_this_year, len(done))
    return done


def scrape_targets(s: requests.Session, dups: dict[int, dict]) -> None:
    wanted: set[int] = set()
    for rec in dups.values():
        if rec.get("dup_target"):
            wanted.add(int(rec["dup_target"]))
        wanted.update(int(x) for x in rec.get("body_refs_current", []))
        # extended refs are candidate-grade; fetch them too — cheap relative to their value
        wanted.update(int(x) for x in rec.get("body_refs_extended", []))
    wanted -= set(dups.keys())
    have = _load_ckpt(TARGETS_CKPT)
    todo = sorted(wanted - set(have.keys()))
    log.info("targets: %d wanted, %d already fetched, %d to go", len(wanted), len(have), len(todo))
    for i, t in enumerate(todo):
        r = _get(s, f"{API}/repos/{REPO}/issues/{t}")
        time.sleep(0.55)
        if r is None:
            _append_ckpt(TARGETS_CKPT, {"number": t, "missing": True})
            continue
        it = r.json()
        rec = {
            "number": t,
            "is_pr": "pull_request" in it,
            "title": it.get("title"),
            "body": str(it.get("body") or ""),
            "created_at": it.get("created_at"),
            "state": it.get("state"),
            "labels": [lb["name"] for lb in it.get("labels", [])],
        }
        _append_ckpt(TARGETS_CKPT, rec)
        if not rec["is_pr"]:
            _save_raw(t, it)
        if (i + 1) % 200 == 0:
            log.info("targets fetched: %d/%d", i + 1, len(todo))


def main() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s = _session()
    dups = scrape_dup_issues(s)
    scrape_targets(s, dups)
    with_target = sum(1 for d in dups.values() if d.get("dup_target"))
    log.info(
        "DONE. dup issues: %d, with parsed dup target: %d (%.0f%%)",
        len(dups),
        with_target,
        100 * with_target / max(len(dups), 1),
    )


if __name__ == "__main__":
    main()
