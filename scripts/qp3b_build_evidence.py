"""Build the resolution-evidence artifact for flag 2 (TRIAGE_RESOLUTION_EVIDENCE), k8s.

Offline CANDIDATE artifact (docs/QP3B_DESIGN_2026-10.md, flag 2): one record per derived_v2 issue
with (a) its audited fix PRs (closer + merged, `audited_fix_links`): number, title and the first
~300 chars of the cleaned PR body, and (b) the last non-bot, non-command comment among the (up to
5) comments stored in `issues.closing_comments`, only for CLOSED issues (an open issue has no
closing comment; its stored comments are just the tail of a live thread). Empty when none.
Nothing here is served and nothing is written under data/models.

Usage: python scripts/qp3b_build_evidence.py [--derived DIR] [--out PARQUET]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from qp0_derive_stats import audited_fix_links

ROOT = Path(__file__).resolve().parent.parent
DERIVED = Path(r"D:\triageiq-data\derived_v2\kubernetes__kubernetes")
OUT = Path(r"D:\triageiq-data\models\candidates\resolution_evidence_kubernetes_kubernetes_v1.parquet")

MAX_CHARS = 300  # design: each evidence item truncated; budget is checked later by C4
# Exact bot logins (lower-case). The design names k8s-ci-robot / kubernetes-prow / vscode-triage-bot;
# k8s-triage-robot (stale/rotten lifecycle spam, 5,009 stored comments) is added because it is
# unambiguously a bot and would otherwise be most "closing comments".
BOT_LOGINS = frozenset({"k8s-ci-robot", "kubernetes-prow", "k8s-triage-robot", "vscode-triage-bot"})
BOT_SUFFIXES = ("[bot]", "-robot", "-bot")  # e.g. dependabot[bot], redhat-chai-bot
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE = re.compile(r"```[\s\S]*?```")
_HEADING_ONLY = re.compile(r"^\s*#{1,6}\s")
_WS = re.compile(r"\s+")
# Headings of the k8s PR template whose section holds the substantive description.
_WHAT_PR_DOES = re.compile(r"#+\s*What this PR does[^\n]*\n", re.IGNORECASE)


def is_bot(login: str | None) -> bool:
    """True for automation accounts (see BOT_LOGINS / BOT_SUFFIXES); None (deleted user) is not."""
    if not login:
        return False
    low = login.lower()
    return low in BOT_LOGINS or low.endswith(BOT_SUFFIXES)


def truncate(text: str, n: int = MAX_CHARS) -> str:
    """Collapse whitespace and cut to n chars, ending with an ellipsis when cut."""
    t = _WS.sub(" ", text).strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "\u2026"


def clean_pr_body(body: str | None, n: int = MAX_CHARS) -> str:
    """PR body for evidence: drop HTML comments/code fences/prow command lines/headings.

    k8s PR bodies start with a template; when the 'What this PR does' section exists the text
    after it is used, otherwise the whole body.
    """
    if not body:
        return ""
    t = _HTML_COMMENT.sub("", body)
    m = _WHAT_PR_DOES.search(t)
    if m:
        t = t[m.end() :]
    t = _FENCE.sub("", t)
    keep = [
        ln
        for ln in t.splitlines()
        if ln.strip() and not ln.strip().startswith("/") and not _HEADING_ONLY.match(ln)
    ]
    return truncate("\n".join(keep), n)


def _is_command_only(body: str) -> bool:
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("/") for ln in lines)


def last_human_comment(comments: Iterable[Mapping[str, Any]] | None) -> tuple[str, str]:
    """(author, truncated body) of the last non-bot comment with real text; ("", "") if none.

    Comments that are only prow commands (e.g. "/sig node", "/lgtm") carry no resolution
    information and are skipped. `comments` are in chronological order.
    """
    best = ("", "")
    for c in comments if comments is not None else []:
        body = (c.get("body") or "").strip()
        if is_bot(c.get("author")) or not body or _is_command_only(body):
            continue
        best = (c.get("author") or "", truncate(body))
    return best


def build_evidence(
    issues: pd.DataFrame, fix_links: pd.DataFrame, prs: pd.DataFrame
) -> pd.DataFrame:
    """One row per issue: audited fix PRs (list of structs) + last non-bot closing comment."""
    pr_by_num = prs.drop_duplicates("number").set_index("number")
    audited = audited_fix_links(fix_links).drop_duplicates(["issue", "pr"])
    fix_by_issue: dict[int, list[dict[str, Any]]] = {}
    for issue, pr in sorted(zip(audited["issue"], audited["pr"], strict=True)):
        if pr in pr_by_num.index:
            r = pr_by_num.loc[pr]
            rec = {
                "number": int(pr),
                "title": truncate(r["title"] or ""),
                "body_excerpt": clean_pr_body(r["body"]),
            }
        else:  # link target absent from prs.parquet: no title to cite, so no evidence
            continue
        fix_by_issue.setdefault(int(issue), []).append(rec)
    rows = []
    for r in issues.itertuples(index=False):
        author, comment = ("", "")
        if r.state == "CLOSED":
            author, comment = last_human_comment(r.closing_comments)
        fixes = fix_by_issue.get(int(r.number), [])
        rows.append(
            {
                "number": int(r.number),
                "state": r.state,
                "fix_prs": fixes,
                "closing_comment_author": author,
                "closing_comment": comment,
                "has_fix_evidence": bool(fixes),
                "has_comment_evidence": bool(comment),
                "has_evidence": bool(fixes) or bool(comment),
            }
        )
    return pd.DataFrame(rows)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path, default=DERIVED)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    cols = ["number", "state", "closing_comments"]
    iss = pd.read_parquet(a.derived / "issues.parquet", columns=cols)
    fl = pd.read_parquet(a.derived / "fix_links.parquet")
    prs = pd.read_parquet(a.derived / "prs.parquet", columns=["number", "title", "body"])
    ev = build_evidence(iss, fl, prs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(a.out, index=False)
    manifest = {
        "kind": "CANDIDATE (offline; not served)",
        "rows": len(ev),
        "with_fix_evidence": int(ev["has_fix_evidence"].sum()),
        "with_comment_evidence": int(ev["has_comment_evidence"].sum()),
        "with_any_evidence": int(ev["has_evidence"].sum()),
        "max_chars": MAX_CHARS,
        "bot_logins": sorted(BOT_LOGINS),
        "bot_suffixes": list(BOT_SUFFIXES),
        "sources": {
            n: {"path": str(a.derived / n), "sha256": _sha(a.derived / n)}
            for n in ("issues.parquet", "fix_links.parquet", "prs.parquet")
        },
        "output_sha256": _sha(a.out),
        "script": "scripts/qp3b_build_evidence.py",
    }
    a.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in manifest if k.startswith(("rows", "with"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
