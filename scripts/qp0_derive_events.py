from __future__ import annotations

# Timeline/comment extraction for the QP0 derived tables: shared raw-record primitives, the
# same-repo ClosedEvent closer filter, and duplicate-pair extraction. Split out of
# qp0_derive_tables.py to keep each script reviewable. All functions are pure.
import re
from collections import Counter
from datetime import datetime
from typing import Any

# lower rank wins when several methods find the same (source, canonical) pair
METHOD_RANK = {
    "marked_as_duplicate_event": 0,
    "vscode_bot_comment": 1,
    "k8s_close_comment": 2,
    "dup_label_with_ref": 3,
}
_REF = (
    r"(?:https?://github\.com/(?P<ur>[\w.-]+/[\w.-]+)/issues/(?P<un>\d+)"
    r"|(?P<sr>[\w.-]+/[\w.-]+)#(?P<sn>\d+)|#(?P<n>\d+))"
)
REF_RE = re.compile(_REF)
# a connector word is required ("duplicate of/with/to/as"): "the duplicate issue #44 was fixed"
# is not a duplicate statement. "Duplicate of issue #5" / "of [x](url)" are accepted.
DUP_RE = re.compile(
    r"\b(?:duplicate[sd]?|dups?)\b(?:\s+(?:of|with|to|as))+[\s:]*(?:issue\s+)?"
    r"(?:\[[^\]]*\]\(\s*)?" + _REF,
    re.I,
)
SLASH_DUP_RE = re.compile(r"(?:^|\s)/duplicate\s+(?:" + _REF + r"|(?P<sd>\d+))", re.I)
CLOSE_RE = re.compile(r"(?:^|\s)/close\b")
# negation within ~40 chars before the phrase: "is not a duplicate of", "isn't", "no longer"
NEG_RE = re.compile(r"\bnot\b|n['’]t\b|\bno longer\b|\bnever\b", re.I)
SEGMENT_RE = re.compile(r"\n\s*\n|(?<=[.!?])\s+")  # paragraph / sentence boundaries


class RawDataError(ValueError):
    """A raw record is malformed; the message names file:line and issue number where known."""


def parse_ts(s: str | None) -> datetime | None:
    """Parse a GitHub ISO timestamp (UTC 'Z') into an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def login(node: dict[str, Any] | None) -> str | None:
    return ((node or {}).get("author") or (node or {}).get("actor") or {}).get("login")


def timeline(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Non-null timeline nodes; unknown __typename values are fine, a missing one is not."""
    tl = rec.get("timelineItems")
    if not isinstance(tl, dict) or not isinstance(tl.get("nodes"), list):
        raise RawDataError("missing or null timelineItems")
    nodes = [n for n in tl["nodes"] if n]
    if any("__typename" not in n for n in nodes):
        raise RawDataError("timeline node without __typename")
    return nodes


def _same_repo(
    node: dict[str, Any], rec: dict[str, Any], repo: str, cnt: Counter[str], p: str
) -> bool:
    """True if node.repository is ``repo``; foreign / unknown (incl. schema-1 rows) are counted."""
    r = (node.get("repository") or {}).get("nameWithOwner")
    if rec.get("_schema", 1) < 2 or not r:
        cnt[f"{p}_repo_unknown"] += 1
    elif r.lower() != repo.lower():
        cnt[f"{p}_foreign_repo"] += 1
    else:
        return True
    return False


def closers(rec: dict[str, Any], repo: str) -> tuple[list[dict[str, Any]], Counter[str]]:
    """ClosedEvent closer PullRequests in ``repo``; others are only counted. A PR number alone
    is ambiguous across repos, so a closer without a verifiable repository is never used."""
    ok, cnt = [], Counter[str]()
    for n in timeline(rec):
        c = n.get("closer") or {}
        if (
            n["__typename"] == "ClosedEvent"
            and "number" in c
            and _same_repo(c, rec, repo, cnt, "closer")
        ):
            ok.append(c)
    return ok, cnt


def label_events(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Labeled/Unlabeled timeline events in timeline (chronological) order."""
    out = []
    for n in timeline(rec):
        if n["__typename"] in ("LabeledEvent", "UnlabeledEvent") and n.get("label"):
            act = "labeled" if n["__typename"] == "LabeledEvent" else "unlabeled"
            ev = {"number": rec["number"], "ts": parse_ts(n["createdAt"]), "action": act}
            out.append({**ev, "label": n["label"]["name"], "actor": login(n)})
    return out


def has_dup_label(rec: dict[str, Any]) -> bool:
    return any("duplicate" in (x or {}).get("name", "").lower() for x in rec["labels"]["nodes"])


def closed_as_duplicate(rec: dict[str, Any]) -> bool:
    """GitHub's own 'closed as duplicate' reason (strong signal even without a target)."""
    if rec.get("stateReason") == "DUPLICATE":
        return True
    return any(
        n["__typename"] == "ClosedEvent" and n.get("stateReason") == "DUPLICATE"
        for n in timeline(rec)
    )


def _ref_number(m: re.Match[str], repo: str) -> int | None:
    """Same-repo issue number of a REF match; other repos are not usable pairs."""
    if m["n"]:
        return int(m["n"])
    r, num = (m["ur"], m["un"]) if m["ur"] else (m["sr"], m["sn"])
    return int(num) if r.lower() == repo.lower() else None


def _comment_text(c: dict[str, Any]) -> tuple[str | None, str]:
    """(author, text) without quoted lines. The exception is k8s-ci-robot, which echoes the
    human's "/close" comment as a ">" quote: that echo is the only evidence when the original
    is older than the last-5 saved comments, so there ONLY the quoted lines are kept."""
    who, body = login(c), (c.get("body") or "").replace("\r", "")
    quoted = [ln.lstrip()[1:] for ln in body.splitlines() if ln.lstrip()[:1] == ">"]
    if who == "k8s-ci-robot":
        return who, "\n".join(quoted) if "In response to" in body else ""
    return who, "\n".join(ln for ln in body.splitlines() if ln.lstrip()[:1] != ">")


def _targets(text: str, repo: str) -> tuple[list[int], str]:
    """Same-repo targets named by non-negated duplicate phrases (each taken from the phrase's
    own sentence, in order) and the evidence snippet of the first one."""
    found: list[int] = []
    evidence = ""
    for seg in SEGMENT_RE.split(text):
        for rx in (DUP_RE, SLASH_DUP_RE):
            m = rx.search(seg)
            if not m or NEG_RE.search(seg[max(0, m.start() - 40) : m.start()]):
                continue
            bare = m.groupdict().get("sd")  # "/duplicate 12" without a '#'
            nums = (
                [int(bare)]
                if bare
                else [_ref_number(r, repo) for r in REF_RE.finditer(seg, m.start())]
            )
            found += [x for x in nums if x is not None]
            if not evidence and any(x is not None for x in nums):
                evidence = " ".join(seg[max(0, m.start() - 40) : m.end() + 60].split())
    return found, evidence


def extract_duplicates(
    rec: dict[str, Any], repo: str, diag: Counter[str] | None = None
) -> list[dict[str, Any]]:
    """Duplicate pairs (source=this issue) from events, comments and a ``*duplicate`` label.

    Per canonical the best-ranked method wins. Comment pairs need a CLOSED issue; a label
    alone yields nothing, it only lends its method to a target found in a duplicate sentence.
    Foreign / unverifiable-repo event canonicals are dropped and counted in ``diag``.
    """
    src, diag = rec["number"], diag if diag is not None else Counter()
    found: dict[int, dict[str, Any]] = {}

    def add(canon: int | None, method: str, ts: str, ev: str, n_targets: int = 1) -> None:
        if canon is None or canon == src:
            return
        row = {"source": src, "canonical": canon, "method": method, "ts": parse_ts(ts)}
        row |= {"evidence": ev[:200], "n_targets": n_targets}
        old = found.get(canon)
        if old is None or METHOD_RANK[method] < METHOD_RANK[old["method"]]:
            found[canon] = row

    for n in timeline(rec):
        canon = n.get("canonical") or {}
        if n["__typename"] == "MarkedAsDuplicateEvent":
            if _same_repo(canon, rec, repo, diag, "canon"):
                add(
                    canon.get("number"),
                    "marked_as_duplicate_event",
                    n["createdAt"],
                    f"marked by {login(n)}",
                )
        elif n["__typename"] == "UnmarkedAsDuplicateEvent":
            found.pop(canon.get("number"), None)  # un-marking later retracts the pair
    if rec["state"] != "CLOSED":
        return list(found.values())
    for c in (c for c in rec["lastComments"]["nodes"] if c):
        who, text = _comment_text(c)
        targets, ev = _targets(text, repo)
        if not targets:
            continue
        if who and who != "k8s-ci-robot" and who.lower().endswith(("bot", "[bot]")):
            method = "vscode_bot_comment"
        elif CLOSE_RE.search(text):
            method = "k8s_close_comment"
        elif has_dup_label(rec):
            method = "dup_label_with_ref"
        else:
            continue  # a bare mention without close / label / bot author is not a pair
        add(targets[0], method, c["createdAt"], ev, len(set(targets)))
    return list(found.values())
