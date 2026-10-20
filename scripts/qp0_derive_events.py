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
    "closed_event_duplicate_of": 0,  # GitHub's own target of "close as duplicate" (dup_targets scrape)
    "marked_as_duplicate_event": 1,  # .duplicate of a MarkedAsDuplicateEvent, direction resolved
    "vscode_bot_comment": 2,
    "slash_duplicate_cmd": 3,  # a maintainer's explicit /duplicate command (vscode)
    "k8s_close_comment": 4,
    "dup_label_with_ref": 5,
    "close_as_duplicate_prose": 6,  # weakest: a maintainer says "close this as a duplicate" + one link
}
_REF = (
    r"(?:https?://github\.com/(?P<ur>[\w.-]+/[\w.-]+)/issues/(?P<un>\d+)"
    r"|(?P<sr>[\w.-]+/[\w.-]+)#(?P<sn>\d+)|#(?P<n>\d+))"
)
REF_RE = re.compile(_REF)
# a connector word is required ("duplicate of/with/to/as"): "the duplicate issue #44 was fixed"
# is not a duplicate statement. "Duplicate of issue #5" / "of [x](url)" are accepted.
DUP_RE = re.compile(
    r"\b(?:duplicate[sd]?|dup(?:e[sd]?|s)?)\b(?:\s+(?:of|with|to|as))+[\s:]*(?:issue\s+)?"
    r"(?:\[[^\]]*\]\(\s*)?" + _REF,
    re.I,
)
SLASH_DUP_RE = re.compile(r"(?:^|\s)/duplicate\s+(?:" + _REF + r"|(?P<sd>\d+))", re.I)
CLOSE_RE = re.compile(r"(?:^|\s)/close\b|\bclos(?:e|ing|ed)\s+(?:this\s+)?(?:issue\s+)?as\b", re.I)
# hedged / question / conditional wording in the phrase's own line: not a duplicate claim
HEDGE_RE = re.compile(
    r"\?|\b(?:if|whether|may|might|could|would|possible|possibly|maybe|perhaps|probably|unsure"
    r"|doubt|unless)\b|not sure|but it is not|is wrong|can you|please check|i think",
    re.I,
)
# negation / disagreement AFTER the phrase in the same line ("... #30 but it is not")
NEG_AFTER_RE = re.compile(r"\bnot\b|n['’]t\b|\bnope\b|disagree|\bwrong\b", re.I)
# negation within ~40 chars before the phrase: "is not a duplicate of", "isn't", "no longer"
NEG_RE = re.compile(r"\bnot\b|n['’]t\b|\bno longer\b|\bnever\b", re.I)
# vscode maintainers close with "/duplicate <ref>" (also a backslash; the ref may follow other words
# on later lines) or prose like "I will go ahead and close this issue as a duplicate"
SLASH_CMD_RE = re.compile(r"(?:^|\s)[/\\]duplicate(?![\w-])", re.I)  # not "/duplicate-check"
NEG_CMD_RE = re.compile(
    r"\b(?:not|n['’]t|no longer|never)(?:\s+(?:a|an|the))?(?:\s+dup\w*\W*)?\s*$", re.I
)
PULL_RE = re.compile(r"/pull/\d+")
# hedges that matter AFTER a command's target ("#28 maybe"); "if you need ..." is just explanation
TRAIL_HEDGE_RE = re.compile(
    r"\?(?=\s|$)|\b(?:maybe|perhaps|probably|possibly|unsure|not sure)\b|i think", re.I
)
PROSE_RE = re.compile(r"\bclos(?:e|ing|ed)\b[^\n.]{0,30}\bas (?:a )?dup", re.I)
# a staff comment that is just "Duplicate <ref>" (no "of"), as vscode maintainers often write
PLAIN_DUP_RE = re.compile(
    r"^[ \t>*_-]*dup(?:licate|e)?\b[\s:\-]*(?=https?://|#\d|[\w.-]+/[\w.-]+#\d)", re.I | re.M
)
BARE_NUM_RE = re.compile(r"[ \t]*(?:of[ \t]+)?#?(\d+)\b")  # same line only
STAFF = {"OWNER", "MEMBER", "COLLABORATOR"}
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
    if who == "k8s-ci-robot":
        # ">>" and deeper are someone else's words quoted inside the closer's comment
        one = [ln.lstrip()[1:] for ln in body.splitlines() if ln.lstrip()[:1] == ">"]
        text = "\n".join(ln for ln in one if ln[:1] != ">")
        # only a closer's own "/close" inside the echoed block makes it a closing comment
        return who, text if "In response to" in body and CLOSE_RE.search(text) else ""
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
            end = seg.find("\n", m.end())
            end = len(seg) if end < 0 else end
            line = seg[seg.rfind("\n", 0, m.start()) + 1 : end]
            if HEDGE_RE.search(line) or NEG_AFTER_RE.search(seg[m.end() : end]):
                continue
            bare = m.groupdict().get("sd")  # "/duplicate 12" without a '#'
            nums = (
                [int(bare)]
                if bare
                else [_ref_number(r, repo) for r in REF_RE.finditer(seg, m.start(), end)]
            )
            found += [x for x in nums if x is not None]
            if not evidence and any(x is not None for x in nums):
                evidence = " ".join(seg[max(0, m.start() - 40) : m.end() + 60].split())
    return found, evidence


def _cmd_target(text: str, repo: str) -> tuple[int, str, int] | None:
    """Same-repo target (number, evidence, n_targets) of a "/duplicate" command.

    The target is the first reference after the command: at most 3 words may separate them (a
    bare number must follow on the same line) and, when words do separate them, no other
    reference may follow on that line. Refs joined by and / & / , / or count in n_targets. The
    command's own line must carry no hedge ("?", maybe, could, ...), no negation tied to the
    command ("not a /duplicate", "not a duplicate: /duplicate"), and no disagreement after the
    target ("... but it is not"). A "don't" in an unrelated clause does not matter.
    """
    m = SLASH_CMD_RE.search(text) or PLAIN_DUP_RE.search(text)
    if not m:
        return None
    ls = text.rfind("\n", 0, m.start()) + 1
    if NEG_CMD_RE.search(text[ls : m.start()]):
        return None
    tail = text[m.end() : m.end() + 300]
    bare, ref = BARE_NUM_RE.match(tail), REF_RE.search(tail)
    if bare and (not ref or bare.start(1) <= ref.start()):
        pos, num, first_end = bare.start(1), int(bare.group(1)), bare.end(1)
    elif ref:
        pos, num, first_end = ref.start(), _ref_number(ref, repo), ref.end()
    else:
        return None
    words = len(re.findall(r"[A-Za-z']+", tail[:pos]))
    if num is None or words > 3:
        return None
    eol = tail.find("\n", first_end)
    eol = len(tail) if eol < 0 else eol
    after = tail[first_end:eol]
    # full hedge list before the target (the URL itself may contain "?"); only a few markers after
    if HEDGE_RE.search(text[ls : m.end() + pos]) or TRAIL_HEDGE_RE.search(after):
        return None
    if NEG_AFTER_RE.search(after):
        return None
    others = [r for r in REF_RE.finditer(tail, first_end, eol) if _ref_number(r, repo) != num]
    n_targets, prev, later = (
        1,
        first_end,
        others,
    )  # "[#5](url)" repeats the same target: not "other"
    for r in later:
        if re.fullmatch(r"\s*(?:,|&|and|or)?\s*", tail[prev : r.start()], re.I):
            n_targets, prev = n_targets + 1, r.end()
        else:
            break
    if later and words:
        return None  # words before the first ref AND another ref on the line: ambiguous
    return num, " ".join(text[max(0, m.start() - 20) : m.end() + 120].split()), n_targets


def _prose_target(text: str, repo: str, prs: set[int] | None) -> tuple[int, str] | None:
    """Target of "close this as a duplicate" prose: exactly one reference, and it is not a PR."""
    refs, m = list(REF_RE.finditer(text)), PROSE_RE.search(text)
    if not m or len(refs) != 1 or PULL_RE.search(text):
        return None
    num = _ref_number(refs[0], repo)
    if num is None or (prs and num in prs):
        return None
    return num, " ".join(text[max(0, m.start() - 60) : m.end() + 60].split())


def _human(c: dict[str, Any]) -> bool:
    """A real user: not typed Bot and a login that does not end in bot / [bot]."""
    a = c.get("author") or {}
    return a.get("__typename", "User") == "User" and not (a.get("login") or "").lower().endswith(
        ("bot", "[bot]")
    )


def extract_duplicates(
    rec: dict[str, Any],
    repo: str,
    diag: Counter[str] | None = None,
    pr_numbers: set[int] | None = None,
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
            if canon.get("number") == src:
                diag["canon_self"] += 1  # vscode: canonical is the issue itself, target not scraped
            elif _same_repo(canon, rec, repo, diag, "canon"):
                add(
                    canon.get("number"),
                    "marked_as_duplicate_event",
                    n["createdAt"],
                    f"marked by {login(n)}",
                )
        elif n["__typename"] == "ClosedEvent" and n.get("duplicateOf"):
            if _same_repo(n["duplicateOf"], rec, repo, diag, "canon"):
                add(
                    n["duplicateOf"].get("number"),
                    "closed_event_duplicate_of",
                    n["createdAt"],
                    "closed as duplicate",
                )
        elif n["__typename"] == "UnmarkedAsDuplicateEvent":
            found.pop(canon.get("number"), None)  # un-marking later retracts the pair
    if rec["state"] != "CLOSED":
        return list(found.values())
    for c in (c for c in rec["lastComments"]["nodes"] if c):
        who, text = _comment_text(c)
        staff = c.get("authorAssociation") in STAFF and _human(c)
        if staff and (hit := _cmd_target(text, repo)):
            add(hit[0], "slash_duplicate_cmd", c["createdAt"], hit[1], hit[2])
            continue
        targets, ev = _targets(text, repo)
        if not targets and staff and (hit := _prose_target(text, repo, pr_numbers)):
            add(hit[0], "close_as_duplicate_prose", c["createdAt"], hit[1])
            continue
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


def _dup_side(rec: dict[str, Any]) -> bool:
    """True if the issue itself says it is the duplicate: closed with a duplicate label / reason."""
    return rec["state"] == "CLOSED" and (has_dup_label(rec) or closed_as_duplicate(rec))


def target_pairs(
    dt: dict[str, Any], recs: dict[int, dict[str, Any]], repo: str, diag: Counter[str]
) -> list[dict[str, Any]]:
    """Pairs from a dup_targets row (the issue's CLOSED / MarkedAsDuplicate / Unmarked events).

    ClosedEvent.duplicateOf is the real target of a UI "close as duplicate" (null for slash
    commands). On an issue's own timeline MarkedAsDuplicateEvent.canonical is that issue itself and
    .duplicate the OTHER issue of the pair, so the direction comes from which side is the closed
    duplicate (label / stateReason in ``recs``); both or neither, or a foreign repo, is skipped
    (ambiguity counted in ``diag``). Un-marking retracts. Row source = the duplicate side.
    """
    x = dt["number"]
    found: dict[tuple[int, int], dict[str, Any]] = {}

    def add(src: int, canon: int | None, method: str, node: dict[str, Any], ev: str) -> None:
        if canon is not None and canon != src:
            row = {
                "source": src,
                "canonical": canon,
                "method": method,
                "ts": parse_ts(node["createdAt"]),
            }
            found[(src, canon)] = row | {"evidence": ev, "n_targets": 1}

    if dt.get("missing"):
        return []
    for n in dt["timelineItems"]["nodes"]:
        if not n:
            continue
        kind, other = n["__typename"], n.get("duplicate") or {}
        if kind == "ClosedEvent" and n.get("duplicateOf"):
            if _same_repo(n["duplicateOf"], dt, repo, diag, "canon"):
                add(
                    x,
                    n["duplicateOf"]["number"],
                    "closed_event_duplicate_of",
                    n,
                    "closed as duplicate",
                )
        elif kind == "MarkedAsDuplicateEvent":
            if not _same_repo(other, dt, repo, diag, "canon"):
                continue
            y = other["number"]
            x_dup = x in recs and _dup_side(recs[x])
            y_dup = y in recs and _dup_side(recs[y])
            if x_dup == y_dup:
                diag["dup_target_ambiguous"] += 1
            else:
                src, canon = (x, y) if x_dup else (y, x)
                add(src, canon, "marked_as_duplicate_event", n, f"marked by {login(n)}")
        elif kind == "UnmarkedAsDuplicateEvent" and other.get("number"):
            found.pop((x, other["number"]), None)
            found.pop((other["number"], x), None)
    return list(found.values())
