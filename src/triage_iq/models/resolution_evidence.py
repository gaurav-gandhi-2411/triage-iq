"""Resolution evidence for the prompt (Phase 3b flag 2, docs/QP3B_DESIGN_2026-10.md).

An offline artifact (scripts/qp3b_build_evidence.py) maps an issue number to its audited fix PRs
(title + short excerpt) and the last non-bot closing comment. For the top-3 retrieved neighbours
this module renders one compact block for the user prompt. Pure formatting + one parquet read;
no network. Caps are hard: the whole block never exceeds EVIDENCE_BLOCK_MAX_CHARS.

The evidence text is third-party content (PR bodies, comments) placed in a prompt: it is
whitespace-collapsed, double quotes are neutralised, and "#<digits>" references inside it are
rewritten so the only "#N" tokens in the block are the neighbours' own numbers (the grounding
metric scores "#N" citations against the retrieved set; a copied "Fixes #123" would be an
ungrounded citation).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

EVIDENCE_TOP_NEIGHBOURS = 3  # design: top-3 retrieved neighbours
EVIDENCE_BLOCK_MAX_CHARS = 900  # design: total evidence block <= ~900 chars (hard cap here)
EVIDENCE_ITEM_MAX_CHARS = 270  # per neighbour; 3 * 270 + header (<= 80) + 2 newlines <= 900
EVIDENCE_HEADER = "--- RESOLUTION EVIDENCE (how the top similar issues were resolved) ---"
_FIX_SHARE = 0.6  # share of an item's text budget for the fix PR when a comment is also present
_TITLE_MAX = 90
_WS = re.compile(r"\s+")
_HASH_REF = re.compile(r"#(\d+)")


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidence for one issue.

    fix_prs: (number, title, excerpt, merged_at ISO UTC or ""); closing_comment with its
    created_at. An empty date means unknown and is dropped by `before` (fail closed).
    """

    fix_prs: tuple[tuple[int, str, str, str], ...] = ()
    closing_comment: str = ""
    closing_comment_at: str = ""

    @property
    def has_evidence(self) -> bool:
        """True when there is a fix PR (with a title) or a closing comment to cite."""
        return bool(self.fix_prs) or bool(self.closing_comment)

    def before(self, as_of: datetime) -> EvidenceRecord:
        """The items dated strictly before `as_of`; undated items are dropped."""
        t = _utc(as_of)
        prs = tuple(p for p in self.fix_prs if p[3] and _utc(p[3]) < t)
        keep_comment = bool(self.closing_comment_at) and _utc(self.closing_comment_at) < t
        return replace(
            self,
            fix_prs=prs,
            closing_comment=self.closing_comment if keep_comment else "",
            closing_comment_at=self.closing_comment_at if keep_comment else "",
        )


def _utc(value: Any) -> pd.Timestamp:
    """tz-aware UTC timestamp (naive input is taken as UTC)."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _clean(text: str) -> str:
    """One line, no double quotes, no '#<digits>' tokens (see module docstring)."""
    return _HASH_REF.sub(r"\1", _WS.sub(" ", text or "")).replace('"', "'").strip()


def _cut(text: str, n: int) -> str:
    """Cut to at most n chars, ending with an ellipsis when something was removed."""
    if n <= 0:
        return ""
    return text if len(text) <= n else text[: max(n - 1, 0)].rstrip() + "…"


def _fix_part(rec: EvidenceRecord, budget: int) -> str:
    """'fix PR 123 "title": excerpt' for the first audited fix PR, within `budget` chars."""
    if not rec.fix_prs or budget <= 0:
        return ""
    number, title, excerpt, _ = rec.fix_prs[0]
    head = f'fix PR {number} "{_cut(_clean(title), min(_TITLE_MAX, budget))}"'
    room = budget - len(head) - 2
    ex = _cut(_clean(excerpt), room) if room >= 20 else ""
    return _cut(f"{head}: {ex}" if ex else head, budget)


def _comment_part(rec: EvidenceRecord, budget: int) -> str:
    """'closing comment "..."' within `budget` chars ('' when too tight to say anything)."""
    if not rec.closing_comment or budget < 24:
        return ""
    label = "closing comment "
    return f'{label}"{_cut(_clean(rec.closing_comment), budget - len(label) - 2)}"'


def render_item(number: int, rec: EvidenceRecord, max_chars: int = EVIDENCE_ITEM_MAX_CHARS) -> str:
    """One neighbour line '#N: ...' within max_chars; '' when the record has no evidence."""
    if not rec.has_evidence:
        return ""
    budget = max_chars - len(f"#{number}: ")
    has_fix, has_comment = bool(rec.fix_prs), bool(rec.closing_comment)
    if has_fix and has_comment:
        fix_budget = int(budget * _FIX_SHARE)
        parts = [_fix_part(rec, fix_budget), _comment_part(rec, budget - fix_budget - 2)]
    elif has_fix:
        parts = [_fix_part(rec, budget)]
    else:
        parts = [_comment_part(rec, budget)]
    body = "; ".join(p for p in parts if p)
    return f"#{number}: {body}" if body else ""


class ResolutionEvidence:
    """Number -> EvidenceRecord lookup plus the block renderer.

    `dated` is False for an artifact without item dates (v1): block_for then refuses to run,
    because an as-of filter cannot be applied to it and silently ignoring it would leak evidence
    from after the query.
    """

    def __init__(self, records: Mapping[int, EvidenceRecord], dated: bool = True) -> None:
        self.records = dict(records)
        self.dated = dated

    @classmethod
    def load(cls, path: Path) -> ResolutionEvidence:
        """Read the parquet artifact (number, fix_prs[.merged_at], closing_comment[_at], ...).

        Only rows with evidence are kept. `dated` is True only when the artifact carries item
        dates (v2). Raises on a missing core column or unreadable file.
        """
        df = pd.read_parquet(path)
        dated = "closing_comment_at" in df.columns
        recs: dict[int, EvidenceRecord] = {}
        for r in df[df["has_evidence"].astype(bool)].itertuples(index=False):
            prs = tuple(
                (
                    int(p["number"]),
                    str(p["title"] or ""),
                    str(p["body_excerpt"] or ""),
                    str(p.get("merged_at") or ""),
                )
                for p in (r.fix_prs if r.fix_prs is not None else [])
                if p["title"]
            )
            rec = EvidenceRecord(
                prs,
                str(r.closing_comment or ""),
                str(getattr(r, "closing_comment_at", "") or ""),
            )
            if rec.has_evidence:
                recs[int(r.number)] = rec
        return cls(recs, dated=dated)

    def block_for(
        self, neighbours: Sequence[Mapping[str, Any]], as_of: datetime
    ) -> tuple[str, list[int]]:
        """(block text, neighbour numbers cited) for the top-3 neighbours; ('', []) if none.

        Only evidence items dated strictly before `as_of` (the query issue's creation time) are
        used; a neighbour with no remaining item contributes nothing. Raises ValueError on an
        undated artifact rather than ignoring `as_of`. The block is header + one line per
        neighbour that has evidence, never longer than EVIDENCE_BLOCK_MAX_CHARS.
        """
        if not self.dated:
            raise ValueError("evidence artifact has no item dates: cannot apply the as-of filter")
        lines: list[str] = []
        used: list[int] = []
        for s in neighbours[:EVIDENCE_TOP_NEIGHBOURS]:
            n = int(s["number"])
            rec = self.records.get(n)
            rec = rec.before(as_of) if rec is not None else None
            line = render_item(n, rec) if rec is not None else ""
            if line:
                lines.append(line)
                used.append(n)
        if not lines:
            return "", []
        block = "\n".join([EVIDENCE_HEADER, *lines])
        return _cut(block, EVIDENCE_BLOCK_MAX_CHARS), used
