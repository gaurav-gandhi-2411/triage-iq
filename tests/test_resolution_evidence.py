"""Phase 3b flag 2 plumbing: evidence rendering caps + artifact loader. Network-free."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from triage_iq.api import loader
from triage_iq.models.resolution_evidence import (
    EVIDENCE_BLOCK_MAX_CHARS,
    EVIDENCE_HEADER,
    EVIDENCE_ITEM_MAX_CHARS,
    EvidenceRecord,
    ResolutionEvidence,
    render_item,
)

LONG = "word " * 400


def _rec(prs=(), comment: str = "") -> EvidenceRecord:  # noqa: ANN001
    return EvidenceRecord(tuple(prs), comment)


def test_item_without_evidence_is_empty() -> None:
    assert render_item(5, _rec()) == ""


def test_fix_item_names_pr_title_and_excerpt() -> None:
    line = render_item(7, _rec([(900, "Fix kubelet crash", "Handles nil pod in sync loop")]))
    assert line == '#7: fix PR 900 "Fix kubelet crash": Handles nil pod in sync loop'


def test_comment_only_item() -> None:
    line = render_item(7, _rec(comment="Fixed by bumping etcd."))
    assert line == '#7: closing comment "Fixed by bumping etcd."'


def test_item_with_both_stays_within_cap_and_keeps_both_parts() -> None:
    line = render_item(1234, _rec([(99, "T" * 300, LONG)], LONG))
    assert len(line) <= EVIDENCE_ITEM_MAX_CHARS
    assert "fix PR 99" in line and "closing comment" in line


def test_hash_references_and_quotes_are_neutralised() -> None:
    line = render_item(7, _rec([(9, 'Fixes #123 "x"', "see #456 and #78")], "dup of #999"))
    assert line.count("#") == 1  # only the neighbour's own "#7"
    assert '"x"' not in line  # inner double quotes replaced


def test_block_is_hard_capped_and_header_first() -> None:
    ev = ResolutionEvidence({n: _rec([(n, "T" * 300, LONG)], LONG) for n in (1, 2, 3, 4)})
    block, used = ev.block_for([{"number": n} for n in (1, 2, 3, 4)])
    assert len(block) <= EVIDENCE_BLOCK_MAX_CHARS
    assert block.startswith(EVIDENCE_HEADER)
    assert used == [1, 2, 3]  # only the top-3 neighbours are considered
    assert block.count("\n#") == 3  # no line was cut off by the global cap


def test_block_skips_neighbours_without_evidence_and_is_empty_when_none() -> None:
    ev = ResolutionEvidence({2: _rec(comment="done")})
    block, used = ev.block_for([{"number": 1}, {"number": 2}, {"number": 3}])
    assert used == [2] and "#1" not in block
    assert ev.block_for([{"number": 1}, {"number": 3}]) == ("", [])
    assert ev.block_for([]) == ("", [])


def test_load_keeps_only_rows_with_evidence(tmp_path: Path) -> None:
    p = tmp_path / "e.parquet"
    pd.DataFrame(
        [
            {
                "number": 1,
                "fix_prs": [{"number": 10, "title": "Fix A", "body_excerpt": "x"}],
                "closing_comment": "",
                "has_evidence": True,
            },
            {"number": 2, "fix_prs": [], "closing_comment": "done", "has_evidence": True},
            {"number": 3, "fix_prs": [], "closing_comment": "", "has_evidence": False},
        ]
    ).to_parquet(p, index=False)
    ev = ResolutionEvidence.load(p)
    assert sorted(ev.records) == [1, 2]
    assert ev.records[1].fix_prs == ((10, "Fix A", "x"),)
    assert ev.records[2].closing_comment == "done"


# --- loader ---------------------------------------------------------------------------------


def test_loader_missing_artifact_warns_once_and_returns_none(tmp_path: Path) -> None:
    records: list[logging.LogRecord] = []

    class _Grab(logging.Handler):  # direct handler: other tests may cut propagation to root
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _Grab(logging.WARNING)
    loader.logger.addHandler(h)
    try:
        assert loader._load_resolution_evidence(tmp_path, "kubernetes_kubernetes") is None
    finally:
        loader.logger.removeHandler(h)
    warned = [r for r in records if "TRIAGE_RESOLUTION_EVIDENCE" in r.getMessage()]
    assert len(warned) == 1
    assert warned[0].event == "resolution_evidence_missing"  # type: ignore[attr-defined]


def test_loader_unreadable_artifact_returns_none(tmp_path: Path) -> None:
    (tmp_path / "resolution_evidence_kubernetes_kubernetes.parquet").write_bytes(b"nope")
    assert loader._load_resolution_evidence(tmp_path, "kubernetes_kubernetes") is None


def test_loader_reads_the_slug_named_artifact(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"number": 2, "fix_prs": [], "closing_comment": "done", "has_evidence": True}]
    ).to_parquet(tmp_path / "resolution_evidence_kubernetes_kubernetes.parquet", index=False)
    ev = loader._load_resolution_evidence(tmp_path, "kubernetes_kubernetes")
    assert ev is not None and list(ev.records) == [2]


def test_loader_is_k8s_only() -> None:
    slugs = set(loader._RESOLUTION_EVIDENCE_SLUGS)  # lower-case: ruff SIM300 reads UPPER as const
    assert slugs == {"kubernetes_kubernetes"}
