"""Phase 3b flag 2 loader: resolution_evidence_<slug>.parquet handling. Network-free."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from triage_iq.api import loader

OLD = "2020-01-01T00:00:00Z"


def _v1_frame() -> pd.DataFrame:
    """The v1 artifact shape: no merged_at / closing_comment_at."""
    return pd.DataFrame(
        [
            {
                "number": 2,
                "fix_prs": [{"number": 10, "title": "Fix A", "body_excerpt": "x"}],
                "closing_comment": "done",
                "has_evidence": True,
            }
        ]
    )


def test_loader_refuses_undated_artifact_with_one_warning(tmp_path: Path) -> None:
    _v1_frame().to_parquet(
        tmp_path / "resolution_evidence_kubernetes_kubernetes.parquet", index=False
    )
    records: list[logging.LogRecord] = []

    class _Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _Grab(logging.WARNING)
    loader.logger.addHandler(h)
    try:
        assert loader._load_resolution_evidence(tmp_path, "kubernetes_kubernetes") is None
    finally:
        loader.logger.removeHandler(h)
    assert [r.event for r in records if hasattr(r, "event")] == ["resolution_evidence_undated"]  # type: ignore[attr-defined]


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
        [
            {
                "number": 2,
                "fix_prs": [],
                "closing_comment": "done",
                "closing_comment_at": OLD,
                "has_evidence": True,
            }
        ]
    ).to_parquet(tmp_path / "resolution_evidence_kubernetes_kubernetes.parquet", index=False)
    ev = loader._load_resolution_evidence(tmp_path, "kubernetes_kubernetes")
    assert ev is not None and list(ev.records) == [2]


def test_loader_is_k8s_only() -> None:
    slugs = set(loader._RESOLUTION_EVIDENCE_SLUGS)  # lower-case: ruff SIM300 reads UPPER as const
    assert slugs == {"kubernetes_kubernetes"}
