from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qprc", _S / "qp_recheck_production_vectors.py")
rc = importlib.util.module_from_spec(_SPEC)
sys.modules["qprc"] = rc
_SPEC.loader.exec_module(rc)

PROD_NUMS = np.array([30, 10, 20])  # deliberately not sorted: alignment must go by number
PROD_VECS = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)


def test_align_by_issue_number_not_position() -> None:
    out = rc.align_vectors(np.array([10, 20, 30, 10]), PROD_NUMS, PROD_VECS)
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0, 1.0]


def test_missing_number_raises_and_names_it() -> None:
    with pytest.raises(KeyError, match="99"):
        rc.align_vectors(np.array([10, 99]), PROD_NUMS, PROD_VECS)


def test_missing_number_uses_fallback_row() -> None:
    fb = np.array([[7.0, 7.0], [8.0, 8.0]], dtype=np.float32)
    out = rc.align_vectors(np.array([10, 99]), PROD_NUMS, PROD_VECS, fallback=fb)
    assert out.tolist() == [[1.0, 0.0], [8.0, 8.0]]


def test_duplicate_production_numbers_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        rc.align_vectors(np.array([1]), np.array([1, 1]), PROD_VECS[:2])


def _fake_flat_ip(path: Path, vecs: np.ndarray) -> None:
    n, d = vecs.shape
    head = b"IxFI" + struct.pack("<iqqqBiQ", d, n, 1 << 20, 1 << 20, 1, 0, n * d * 4)
    assert len(head) == rc.FAISS_HEADER_BYTES
    path.write_bytes(head + vecs.astype("<f4").tobytes())


def test_read_flat_ip_vectors_roundtrip_and_size_check(tmp_path: Path) -> None:
    v = np.arange(6, dtype=np.float32).reshape(3, 2)
    _fake_flat_ip(tmp_path / "i.faiss", v)
    assert np.array_equal(rc.read_flat_ip_vectors(tmp_path / "i.faiss", 2, 3), v)
    with pytest.raises(ValueError, match="size"):
        rc.read_flat_ip_vectors(tmp_path / "i.faiss", 2, 4)
    (tmp_path / "bad.faiss").write_bytes(b"XXXX" + b"\0" * 60)
    with pytest.raises(ValueError, match="not a faiss"):
        rc.read_flat_ip_vectors(tmp_path / "bad.faiss", 2, 3)
