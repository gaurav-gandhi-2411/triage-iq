"""Tests for kaggle/kernels/xenc_rerank/xenc_rerank.py: leak assertion, group loss, batching."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def kernel():  # noqa: ANN201
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = ROOT / "kaggle" / "kernels" / "xenc_rerank" / "xenc_rerank.py"
    spec = importlib.util.spec_from_file_location("xenc_rerank_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _groups(**over: list[int]) -> pd.DataFrame:
    cols = {"anchor": [1], "pos": [3], **{f"neg{i}": [1] for i in range(7)}}
    cols.update(over)
    return pd.DataFrame(cols)


def test_assert_no_heldout(kernel) -> None:  # noqa: ANN001
    splits = pd.DataFrame({"number": [1, 2, 3], "split": ["train", "validation", "train"]})
    kernel.assert_no_heldout(_groups(), splits)
    with pytest.raises(RuntimeError):
        kernel.assert_no_heldout(_groups(neg4=[2]), splits)  # validation issue as a negative
    with pytest.raises(RuntimeError):
        kernel.assert_no_heldout(_groups(pos=[9]), splits)  # unknown issue is not train either


def test_group_loss_prefers_positive_in_column_zero(kernel) -> None:  # noqa: ANN001
    import torch

    good = torch.tensor([[5.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
    bad = torch.tensor([[0.0, 5.0, 0.0], [0.0, 1.0, 4.0]])
    assert float(kernel.group_loss(good)) < float(kernel.group_loss(bad))
    assert float(kernel.group_loss(torch.zeros(1, 8))) == pytest.approx(np.log(8), abs=1e-6)


def test_group_batches_cover_all_groups_deterministically(kernel) -> None:  # noqa: ANN001
    a = kernel.group_batches(10, np.random.default_rng(42))
    b = kernel.group_batches(10, np.random.default_rng(42))
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert sorted(np.concatenate(a).tolist()) == list(range(10))
    assert [len(x) for x in a] == [4, 4, 2]


def test_side_text_recipe_and_cut(kernel) -> None:  # noqa: ANN001
    assert kernel.side_text("  T ", " body ") == "T. body"
    assert len(kernel.side_text("t", "x" * 10_000)) == kernel.SIDE_CHARS


def test_pinned_revisions_and_grid(kernel) -> None:  # noqa: ANN001
    assert kernel.MODELS["M"][1].startswith("7b0235") and kernel.MODELS["B"][1].startswith("2cfc18")
    assert list(kernel.CONFIGS) == ["X1", "X2", "X3", "X4"] and kernel.SEED == 42
