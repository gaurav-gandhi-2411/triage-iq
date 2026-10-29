"""Tests for kaggle/kernels/dup_finetune/dup_finetune.py: leak assertion, loss mask, batching."""

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
    path = ROOT / "kaggle" / "kernels" / "dup_finetune" / "dup_finetune.py"
    spec = importlib.util.spec_from_file_location("dup_finetune_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assert_no_heldout(kernel) -> None:  # noqa: ANN001
    splits = pd.DataFrame({"number": [1, 2, 3], "split": ["train", "validation", "train"]})
    ok = pd.DataFrame({"anchor": [1], "pos": [3], "neg": [1]})
    kernel.assert_no_heldout(ok, splits)
    with pytest.raises(RuntimeError):
        kernel.assert_no_heldout(pd.DataFrame({"anchor": [1], "pos": [3], "neg": [2]}), splits)
    with pytest.raises(RuntimeError):  # an issue unknown to the split table is not train either
        kernel.assert_no_heldout(pd.DataFrame({"anchor": [1], "pos": [3], "neg": [9]}), splits)


def test_loss_masks_same_group_candidates(kernel) -> None:  # noqa: ANN001
    import torch

    a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pos = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    comp = torch.tensor([5, 5, 9])  # rows 0 and 1 are one duplicate group
    masked = kernel.mnrl_loss(a, pos, comp, comp, 20.0)
    plain = kernel.mnrl_loss(a, pos, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 20.0)
    assert float(masked) < float(plain)  # the identical positive no longer competes as a negative
    assert float(masked) == pytest.approx(0.0, abs=1e-6)


def test_loss_with_hard_negatives_masks_group_member(kernel) -> None:  # noqa: ANN001
    import torch

    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cand = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])  # 2 pos + 2 negatives
    comp_a = torch.tensor([5, 9])
    comp_c = torch.tensor([5, 9, 5, -7])  # negative 0 is in anchor 0's own group: masked
    got = kernel.mnrl_loss(a, cand, comp_a, comp_c, 20.0)
    unmasked = kernel.mnrl_loss(a, cand, comp_a, torch.tensor([5, 9, -1, -2]), 20.0)
    assert float(got) < float(unmasked)


def test_batches_cover_all_rows_deterministically(kernel) -> None:  # noqa: ANN001
    b1 = kernel.batches(70, np.random.default_rng(42))
    b2 = kernel.batches(70, np.random.default_rng(42))
    assert [list(x) for x in b1] == [list(x) for x in b2]
    assert sorted(np.concatenate(b1)) == list(range(70))
    assert all(len(x) >= 2 for x in kernel.batches(65, np.random.default_rng(1)))


class _FakeTok:
    """Whitespace tokenizer with the three methods production_text uses."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,  # noqa: FBT001, FBT002
        truncation: bool = False,  # noqa: FBT001, FBT002
        max_length: int | None = None,
    ) -> list[str]:
        toks = text.split()
        return toks[:max_length] if truncation and max_length is not None else toks

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:  # noqa: FBT001, FBT002
        return " ".join(ids)


def test_production_text_matches_the_embedding_kernel(kernel) -> None:  # noqa: ANN001
    """The recipe must stay identical to the #345 vectors' kernel (embed_bge.py)."""
    path = ROOT / "kaggle" / "kernels" / "embed_bge" / "embed_bge.py"
    spec = importlib.util.spec_from_file_location("embed_bge_kernel", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    title, body = "  A title ", " ".join(f"w{i}" for i in range(2000))
    got = kernel.production_text(title, body, _FakeTok())
    assert got == ref.production_text(title, body, _FakeTok())
    assert len(got.split()) <= kernel.MAX_SEQ_LENGTH
