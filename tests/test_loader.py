"""Tests for the _load_detector loader branch logic.

Guards the two branches introduced in ADR-0010:
  - fine-tuned artifacts present → loads fine-tuned SimilarIssueRetriever (source="finetuned")
  - only baseline artifacts present → falls back to baseline (source="baseline")

This test suite exists because the loader branch failed silently in the W3 judge eval:
the judge run used the baseline retriever even though fine-tuned artifacts were on disk,
because the eval process launched before the loader change was in effect and there was no
observable evidence of which model was actually serving.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage_iq.api.loader import _load_detector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finetuned_artifacts(models_dir: Path, alias: str) -> tuple[Path, Path]:
    """Create the directory structure that triggers the fine-tuned branch."""
    ft_idx = models_dir / f"bge_finetuned_{alias}_index"
    ft_idx.mkdir(parents=True)
    (ft_idx / "index.faiss").touch()
    np.save(str(ft_idx / "numbers.npy"), np.array([1, 2, 3], dtype=np.int64))
    (ft_idx / "texts.json").write_text(json.dumps(["text_a", "text_b", "text_c"]))
    ft_model = models_dir / "bge_finetuned_combined"
    ft_model.mkdir()
    return ft_idx, ft_model


def _make_baseline_artifact(models_dir: Path, slug: str) -> Path:
    """Create the directory structure that triggers the baseline fallback."""
    baseline = models_dir / f"dup_index_{slug}_bge"
    baseline.mkdir(parents=True)
    return baseline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectorLoader:

    def test_loads_finetuned_when_artifacts_present(self, tmp_path):
        """Fine-tuned index + combined model present → retriever.source == 'finetuned'."""
        ft_idx, ft_model = _make_finetuned_artifacts(tmp_path, "vsc")

        mock_index = MagicMock()
        with (
            patch("triage_iq.models.similar_issues.faiss.read_index", return_value=mock_index),
            patch("triage_iq.models.similar_issues.SentenceTransformer", return_value=MagicMock()),
        ):
            result = _load_detector(tmp_path, "microsoft_vscode")

        assert result.source == "finetuned", (
            f"Expected source='finetuned', got source='{result.source}'. "
            "Loader is not picking up the fine-tuned artifacts."
        )
        assert str(ft_idx) in result.index_dir
        assert result.issue_numbers is not None
        assert len(result.issue_numbers) == 3

    def test_finetuned_branch_uses_correct_repo_name(self, tmp_path):
        """load_finetuned receives the canonical repo slug, not the alias."""
        _make_finetuned_artifacts(tmp_path, "k8s")

        with (
            patch("triage_iq.models.similar_issues.faiss.read_index", return_value=MagicMock()),
            patch("triage_iq.models.similar_issues.SentenceTransformer", return_value=MagicMock()),
        ):
            result = _load_detector(tmp_path, "kubernetes_kubernetes")

        assert result.source == "finetuned"
        assert result.repo == "kubernetes_kubernetes"

    def test_falls_back_to_baseline_when_no_finetuned_artifacts(self, tmp_path):
        """No fine-tuned artifacts → falls back to baseline retriever (source='baseline')."""
        baseline = _make_baseline_artifact(tmp_path, "microsoft_vscode")

        fake_retriever = MagicMock()
        fake_retriever.source = "baseline"
        fake_retriever.index_dir = str(baseline)

        # SimilarIssueRetriever is imported inside _load_detector — patch on the class itself
        with patch(
            "triage_iq.models.similar_issues.SimilarIssueRetriever.load",
            return_value=fake_retriever,
        ) as mock_load:
            result = _load_detector(tmp_path, "microsoft_vscode")

        mock_load.assert_called_once_with(str(baseline))
        assert result.source == "baseline", (
            f"Expected source='baseline', got source='{result.source}'. "
            "Fallback to baseline not working correctly."
        )

    def test_missing_numbers_npy_skips_finetuned(self, tmp_path):
        """Fine-tuned index directory exists but numbers.npy is absent → baseline fallback."""
        # Create index dir WITHOUT numbers.npy
        ft_idx = tmp_path / "bge_finetuned_vsc_index"
        ft_idx.mkdir()
        (ft_idx / "index.faiss").touch()
        # Note: numbers.npy intentionally absent
        (tmp_path / "bge_finetuned_combined").mkdir()

        baseline = _make_baseline_artifact(tmp_path, "microsoft_vscode")
        fake_retriever = MagicMock()
        fake_retriever.source = "baseline"
        fake_retriever.index_dir = str(baseline)

        with patch(
            "triage_iq.models.similar_issues.SimilarIssueRetriever.load",
            return_value=fake_retriever,
        ):
            result = _load_detector(tmp_path, "microsoft_vscode")

        assert result.source == "baseline"

    def test_raises_when_no_artifacts_at_all(self, tmp_path):
        """Neither fine-tuned nor baseline artifacts present → FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Detector not found"):
            _load_detector(tmp_path, "microsoft_vscode")
