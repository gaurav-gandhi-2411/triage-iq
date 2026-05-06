"""Model loading for the triage API.

Loads all per-repo models once at startup and caches them as app state.
Models are intentionally not reloaded between requests.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_SLUGS = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}


@dataclass
class RepoBundle:
    classifier: Any
    detector: Any
    predictor: Any
    train_df: pd.DataFrame
    assistant: Any


class ModelStore:
    """Per-repo model bundles, loaded once at startup."""

    def __init__(self, bundles: dict[str, RepoBundle], start_time: float) -> None:
        self._bundles = bundles
        self.start_time = start_time

    @property
    def repos(self) -> list[str]:
        return list(self._bundles.keys())

    def get(self, repo: str) -> RepoBundle:
        if repo not in self._bundles:
            raise KeyError(f"Repo '{repo}' not loaded. Available: {self.repos}")
        return self._bundles[repo]

    @classmethod
    def load_all(
        cls,
        data_dir: Path | None = None,
        groq_api_key: str | None = None,
    ) -> "ModelStore":
        import time

        from triage_iq.models.triage import TriageAssistant

        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data"
        data_dir = Path(data_dir)
        models_dir = data_dir / "models"
        processed_dir = data_dir / "processed"

        key = (groq_api_key if groq_api_key else os.environ.get("GROQ_API_KEY", "")).strip()

        bundles: dict[str, RepoBundle] = {}
        for repo, slug in _REPO_SLUGS.items():
            try:
                logger.info("Loading models for %s …", repo)
                clf = _load_classifier(models_dir, slug)
                det = _load_detector(models_dir, slug)
                pred = _load_predictor(models_dir, slug)
                train_df = _load_train(processed_dir, slug)
                asst = TriageAssistant(
                    repo=repo,
                    classifier=clf,
                    detector=det,
                    predictor=pred,
                    train_df=train_df,
                    groq_api_key=key,
                )
                bundles[repo] = RepoBundle(clf, det, pred, train_df, asst)
                logger.info("Loaded %s — OK", repo)
            except Exception as exc:
                logger.error("Failed to load %s: %s — skipping", repo, exc)

        if not bundles:
            raise RuntimeError("No repo models could be loaded; check data/models/")

        return cls(bundles, start_time=time.monotonic())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_classifier(models_dir: Path, slug: str):
    from triage_iq.models.component_classifier import TFIDFComponentClassifier
    for prefix in ["component_classifier", "tfidf_classifier"]:
        p = models_dir / f"{prefix}_{slug}.pkl"
        if p.exists():
            return TFIDFComponentClassifier.load(str(p))
    raise FileNotFoundError(f"No classifier for {slug} in {models_dir}")


def _load_detector(models_dir: Path, slug: str):
    from triage_iq.models.duplicates import DuplicateDetector
    p = models_dir / f"dup_index_{slug}_bge"
    if p.exists():
        return DuplicateDetector.load(str(p))
    raise FileNotFoundError(f"Detector not found: {p}")


def _load_predictor(models_dir: Path, slug: str):
    from triage_iq.models.resolution import ResolutionTimePredictor
    p = models_dir / f"resolution_predictor_{slug}.pkl"
    if p.exists():
        return ResolutionTimePredictor.load(str(p))
    raise FileNotFoundError(f"Predictor not found: {p}")


def _load_train(processed_dir: Path, slug: str) -> pd.DataFrame:
    for suffix in ["temporal_train", "classifier_train", "train"]:
        p = processed_dir / f"{slug}_{suffix}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"No train split for {slug} in {processed_dir}")
