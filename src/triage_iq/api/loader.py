"""Model loading for the triage API.

Loads all per-repo models once at startup and caches them as app state.
Models are intentionally not reloaded between requests.
"""
from __future__ import annotations

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


def _load_conformal_adjustments(models_dir: Path) -> dict[str, dict]:
    """Load per-repo CQR conformal adjustments from JSON.

    Returns a dict keyed by repo canonical name (e.g. "microsoft/vscode").
    Returns empty dict and logs a warning if the file is missing or invalid.
    Falls back gracefully — callers must handle missing repos.
    """
    import json
    p = models_dir / "cqr_conformal_adjustments.json"
    if not p.exists():
        logger.warning(
            "cqr_conformal_adjustments.json not found at %s — "
            "resolution_interval_conformal will be None for all repos. "
            "Upload the file to GCS and rebuild the image to enable conformal intervals.",
            p,
        )
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Failed to parse cqr_conformal_adjustments.json: %s — falling back to raw intervals",
            exc,
        )
        return {}

    result: dict[str, dict] = {}
    repos_data = raw.get("repos", {})
    target_coverage = float(raw.get("target_coverage", 0.80))

    for repo, data in repos_data.items():
        # vscode has split sub-dicts; kubernetes has a flat dict
        if "40_60" in data:
            adj = data["40_60"]
        elif "30_70" in data:
            adj = data["30_70"]
        else:
            adj = data  # flat dict (kubernetes case)
        result[repo] = {
            "q_adjustment_hours": float(adj["q_adjustment_hours"]),
            "target_coverage": target_coverage,
            "empirical_coverage": float(adj["empirical_test_coverage"]),
            "coverage_ci95_lower": float(adj["coverage_ci95_lower"]),
            "coverage_ci95_upper": float(adj["coverage_ci95_upper"]),
        }
        logger.info(
            "Conformal adjustment loaded for %s: Q=%.4fh empirical_coverage=%.3f [%.3f, %.3f]",
            repo,
            result[repo]["q_adjustment_hours"],
            result[repo]["empirical_coverage"],
            result[repo]["coverage_ci95_lower"],
            result[repo]["coverage_ci95_upper"],
        )

    return result


class ModelStore:
    """Per-repo model bundles, loaded once at startup."""

    def __init__(
        self,
        bundles: dict[str, RepoBundle],
        start_time: float,
        conformal_adjustments: dict[str, dict] | None = None,
    ) -> None:
        self._bundles = bundles
        self.start_time = start_time
        self.conformal_adjustments: dict[str, dict] = conformal_adjustments if conformal_adjustments is not None else {}

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
        cache=None,
    ) -> ModelStore:
        import time

        from triage_iq.models.triage import TriageAssistant

        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data"
        data_dir = Path(data_dir)
        models_dir = data_dir / "models"
        processed_dir = data_dir / "processed"

        _check_manifest_drift(data_dir)  # warn-not-crash: image-baked artifact integrity

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
                    cache=cache,
                )
                bundles[repo] = RepoBundle(clf, det, pred, train_df, asst)
                logger.info("Loaded %s — OK", repo)
            except Exception as exc:
                logger.error("Failed to load %s: %s — skipping", repo, exc)

        if not bundles:
            raise RuntimeError("No repo models could be loaded; check data/models/")

        conformal = _load_conformal_adjustments(models_dir)
        return cls(bundles, start_time=time.monotonic(), conformal_adjustments=conformal)


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
    from triage_iq.models.similar_issues import SimilarIssueRetriever
    p = models_dir / f"dup_index_{slug}_bge"  # TODO(#3): GCS artifact rename pending. See https://github.com/gaurav-gandhi-2411/triage-iq/issues/3
    if p.exists():
        return SimilarIssueRetriever.load(str(p))
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


def _check_manifest_drift(data_dir: Path) -> None:
    """Compare image-baked artifacts against MANIFEST.sha256.

    Catches image-level corruption (stale artifact baked into the Docker layer).
    GCS-level drift is caught earlier by the deploy-gate step in deploy.yml.
    Warns with ARTIFACT_DRIFT: prefix and does NOT crash — the API stays up.
    """
    import hashlib

    manifest = data_dir / "models" / "MANIFEST.sha256"
    if not manifest.exists():
        logger.warning(
            "ARTIFACT_DRIFT: MANIFEST.sha256 not found at %s — skipping drift check. "
            "Run python scripts/publish_models.py to generate and commit the manifest.",
            manifest,
        )
        return

    lines = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    drifted: list[str] = []
    for line in lines:
        expected, rel_path = line.split("  ", 1)
        # rel_path is repo-relative (e.g. data/models/foo.pkl or data/processed/bar.parquet)
        filename = rel_path.removeprefix("data/")
        p = data_dir / filename
        if not p.exists():
            logger.warning("ARTIFACT_DRIFT: missing %s", rel_path)
            drifted.append(rel_path)
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            logger.warning(
                "ARTIFACT_DRIFT: %s — manifest=%s actual=%s",
                rel_path, expected[:16], actual[:16],
            )
            drifted.append(rel_path)

    if drifted:
        logger.warning(
            "ARTIFACT_DRIFT: %d/%d artifact(s) do not match MANIFEST.sha256. "
            "Re-deploy after running python scripts/publish_models.py.",
            len(drifted), len(lines),
        )
    else:
        logger.info("Manifest check: all %d artifacts verified OK", len(lines))
