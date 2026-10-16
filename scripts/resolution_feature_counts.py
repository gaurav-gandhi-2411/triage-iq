"""Count the input features of each served resolution predictor (reads data/models, writes a report).

README #151's pipeline diagram quotes the feature count; this makes it re-checkable
(scripts/readme_numbers_check.py reads reports/resolution_feature_counts.json). Run from a checkout that holds
data/models (gitignored): python scripts/resolution_feature_counts.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage_iq.models.resolution import ResolutionTimePredictor  # noqa: E402

OUT = ROOT / "reports" / "resolution_feature_counts.json"
REPOS = {"microsoft/vscode": "microsoft_vscode", "kubernetes/kubernetes": "kubernetes_kubernetes"}


def main() -> None:
    result: dict = {}
    for repo, slug in REPOS.items():
        path = ROOT / "data" / "models" / f"resolution_predictor_{slug}.pkl"
        pred = ResolutionTimePredictor.load(str(path))
        names = list(pred.feature_names)
        pca = getattr(pred, "pca", None)
        result[repo] = {
            "n_features": len(names),
            "n_embedding_features": sum(n.startswith("emb_") for n in names),
            "n_structured_features": sum(not n.startswith("emb_") for n in names),
            "has_pca": pca is not None,
            # PCA.n_samples_ is the number of training rows the embedding PCA (and so the predictor) was fitted on
            "pca_n_samples": int(pca.n_samples_) if pca is not None else None,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
