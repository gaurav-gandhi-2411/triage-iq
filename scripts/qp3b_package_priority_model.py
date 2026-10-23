"""Package the P4 priority model as a src-loadable artifact (Phase 3b flag 1).

The P4 joblibs pickle `qp1_tfidf_features.Featurizer` (scripts/), which src cannot import. This
loads them, keeps only the fitted TfidfVectorizers + LogisticRegression + the field-splitting
config, and writes a joblib of plain sklearn objects and data. With --verify it also proves the
packaged predictor reproduces P4's predict_proba on the saved test issues (needs the D: data) and
writes the proof report. The output is NOT committed: data/models artifacts are image-baked and
pinned by MANIFEST.sha256 (GG decides the cutover).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # unpickling needs qp1_tfidf_features
sys.path.insert(0, str(ROOT / "src"))

from triage_iq.models.priority import (  # noqa: E402
    BODY_CHARS,
    FORMAT_VERSION,
    PRIORITY_CLASSES,
    PriorityModel,
    file_sha256,
)

# Field function name (qp1_tfidf_features) -> packaged field name.
_FIELD_NAMES = {"field_title": "title", "field_body": "body", "field_full": "full"}
DATA_ROOT = Path(r"D:\triageiq-data")
P4_DIR = DATA_ROOT / "models" / "kubernetes__kubernetes" / "priority" / "p4"
DEFAULT_OUT = ROOT / "data" / "models" / "priority_model_kubernetes_kubernetes.joblib"


def package(featurizer_path: Path, model_path: Path) -> dict:
    """Return the packaged artifact dict for the P4 joblibs (needs scripts/ on sys.path)."""
    feat, lr = joblib.load(featurizer_path), joblib.load(model_path)
    return {
        "format_version": FORMAT_VERSION,
        "classes": list(PRIORITY_CLASSES),
        "body_chars": BODY_CHARS,
        "blocks": [{"vectorizer": v, "field": _FIELD_NAMES[f.__name__]} for v, f in feat.blocks],
        "lr": lr,
        "source_sha256": {
            "featurizer.joblib": file_sha256(featurizer_path),
            "model.joblib": file_sha256(model_path),
        },
    }


def verify(p4_dir: Path, packaged: Path, data_root: Path) -> dict:
    """Max abs diff between P4's predict_proba and the packaged predictor on the test issues."""
    import pandas as pd

    feat, lr = joblib.load(p4_dir / "featurizer.joblib"), joblib.load(p4_dir / "model.joblib")
    saved = pd.read_parquet(p4_dir / "test.parquet")  # P4's own saved test probabilities
    iss = pd.read_parquet(data_root / "derived_v2" / "kubernetes__kubernetes" / "issues.parquet")
    iss = iss.drop_duplicates("number").set_index("number").loc[saved["number"]]
    texts = [
        f"{t}\n{b[:BODY_CHARS]}"
        for t, b in zip(iss["title"].fillna(""), iss["body"].fillna(""), strict=True)
    ]
    orig = np.zeros((len(texts), len(PRIORITY_CLASSES)))
    cols = [PRIORITY_CLASSES.index(c) for c in lr.classes_]
    orig[:, cols] = lr.predict_proba(feat.transform(pd.Series(texts)))
    new = PriorityModel.load(packaged).proba_texts(texts)
    return {
        "n": len(texts),
        "max_abs_diff_vs_original_predict_proba": float(np.abs(orig - new).max()),
        "max_abs_diff_vs_saved_test_parquet": float(
            np.abs(saved[list(PRIORITY_CLASSES)].to_numpy() - new).max()
        ),
        "argmax_identical_to_original": bool((orig.argmax(1) == new.argmax(1)).all()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Package the P4 priority model for src")
    ap.add_argument("--p4-dir", type=Path, default=P4_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", type=Path, help="data root (e.g. D:/triageiq-data); writes proof")
    ap.add_argument(
        "--report", type=Path, default=ROOT / "reports" / "qp3b_priority_packaging_2026-10.json"
    )
    a = ap.parse_args(argv)
    fp, mp = a.p4_dir / "featurizer.joblib", a.p4_dir / "model.joblib"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package(fp, mp), a.out)
    print(f"wrote {a.out} sha256={file_sha256(a.out)}")
    if a.verify:
        proof = verify(a.p4_dir, a.out, a.verify) | {
            "source_sha256": {
                "featurizer.joblib": file_sha256(fp),
                "model.joblib": file_sha256(mp),
            },
            "packaged_artifact": a.out.name,
            "packaged_sha256": file_sha256(a.out),
        }
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(proof, indent=1), encoding="utf-8")
        print(json.dumps(proof, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
