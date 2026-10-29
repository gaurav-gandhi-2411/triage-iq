"""Bit-identical check: retrain-evaluation scoring vs the serving path for a family_v2 bundle.

Offline side: the retrain window's text recipe (title + "\\n" + raw body[:4000], the text
scripts/vscode_retrain_lib.assemble builds) scored in one batch with qp1_tfidf_lr.proba_full, the
call vscode_retrain_lib.bundle_proba makes. Serving side: TriageAssistant._collect_signals with the
FamilyV2Classifier wrapper, fed an issue shaped as POST /triage shapes it with
TRIAGE_CLEAN_REQUEST_BODY=1 (body_clean = clean_text(body), body_raw = body). Retriever and
resolution predictor are absent (their failures are caught by _collect_signals); no LLM is called.

Usage (the bundle, window and sha are arguments so the same check re-runs on any later bundle):
  python scripts/verify_family_v2_serving.py --bundle B.joblib --window RUN/window.parquet \
      --expected-sha256 HEX --issues derived_v2/microsoft__vscode/issues.parquet --out OUT.json
Exit 0 iff top-3 labels and order agree on 100% of the sample and max |prob diff| <= --tol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 42


def offline_scores(bundle: dict[str, Any], d: pd.DataFrame) -> np.ndarray:
    """Retrain-evaluation scoring: recipe text, one batch, probabilities in bundle class order."""
    import qp1_tfidf_lr as t1

    text = d["title"].fillna("") + "\n" + d["body"].fillna("").str.slice(0, 4000)
    classes = [str(c) for c in bundle["classes"]]
    return t1.proba_full(bundle["model"], bundle["vectorizer"].transform(text), classes)


def serving_top3(bundle: dict[str, Any], d: pd.DataFrame) -> tuple[list[list[dict]], np.ndarray]:
    """classifier_top3 per issue from the real _collect_signals, plus the wrapper's full vectors."""
    from triage_iq.data.preprocess import clean_text
    from triage_iq.models.component_classifier import FamilyV2Classifier
    from triage_iq.models.triage import TriageAssistant

    clf = FamilyV2Classifier(bundle)
    asst = TriageAssistant(
        repo="microsoft/vscode", classifier=clf, detector=None, predictor=None,
        train_df=pd.DataFrame(), groq_api_key="dummy",
    )  # fmt: skip
    tops, full = [], []
    for r in d.itertuples():
        body = r.body if isinstance(r.body, str) else ""
        issue = pd.Series({
            "number": int(r.number), "title": r.title, "body_clean": clean_text(body)[0],
            "body_raw": body, "created_at": pd.Timestamp("2026-10-22", tz="UTC"),
        })  # fmt: skip
        tops.append(asst._collect_signals(issue)["classifier_top3"])
        full.append(
            clf.predict_proba_calibrated(
                pd.Series([FamilyV2Classifier.serving_text(r.title, body)])
            )[0]
        )
    return tops, np.array(full)


def compare(bundle: dict[str, Any], d: pd.DataFrame) -> dict[str, Any]:
    """Top-3 label/order agreement and max abs probability differences."""
    classes = [str(c) for c in bundle["classes"]]
    p_off = offline_scores(bundle, d)
    tops, p_srv = serving_top3(bundle, d)
    off_idx = np.argsort(-p_off, axis=1, kind="stable")[:, :3]
    label_ok = order_ok = 0
    top_diff = 0.0
    for i, t in enumerate(tops):
        off_labels = [classes[j] for j in off_idx[i]]
        got = [e["label"] for e in t]
        label_ok += sorted(got) == sorted(off_labels)
        order_ok += got == off_labels
        top_diff = max(
            top_diff,
            *(abs(e["confidence"] - p_off[i, j]) for e, j in zip(t, off_idx[i], strict=True)),
        )
    return {
        "n": len(d), "top3_labels_identical": label_ok, "top3_labels_and_order_identical": order_ok,
        "max_abs_diff_top3_confidence": top_diff,
        "max_abs_diff_full_vector": float(np.abs(p_srv - p_off).max()),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--window", type=Path, required=True, help="retrain window.parquet (split col)")
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--issues", type=Path, required=True, help="derived issues.parquet (raw body)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args(argv)
    logging.disable(logging.CRITICAL)  # expected retriever/predictor-absent warnings x n
    sha = hashlib.sha256(a.bundle.read_bytes()).hexdigest()
    if sha != a.expected_sha256.lower():
        print(f"bundle sha256 {sha} != expected {a.expected_sha256}", file=sys.stderr)
        return 2
    bundle = joblib.load(a.bundle)
    w = pd.read_parquet(a.window, columns=["number", "split"])
    ho = w[w["split"] == "holdout"].sort_values("number")
    iss = pd.read_parquet(a.issues, columns=["number", "title", "body"])
    d = ho.merge(iss, on="number").reset_index(drop=True)
    if len(d) < a.n:
        print(f"only {len(d)} holdout issues < n={a.n}", file=sys.stderr)
        return 2
    d = d.iloc[np.sort(np.random.default_rng(SEED).choice(len(d), a.n, replace=False))]
    res = compare(bundle, d.reset_index(drop=True))
    res |= {
        "bundle_sha256": sha,
        "seed": SEED,
        "tol": a.tol,
        "issue_numbers": [int(x) for x in d["number"]],
    }
    ok = (
        res["top3_labels_and_order_identical"] == res["n"]
        and max(res["max_abs_diff_top3_confidence"], res["max_abs_diff_full_vector"]) <= a.tol
    )
    res["pass"] = bool(ok)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "issue_numbers"}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
