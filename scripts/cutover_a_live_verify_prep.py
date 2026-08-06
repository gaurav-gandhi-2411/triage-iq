"""CUTOVER A live-verification prep: pick one k8s and one vscode D1-eval-set query where the
per-repo query-instruction setting (ADR-0040) actually changes the top-5 result set (not just
scores), plus compute the CURRENT (unchanged) classifier's prediction for the same two issues
offline -- so the live /triage response can be checked against known-correct expectations rather
than just "did it return 200."

Reads:  data/models/dup_index_{repo}_bge/            (the NOW-DEPLOYED, just-published index)
        data/models/component_classifier_{repo}.pkl  (unchanged by this cutover)
        reports/d1_eval_set_{k8s_related,vscode_duplicate}.json
Writes: reports/cutover_a_live_verify_expected.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.component_classifier import load_classifier  # noqa: E402
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")
REPORTS = Path("reports")


def query_text(row: dict) -> str:
    return f"{row['query_title']}. {row.get('query_body', '')}"


def find_divergent_case(
    detector: SimilarIssueRetriever, pairs: list[dict], repo_default: bool
) -> dict | None:
    """Find a query where instruction ON vs OFF top-5 SETS actually differ (not just order)."""
    live_numbers = {int(n) for n in detector.issue_numbers}
    for row in pairs:
        if row["query_number"] not in live_numbers or row["original_number"] not in live_numbers:
            continue
        qt = query_text(row)
        qnum = int(row["query_number"])
        top5_on = {
            r["number"]
            for r in detector.retrieve(qt, k=5, exclude_number=qnum, apply_query_instruction=True)
        }
        top5_off = {
            r["number"]
            for r in detector.retrieve(qt, k=5, exclude_number=qnum, apply_query_instruction=False)
        }
        if top5_on != top5_off:
            top5_default = detector.retrieve(
                qt, k=5, exclude_number=qnum
            )  # None -> per-repo default
            return {
                "query_number": qnum,
                "query_title": row["query_title"],
                "query_body": row.get("query_body", ""),
                "top5_with_instruction": sorted(top5_on),
                "top5_without_instruction": sorted(top5_off),
                "top5_via_per_repo_default": [r["number"] for r in top5_default],
                "repo_default_is_instruction_on": repo_default,
                "expected_top5_scores_default": [
                    {"number": r["number"], "score": round(r["score"], 4)} for r in top5_default
                ],
            }
    return None


def main() -> None:
    out: dict = {}

    log.info("=== k8s (instruction should be ON by default, ADR-0040) ===")
    k8s_detector = SimilarIssueRetriever.load(
        str(MODELS_DIR / "dup_index_kubernetes_kubernetes_bge")
    )
    k8s_pairs = json.loads((REPORTS / "d1_eval_set_k8s_related.json").read_text(encoding="utf-8"))
    k8s_case = find_divergent_case(k8s_detector, k8s_pairs, repo_default=True)
    out["kubernetes_kubernetes"] = k8s_case
    log.info(json.dumps(k8s_case, indent=2)[:2000])

    log.info("=== vscode (instruction should be OFF by default, ADR-0040) ===")
    vsc_detector = SimilarIssueRetriever.load(str(MODELS_DIR / "dup_index_microsoft_vscode_bge"))
    vsc_pairs = json.loads(
        (REPORTS / "d1_eval_set_vscode_duplicate.json").read_text(encoding="utf-8")
    )
    vsc_case = find_divergent_case(vsc_detector, vsc_pairs, repo_default=False)
    out["microsoft_vscode"] = vsc_case
    log.info(json.dumps(vsc_case, indent=2)[:2000])

    # Classifier predictions (unchanged model, ADR-0036) for the SAME two query issues.
    for repo_slug, case, repo_name in [
        ("microsoft_vscode", vsc_case, "microsoft/vscode"),
        ("kubernetes_kubernetes", k8s_case, "kubernetes/kubernetes"),
    ]:
        if case is None:
            continue
        clf = load_classifier(str(MODELS_DIR), repo_slug)
        text = pd.Series([f"{case['query_title']}. {case['query_body']}"])
        proba = clf.predict_proba_calibrated(text)
        classes = clf.classes_()
        import numpy as np

        top_idx = np.argsort(proba[0])[::-1][:3]
        top3 = [{"label": classes[i], "confidence": float(proba[0][i])} for i in top_idx]
        case["expected_classifier_top3"] = top3
        log.info("[%s] expected classifier top3: %s", repo_name, top3)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cutover_a_live_verify_expected.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    log.info("Wrote reports/cutover_a_live_verify_expected.json")


if __name__ == "__main__":
    main()
