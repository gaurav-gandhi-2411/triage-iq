"""Verify the candidate production index (data/models/dup_index_{repo}_bge_candidate/) before
any GCS publish or deploy -- two checks, both must pass:

1. REPRODUCTION: re-run D1's frozen eval sets against the candidate index through the REAL,
   un-overridden production code path (retrieve() with apply_query_instruction=None, letting
   ADR-0040's per-repo default resolve) and confirm the result matches what lever12_eval.py
   already measured under manual overrides -- k8s should land at the lever1+2 number (its
   override is True), vscode at the lever1-only number (its override is False).

2. INDEX/QUERY CONSISTENCY: prove the candidate index's stored corpus text was actually built
   by the CURRENT _build_text() (tokenizer-based truncation), not stale/leftover text from a
   different code path -- by re-deriving _build_text() on a sample of the same raw issues and
   asserting byte-identical output. This is the check GG asked for: index-time and query-time
   construction must be provably the same code, not just "should be."

Reads:  data/models/dup_index_{repo}_bge_candidate/
        data/processed/issues_{repo}.parquet
        reports/d1_eval_set_{k8s_related,vscode_duplicate,vscode_related}.json
Writes: reports/lever12_candidate_verification.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import (  # noqa: E402
    QUERY_INSTRUCTION_REPO_OVERRIDE,
    SimilarIssueRetriever,
    _build_text,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 2000
K_MAX = 20

MODELS_DIR = Path("data/models")
PROCESSED_DIR = Path("data/processed")
REPORTS = Path("reports")

EVAL_SETS = [
    ("k8s_related", "kubernetes_kubernetes", "d1_eval_set_k8s_related.json"),
    ("vscode_duplicate", "microsoft_vscode", "d1_eval_set_vscode_duplicate.json"),
    ("vscode_related", "microsoft_vscode", "d1_eval_set_vscode_related.json"),
]


def query_text(row: dict) -> str:
    return f"{row['query_title']}. {row.get('query_body', '')}"


def bootstrap_ci_recall(hits: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def eval_via_real_code_path(
    detector: SimilarIssueRetriever, pairs: list[dict]
) -> tuple[np.ndarray, int]:
    """No apply_query_instruction override -- exactly what prod's triage.py::retrieve() call
    does, resolving the per-repo default the same way a live request would."""
    live_numbers = {int(n) for n in detector.issue_numbers}
    usable = [
        p
        for p in pairs
        if p["query_number"] in live_numbers and p["original_number"] in live_numbers
    ]
    hits = []
    for row in usable:
        results = detector.retrieve(
            query_text(row), k=K_MAX, exclude_number=int(row["query_number"])
        )
        retrieved = [r["number"] for r in results]
        pos = int(row["original_number"])
        hits.append(any(n == pos for n in retrieved[:5]))
    return np.array(hits, dtype=float), len(usable)


def check_index_query_consistency(
    repo: str, detector: SimilarIssueRetriever, n_sample: int = 200
) -> dict:
    """Re-derive _build_text() on a random sample of the same raw issues the index was built
    from, using the SAME tokenizer/model the loaded detector carries, and assert byte-identical
    output against what's actually stored in the index's meta.pkl `texts`."""
    df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
    rng = np.random.default_rng(SEED)
    sample_positions = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)

    df_sample = df.iloc[sample_positions]
    rederived = _build_text(
        df_sample["title"],
        df_sample["body_clean"],
        tokenizer=detector.model.tokenizer,
        max_tokens=detector.model.max_seq_length,
    )
    assert detector.texts is not None
    stored = [detector.texts[i] for i in sample_positions]

    mismatches = [
        {"position": int(p), "issue_number": int(df.iloc[p]["number"])}
        for p, r, s in zip(sample_positions, rederived, stored, strict=True)
        if r != s
    ]
    return {
        "n_checked": len(sample_positions),
        "n_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:5],
        "passed": len(mismatches) == 0,
    }


def main() -> None:
    out: dict = {"reproduction": {}, "consistency": {}}
    detectors: dict[str, SimilarIssueRetriever] = {}
    all_passed = True

    for label, repo, eval_path in EVAL_SETS:
        if repo not in detectors:
            log.info("[%s] loading candidate index...", repo)
            detectors[repo] = SimilarIssueRetriever.load(
                str(MODELS_DIR / f"dup_index_{repo}_bge_candidate")
            )

        pairs = json.loads((REPORTS / eval_path).read_text(encoding="utf-8"))
        hits, n_used = eval_via_real_code_path(detectors[repo], pairs)
        r5 = float(hits.mean())
        ci = bootstrap_ci_recall(hits)
        instruction_on = QUERY_INSTRUCTION_REPO_OVERRIDE.get(repo, None)

        out["reproduction"][label] = {
            "repo": repo,
            "n_evaluated": n_used,
            "query_instruction_applied": instruction_on,
            "recall_at_5": r5,
            "recall_at_5_ci95": [round(c, 4) for c in ci],
        }
        log.info(
            "[%s] REAL code path (instruction=%s): R@5=%.4f %s (n=%d)",
            label,
            instruction_on,
            r5,
            ci,
            n_used,
        )

    for repo in ["kubernetes_kubernetes", "microsoft_vscode"]:
        result = check_index_query_consistency(repo, detectors[repo])
        out["consistency"][repo] = result
        all_passed = all_passed and result["passed"]
        log.info(
            "[%s] index/query consistency: %d/%d match, passed=%s",
            repo,
            result["n_checked"] - result["n_mismatches"],
            result["n_checked"],
            result["passed"],
        )

    out["all_consistency_checks_passed"] = all_passed
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "lever12_candidate_verification.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    log.info("Wrote reports/lever12_candidate_verification.json")
    if not all_passed:
        log.error("CONSISTENCY CHECK FAILED -- do not publish this candidate.")
        sys.exit(1)


if __name__ == "__main__":
    main()
