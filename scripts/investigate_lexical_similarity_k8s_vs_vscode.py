"""Investigation: characterize vscode_duplicate's lexical structure vs k8s_related's.

Context: ADR-0040 found the BGE query-instruction prefix helps k8s_related (+6.7pp R@5) but
hurts vscode_duplicate (isolated delta -2.0pp, CI[-5.0,+1.0], crosses zero but consistently
negative in direction across two independent measurements). ADR-0040's working hypothesis,
explicitly NOT confirmed: vscode's task is near-duplicate matching where exact lexical/surface
overlap is plausibly the dominant relevance signal, and BGE's semantic "searching relevant
passages" framing dilutes that signal -- whereas k8s's task is genuine semantic relatedness,
which fits what the instruction was designed for.

This script tests that hypothesis with data: for each eval set's (query, true target) pair,
compute three lexical-similarity metrics and compare the two eval sets' distributions.

Metrics (all in [0, 1], higher = more lexically similar):
  1. Jaccard token overlap, title+body (lowercased, [a-z0-9]+ tokenized)
  2. Jaccard token overlap, title-only
  3. Normalized Levenshtein similarity, title+body (rapidfuzz.distance.Levenshtein), the
     standard edit-distance-family metric -- character-level alignment, complementary to
     Jaccard's set-level (bag-of-tokens) view. TF-IDF cosine was considered instead but
     Jaccard+TFIDF-cosine would double up on "bag of tokens" signal without adding an
     orthogonal view; Levenshtein captures word-order/phrasing similarity that near-duplicate
     issues (same bug, re-titled/re-worded) plausibly share more of than genuinely-related-but-
     distinct issues do.

Target-side text: the eval JSON only carries `original_title`, not `original_body` -- so the
target's full text is pulled from the SAME full-corpus index used for the retrieval measurement
(`data/models/d1_full_corpus_index_{repo}_bge_lever1`, confirmed byte-identical to the current
live-serving `data/models/similar_issue_index_{repo}_bge` -- see this investigation's write-up),
i.e. exactly what the retriever has indexed for that issue (title + tokenizer-truncated body,
Lever 1's fix). This is the fairest comparison: query text vs. what the retriever actually sees
for the target, not a separately-scraped raw body that might not even be what's indexed.

Reads:
  reports/d1_eval_set_k8s_related.json
  reports/d1_eval_set_vscode_duplicate.json
  data/models/d1_full_corpus_index_{kubernetes_kubernetes,microsoft_vscode}_bge_lever1/
Writes:
  reports/lexical_similarity_k8s_vs_vscode.json
Reproduce: PYTHONPATH=src python scripts/investigate_lexical_similarity_k8s_vs_vscode.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _retrieval_eval_common import SEED, load_d1_eval_pairs  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

N_BOOTSTRAP = 2000
TOKEN_RE = re.compile(r"[a-z0-9]+")

MODELS_DIR = Path("data/models")
REPORTS = Path("reports")
OUTPUT_PATH = REPORTS / "lexical_similarity_k8s_vs_vscode.json"

EVAL_SETS = [
    ("k8s_related", "kubernetes_kubernetes", "d1_full_corpus_index_kubernetes_kubernetes_bge_lever1"),
    ("vscode_duplicate", "microsoft_vscode", "d1_full_corpus_index_microsoft_vscode_bge_lever1"),
]


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def two_sample_bootstrap_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Percentile bootstrap CI on mean(b) - mean(a), independent resampling per arm (these are
    two DIFFERENT eval sets/populations, not paired subjects -- paired_bootstrap_ci in
    _retrieval_eval_common.py assumes shared resample indices across arms, which only makes
    sense for the same pairs measured under two conditions; that assumption doesn't hold here).
    Same seed/N_BOOTSTRAP as the rest of the project for consistency.
    """
    rng = np.random.default_rng(SEED)
    na, nb = len(a), len(b)
    deltas = [
        b[rng.integers(0, nb, nb)].mean() - a[rng.integers(0, na, na)].mean()
        for _ in range(N_BOOTSTRAP)
    ]
    return (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
        float(np.mean(b) - np.mean(a)),
    )


def describe(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "mean": round(float(np.mean(x)), 4),
        "median": round(float(np.median(x)), 4),
        "std": round(float(np.std(x)), 4),
        "q1": round(float(np.percentile(x, 25)), 4),
        "q3": round(float(np.percentile(x, 75)), 4),
        "iqr": round(float(np.percentile(x, 75) - np.percentile(x, 25)), 4),
        "min": round(float(np.min(x)), 4),
        "max": round(float(np.max(x)), 4),
    }


def measure_repo(label: str, repo: str, index_dir: str) -> dict:
    pairs = load_d1_eval_pairs(repo)
    detector = SimilarIssueRetriever.load(str(MODELS_DIR / index_dir))
    number_to_text = dict(
        zip((int(n) for n in detector.issue_numbers), detector.texts, strict=True)
    )
    log.info("[%s] loaded index (%d records), %d eval pairs", label, len(number_to_text), len(pairs))

    title_jaccard, full_jaccard, full_levenshtein = [], [], []
    n_missing_target = 0

    for row in pairs.itertuples():
        target_num = int(row.original_number)
        target_text = number_to_text.get(target_num)
        if target_text is None:
            n_missing_target += 1
            continue

        # query_text() (imported above) expects a pandas Series; itertuples() rows are
        # namedtuples, so build the identical "{title}. {body}" text directly here.
        qfull = str(row.query_title) + ". " + str(row.query_body)

        title_jaccard.append(jaccard(tokenize(str(row.query_title)), tokenize(str(row.original_title))))
        full_jaccard.append(jaccard(tokenize(qfull), tokenize(target_text)))
        full_levenshtein.append(Levenshtein.normalized_similarity(qfull, target_text))

    if n_missing_target:
        log.warning("[%s] %d/%d pairs' target not found in full-corpus index -- excluded", label, n_missing_target, len(pairs))

    return {
        "label": label,
        "repo": repo,
        "n_pairs_total": len(pairs),
        "n_missing_target": n_missing_target,
        "n_measured": len(title_jaccard),
        "title_jaccard": describe(np.array(title_jaccard)),
        "full_jaccard": describe(np.array(full_jaccard)),
        "full_levenshtein_normalized_similarity": describe(np.array(full_levenshtein)),
        "_raw": {
            "title_jaccard": title_jaccard,
            "full_jaccard": full_jaccard,
            "full_levenshtein_normalized_similarity": full_levenshtein,
        },
    }


def main() -> None:
    results = [measure_repo(label, repo, index_dir) for label, repo, index_dir in EVAL_SETS]
    by_label = {r["label"]: r for r in results}

    comparisons = {}
    for metric in ["title_jaccard", "full_jaccard", "full_levenshtein_normalized_similarity"]:
        a = np.array(by_label["k8s_related"]["_raw"][metric])
        b = np.array(by_label["vscode_duplicate"]["_raw"][metric])
        lo, hi, delta = two_sample_bootstrap_diff(a, b)
        comparisons[metric] = {
            "k8s_mean": round(float(np.mean(a)), 4),
            "vscode_mean": round(float(np.mean(b)), 4),
            "vscode_minus_k8s_delta": round(delta, 4),
            "vscode_minus_k8s_ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": lo > 0 or hi < 0,
        }
        log.info(
            "[%s] k8s=%.4f  vscode=%.4f  delta(vscode-k8s)=%+.4f CI[%+.4f,%+.4f]  excludes_zero=%s",
            metric, np.mean(a), np.mean(b), delta, lo, hi, comparisons[metric]["excludes_zero"],
        )

    for r in results:
        del r["_raw"]

    out = {
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "two-sample percentile (independent resampling per arm)"},
        "results": results,
        "comparisons_vscode_minus_k8s": comparisons,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
