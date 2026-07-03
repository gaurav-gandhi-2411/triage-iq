from __future__ import annotations

"""Quality regression gate: compares current pipeline scores against the frozen baseline.

Requires reports/eval_baseline.json to exist (written by run_eval.py --update-baseline).
All LLM calls are served from the cassette — no live API calls in CI.

Run with:
    pytest eval/test_quality_regression.py -v
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BASELINE_PATH = ROOT / "reports" / "eval_baseline.json"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"

DIMENSION_KEYS: list[str] = [
    "component_match",
    "similar_issues_relevance",
    "resolution_estimate_reasonableness",
    "priority_alignment",
    "next_steps_actionability",
    "overall_quality",
]


@pytest.fixture(scope="session")
def baseline() -> dict:
    """Load and return the frozen baseline JSON.

    Skips the session if the file does not exist — this allows CI to run
    the invariant suite before a baseline has been generated.
    """
    import json

    if not BASELINE_PATH.exists():
        pytest.skip(reason=f"Baseline file not found: {BASELINE_PATH} — run eval/run_eval.py --update-baseline first")

    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def current_scores() -> dict:
    """Run the pipeline in replay-only mode and return scores.

    Import is deferred to the fixture body to avoid heavy model imports
    at pytest collection time.
    """
    from run_eval import compute_scores

    return compute_scores()


def _check_repo_quality(repo: str, current_scores: dict, baseline: dict) -> None:
    """Assert that the current mean score for *repo* has not regressed below the baseline.

    Computes drop = baseline_mean - current_mean and fails if drop exceeds
    (absolute_drop + epsilon) from the baseline threshold block.
    """
    baseline_repo = baseline["per_repo"][repo]
    current_repo = current_scores["per_repo"][repo]

    baseline_mean: float = baseline_repo["mean"]
    current_mean: float = current_repo["mean"]
    drop: float = baseline_mean - current_mean

    threshold: float = (
        baseline["threshold"]["absolute_drop"] + baseline["threshold"]["epsilon"]
    )

    if drop > threshold:
        baseline_dims: dict[str, float] = baseline_repo.get("dimensions", {})
        current_dims: dict[str, float] = current_repo.get("dimensions", {})

        lines = [
            f"Quality regression detected for {repo}",
            f"  baseline mean : {baseline_mean:.4f}/15",
            f"  current mean  : {current_mean:.4f}/15",
            f"  drop          : {drop:.4f}  (threshold = {threshold:.4e})",
            "",
            "  Per-criterion breakdown:",
            f"  {'dimension':<40} {'baseline':>9} {'current':>9} {'delta':>9}",
            f"  {'-'*40} {'-'*9} {'-'*9} {'-'*9}",
        ]
        for key in DIMENSION_KEYS:
            b_val = baseline_dims.get(key, float("nan"))
            c_val = current_dims.get(key, float("nan"))
            delta = c_val - b_val
            lines.append(
                f"  {key:<40} {b_val:>9.4f} {c_val:>9.4f} {delta:>+9.4f}"
            )

        pytest.fail("\n".join(lines))


def test_vscode_quality_regression(current_scores: dict, baseline: dict) -> None:
    """microsoft/vscode mean score must not drop below baseline."""
    _check_repo_quality("microsoft/vscode", current_scores, baseline)


def test_k8s_quality_regression(current_scores: dict, baseline: dict) -> None:
    """kubernetes/kubernetes mean score must not drop below baseline."""
    _check_repo_quality("kubernetes/kubernetes", current_scores, baseline)


def test_cassette_hash_matches_baseline(baseline: dict) -> None:
    """Cassette on disk must match the hash recorded in eval_baseline.json.

    A mismatch means scores in the baseline were computed from a different cassette
    than the one currently on disk — the baseline cannot be trusted for regression
    gating until they are re-synced via run_eval.py --update-baseline.
    """
    import json

    baseline_hash: str = baseline.get("cassette_hash", "")
    if not baseline_hash:
        pytest.fail("eval_baseline.json is missing 'cassette_hash' — re-run run_eval.py --update-baseline")

    if not CASSETTE_PATH.exists():
        pytest.fail(f"Cassette not found: {CASSETTE_PATH}")

    actual_hash = hashlib.sha256(CASSETTE_PATH.read_bytes()).hexdigest()

    assert actual_hash == baseline_hash, (
        f"Cassette on disk does not match eval_baseline.json:\n"
        f"  baseline cassette_hash : {baseline_hash[:32]}\n"
        f"  disk cassette SHA-256  : {actual_hash[:32]}\n"
        f"Re-run: python eval/run_eval.py --update-baseline"
    )
