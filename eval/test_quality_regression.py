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
        pytest.skip(
            reason=f"Baseline file not found: {BASELINE_PATH} — run eval/run_eval.py --update-baseline first"
        )

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
    """Assert that the current per-repo mean has not regressed below baseline - band.

    One-directional (ADR-0019): only fires if drop = baseline_mean - current_mean exceeds
    the repo's tolerance band. An improvement (negative drop) never trips this — the band
    only bounds regressions, not any movement. The band is a data-derived 2xSEM figure from
    the measured re-record jitter (see threshold.per_repo_band in eval_baseline.json), not a
    tuned-to-pass number — per-repo because vscode's n=11 genuinely has a wider band than
    k8s's n=54, reflecting real statistical power, not a fudge.
    """
    baseline_repo = baseline["per_repo"][repo]
    current_repo = current_scores["per_repo"][repo]

    baseline_mean: float = baseline_repo["mean"]
    current_mean: float = current_repo["mean"]
    drop: float = baseline_mean - current_mean

    threshold: float = baseline["threshold"]["per_repo_band"][repo]["band"]

    if drop > threshold:
        baseline_dims: dict[str, float] = baseline_repo.get("dimensions", {})
        current_dims: dict[str, float] = current_repo.get("dimensions", {})

        lines = [
            f"Quality regression detected for {repo}",
            f"  baseline mean : {baseline_mean:.4f}/15",
            f"  current mean  : {current_mean:.4f}/15",
            f"  drop          : {drop:.4f}  (band = {threshold:.4f})",
            "",
            "  Per-criterion breakdown:",
            f"  {'dimension':<40} {'baseline':>9} {'current':>9} {'delta':>9}",
            f"  {'-' * 40} {'-' * 9} {'-' * 9} {'-' * 9}",
        ]
        for key in DIMENSION_KEYS:
            b_val = baseline_dims.get(key, float("nan"))
            c_val = current_dims.get(key, float("nan"))
            delta = c_val - b_val
            lines.append(f"  {key:<40} {b_val:>9.4f} {c_val:>9.4f} {delta:>+9.4f}")

        pytest.fail("\n".join(lines))


def test_vscode_quality_regression(current_scores: dict, baseline: dict) -> None:
    """microsoft/vscode mean score must not drop below baseline."""
    _check_repo_quality("microsoft/vscode", current_scores, baseline)


def test_k8s_quality_regression(current_scores: dict, baseline: dict) -> None:
    """kubernetes/kubernetes mean score must not drop below baseline.

    The xfail marker that lived here (2026-08-05 -> 2026-08-10) is gone: the baseline was
    re-recorded to the ADR-0036 multi-label classifier cutover's own scores (k8s 10.2642,
    ADR-0043), so this now compares the recording against itself rather than against the
    frozen pre-cutover target. The -0.2452 residual vs. the OLD frozen baseline is not a
    regression this gate needs to catch going forward -- it's baked into the new floor as a
    deliberately accepted tradeoff (ADR-0037/ADR-0039/ADR-0043). This test still protects
    against *future* regressions below the new floor.
    """
    _check_repo_quality("kubernetes/kubernetes", current_scores, baseline)


def _check_no_fabrication(repo: str, current_scores: dict) -> None:
    """Assert zero fabricated claims for `repo` (ADR-0028 Phase B3).

    A fabricated component/similar-issue claim is qualitatively worse than a soft
    quality miss and the mean-band gate above cannot detect it at all (the judge never
    sees classifier_top3/retrieved_numbers). BLOCKING since PR #57 (this file's CI job
    lost continue-on-error:true 2026-08-10). Deliberately zero-tolerance, no per-repo
    slack even for vscode's n=11 -- see ADR-0044 for why (grounding is replay-deterministic,
    so this can only move on a deliberate, human-reviewed cassette re-record, not on an
    ordinary PR).
    """
    rate = current_scores["per_repo"][repo]["fabrication_rate"]
    n = current_scores["per_repo"][repo]["n"]
    assert rate == 0.0, (
        f"Fabrication detected for {repo}: fabrication_rate={rate:.4f} (n={n}). "
        "A fabricated component/similar-issue claim is a hard-fail correctness issue, "
        "not a soft quality miss -- see plan.grounding_status for the offending plan(s)."
    )


def test_vscode_no_fabrication(current_scores: dict) -> None:
    """microsoft/vscode must have zero grounding-verified fabricated claims."""
    _check_no_fabrication("microsoft/vscode", current_scores)


def test_k8s_no_fabrication(current_scores: dict) -> None:
    """kubernetes/kubernetes must have zero grounding-verified fabricated claims."""
    _check_no_fabrication("kubernetes/kubernetes", current_scores)


def _check_no_prose_number_contradiction(repo: str, current_scores: dict) -> None:
    """Assert zero prose/number contradictions for `repo` (ADR-0042, LEVER 4).

    Motivating case (ADR-0037, k8s-14756): expected_resolution_summary said "typically 1
    day or less" against a numeric interval of [2.8d, 21.6d] -- the model contradicting
    numbers it was directly given, a real correctness defect distinct from the hedging-tone
    investigation in ADR-0037. BLOCKING since PR #57 (this file's CI job lost
    continue-on-error:true 2026-08-10): measured 0/64 on the current cassette
    (reports/lever4_prose_number_consistency.json) -- same zero-tolerance rationale as
    fabrication_rate above (ADR-0044): replay-deterministic, so it only moves on a
    deliberate cassette re-record, not on an ordinary PR.
    """
    rate = current_scores["per_repo"][repo]["prose_number_contradiction_rate"]
    n = current_scores["per_repo"][repo]["n"]
    assert rate == 0.0, (
        f"Prose/number contradiction detected for {repo}: rate={rate:.4f} (n={n}). "
        "The free-text expected_resolution_summary claims a time range with zero overlap "
        "against expected_resolution_lower_days/upper_days -- see "
        "src/triage_iq/models/resolution_consistency.py for the offending plan(s)."
    )


def test_vscode_no_prose_number_contradiction(current_scores: dict) -> None:
    """microsoft/vscode's resolution summaries must not contradict their own numeric interval."""
    _check_no_prose_number_contradiction("microsoft/vscode", current_scores)


def test_k8s_no_prose_number_contradiction(current_scores: dict) -> None:
    """kubernetes/kubernetes's resolution summaries must not contradict their own numeric interval."""
    _check_no_prose_number_contradiction("kubernetes/kubernetes", current_scores)


def test_cassette_hash_matches_baseline(baseline: dict) -> None:
    """Cassette on disk must match the hash recorded in eval_baseline.json.

    A mismatch means scores in the baseline were computed from a different cassette
    than the one currently on disk — the baseline cannot be trusted for regression
    gating until they are re-synced via run_eval.py --update-baseline.
    """
    import json

    baseline_hash: str = baseline.get("cassette_hash", "")
    if not baseline_hash:
        pytest.fail(
            "eval_baseline.json is missing 'cassette_hash' — re-run run_eval.py --update-baseline"
        )

    if not CASSETTE_PATH.exists():
        pytest.fail(f"Cassette not found: {CASSETTE_PATH}")

    actual_hash = hashlib.sha256(CASSETTE_PATH.read_bytes()).hexdigest()

    assert actual_hash == baseline_hash, (
        f"Cassette on disk does not match eval_baseline.json:\n"
        f"  baseline cassette_hash : {baseline_hash[:32]}\n"
        f"  disk cassette SHA-256  : {actual_hash[:32]}\n"
        f"Re-run: python eval/run_eval.py --update-baseline"
    )
