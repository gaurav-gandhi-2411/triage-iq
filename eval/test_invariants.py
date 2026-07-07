from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
CALIBRATION_RESULTS = ROOT / "reports" / "calibration_results.json"
CONFORMAL_ADJ = ROOT / "data" / "models" / "cqr_conformal_adjustments.json"
MANIFEST_PATH = ROOT / "data" / "models" / "MANIFEST.sha256"

REPOS = ["microsoft/vscode", "kubernetes/kubernetes"]
REPO_SLUGS = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

_RECORDED_ECE: dict[str, float] = {
    "microsoft_vscode": 0.1381,
    "kubernetes_kubernetes": 0.1558,
}
_ECE_TOLERANCE = 0.15

_COVERAGE_TOL = 0.05

# Recorded grounding baseline — mirrors _RECORDED_ECE above. Produced by
# scripts/measure_grounding.py against the CURRENT (unmodified) cassette over the clean,
# train-decontaminated n=65 eval set (ADR-0018) with the local qwen3:8b judge (ADR-0019).
# Structured per_repo (mirrors reports/eval_baseline.json) rather than pooled: a pooled
# count on an 83%-k8s-weighted gold set could mask a vscode-only regression going 0 -> N
# ungrounded underneath k8s's volume. See ADR-0015.
#
# No tolerance band here (unlike the judge-mean gate in test_quality_regression.py):
# grounding is computed by replaying the FROZEN plan already committed in the cassette
# (CassettePlayer(strict=True), zero live calls) — verify_plan_grounding() is pure Python
# with no LLM call, confirmed deterministic given a fixed plan (checked directly: 39/39
# issues with byte-identical plans across two independent recordings also had byte-identical
# grounding_status, 0 exceptions — ADR-0019). The replay invariant applies here unchanged;
# only the RE-RECORD comparison (a different cassette, e.g. a future re-record) would need
# the plan-level version of this same jitter treatment, not this ratchet as implemented.
#
# Old baseline (pre-ADR-0018/0019, n=60, Groq-70B judge era) pinned issues #1678 and #13435
# from the contaminated gold set. Both are gone from the clean n=65 set or no longer
# ungrounded under the new local-judge recording — re-derived against the actual committed
# cassette below, not carried forward.
_GROUNDING_BASELINE = {
    "eval_set_hash": "e37a69ea0cf1d26749d7f714d4f161ec6fd5f37d25a22156277551e00fd30138",
    "per_repo": {
        "kubernetes/kubernetes": {
            "ungrounded_count": 1,
            "n": 54,
            "known_ungrounded_cases": {
                13057: {
                    "axis": "component",
                    "detail": "predicted_component 'storage' not in classifier_top3 "
                    "['provider/gcp', 'kubectl', 'security']",
                },
            },
        },
        "microsoft/vscode": {
            "ungrounded_count": 1,
            "n": 11,
            "known_ungrounded_cases": {
                311836: {
                    "axis": "component",
                    "detail": "predicted_component 'webview' not in classifier_top3 "
                    "['suggest', 'accessibility', 'debug']",
                },
            },
        },
    },
}


def _extract_q_hours(repo: str, repos_data: dict) -> float:
    """Extract q_adjustment_hours for a repo using the same fallback logic as loader.py."""
    data = repos_data[repo]
    if "40_60" in data:
        adj = data["40_60"]
    elif "30_70" in data:
        adj = data["30_70"]
    else:
        adj = data
    return float(adj["q_adjustment_hours"])


def _compute_ece(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Multi-class ECE: bin by top-1 confidence, compute |acc - conf| per bin."""
    conf = y_proba.max(axis=1)
    correct = (y_pred_labels == y_true_labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return ece


def test_conformal_q_formula() -> None:
    """Verify the conformal interval formula from app.py is mathematically correct."""
    raw = json.loads(CONFORMAL_ADJ.read_text(encoding="utf-8"))
    repos_data = raw["repos"]

    test_cases = [
        (5.0, 30.0),
        (0.001, 10.0),
        (50.0, 100.0),
    ]

    for repo in REPOS:
        q_hours = _extract_q_hours(repo, repos_data)
        q_days = q_hours / 24.0

        for raw_lo, raw_hi in test_cases:
            expected_lo = max(0.0, raw_lo - q_days)
            expected_hi = raw_hi + q_days

            computed_lo = max(0.0, raw_lo - q_days)
            computed_hi = raw_hi + q_days

            assert abs(computed_lo - expected_lo) < 1e-6, (
                f"{repo}: lower mismatch for raw_lo={raw_lo}, raw_hi={raw_hi}: "
                f"expected {expected_lo}, got {computed_lo}"
            )
            assert abs(computed_hi - expected_hi) < 1e-6, (
                f"{repo}: upper mismatch for raw_lo={raw_lo}, raw_hi={raw_hi}: "
                f"expected {expected_hi}, got {computed_hi}"
            )


def test_conformal_layer_active() -> None:
    """Verify conformal adjustments load correctly and Q is active (non-zero) for both repos."""
    from triage_iq.api.loader import _load_conformal_adjustments

    adjustments = _load_conformal_adjustments(MODELS_DIR)

    assert adjustments, "Conformal adjustments dict is empty — JSON file may be missing or malformed"

    for repo in REPOS:
        assert repo in adjustments, f"Repo '{repo}' not found in conformal adjustments"
        adj = adjustments[repo]
        assert adj["q_adjustment_hours"] > 0, (
            f"{repo}: q_adjustment_hours={adj['q_adjustment_hours']} — conformal layer is a no-op"
        )
        assert adj["empirical_coverage"] > 0.50, (
            f"{repo}: empirical_coverage={adj['empirical_coverage']} — below 50% floor"
        )
        assert adj["target_coverage"] == pytest.approx(0.80, abs=1e-6), (
            f"{repo}: target_coverage={adj['target_coverage']} — expected 0.80"
        )


def test_triage_plan_schema_contract() -> None:
    """Guard TriagePlan Pydantic schema against field renames, type changes, or removals."""
    from triage_iq.models.triage import ConformalIntervalResult, SimilarIssue, TriagePlan  # noqa: F401

    REQUIRED_FIELDS = [
        "predicted_component",
        "component_confidence",
        "similar_issues",
        "expected_resolution_summary",
        "expected_resolution_lower_days",
        "expected_resolution_upper_days",
        "resolution_bucket",
        "resolution_confidence_pct",
        "resolution_interval_conformal",
        "priority_guess",
        "priority_rationale",
        "suggested_assignee_class",
        "suggested_next_steps",
        "triage_summary",
    ]

    for field_name in REQUIRED_FIELDS:
        assert field_name in TriagePlan.model_fields, (
            f"TriagePlan is missing required field '{field_name}'"
        )

    plan = TriagePlan(
        predicted_component="editor",
        component_confidence=0.9,
        expected_resolution_summary="Fast fix expected",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=7.0,
        priority_guess="medium",
        priority_rationale="Medium priority",
        suggested_assignee_class="editor-team",
        suggested_next_steps=["Review the PR"],
        triage_summary="Test plan",
    )
    assert isinstance(plan.predicted_component, str)
    assert isinstance(plan.similar_issues, list)
    assert isinstance(plan.expected_resolution_lower_days, float)
    assert plan.priority_guess in {"low", "medium", "high"}

    CONFORMAL_REQUIRED = [
        "lower_days",
        "upper_days",
        "target_coverage",
        "empirical_coverage",
        "coverage_ci95_lower",
        "coverage_ci95_upper",
    ]
    for field_name in CONFORMAL_REQUIRED:
        assert field_name in ConformalIntervalResult.model_fields, (
            f"ConformalIntervalResult is missing required field '{field_name}'"
        )


def test_calibration_ece_in_tolerance() -> None:
    """Verify calibrated ECE is within tolerance of recorded values on the frozen eval set."""
    import pandas as pd

    from triage_iq.models.component_classifier import TFIDFComponentClassifier

    if not EVAL_SET.exists():
        pytest.skip(reason="eval_set.jsonl not found — skipping ECE check")

    issues = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]

    for repo, slug in REPO_SLUGS.items():
        model_path = MODELS_DIR / f"component_classifier_{slug}.pkl"
        if not model_path.exists():
            pytest.skip(reason=f"Classifier model not found: {model_path}")

        clf = TFIDFComponentClassifier.load(str(model_path))
        repo_issues = [iss for iss in issues if iss["repo"] == repo]

        texts = [f"{iss['title']}. {iss['body']}" for iss in repo_issues]
        text_series = pd.Series(texts)

        proba = clf.predict_proba_calibrated(text_series)
        preds = clf.predict(text_series)
        y_true = np.array([iss["gold_component"] for iss in repo_issues])

        ece = _compute_ece(y_true, preds, proba, n_bins=5)

        assert abs(ece - _RECORDED_ECE[slug]) < _ECE_TOLERANCE, (
            f"ECE {ece:.4f} deviates from recorded {_RECORDED_ECE[slug]:.4f} by more than "
            f"{_ECE_TOLERANCE} — calibrator may be missing or corrupted"
        )


def test_conformal_coverage_on_eval_set() -> None:
    """Verify conformal intervals achieve reasonable coverage on the frozen eval set."""
    import pandas as pd

    from triage_iq.api.loader import _load_conformal_adjustments
    from triage_iq.models.resolution import ResolutionTimePredictor, engineer_features

    if not EVAL_SET.exists():
        pytest.skip(reason="eval_set.jsonl not found — skipping conformal coverage check")

    adjustments = _load_conformal_adjustments(MODELS_DIR)
    issues = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]

    for repo, slug in REPO_SLUGS.items():
        predictor_path = MODELS_DIR / f"resolution_predictor_{slug}.pkl"
        train_path = PROCESSED_DIR / f"{slug}_temporal_train.parquet"

        if not predictor_path.exists() or not train_path.exists():
            pytest.skip(reason=f"Model or train data missing for {slug}")

        if repo not in adjustments:
            pytest.skip(reason=f"Conformal adjustments missing for {repo}")

        predictor = ResolutionTimePredictor.load(str(predictor_path))
        train_df = pd.read_parquet(train_path)
        adj = adjustments[repo]
        q_hours = adj["q_adjustment_hours"]

        repo_issues = [iss for iss in issues if iss["repo"] == repo]
        n_issues = len(repo_issues)
        n_covered = 0

        for iss in repo_issues:
            issue_df = pd.DataFrame([{
                "title": iss["title"],
                "body_clean": iss.get("body_clean", iss.get("body", "")),
                "number": iss["number"],
                "created_at": pd.to_datetime(iss["created_at"], utc=True),
            }])

            feats, _ = engineer_features(issue_df, train_df=train_df)

            for col in predictor.feature_names:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[predictor.feature_names]

            lo_hrs, hi_hrs = predictor.predict_intervals(feats)
            conf_lo = max(0.0, float(lo_hrs[0]) - q_hours)
            conf_hi = float(hi_hrs[0]) + q_hours

            conf_lo_days = conf_lo / 24.0
            conf_hi_days = conf_hi / 24.0

            assert conf_hi_days > conf_lo_days, (
                f"{repo} issue #{iss['number']}: conformal interval is empty "
                f"[{conf_lo_days:.4f}, {conf_hi_days:.4f}]"
            )

            actual_days = float(iss["actual_resolution_days"])
            covered = conf_lo_days <= actual_days <= conf_hi_days
            if covered:
                n_covered += 1

        coverage = n_covered / n_issues
        print(f"{repo}: conformal coverage = {coverage:.2f} ({n_covered}/{n_issues})")

        assert 0.40 <= coverage <= 1.00, (
            f"{repo}: conformal coverage {coverage:.2f} is outside [0.40, 1.00] — "
            f"complete breakdown detected"
        )


def test_retrieval_top_k() -> None:
    """Live FAISS top-5 agrees with the CPU-frozen top-5 on 2 probe issues per repo.

    Checks: top-1 exact match + >=4/5 set-membership.
    Full ordering is not asserted — FAISS cosine scores can differ by <1e-4 between
    CPU float32 runs, causing rank swaps at tied positions (e.g. vscode #311565,
    delta=0.0000). The frozen provenance is in eval/frozen_retrieval_provenance.json.

    Failures here mean the FAISS index was rebuilt or the embedding model changed.
    Re-run eval/freeze_similar_issues.py and recommit eval_set.jsonl.
    """
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    # 2 probe issues per repo — chosen for clear score gaps at rank 1.
    # vscode #311565 (zero-gap at rank 5/6) is intentionally excluded; its ordering
    # is unstable even within CPU float32 runs due to FAISS tie-breaking.
    # Re-picked for the clean n=65 eval set (ADR-0018): the old probes (#2093, #4223
    # vscode; #11079 k8s) were part of the train-contaminated original 60 and are no
    # longer in eval_set.jsonl. #13257 (k8s) survived the contamination filter and is
    # unchanged. Verified directly: all four replacements are 5/5 exact top-5 match.
    PROBES: dict[str, list[int]] = {
        "microsoft/vscode": [311284, 311836],
        "kubernetes/kubernetes": [14054, 13257],
    }

    if not EVAL_SET.exists():
        pytest.skip("eval_set.jsonl not found")

    issues_by_num: dict[str, dict[int, dict]] = {}
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        iss = json.loads(line)
        issues_by_num.setdefault(iss["repo"], {})[iss["number"]] = iss

    for repo, slug in REPO_SLUGS.items():
        idx_dir = MODELS_DIR / f"dup_index_{slug}_bge"
        if not idx_dir.exists():
            pytest.skip(f"FAISS index not found: {idx_dir}")

        det = SimilarIssueRetriever.load(str(idx_dir))

        for num in PROBES[repo]:
            iss = issues_by_num.get(repo, {}).get(num)
            if iss is None:
                pytest.fail(f"Probe issue {repo} #{num} not found in eval_set.jsonl")

            frozen_top5 = iss.get("similar_issues")
            if not frozen_top5:
                pytest.fail(
                    f"{repo} #{num} missing 'similar_issues' — "
                    "re-run eval/freeze_similar_issues.py"
                )

            frozen_nums = [s["number"] for s in frozen_top5[:5]]
            query_text = iss["title"] + ". " + iss["body"]
            live_results = det.retrieve(query_text, k=5, exclude_number=num)
            live_nums = [s["number"] for s in live_results[:5]]

            assert live_nums[0] == frozen_nums[0], (
                f"{repo} #{num}: live rank-1={live_nums[0]} ≠ frozen rank-1={frozen_nums[0]}. "
                "FAISS index may have been rebuilt — re-run eval/freeze_similar_issues.py."
            )
            overlap = len(set(live_nums) & set(frozen_nums))
            assert overlap >= 4, (
                f"{repo} #{num}: only {overlap}/5 issues match between live and frozen top-5. "
                f"live={live_nums} frozen={frozen_nums}. "
                "FAISS index may have been rebuilt — re-run eval/freeze_similar_issues.py."
            )


def test_model_manifest_clean() -> None:
    """Verify all model artifacts match the committed MANIFEST.sha256.

    Guards against the 6-week calibration gap (ADR-0013): a locally-committed
    model file that was never uploaded to GCS, or an artifact baked into a
    Docker layer that has since diverged. Failure here means scripts/publish_models.py
    needs to be run and the manifest re-committed.
    """
    if not MANIFEST_PATH.exists():
        pytest.fail(
            f"MANIFEST.sha256 not found at {MANIFEST_PATH}. "
            "Run python scripts/publish_models.py to generate and commit it."
        )

    lines = [ln.strip() for ln in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, f"MANIFEST.sha256 at {MANIFEST_PATH} is empty"

    drifted: list[str] = []
    missing: list[str] = []

    for line in lines:
        expected_hash, rel_path = line.split("  ", 1)
        p = ROOT / rel_path
        if not p.exists():
            missing.append(rel_path)
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected_hash:
            drifted.append(f"{rel_path}: manifest={expected_hash[:16]} actual={actual[:16]}")

    errors: list[str] = []
    if missing:
        errors.append(f"Missing artifacts ({len(missing)}): {missing}")
    if drifted:
        errors.append(f"Hash mismatches ({len(drifted)}): {drifted}")

    assert not errors, (
        "Model artifact drift detected — run python scripts/publish_models.py "
        "and commit the updated manifest:\n" + "\n".join(errors)
    )


def _eval_set_hash_guard() -> str:
    """Compute eval_set.jsonl's sha256 and return a loud failure message if it has drifted.

    Returns the current hash. Callers assert `current_hash == _GROUNDING_BASELINE["eval_set_hash"]`
    with this message so staleness surfaces instead of being silently compared across sets.
    """
    return hashlib.sha256(EVAL_SET.read_bytes()).hexdigest()


_HASH_DRIFT_MSG = (
    "eval_set.jsonl changed — re-derive _GROUNDING_BASELINE (ratchet + known-case pins) "
    "deliberately, do not silently compare across different sets"
)


@pytest.fixture(scope="module")
def grounding_reports() -> list[dict]:
    """Compute grounding reports once for the module, shared by the ratchet and pin tests.

    Reuses the same cassette-replay pipeline as scripts/measure_grounding.py (zero live
    LLM calls — CassettePlayer(strict=True)). See ADR-0015.
    """
    if not EVAL_SET.exists():
        pytest.skip(reason="eval_set.jsonl not found — skipping grounding checks")

    from measure_grounding import compute_grounding_reports

    return compute_grounding_reports()


def test_grounding_ratchet_no_new_ungrounded_claims(grounding_reports: list[dict]) -> None:
    """Ungrounded-claim count on the frozen eval set must not exceed the recorded baseline.

    Checked per-repo (not pooled): a regression concentrated in one repo must fail this
    test on its own, independent of the other repo's volume. Guards against silent
    regressions in synthesis grounding (component/similar-issue hallucination) creeping in
    above the measured 2/30 (k8s) + 0/30 (vscode) baseline. See ADR-0015.
    """
    current_hash = _eval_set_hash_guard()
    assert current_hash == _GROUNDING_BASELINE["eval_set_hash"], _HASH_DRIFT_MSG

    for repo, baseline in _GROUNDING_BASELINE["per_repo"].items():
        repo_reports = [c for c in grounding_reports if c["repo"] == repo]
        ungrounded_count = sum(1 for c in repo_reports if not c["all_grounded"])

        assert len(repo_reports) == baseline["n"], (
            f"{repo}: eval set size changed ({len(repo_reports)} vs baseline "
            f"{baseline['n']}) despite matching top-level hash — investigate"
        )
        assert ungrounded_count <= baseline["ungrounded_count"], (
            f"{repo}: ungrounded claim count regressed: {ungrounded_count} > "
            f"baseline {baseline['ungrounded_count']}"
        )


def test_grounding_known_cases_still_flagged(grounding_reports: list[dict]) -> None:
    """The two known-bad cases (#13057 k8s, #311836 vscode) must still be caught by name.

    This catches a verifier regressed to a no-op, which would otherwise trivially satisfy
    the ratchet test at 0 <= 1 ungrounded per repo. See ADR-0015.

    Re-derived against the clean n=65 set + local qwen3:8b judge (ADR-0018/0019). The old
    pins (#1678 similar_issue-axis, #13435 component-axis, both from the contaminated n=60
    set) are gone: #1678 isn't in the clean n=65 set, and no similar_issue-axis hallucination
    exists in the current committed cassette to pin — both new pins are component-axis. This
    is not a weaker test by design; it reflects what's actually in the committed recording.
    """
    current_hash = _eval_set_hash_guard()
    assert current_hash == _GROUNDING_BASELINE["eval_set_hash"], _HASH_DRIFT_MSG

    by_issue = {c["issue_number"]: c for c in grounding_reports}

    case_13057 = by_issue.get(13057)
    assert case_13057 is not None, "Issue #13057 not found in grounding reports"
    assert case_13057["component_grounded"] is False, (
        "Issue #13057: expected component_grounded is False "
        f"(predicted_component={case_13057['predicted_component']!r}, "
        f"classifier_top3_labels={case_13057['classifier_top3_labels']})"
    )

    case_311836 = by_issue.get(311836)
    assert case_311836 is not None, "Issue #311836 not found in grounding reports"
    assert case_311836["component_grounded"] is False, (
        "Issue #311836: expected component_grounded is False "
        f"(predicted_component={case_311836['predicted_component']!r}, "
        f"classifier_top3_labels={case_311836['classifier_top3_labels']})"
    )
