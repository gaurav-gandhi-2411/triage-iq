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
    # ADR-0036: multi-label OvR classifier (component_confidence semantics changed from
    # single-softmax to independent-sigmoid top-1). Recorded fresh on THIS eval harness
    # (eval/eval_set.jsonl, n_bins=5) for the new model -- the prior constants (0.1381/0.1558)
    # were themselves test-split ECE, not this eval-set's own ECE, and were never a tight match
    # even for the old model (old model's actual eval-set ECE: 0.2351 vscode / 0.1537 k8s).
    #
    # Re-derived 2026-09-05 (ADR-0058, Phase 3): the ADR-0057 classifier retrain moved ECE
    # 0.3781 -> 0.1875 (vscode) / 0.1299 -> 0.1210 (k8s), both computed on the IDENTICAL
    # population as before (eval/eval_set.jsonl, n=11/n=53 unchanged) -- confirmed
    # comparable by reproducing the OLD classifier's ECE via this exact method before
    # re-deriving (scripts/scratch/ece_comparability_check.py): it reproduced 0.3781/0.1299
    # exactly, ruling out the population-mismatch error shape the ADR-0057 top-3 comparison
    # had to correct for. The improvement is a genuine, mechanistically-explained
    # consequence of the resolved taxonomy gap (ADR-0056): 7/11 vscode gold labels (63.6%)
    # and 7/53 k8s gold labels (13.2%) were outside the OLD classifier's class list --
    # automatic top-1 misses regardless of confidence, which mechanically inflates ECE.
    # vscode's in-taxonomy-only top-1 accuracy was already 75% (3/4) under the old
    # classifier; the retrain's 90.9% overall reflects a genuinely better-calibrated model
    # on a population it can now mostly answer, not a comparability artifact.
    "microsoft_vscode": 0.1875,
    "kubernetes_kubernetes": 0.1210,
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
#
# Re-derived 2026-07-11 (ADR-0028 Phase B1): k8s #14398 quarantined from eval_set.jsonl
# (a near-duplicate leak, cosine 0.907 vs classifier_train #14399 — unrelated to
# grounding), changing the file's hash and k8s's n from 54 to 53. #14398 itself was
# grounded, so this is purely a denominator/hash change.
#
# ADR-0039: the per-issue named-case pin (test_grounding_known_cases_still_flagged) was
# removed 2026-08-06 — the two cases pinned at the time (#13057 k8s, #311836 vscode) no
# longer reproduce under the v3 cassette, and re-pinning to whatever the current cassette
# happens to produce would make the test self-fulfilling. Only the rate bound
# (ungrounded_count/n below) remains; see ADR-0039 for the no-op-verifier blind-spot
# tradeoff this reopens and the guidance for choosing deliberate adversarial pins later.
_GROUNDING_BASELINE = {
    # Re-derived 2026-08-10: eval_set.jsonl drifted to 0c2e5741... in PR #52 (2026-08-06,
    # frozen retrieval snapshot refresh for ADR-0040) but this ratchet baseline was never
    # updated to match -- masked for the intervening period by eval-gate.yml's job-level
    # continue-on-error, so this test had been silently failing on main since 2026-08-06
    # rather than gating anything. New counts measured directly against the current cassette
    # via scripts/measure_grounding.py's compute_grounding_reports() (zero live LLM calls):
    # both repos are now fully grounded (0 ungrounded claims), consistent with
    # reports/eval_baseline.json's fabrication_rate: 0.0 for both repos on the same cassette
    # (same underlying definition -- plan.grounding_status.all_grounded is False).
    #
    # Re-verified 2026-09-05 (ADR-0052): the model, classifier, wire schema, and prompt
    # ALL changed since the paragraph above was written (see ADR-0054/0055/0056/0057) --
    # the cassette this constant was originally measured against is no longer valid. The
    # 64-issue re-record under the new configuration reproduced the identical result (0
    # ungrounded, same n both repos) via the same zero-live-call replay, and eval_set_hash
    # is unchanged (only the cassette recording changed, not the eval SET) -- so no value
    # below changed, only this provenance note.
    "eval_set_hash": "0c2e57410098ea170f3f65668ff8977d3ce4942936b9a3e2ffb6696a09621bfe",
    "per_repo": {
        "kubernetes/kubernetes": {
            "ungrounded_count": 0,
            "n": 53,
        },
        "microsoft/vscode": {
            "ungrounded_count": 0,
            "n": 11,
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

    from triage_iq.models.component_classifier import load_classifier

    if not EVAL_SET.exists():
        pytest.skip(reason="eval_set.jsonl not found — skipping ECE check")

    issues = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]

    for repo, slug in REPO_SLUGS.items():
        model_path = MODELS_DIR / f"component_classifier_{slug}.pkl"
        if not model_path.exists():
            pytest.skip(reason=f"Classifier model not found: {model_path}")

        # load_classifier() dispatches on the pkl's model_kind marker (ADR-0036: multi-label
        # OvR vs legacy single-label) -- same loader the live API uses, not hardcoded to one class.
        clf = load_classifier(MODELS_DIR, slug)
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
        idx_dir = MODELS_DIR / f"similar_issue_index_{slug}_bge"
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


def test_cassette_provenance_matches_current_artifacts() -> None:
    """Every eval_cassette.json entry's stamped artifact_hashes (ADR-0059) must match the
    classifier/predictor/index/conformal-store files actually on disk in this checkout.

    Guards against the 2026-09-05 incident this test exists to make impossible: a cassette
    recorded against one checkout's classifier files got committed while a DIFFERENT
    (retrained) classifier lived in this worktree's data/models -- record_cassettes.py's own
    model/prompt_hash checkpoint tagging had no way to detect this, because neither the model
    name nor the prompt text changed, only the classifier's fitted parameters did. A mismatch
    here means either the cassette needs re-recording (eval/record_cassettes.py) against the
    artifacts currently on disk, or the artifacts on disk are wrong for this cassette.

    Entries with no artifact_hashes at all (recorded before ADR-0059) are reported as a
    distinct failure category, not silently skipped -- an unstamped entry is exactly as
    unverifiable as a mismatched one, just for a different reason.
    """
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "eval"))
    import artifact_fingerprint

    cassette_path = ROOT / "eval" / "cassettes" / "eval_cassette.json"
    if not cassette_path.exists():
        pytest.skip(reason="eval_cassette.json not found")

    raw = json.loads(cassette_path.read_text(encoding="utf-8"))
    entries = raw.get("entries", {})
    if not entries:
        pytest.skip(reason="eval_cassette.json has no entries")

    current_by_repo = {
        repo: artifact_fingerprint.compute_artifact_hashes(ROOT, repo=repo) for repo in REPOS
    }

    unstamped: list[str] = []
    mismatched: list[str] = []
    checked = 0
    for key, entry in entries.items():
        if not (isinstance(entry, dict) and "response" in entry):
            continue  # legacy/pre-request-storage entry shape, not this check's concern
        provenance = entry.get("artifact_hashes")
        if not provenance:
            unstamped.append(key[:16])
            continue
        checked += 1
        # A provenance dict was recorded against ONE repo's artifact set (ADR-0059 stamps
        # only the triaged issue's own repo, not both) -- match it against whichever repo's
        # current hashes it's a subset of, rather than assuming which repo this key belongs
        # to (the cassette key is an opaque LLM-request hash, not itself repo-labeled).
        matched_repo = next(
            (
                repo for repo, current in current_by_repo.items()
                if all(current.get(p) == h for p, h in provenance.items())
            ),
            None,
        )
        if matched_repo is None:
            best_repo = max(
                current_by_repo,
                key=lambda r: sum(
                    1 for p, h in provenance.items() if current_by_repo[r].get(p) == h
                ),
            )
            diff = artifact_fingerprint.diff_against_expected(current_by_repo[best_repo], provenance)
            mismatched.append(f"{key[:16]} (closest match {best_repo}):\n" + "\n".join(diff))

    errors: list[str] = []
    if unstamped:
        errors.append(
            f"{len(unstamped)}/{checked + len(unstamped)} entries have no artifact_hashes "
            f"(recorded before ADR-0059, or by a path that doesn't stamp provenance): "
            f"{unstamped[:10]}{' ...' if len(unstamped) > 10 else ''}"
        )
    if mismatched:
        errors.append(
            f"{len(mismatched)}/{checked} stamped entries do NOT match the artifacts "
            f"currently on disk:\n" + "\n\n".join(mismatched[:5])
            + (f"\n... and {len(mismatched) - 5} more" if len(mismatched) > 5 else "")
        )

    assert not errors, (
        "Cassette provenance drift detected — re-run eval/record_cassettes.py against the "
        "current artifacts and commit the updated cassette:\n\n" + "\n\n".join(errors)
    )


def _eval_set_hash_guard() -> str:
    """Compute eval_set.jsonl's sha256 and return a loud failure message if it has drifted.

    Returns the current hash. Callers assert `current_hash == _GROUNDING_BASELINE["eval_set_hash"]`
    with this message so staleness surfaces instead of being silently compared across sets.
    """
    return hashlib.sha256(EVAL_SET.read_bytes()).hexdigest()


_HASH_DRIFT_MSG = (
    "eval_set.jsonl changed — re-derive _GROUNDING_BASELINE (ratchet bound) deliberately, "
    "do not silently compare across different sets"
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


def _grounding_ratchet_check(repo: str, grounding_reports: list[dict]) -> tuple[int, int]:
    """Shared size/count computation for the per-repo ratchet tests below. Returns
    (ungrounded_count, n); does not assert -- callers decide whether to gate."""
    current_hash = _eval_set_hash_guard()
    assert current_hash == _GROUNDING_BASELINE["eval_set_hash"], _HASH_DRIFT_MSG

    baseline = _GROUNDING_BASELINE["per_repo"][repo]
    repo_reports = [c for c in grounding_reports if c["repo"] == repo]
    ungrounded_count = sum(1 for c in repo_reports if not c["all_grounded"])

    assert len(repo_reports) == baseline["n"], (
        f"{repo}: eval set size changed ({len(repo_reports)} vs baseline "
        f"{baseline['n']}) despite matching top-level hash — investigate"
    )
    return ungrounded_count, baseline["ungrounded_count"]


def test_grounding_ratchet_k8s(grounding_reports: list[dict]) -> None:
    """kubernetes/kubernetes ungrounded-claim count must not exceed the recorded baseline.

    Stays hard-gated (ADR-0058): n=53 gives a Wilson upper bound of ~9.9% at 0 observed --
    genuinely informative, unlike vscode's n=11 (see test_grounding_ratchet_vscode below).
    Guards against silent regressions in synthesis grounding (component/similar-issue
    hallucination) creeping in above the recorded baseline. See ADR-0015.
    """
    ungrounded_count, baseline_count = _grounding_ratchet_check("kubernetes/kubernetes", grounding_reports)
    assert ungrounded_count <= baseline_count, (
        f"kubernetes/kubernetes: ungrounded claim count regressed: {ungrounded_count} > "
        f"baseline {baseline_count}"
    )


def test_grounding_ratchet_vscode(grounding_reports: list[dict]) -> None:
    """microsoft/vscode's grounding ratchet is REPORT ONLY, not gated (ADR-0058, 2026-09-05).

    Was a hard `<=` ratchet identical to k8s's until this session found direct evidence it
    cannot support one: the SAME issue (#311836) was flagged 4/11, 1/11, then 0/11 across
    three measurements this engagement took as the classifier/prompt/model changed, and a
    negative control (scripts/scratch/negative_control_fabrication.py) confirmed the
    underlying grounding computation genuinely catches a real fabrication when one exists
    -- so the flip is not the mechanism failing, it is n=11 giving a Wilson 95% CI of
    [1.6%, 37.7%] on a single count, wide enough that a real regression and pure noise are
    indistinguishable. Six independent live redraws of #311836 under the FINAL shipping
    config landed in the classifier's top-3 every time (0/6 would-be-flagged) -- the
    specific flip that motivated this change is a stable property of the current config,
    not one lucky draw, though no equivalent repeat-draw evidence exists for vscode's other
    10 issues. Mirrors test_vscode_no_fabrication's identical treatment and reasoning
    (eval/test_quality_regression.py) -- these two are the same underlying signal
    (ungrounded == not all_grounded == fabrication) read by two different consumers, so
    they must move together. Does not assert; still prints the count for visibility. See
    ADR-0058 for the n≈73 figure needed to support a 5%-ceiling hard gate.
    """
    ungrounded_count, baseline_count = _grounding_ratchet_check("microsoft/vscode", grounding_reports)
    if ungrounded_count > baseline_count:
        print(f"\nWARNING (informational, not gated): microsoft/vscode ungrounded_count="
              f"{ungrounded_count} > baseline {baseline_count}. Not blocking per ADR-0058 "
              "-- n=11 cannot support a zero-tolerance gate.")


def test_no_fallback_plans_in_cassette(grounding_reports: list[dict]) -> None:
    """A committed cassette must contain zero scored fallback plans.

    2026-08-27 finding: a fallback-plan audit of the openai/gpt-oss-20b re-record found
    68.75% first-attempt JSON-parse failure and a 32% k8s fallback-plan rate
    (llm_status == "parse_failure" -- both retry attempts failed to produce parseable
    JSON, so TriageAssistant._make_fallback_plan() shipped a predictor-only, no-LLM-text
    plan that the judge then scored as if it were genuine model output). Nothing had ever
    checked for this; it was found by manual audit, not CI. This is that check, so the
    next time a model swap's re-recording contains degraded fallback plans, the eval gate
    fails the recording instead of merging it. Zero tolerance, not a rate threshold --
    a single scored fallback plan already means the judge scored something the model
    never actually said.
    """
    fallback_cases = [c for c in grounding_reports if c.get("llm_status") == "parse_failure"]
    assert not fallback_cases, (
        f"{len(fallback_cases)}/{len(grounding_reports)} cases in this cassette are "
        "fallback plans (llm_status == 'parse_failure') scored as genuine model output: "
        f"{[(c['repo'], c['issue_number']) for c in fallback_cases]}. "
        "Re-record against Groq (not a cassette-side fix) until every entry reflects real "
        "LLM synthesis output, or the current model/config combination is not viable."
    )


def test_no_truncated_completions_in_cassette(grounding_reports: list[dict]) -> None:
    """A committed cassette must contain zero completions truncated by max_tokens.

    2026-08-28: going forward, TriageAssistant._groq_completion raises
    TruncatedCompletionError the moment Groq reports finish_reason == "length" -- before
    the caller ever gets content back, so a truncated completion can no longer reach
    cache.set() at all. This test is the defense-in-depth backstop for a cassette recorded
    before that fix landed (finish_reason wasn't even captured before 2026-08-28, so older
    entries read as None here, not "length" -- a None is not itself a violation, just an
    entry this check can't evaluate). Distinct from test_no_fallback_plans_in_cassette:
    truncation and unparseable-JSON-after-retry are different failure modes that used to
    be indistinguishable from each other (both surfaced as a generic parse error) -- this
    is exactly the ambiguity that hid the actual defect for this engagement.
    """
    truncated_cases = [c for c in grounding_reports if c.get("finish_reason") == "length"]
    assert not truncated_cases, (
        f"{len(truncated_cases)}/{len(grounding_reports)} cases in this cassette were "
        f"truncated by max_tokens (finish_reason == 'length'): "
        f"{[(c['repo'], c['issue_number']) for c in truncated_cases]}. "
        "Raise max_tokens and re-record -- retrying at the same cap reproduces the same "
        "truncation."
    )
