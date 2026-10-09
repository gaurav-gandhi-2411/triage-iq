"""Drift guard: every number /eval/summary publishes must equal its committed source report.

reports/eval_summary.json is hand-maintained, and the /eval page renders it verbatim. It drifted
repeatedly (judge block, 2026-10-05; CQR coverage and classifier calibration, found in the
2026-10-05 page audit), so each published block is pinned here to the committed artifact that
produced it. Same pattern as tests/test_api.py::test_eval_summary_judge_block_matches_committed_baseline.

Surface (85a): this guards the blocks backed by a COMMITTED source. The one exception is
leakage.honest_metrics.*.ci_coverage, whose source reports/resolution_results.json is gitignored;
it is only checked for internal consistency (integer hit-count over n_test), not against its source.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
SLUG = {"microsoft/vscode": "microsoft_vscode", "kubernetes/kubernetes": "kubernetes_kubernetes"}


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def summary() -> dict:
    return _load("eval_summary.json")


def _manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (
        (ROOT / "data" / "models" / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    ):
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def _recorded_ece() -> dict[str, float]:
    """Read eval/test_invariants.py::_RECORDED_ECE without importing the module (heavy deps)."""
    tree = ast.parse((ROOT / "eval" / "test_invariants.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_RECORDED_ECE":
            return ast.literal_eval(node.value)
    raise AssertionError("_RECORDED_ECE not found in eval/test_invariants.py")


# --- leakage / resolution ---------------------------------------------------------------------


def test_split_label_has_no_duplicated_suffix(summary):
    """Regression: the page rendered 'created_at (was: closed_at) (was: closed_at)' because the data
    already carried the parenthetical and the UI appended another."""
    lk = summary["leakage"]
    assert "(was" not in lk["fixed_split"]
    assert lk["fixed_split"] == "created_at"
    assert lk["previous_split"] == "closed_at"


@pytest.mark.parametrize(
    "key,repo", [("k8s", "kubernetes_kubernetes"), ("vscode", "microsoft_vscode")]
)
def test_honest_metrics_match_w6_diagnosis(summary, key, repo):
    w6 = _load("w6_resolution_diagnosis.json")["repos"][repo]
    got = summary["leakage"]["honest_metrics"][key]
    pr = w6["point_regression"]
    assert got["lgbm_mae_days"] == pr["model_mae_days"]
    assert got["naive_mae_days"] == pr["naive_mae_days"]
    assert got["n_test"] == w6["n_test"]
    assert got["improvement_pct"] == f"{pr['model_recovers_pct_of_naive_error']:+.1f}%"
    # the same improvement recomputed from the two MAEs (guards the source's own rounding)
    recomputed = (pr["naive_mae_days"] - pr["model_mae_days"]) / pr["naive_mae_days"] * 100
    assert got["improvement_pct"] == f"{recomputed:+.1f}%"


@pytest.mark.parametrize("key", ["k8s", "vscode"])
def test_ci_coverage_is_an_integer_hit_count_over_n_test(summary, key):
    """Only internal-consistency check possible: the source (resolution_results.json) is gitignored."""
    m = summary["leakage"]["honest_metrics"][key]
    hits = round(m["ci_coverage"] * m["n_test"])
    # value is stored rounded to 4 dp, so compare after re-rounding the implied hit-count ratio
    assert round(hits / m["n_test"], 4) == m["ci_coverage"], (m["ci_coverage"], m["n_test"])


def test_vscode_bucket_delta_matches_w6(summary):
    b = _load("w6_resolution_diagnosis.json")["repos"]["microsoft_vscode"][
        "bucket_classifier_vs_naive"
    ]["accuracy_delta_bootstrap"]
    got = summary["leakage"]["honest_metrics"]["vscode"]
    assert got["bucket_vs_naive_delta_pp"] == pytest.approx(b["mean"] * 100, abs=0.01)
    assert got["bucket_vs_naive_ci95_pp"] == pytest.approx(
        [b["ci95_lower"] * 100, b["ci95_upper"] * 100], abs=0.01
    )


# --- classifier calibration ------------------------------------------------------------------


@pytest.mark.parametrize("repo", ["microsoft_vscode", "kubernetes_kubernetes"])
def test_calibration_matches_production_classifier_report(summary, repo):
    """The page showed the retired 28/35-class classifier's T_opt/ECE (0.2981/0.1381) after the
    47-class retrain shipped. Pin T_opt/ECE to the retrain report and the artifact hash to MANIFEST."""
    final = _load("multilabel_classifier_final_training.json")[repo]
    got = summary["calibration"]["repos"][repo]
    assert got["T_opt"] == pytest.approx(final["T_opt"], abs=1e-4)
    assert got["ece_test"] == final["ece_test"]
    assert got["classifier_sha256"] == _manifest()[f"data/models/component_classifier_{repo}.pkl"]
    # never carry the retired classifier's numbers (0.2981 / 0.3234) as current
    retired = summary["calibration"]["previous_classifier"]["repos"][repo]
    assert got["T_opt"] != retired["T_opt"]
    assert got["ece_test"] != retired["ece_test"]
    # validation-split ECE was never recorded for the retrain: it must be explicitly absent, not stale
    assert got["ece_before_val"] is None
    assert got["ece_after_val"] is None


@pytest.mark.parametrize("repo", ["microsoft_vscode", "kubernetes_kubernetes"])
def test_calibration_eval_set_ece_matches_recorded_invariant(summary, repo):
    assert summary["calibration"]["repos"][repo]["ece_eval_set"] == _recorded_ece()[repo]


# --- reranker --------------------------------------------------------------------------------


def test_reranker_block_matches_phase2_robustness(summary):
    p2 = _load("phase2_robustness.json")
    rr = summary["reranker"]
    assert rr["model_tested"] == p2["model_id"].split("/")[-1]
    assert rr["phase2_robustness_n"] == p2["n_eval"]
    assert rr["phase2_n_bootstrap"] == p2["n_bootstrap"]
    assert rr["phase2_baseline_r5"] == pytest.approx(p2["baseline_r5"], abs=1e-4)
    assert rr["phase2_reranker_r5"] == pytest.approx(p2["reranker_r5"], abs=1e-4)
    assert rr["phase2_delta_pp"] == f"{p2['delta_mean'] * 100:+.2f}pp"
    assert rr["phase2_ci_95"] == f"[{p2['ci_lo_95'] * 100:+.1f}pp, {p2['ci_hi_95'] * 100:+.1f}pp]"


# --- conformal (CQR) -------------------------------------------------------------------------


def test_cqr_snapshot_is_the_manifest_pinned_production_artifact():
    """reports/cqr_conformal_adjustments_v2.snapshot.json is a byte copy of the gitignored
    data/models/cqr_conformal_adjustments_v2.json (the file loader.py serves); the MANIFEST hash
    proves it is the served one."""
    digest = hashlib.sha256(
        (REPORTS / "cqr_conformal_adjustments_v2.snapshot.json").read_bytes()
    ).hexdigest()
    assert digest == _manifest()["data/models/cqr_conformal_adjustments_v2.json"]


def test_cqr_v1_snapshot_still_matches_its_manifest_pin():
    """v1 is no longer served but is still published (eval artifact fingerprint pins it)."""
    digest = hashlib.sha256(
        (REPORTS / "cqr_conformal_adjustments.snapshot.json").read_bytes()
    ).hexdigest()
    assert digest == _manifest()["data/models/cqr_conformal_adjustments.json"]


def test_cqr_v2_vscode_entry_is_unchanged_from_v1():
    """v2 only re-calibrates k8s; the vscode entry must be carried over verbatim."""
    v1 = _load("cqr_conformal_adjustments.snapshot.json")["repos"]["microsoft/vscode"]
    v2 = _load("cqr_conformal_adjustments_v2.snapshot.json")["repos"]["microsoft/vscode"]
    assert v1 == v2


@pytest.mark.parametrize(
    "repo,src_key",
    [("kubernetes/kubernetes", None), ("microsoft/vscode", "40_60")],  # loader.py prefers 40_60
)
def test_conformal_block_matches_cqr_artifact(summary, repo, src_key):
    art = _load("cqr_conformal_adjustments_v2.snapshot.json")["repos"][repo]
    src = art[src_key] if src_key else art
    got = summary["conformal"]["by_repo"][repo]
    assert got["n_calibration"] == src["n_calibration"]
    assert got["n_true_test"] == src["n_true_test"]
    assert got["q_adjustment_hours"] == src["q_adjustment_hours"]
    assert got["q_adjustment_days"] == src["q_adjustment_days"]
    assert got["empirical_coverage"] == src["empirical_test_coverage"]
    assert got["coverage_ci95_lower"] == src["coverage_ci95_lower"]
    assert got["coverage_ci95_upper"] == src["coverage_ci95_upper"]
    assert got["raw_interval_coverage"] == src["raw_interval_coverage"]
    assert got["median_width_raw_days"] == src["median_width_raw_days"]
    assert got["median_width_conformal_days"] == src["median_width_conformal_days"]


def test_vscode_split_sensitivity_matches_cqr_artifact(summary):
    art = _load("cqr_conformal_adjustments_v2.snapshot.json")["repos"]["microsoft/vscode"]
    s = summary["conformal"]["by_repo"]["microsoft/vscode"]["split_sensitivity"]
    assert s["split_30_70"]["empirical_coverage"] == art["30_70"]["empirical_test_coverage"]
    assert s["split_40_60"]["empirical_coverage"] == art["40_60"]["empirical_test_coverage"]
    want = round(
        (art["40_60"]["empirical_test_coverage"] - art["30_70"]["empirical_test_coverage"]) * 100, 1
    )
    assert s["divergence_pp"] == pytest.approx(want, abs=0.05)


def test_conformal_note_quotes_the_block_coverage(summary):
    v = summary["conformal"]["by_repo"]["microsoft/vscode"]
    assert f"{v['empirical_coverage'] * 100:.1f}%" in v["exchangeability_note"]


# --- resolution_served (ADR-0064): what production serves, tied to code and manifest -----------


def _builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_resolution_served_block", ROOT / "scripts" / "build_resolution_served_block.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolution_served_block_equals_its_source_reports(summary):
    assert summary["resolution_served"] == _builder().build()


def test_resolution_served_manifest_hashes_match_the_manifest(summary):
    pinned = summary["resolution_served"]["provenance"]["manifest_sha256"]
    manifest = _manifest()
    assert pinned
    for path, digest in pinned.items():
        assert manifest[path] == digest, path


def test_resolution_served_describes_the_point_the_code_serves(summary):
    """The published claim "the train median is the served point" must match POINT_ESTIMATE_TRUSTED."""
    from triage_iq.models.resolution import POINT_ESTIMATE_TRUSTED, repo_slug

    block = summary["resolution_served"]
    assert block["point_estimate"]["served"] == "train_median"
    for repo in block["repos"]:
        assert POINT_ESTIMATE_TRUSTED[repo_slug(repo)] is False, repo


def test_resolution_served_k8s_interval_and_bucket_were_unchanged_by_d7(summary):
    unchanged = summary["resolution_served"]["repos"]["kubernetes/kubernetes"]["unchanged_by_d7"]
    assert unchanged == {"interval_lo_hi_bit_identical": True, "bucket_identical": True}


def test_old_honest_metrics_table_is_labelled_historical(summary):
    assert summary["leakage"]["honest_metrics_status"].startswith("historical")
