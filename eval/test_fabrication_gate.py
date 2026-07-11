from __future__ import annotations

"""Fabrication gate: HARD, BLOCKING CI check (ADR-0029).

Elevates the deterministic grounding check from informational (ADR-0015's verifier;
ADR-0028 Phase B3's `fabrication_rate`, surfaced but gating nothing) to a hard gate.
A plan that cites a component not in classifier_top3, or a similar-issue number not in
the retrieved set, is the failure mode most likely to mislead a human triager — a
confidently-cited nonexistent source. This file's CI job (`fabrication-gate` in
eval-gate.yml) has NO `continue-on-error`, unlike the two existing eval-gate.yml jobs.

Ratchet semantics, not zero-tolerance (see ADR-0029 for the measured rate that justifies
this): the gate fails on any fabrication BEYOND the pinned, human-approved baseline
below, not on the 2 known cases already in it. Both were re-measured 2026-07-12 and
confirmed genuine (not a top-3-vs-top-5 threshold artifact) — `predicted_component` for
both ('webview' vscode #311836, 'storage' k8s #13057) is not in the classifier's label
space AT ALL (28 vscode / 35 k8s labels checked directly), not merely outside top-3.
A rate this low (2/64 = 3.1% overall) is what makes a hard gate safe: it will not
false-fail legitimate plans, only genuine fabrications beyond what's already measured.

Run with:
    pytest eval/test_fabrication_gate.py -v
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"

# Recorded grounding baseline — mirrors _RECORDED_ECE in test_invariants.py. Produced by
# scripts/measure_grounding.py against the CURRENT (unmodified) cassette over the clean,
# train-decontaminated n=64 eval set (ADR-0018/B1) with the local qwen3:8b judge (ADR-0019).
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
# from the contaminated gold set. Both are gone from the clean n=64 set or no longer
# ungrounded under the new local-judge recording — re-derived against the actual committed
# cassette below, not carried forward.
#
# Re-derived 2026-07-11 (ADR-0028 Phase B1): k8s #14398 quarantined from eval_set.jsonl
# (a near-duplicate leak, cosine 0.907 vs classifier_train #14399 — unrelated to
# grounding), changing the file's hash and k8s's n from 54 to 53. #14398 itself was
# grounded, so this is purely a denominator/hash change — the same two known-bad cases
# (#13057 k8s, #311836 vscode) persist unchanged; re-confirmed directly via
# measure_grounding.compute_grounding_reports() against the current committed cassette,
# and again on 2026-07-12 for ADR-0029 (see this file's module docstring).
#
# Moved here from eval/test_invariants.py 2026-07-12 (ADR-0029): the ratchet was already
# correctly designed (ceiling, not zero-tolerance) but lived in an informational
# (continue-on-error) job alongside unrelated invariants. It's promoted to its own
# always-blocking job — the mechanism did not change, only its CI blast radius.
_GROUNDING_BASELINE = {
    "eval_set_hash": "86c1df0a066c9dce5ece19ff9da4b3298563b8a59c2d6f8d807e4658e43260a4",
    "per_repo": {
        "kubernetes/kubernetes": {
            "ungrounded_count": 1,
            "n": 53,
            "known_ungrounded_cases": {
                13057: {
                    "axis": "component",
                    "detail": "predicted_component 'storage' not in classifier_top3 "
                    "['provider/gcp', 'kubectl', 'security'] (nor anywhere in the "
                    "35-label classifier label space)",
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
                    "['suggest', 'accessibility', 'debug'] (nor anywhere in the "
                    "28-label classifier label space)",
                },
            },
        },
    },
}


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

    HARD GATE (ADR-0029): this job has no continue-on-error — a failure here blocks the
    workflow. Checked per-repo (not pooled): a regression concentrated in one repo must
    fail this test on its own, independent of the other repo's volume. Guards against
    silent regressions in synthesis grounding (component/similar-issue hallucination)
    creeping in above the measured, approved baseline (2/64 total — see module docstring).
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
            f"baseline {baseline['ungrounded_count']} — a plan cited a component or "
            "similar-issue this pipeline's own upstream signals never produced. See "
            "plan.grounding_status for the offending plan(s); this is a hard fail "
            "(ADR-0029), not a soft quality miss."
        )


def test_grounding_known_cases_still_flagged(grounding_reports: list[dict]) -> None:
    """The two known-bad cases (#13057 k8s, #311836 vscode) must still be caught by name.

    This catches a verifier regressed to a no-op, which would otherwise trivially satisfy
    the ratchet test at 0 <= 1 ungrounded per repo. See ADR-0015.

    Re-derived against the clean n=64 set + local qwen3:8b judge (ADR-0018/0019). The old
    pins (#1678 similar_issue-axis, #13435 component-axis, both from the contaminated n=60
    set) are gone: #1678 isn't in the clean n=64 set, and no similar_issue-axis hallucination
    exists in the current committed cassette to pin — both current pins are component-axis.
    This is not a weaker test by design; it reflects what's actually in the committed
    recording.
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
