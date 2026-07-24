"""Deterministic selective-prediction (abstention) gate over existing pipeline signals.

ADR-0021: REJECTED for v1 as a live default -- gated off by default
(TRIAGE_ENABLE_ABSTENTION_GATE in src/triage_iq/api/app.py). Component-stage confidence is a
real but marginal, noisy signal (a deferred product-value call, not shipped). Resolution-stage
interval width was checked directly against the data and does NOT predict coverage failure
(mean width is statistically indistinguishable between covered and uncovered issues on k8s) --
rejected outright, not just deferred. Kept here, not deleted, so a future revisit (once a
coverage-discriminative uncertainty signal exists) has a working starting point.

Not a new model: component-stage abstention reads component_confidence (calibrated TF-IDF,
ADR-0004) and grounding_status.component_grounded (ADR-0015); resolution-stage abstention reads
the CQR-adjusted interval width (ADR-0010), computed the same way as the /triage handler's
resolution_interval_conformal. Priority stage has no calibrated confidence signal anywhere in
the pipeline and is intentionally not gated — see ADR-0021.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triage_iq.models.triage import AbstentionStatus, TriagePlan

# Thresholds from the n=65 sweep (reports/abstention_tradeoff.json) -- retained as a record of
# what was measured, not as a shipped default (ADR-0021 rejected shipping either stage for v1).
# Missing a repo key means that repo's stage is never gated (fails open, same policy as
# resolution_interval_conformal).
#
# *** STALE as of ADR-0036 (2026-07-24) -- DO NOT ENABLE TRIAGE_ENABLE_ABSTENTION_GATE WITHOUT
# *** RE-DERIVING THESE FIRST. These values were tuned to the single-label classifier's
# *** confidence distribution. ADR-0036 shipped a multi-label one-vs-rest classifier with a
# *** different, independently-recalibrated confidence stream -- under it, these SAME fixed
# *** thresholds fire at a wildly different rate on the held-out test set: k8s 59.8%->0.0%,
# *** vscode 13.9%->0.5% (reports/tfidf_multilabel_calibration_and_threshold_check.json). The
# *** gate has been off throughout, so nothing live changed -- but flipping the flag today with
# *** these constants unchanged would enable a silently-dead gate (fires ~never on k8s), not the
# *** behavior this ADR's tradeoff analysis measured. See ADR-0021's Consequences section.
COMPONENT_CONFIDENCE_THRESHOLD: dict[str, float] = {
    "kubernetes/kubernetes": 0.45,
    "microsoft/vscode": 0.29,
}
RESOLUTION_WIDTH_THRESHOLD_DAYS: dict[str, float] = {
    "kubernetes/kubernetes": 91.4236,
    "microsoft/vscode": 195.2045,
}


def compute_abstention_status(
    plan: TriagePlan,
    repo: str,
    component_grounded: bool,
    conformal_width_days: float | None,
) -> AbstentionStatus:
    """Return the selective-prediction gate result for one triaged issue.

    Args:
        plan: The synthesized TriagePlan (reads component_confidence).
        repo: Repository slug, used to look up the per-repo thresholds.
        component_grounded: grounding_status.component_grounded for this plan (ADR-0015) —
            a hard abstain trigger, not swept: a component the classifier's own top-3 doesn't
            support abstains regardless of confidence.
        conformal_width_days: upper_days - lower_days of the CQR-adjusted interval, or None
            when conformal adjustments are unavailable for this repo.
    """
    from triage_iq.models.triage import AbstentionStatus, StageAbstention

    conf_threshold = COMPONENT_CONFIDENCE_THRESHOLD.get(repo)
    if not component_grounded:
        component = StageAbstention(abstained=True, reason="ungrounded")
    elif conf_threshold is not None and plan.component_confidence < conf_threshold:
        component = StageAbstention(abstained=True, reason="low_confidence")
    else:
        component = StageAbstention(abstained=False, reason="")

    width_threshold = RESOLUTION_WIDTH_THRESHOLD_DAYS.get(repo)
    if (
        conformal_width_days is not None
        and width_threshold is not None
        and conformal_width_days > width_threshold
    ):
        resolution = StageAbstention(abstained=True, reason="wide_interval")
    else:
        resolution = StageAbstention(abstained=False, reason="")

    return AbstentionStatus(component=component, resolution=resolution)
