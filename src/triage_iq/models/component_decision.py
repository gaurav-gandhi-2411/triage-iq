"""Classifier-decides-component policy (spec-quality-push.md sec. 5, Phase 3a).

Pure and I/O free. The served component is the classifier's top-1 unless the LLM proposed a
different label that is (a) inside the classifier's top-3, (b) declared as a `model_override`, and
(c) backed by a reason that passes the existing `verify_override_reason_grounded` (a cited
retrieved issue number or a verbatim entity from the issue). Post-hoc enforcement: no prompt
change, so recorded cassettes stay valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from triage_iq.models.grounding import (
    verify_declared_attribution,
    verify_override_reason_grounded,
)

DECIDED_BY_CLASSIFIER = "classifier"
DECIDED_BY_OVERRIDE = "llm_validated_override"
# The classifier produced no usable label (its failure fallback is the single label "unknown"):
# there is nothing to enforce, so the LLM's component is served unchanged rather than "unknown".
DECIDED_BY_LLM_NO_CLASSIFIER = "llm_classifier_unavailable"


@dataclass(frozen=True)
class ComponentDecision:
    """component: label to serve (classifier's exact spelling when one matches).
    decided_by: DECIDED_BY_* value. llm_proposal: the LLM's component as emitted.
    changed: served component differs from the LLM's (case/whitespace-insensitive)."""

    component: str
    decided_by: str
    llm_proposal: str
    changed: bool


def _norm(label: str) -> str:
    return " ".join(str(label).split()).lower()


def decide_component(
    llm_component: str,
    classifier_top3: list[dict],
    declared_attribution: Any,
    issue_title: str,
    issue_body: str,
    retrieved_numbers: set[int],
) -> ComponentDecision:
    """Pick the component to serve; see module docstring for the policy."""
    labels = [str(e["label"]) for e in classifier_top3 if str(e.get("label", "")).strip()]
    if not labels or _norm(labels[0]) == "unknown":
        return ComponentDecision(llm_component, DECIDED_BY_LLM_NO_CLASSIFIER, llm_component, False)

    top1 = labels[0]
    by_norm = {_norm(label): label for label in reversed(labels[:3])}  # top-1 wins a tie
    llm_key = _norm(llm_component)

    served, decided_by = top1, DECIDED_BY_CLASSIFIER
    if llm_key in by_norm and llm_key != _norm(top1) and declared_attribution is not None:
        # Reuse grounding.py as-is: it only reads .declared_attribution / .predicted_component.
        shim = SimpleNamespace(
            declared_attribution=declared_attribution, predicted_component=by_norm[llm_key]
        )
        report = verify_declared_attribution(shim, classifier_top3, retrieved_numbers)
        if report.component_declaration == "honest_override" and verify_override_reason_grounded(
            declared_attribution.component_override_reason,
            issue_title,
            issue_body,
            retrieved_numbers,
        ):
            served, decided_by = by_norm[llm_key], DECIDED_BY_OVERRIDE

    return ComponentDecision(served, decided_by, llm_component, _norm(served) != llm_key)


def repair_component_text(text: str, old: str, new: str) -> str:
    """Replace the whole-word, case-insensitive component name `old` by `new` in `text`.

    Word edges are alphanumerics/underscore, so replacing "git" leaves "github" alone. A no-op
    when `old` is empty or equals `new` ignoring case.
    """
    if not old.strip() or _norm(old) == _norm(new):
        return text
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(old.strip())}(?![A-Za-z0-9_])"
    return re.sub(pattern, lambda _m: new, text, flags=re.IGNORECASE)
