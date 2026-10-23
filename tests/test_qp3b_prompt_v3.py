"""Phase 3b prompt v3 (flags 1+2, docs/QP3B_DESIGN_2026-10.md). Network-free, no LLM.

C1: with both flags off the messages sent to the LLM are byte-identical to main (digest pinned).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from tests.test_vscode_naive_point import K8S, VSCODE, _assistant, _issue

from triage_iq.models import triage as triage_mod
from triage_iq.prompts.triage_prompt import build_triage_prompt

TOP3 = [
    {"label": "sig/node", "confidence": 0.61234},
    {"label": "sig/network", "confidence": 0.20111},
    {"label": "sig/api-machinery", "confidence": 0.05},
]
SIMILAR = [
    {"number": 101, "score": 0.91234, "text": "Kubelet fails to start pods after restart " * 5},
    {"number": 202, "score": 0.8, "text": "Pod stuck in ContainerCreating"},
    {"number": 303, "score": 0.7123, "text": "x" * 300},
    {"number": 404, "score": 0.6, "text": "Fourth"},
    {"number": 505, "score": 0.5, "text": "Fifth"},
]


class _Captured(Exception):  # noqa: N818 - control-flow sentinel, not an error
    """Raised by the capturing cache so the exact messages can be read without any LLM call."""

    def __init__(self, messages: list[dict]) -> None:
        self.messages = messages


class _CapturingCache:
    def compute_key(self, provider, model, messages, temperature, max_tokens):  # noqa: ANN001, ANN201
        raise _Captured(messages)


def _messages(asst, issue=None):  # noqa: ANN001, ANN202
    """The exact message list the assistant would send for `issue` (cache key input)."""
    asst._cache = _CapturingCache()
    sig = asst._collect_signals(issue if issue is not None else _issue())
    try:
        asst._call_llm_verbose(sig)
    except _Captured as c:
        return c.messages
    raise AssertionError("no messages captured")


def _off_projection(monkeypatch: pytest.MonkeyPatch) -> str:
    """sha256 over prompts + full message lists for representative inputs, every flag off."""
    prompts = []
    for kw in (
        {},
        {"resolution_bucket": "weeks", "resolution_confidence_pct": 55.0},
        {"resolution_bucket": "days", "resolution_confidence_pct": 10.0},
        {"resolution_point_source": "train_median", "interval_calibrated": True},
        {"max_body_chars": 40},
        {"issue_body": ""},
    ):
        args = {
            "issue_title": "Pod stuck",
            "issue_body": "kubelet never schedules " * 60,
            "classifier_top3": TOP3,
            "similar_issues": SIMILAR,
            "resolution_point_days": 3.04,
            "resolution_lower_days": 0.5,
            "resolution_upper_days": 20.25,
            "repo": K8S,
        } | kw
        prompts.append(build_triage_prompt(**args))
    msgs = []
    for repo in (K8S, VSCODE):
        msgs.append(_messages(_assistant(repo)))
    # Token-budget shrink path (body re-truncated): a 6500 TPM ceiling leaves <800 completion
    # tokens at the 800-char preview (est. 5593) but enough at 400 chars.
    long_issue = _issue()
    long_issue["body_clean"] = "kubelet never schedules " * 200
    full = _messages(_assistant(K8S), long_issue)[-1]["content"]
    monkeypatch.setattr(triage_mod, "_GROQ_TPM_LIMIT", 6500)
    shrunk = _messages(_assistant(K8S), long_issue)
    assert len(shrunk[-1]["content"]) < len(full)  # the guard really rebuilt the prompt
    msgs.append(shrunk)
    blob = json.dumps({"prompts": prompts, "messages": msgs}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# sha256 of _off_projection(), computed on origin/main (7f3362d) BEFORE any prompt-v3 change.
# Any drift means the flags-off prompt changed (QP3B check C1): regenerate only deliberately.
OFF_GOLDEN_SHA256 = "eb66cb345bf3c0c58cbaa5f009caa33c8839ff2be274f88106de2b98e00fa3c5"


def test_flags_off_messages_identical_to_main(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _off_projection(monkeypatch) == OFF_GOLDEN_SHA256
