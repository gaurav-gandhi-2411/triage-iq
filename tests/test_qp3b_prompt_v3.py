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


# --- prompt text (flags 1 and 2) ---------------------------------------------------------------

PROBA = {"low": 0.1, "medium": 0.2, "high": 0.7}
BLOCK = '--- RESOLUTION EVIDENCE (x) ---\n#101: fix PR 9 "Fix it"'
BASE_ARGS = {
    "issue_title": "Pod stuck",
    "issue_body": "kubelet never schedules",
    "classifier_top3": TOP3,
    "similar_issues": SIMILAR,
    "resolution_point_days": 3.04,
    "resolution_lower_days": 0.5,
    "resolution_upper_days": 20.25,
    "repo": K8S,
}


def test_priority_section_states_level_and_asks_to_explain_not_choose() -> None:
    p = build_triage_prompt(**BASE_ARGS, priority_label="high", priority_proba=PROBA)
    assert "Decided priority: high (probabilities: low 0.10, medium 0.20, high 0.70)" in p
    assert 'set priority_guess to "high"' in p
    assert "explaining why this level fits" in p and "Do not choose or argue" in p
    assert p.index("PRIORITY (decided") < p.index("--- TASK ---")
    assert "RESOLUTION EVIDENCE" not in p  # flag 2 text absent when no block


def test_evidence_block_sits_between_system2_and_system3_with_citation_instruction() -> None:
    p = build_triage_prompt(**BASE_ARGS, evidence_block=BLOCK)
    assert p.index("SYSTEM 2") < p.index(BLOCK) < p.index("SYSTEM 3")
    assert "cite that issue in the step text as #N" in p
    assert "Decided priority" not in p  # flag 1 text absent when no label
    off = build_triage_prompt(**BASE_ARGS)
    assert p.replace(f"\n{BLOCK}\n", "", 1).split("--- TASK ---")[0] == off.split("--- TASK ---")[0]


def test_empty_evidence_and_no_label_equal_v2_prompt() -> None:
    assert build_triage_prompt(**BASE_ARGS, evidence_block="", priority_label=None) == (
        build_triage_prompt(**BASE_ARGS)
    )


# --- assistant integration ---------------------------------------------------------------------


class _FakePriority:
    def __init__(self, label: str = "high", fail: bool = False) -> None:
        self.label, self.fail = label, fail

    def predict(self, title: str, body: str):  # noqa: ANN201
        from triage_iq.models.priority import PriorityPrediction

        if self.fail:
            raise RuntimeError("boom")
        return PriorityPrediction(self.label, dict(PROBA))


def _evidence(numbers=(7,), merged: str = "2019-01-01T00:00:00Z"):  # noqa: ANN001, ANN202
    """Evidence for `numbers`; the test issue was created 2020-02-01 (tests/test_vscode_naive_point)."""
    from triage_iq.models.resolution_evidence import EvidenceRecord, ResolutionEvidence

    rec = EvidenceRecord(((907, "Fix kubelet", "nil pod", merged),), "")
    return ResolutionEvidence(dict.fromkeys(numbers, rec))


def test_evidence_dated_after_the_query_is_not_prompted_and_prompt_equals_off() -> None:
    off = _messages(_assistant(K8S))
    asst = _assistant(K8S)
    asst.resolution_evidence = _evidence(merged="2020-02-01T00:00:01Z")  # 1 s after creation
    assert _messages(asst) == off
    asst.resolution_evidence = _evidence(merged="2020-01-31T23:59:59Z")  # 1 s before
    assert "#7: fix PR 907" in _messages(asst)[-1]["content"]
    asst.resolution_evidence = _evidence(merged="2020-02-01T00:00:00Z")  # exactly at: dropped
    assert _messages(asst) == off


def _on(repo: str = K8S, priority=None, evidence=None):  # noqa: ANN001, ANN202
    asst = _assistant(repo)
    asst.priority_model = _FakePriority() if priority is None else priority
    asst.resolution_evidence = _evidence() if evidence is None else evidence
    return asst


def test_flags_on_user_prompt_carries_both_blocks_and_prefix_is_unchanged() -> None:
    off = _messages(_assistant(K8S))
    on = _messages(_on())
    assert on[:-1] == off[:-1]  # system + few-shot untouched: Groq prefix cache stays warm
    user = on[-1]["content"]
    assert "Decided priority: high" in user
    assert '#7: fix PR 907 "Fix kubelet": nil pod' in user
    assert user != off[-1]["content"]


def test_evidence_flag_without_any_neighbour_evidence_leaves_prompt_unchanged() -> None:
    off = _messages(_assistant(K8S))
    asst = _assistant(K8S)
    asst.resolution_evidence = _evidence(numbers=(12345,))  # not a retrieved neighbour
    assert _messages(asst) == off


def test_priority_model_error_leaves_prompt_without_priority_section() -> None:
    asst = _on(priority=_FakePriority(fail=True))
    user = _messages(asst)[-1]["content"]
    assert "Decided priority" not in user and "RESOLUTION EVIDENCE" in user


def test_budget_shrink_rebuild_keeps_both_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    asst = _on()
    issue = _issue()
    issue["body_clean"] = "kubelet never schedules " * 200
    msgs = _messages(asst, issue)
    full = len(msgs[-1]["content"])
    # Leaves 799 (< the 800 floor) completion tokens at the 800-char preview; the 400-char
    # preview frees ~150 tokens, so the guard recovers by re-truncating the body.
    limit = triage_mod._estimate_prompt_tokens(msgs) + triage_mod._PROMPT_SIZE_SAFETY_MARGIN + 799
    monkeypatch.setattr(triage_mod, "_GROQ_TPM_LIMIT", limit)
    user = _messages(asst, issue)[-1]["content"]
    assert len(user) < full  # the guard really rebuilt the prompt with a shorter body
    assert "Decided priority: high" in user and "#7: fix PR 907" in user


# --- safety net + C5 degraded paths ------------------------------------------------------------


def _plan(priority: str):  # noqa: ANN202
    from tests.test_component_decision import _plan as base

    p = base("a")
    p.priority_guess, p.priority_rationale = priority, f"{priority} because reasons"
    return p


def _run(monkeypatch, asst, plan, status: str = "ok"):  # noqa: ANN001, ANN202
    monkeypatch.setattr(asst, "_call_llm_verbose", lambda s: (plan, "", {}, status, False))
    return asst.triage_with_metadata(_issue())


def test_safety_net_fires_when_llm_returns_another_level(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, meta = _run(monkeypatch, _on(), _plan("low"))
    assert plan.priority_guess == "high"  # served level stays the model's
    assert plan.priority_rationale.startswith("Text-only priority model: high")
    assert meta["priority_safety_net_fired"] is True
    assert meta["priority_in_prompt"] is True
    assert meta["evidence_neighbours"] == [7]


def test_safety_net_quiet_when_llm_explains_the_decided_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, meta = _run(monkeypatch, _on(), _plan("high"))
    assert plan.priority_rationale == "high because reasons"  # the LLM's explanation is served
    assert meta["priority_safety_net_fired"] is False


def test_flags_off_metadata_has_no_v3_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _, meta = _run(monkeypatch, _assistant(K8S), _plan("low"))
    assert not [k for k in meta if "priority" in k or "evidence" in k]


def _raise_rate_limit(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
    import groq
    import httpx

    msg = "Rate limit reached ... tokens per day (TPD): Limit 200000, Used 199990, Requested 5000."
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    raise groq.RateLimitError(
        msg,
        response=httpx.Response(429, request=req),
        body={"error": {"message": msg, "type": "t", "code": "rate_limit_exceeded"}},
    )


def _raise_schema(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
    raise triage_mod.SchemaValidationError("json_validate_failed", "missing field")


def _raise_truncated(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
    raise triage_mod.TruncatedCompletionError(10, 10, "cut")


def _junk(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
    return "not json", {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.mark.parametrize(
    ("completion", "status_prefix"),
    [
        (_raise_rate_limit, "degraded_provider"),
        (_raise_schema, "degraded_schema_invalid"),
        (_raise_truncated, "degraded_truncated"),
        (_junk, "parse_failure"),
    ],
)
def test_degraded_paths_carry_every_v3_field(
    monkeypatch: pytest.MonkeyPatch,
    completion,  # noqa: ANN001
    status_prefix: str,
) -> None:
    asst = _on()
    monkeypatch.setattr(asst, "_groq_completion", completion)
    plan, meta = asst.triage_with_metadata(_issue())
    assert meta["llm_status"].startswith(status_prefix), meta["llm_status"]
    assert plan.priority_guess == "high"  # model level served on every fallback plan
    assert "high (p=0.70" in plan.priority_rationale
    assert meta["priority_decided_by"] == "model"
    assert meta["priority_in_prompt"] is True
    assert meta["priority_safety_net_fired"] is False  # no LLM proposal on a fallback plan
    assert meta["priority_llm_proposal"] is None
    assert meta["evidence_neighbours"] == [7]


def test_budget_exhausted_degrade_carries_every_v3_field(monkeypatch: pytest.MonkeyPatch) -> None:
    asst = _on()
    monkeypatch.setattr(triage_mod, "_GROQ_TPM_LIMIT", 100)  # no body shrink can make this fit
    plan, meta = asst.triage_with_metadata(_issue())
    assert meta["llm_status"] == "degraded_insufficient_budget"
    assert plan.priority_guess == "high"
    assert meta["priority_in_prompt"] is True and meta["priority_safety_net_fired"] is False
    assert meta["evidence_neighbours"] == [7]


def test_priority_model_error_degrade_reports_not_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asst = _on(priority=_FakePriority(fail=True))
    monkeypatch.setattr(asst, "_groq_completion", _raise_rate_limit)
    plan, meta = asst.triage_with_metadata(_issue())
    assert plan.priority_guess == "medium"  # no model level to serve: documented default
    assert meta["priority_decided_by"] == "llm"
    assert meta["priority_in_prompt"] is False and meta["priority_safety_net_fired"] is False
    assert meta["evidence_neighbours"] == [7]


def test_evidence_flag_without_artifact_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _, meta = _run(monkeypatch, _assistant(K8S), _plan("low"))  # resolution_evidence is None
    assert "evidence_neighbours" not in meta


def test_api_exposes_v3_keys_only_when_present() -> None:
    from tests.test_vscode_naive_point import _post

    base = _post({})
    assert "_evidence_neighbours" not in base and "_priority_safety_net_fired" not in base
    on = _post(
        {
            "priority_decided_by": "model",
            "priority_llm_proposal": "low",
            "priority_model_proba": PROBA,
            "priority_in_prompt": True,
            "priority_safety_net_fired": True,
            "evidence_neighbours": [7],
        }
    )
    assert on["_priority_in_prompt"] is True and on["_priority_safety_net_fired"] is True
    assert on["_evidence_neighbours"] == [7]

@pytest.fixture(autouse=True)
def _pre_g3_model_interval_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests pin the model-interval / D7 path that every repo without a served constant still
    uses (and the pre-G3 k8s goldens). G3 gives k8s a fixed constant instead, covered by
    tests/test_k8s_constant_serving.py, so the table is emptied here to keep what they pin."""
    monkeypatch.setattr("triage_iq.models.triage.SERVED_CONSTANT_RESOLUTION", {})
