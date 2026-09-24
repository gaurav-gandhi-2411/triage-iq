"""Production's default prompt config must be the config the committed cassette was recorded
under (2026-09-23).

TRIAGE_PROMPT_INCLUDE_ATTRIBUTION was opt-in while the eval cassette and baseline were
recorded with it ON -- so production, run_eval.py and CI all ran a prompt the baseline never
measured, the same defect class as ADR-0059's stale-classifier incident. These tests pin the
default from the process's point of view (env var UNSET), through two independent routes:

1. the recorder's own fingerprint: _compute_prompt_hash() with no env var must equal the
   prompt_hash every committed checkpoint entry was recorded under;
2. the real serving path: TriageAssistant._call_llm_verbose (Groq client mocked, zero live
   calls) must send exactly the system prompt + few-shot turns stored in every cassette entry.

If either fails, prod and eval have diverged: re-record, or change the default back
deliberately -- never just update the expected value here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from triage_iq.models.triage import TriageAssistant  # noqa: E402
from triage_iq.prompts.triage_prompt import attribution_prompt_enabled  # noqa: E402

CASSETTE = ROOT / "eval" / "cassettes" / "eval_cassette.json"
CHECKPOINT = ROOT / "eval" / "cassettes" / "recording_checkpoint.json"


def _synthesis_entries() -> list[dict]:
    entries = json.loads(CASSETTE.read_text(encoding="utf-8"))["entries"].values()
    return [e for e in entries if e.get("provider") == "groq"]


def test_attribution_prompt_is_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION", raising=False)
    assert attribution_prompt_enabled() is True
    monkeypatch.setenv("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION", "0")
    assert attribution_prompt_enabled() is False


def test_default_prompt_hash_matches_recorded_checkpoint(monkeypatch) -> None:
    import record_cassettes as rc

    monkeypatch.delenv("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION", raising=False)
    recorded = {v["prompt_hash"] for v in json.loads(CHECKPOINT.read_text(encoding="utf-8"))["done"].values()}
    assert len(recorded) == 1, f"checkpoint mixes prompt configs: {recorded}"
    assert rc._compute_prompt_hash() == recorded.pop()


def test_serving_path_sends_the_recorded_prompt_prefix(monkeypatch) -> None:
    monkeypatch.delenv("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION", raising=False)
    entries = _synthesis_entries()
    assert entries, "cassette has no synthesis entries to compare against"
    recorded_prefixes = {json.dumps(e["request_messages"][:-1], ensure_ascii=False) for e in entries}
    assert len(recorded_prefixes) == 1, "cassette entries disagree on system prompt/few-shots"

    asst = TriageAssistant.__new__(TriageAssistant)
    asst.repo = "kubernetes/kubernetes"
    asst.model = "openai/gpt-oss-120b"
    asst.temperature = 0.0
    asst.max_tokens = 2048
    asst.seed = 42
    asst._groq_key = "test-key"
    asst.use_structured_output = True  # TriageAssistant.__init__'s default, what prod runs
    asst._cache = None
    signals = {
        "prompt": "Repository: kubernetes/kubernetes\n\n--- ISSUE ---\nTitle: X\nBody:\nShort.\n",
        "classifier_top3": [{"label": "kubectl", "confidence": 0.5}],
        "similar_raw": [], "pred_days": 2.0, "lo_days": 1.0, "hi_days": 5.0,
        "resolution_bucket": "days", "resolution_conf_pct": 40.0,
        "_title": "X", "_body": "Short.", "_include_bucket": False, "_number": 1,
    }
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="{}"), finish_reason="stop")]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=10)
    client.chat.completions.create.return_value = resp
    with patch("groq.Groq", return_value=client):
        asst._call_llm_verbose(signals)

    sent = client.chat.completions.create.call_args_list[0].kwargs["messages"]
    assert json.dumps(sent[:-1], ensure_ascii=False) == recorded_prefixes.pop()
