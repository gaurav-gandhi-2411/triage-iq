"""Judge prompt wording for gold rows with no answer (D95.5, ADR-0075).

Open issues carry actual_resolution_days=None and unprioritised ones gold_priority=None; the judge
must read "unknown - issue still open" / "not prioritised", never the string "None".
Hermetic: judge built with provider="ollama" and never called; prompts are built locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage_iq.evaluation.triage_eval import TriageJudge

ROOT = Path(__file__).resolve().parent.parent
SET_150 = ROOT / "eval" / "sets" / "quality_push_2026-10" / "eval_set.jsonl"


@pytest.fixture(scope="module")
def judge() -> TriageJudge:
    return TriageJudge(model="qwen3:8b", provider="ollama")


def _prompt(judge: TriageJudge, gold: dict) -> str:
    return judge._build_judge_prompt("t", "b", "{}", gold)


def test_none_gold_values_get_the_fixed_wording(judge: TriageJudge) -> None:
    p = _prompt(judge, {"component": None, "priority": None, "actual_resolution_days": None})
    assert "  priority: not prioritised\n" in p
    assert "  actual_resolution_days: unknown - issue still open\n" in p
    assert "  component: unknown\n" in p
    assert "None" not in p


def test_real_gold_values_render_unchanged(judge: TriageJudge) -> None:
    p = _prompt(judge, {"component": "sig-node", "priority": "high", "actual_resolution_days": 0})
    assert "  component: sig-node\n  priority: high\n  actual_resolution_days: 0\n" in p
    p = _prompt(judge, {"component": "x", "priority": "low", "actual_resolution_days": 12.5})
    assert "  actual_resolution_days: 12.5\n" in p


def test_absent_keys_read_like_none(judge: TriageJudge) -> None:
    assert "not prioritised" in _prompt(judge, {})


def test_no_none_in_any_judge_prompt_of_the_150_set(judge: TriageJudge) -> None:
    rows = [json.loads(line) for line in SET_150.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 150
    saw_open = saw_unprioritised = False
    for issue in rows:
        gold = {
            "component": issue["gold_component"],
            "priority": issue["gold_priority"],
            "actual_resolution_days": issue["actual_resolution_days"],
        }
        saw_open |= gold["actual_resolution_days"] is None
        saw_unprioritised |= gold["priority"] is None
        prompt = _prompt(judge, gold)
        gold_block = prompt.split("GOLD STANDARD:")[1].split("TRIAGE PLAN TO EVALUATE:")[0]
        assert "None" not in gold_block, issue["id"]
    assert saw_open and saw_unprioritised  # the set really exercises both None cases
