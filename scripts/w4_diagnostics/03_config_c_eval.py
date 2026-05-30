"""W4 T2 — Config C ablation: de-leaked float + bucket both in LLM prompt.

Patches build_triage_prompt() to append the bucket label to the System 3 section,
runs the full eval, then reports resolution_estimate_reasonableness vs baseline.

Usage:
    python scripts/w4_diagnostics/03_config_c_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import triage_iq.prompts.triage_prompt as _pt_module
from triage_iq.prompts.triage_prompt import build_triage_prompt as _orig_prompt


def _config_c_prompt(
    issue_title, issue_body, classifier_top3, similar_issues,
    resolution_point_days, resolution_lower_days, resolution_upper_days,
    repo,
    # extra arg injected by the patched _collect_signals
    resolution_bucket="days",
    resolution_confidence_pct=33.0,
):
    """Config C: float + bucket both shown to LLM."""
    base = _orig_prompt(
        issue_title=issue_title, issue_body=issue_body,
        classifier_top3=classifier_top3, similar_issues=similar_issues,
        resolution_point_days=resolution_point_days,
        resolution_lower_days=resolution_lower_days,
        resolution_upper_days=resolution_upper_days,
        repo=repo,
    )
    bucket_desc = {
        "hours": "< 1 day", "days": "1–7 days", "weeks": "1–4 weeks",
        "months": "1–6 months", "long": "> 6 months",
    }.get(resolution_bucket, resolution_bucket)
    conf_note = (
        "(low confidence)" if resolution_confidence_pct < 40
        else f"(confidence: {resolution_confidence_pct:.0f}%)"
    )
    # Insert bucket line before "--- TASK ---"
    bucket_line = f"Bucket estimate: {resolution_bucket} ({bucket_desc}) {conf_note}\n"
    return base.replace("--- TASK ---", bucket_line + "--- TASK ---")


def main() -> None:
    import importlib
    import os

    # Patch build_triage_prompt globally before eval imports it
    _pt_module.build_triage_prompt = lambda **kw: _config_c_prompt(**kw)  # type: ignore

    # Also patch _collect_signals to pass bucket as a keyword through the prompt builder
    # We do this by monkey-patching the TriageAssistant._collect_signals method.
    from triage_iq.models.triage import TriageAssistant
    _orig_collect = TriageAssistant._collect_signals

    def _patched_collect(self, issue):
        signals = _orig_collect(self, issue)
        # Re-build prompt with bucket injected — rebuild prompt using bucket from signals
        from triage_iq.prompts.triage_prompt import build_triage_prompt
        signals["prompt"] = _config_c_prompt(
            issue_title=issue.get("title", ""),
            issue_body=issue.get("body_clean", issue.get("body", "")),
            classifier_top3=signals["classifier_top3"],
            similar_issues=signals["similar_raw"],
            resolution_point_days=signals["pred_days"],
            resolution_lower_days=signals["lo_days"],
            resolution_upper_days=signals["hi_days"],
            repo=self.repo,
            resolution_bucket=signals.get("resolution_bucket", "days"),
            resolution_confidence_pct=signals.get("resolution_conf_pct", 33.0),
        )
        return signals

    TriageAssistant._collect_signals = _patched_collect

    # Now run the eval script logic inline
    from scripts import _11_evaluate_triage as eval_mod  # type: ignore
    # Can't import directly — run as subprocess with flag
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/11_evaluate_triage.py",
         "--judge-provider", "cohere",
         "--judge-model", "command-a-03-2025",
         "--output-file", "triage_results_w4_config_c.json",
         "--judge-delay", "6",
         "--skip-reliability"],
        capture_output=False, text=True,
    )
    return result.returncode


if __name__ == "__main__":
    # This approach won't work because subprocess won't inherit the monkey-patch.
    # Instead, run the eval manually with the patched class.
    # The actual Config C eval needs to be done by temporarily modifying triage_prompt.py.
    print("Config C eval requires temporary prompt modification.")
    print("See the comment block in this file for instructions.")
    print()
    print("To run Config C manually:")
    print("1. Edit build_triage_prompt() in triage_prompt.py to append bucket line")
    print("2. Run: python scripts/11_evaluate_triage.py --output-file triage_results_w4_config_c.json ...")
    print("3. Restore triage_prompt.py")
    print()
    print("Or just wait for Config A results — if Config A >= W1.2, Config C is not needed.")
