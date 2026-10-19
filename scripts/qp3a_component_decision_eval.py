"""QP3a measurement: end-to-end component exact match, classifier-decides flag off vs on.

Zero tokens, zero network: strict cassette replay of the two recorded sets (committed 64 issues and
the 63-issue vscode expansion) through the real TriageAssistant (needs data/models +
data/processed). Flag OFF = the recorded LLM plan's predicted_component. Flag ON = what
component_decision.decide_component would serve for that same plan (the recordings predate
declared_attribution, so no override can be validated and ON equals the classifier top-1; the
override path is covered by unit tests only). Writes reports/qp3a_component_decision_2026-10.json.

Paired statistics: exact two-sided McNemar (binomial on discordant pairs), paired percentile
bootstrap of the accuracy difference (2000 resamples, seed 42), Wilson 95% intervals.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

COMMITTED = (ROOT / "eval" / "cassettes" / "eval_cassette.json", ROOT / "eval" / "eval_set.jsonl")
EXPANSION = (
    ROOT / "reports" / "vscode_expansion_2026-10-12" / "eval_cassette.json",
    ROOT / "eval" / "expansions" / "vscode_2026-10-12" / "eval_set.jsonl",
)
OUT = ROOT / "reports" / "qp3a_component_decision_2026-10.json"
N_RESAMPLES = 2000
SEED = 42


def norm(label: str) -> str:
    return " ".join(str(label).split()).lower()


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p: 2 * P(X <= min(b, c)), X ~ Binomial(b + c, 0.5), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(off: np.ndarray, on: np.ndarray) -> list[float]:
    """95% percentile CI of mean(on) - mean(off), resampling issues (seed 42, 2000 draws)."""
    rng = np.random.default_rng(SEED)
    n = len(off)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    diffs = (on[idx] - off[idx]).mean(axis=1)
    return [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    off = np.array([r["correct_off"] for r in rows], dtype=float)
    on = np.array([r["correct_on"] for r in rows], dtype=float)
    clf = np.array([r["correct_classifier_top1"] for r in rows], dtype=float)
    b = int(sum(1 for r in rows if r["correct_off"] and not r["correct_on"]))  # lost by flag
    c = int(sum(1 for r in rows if r["correct_on"] and not r["correct_off"]))  # gained by flag
    departed = [r for r in rows if r["llm_departed_from_top1"]]
    return {
        "n": n,
        "accuracy_off": round(float(off.mean()), 4),
        "accuracy_off_wilson95": wilson(int(off.sum()), n),
        "accuracy_on": round(float(on.mean()), 4),
        "accuracy_on_wilson95": wilson(int(on.sum()), n),
        "classifier_top1_accuracy": round(float(clf.mean()), 4),
        "classifier_top1_wilson95": wilson(int(clf.sum()), n),
        "accuracy_diff_on_minus_off": round(float(on.mean() - off.mean()), 4),
        "accuracy_diff_paired_bootstrap95": paired_bootstrap_ci(off, on),
        "discordant_b_off_right_on_wrong": b,
        "discordant_c_off_wrong_on_right": c,
        "mcnemar_exact_p_two_sided": round(mcnemar_exact(b, c), 6),
        "llm_departed_from_top1": len(departed),
        "llm_departed_and_right": sum(1 for r in departed if r["correct_off"]),
        "llm_departed_and_wrong": sum(1 for r in departed if not r["correct_off"]),
        "llm_departed_but_classifier_top1_right": sum(
            1 for r in departed if r["correct_classifier_top1"]
        ),
        "decided_by_counts": {
            k: sum(1 for r in rows if r["decided_by"] == k)
            for k in sorted({r["decided_by"] for r in rows})
        },
    }


def _use_snapshot_conformal_store() -> None:
    """Point run_eval at the committed CQR snapshot when data/models lacks the gitignored v2 file.

    The recorded prompts carry the conformal interval, so replaying without the store misses every
    cassette key. reports/cqr_conformal_adjustments_v2.snapshot.json is a byte copy of that file
    (tests/test_eval_summary_drift.py). data/models is not modified: a scratch copy is used.
    """
    import shutil
    import tempfile

    import run_eval

    from triage_iq.api.loader import CQR_ADJUSTMENTS_FILENAME, _load_conformal_adjustments

    if (run_eval.MODELS_DIR / CQR_ADJUSTMENTS_FILENAME).exists():
        return
    scratch = Path(tempfile.mkdtemp(prefix="qp3a_cqr_"))
    shutil.copyfile(
        ROOT / "reports" / "cqr_conformal_adjustments_v2.snapshot.json",
        scratch / CQR_ADJUSTMENTS_FILENAME,
    )
    run_eval._load_conformal_adjustments = lambda _models_dir: _load_conformal_adjustments(scratch)


def replay(cassette: Path, eval_set: Path) -> list[dict[str, Any]]:
    """One row per issue: strict replay (no network), flag-off plan vs flag-on decision."""
    from checkpoint_validation import build_replay_context, issue_to_row

    _use_snapshot_conformal_store()

    from triage_iq.models.component_decision import decide_component

    issues, _cassette, assistants, _ = build_replay_context(cassette, eval_set, need_judge=False)
    rows = []
    for issue in issues:
        plan, meta = assistants[issue["repo"]].triage_with_metadata(issue_to_row(issue))
        top3 = meta["classifier_top3"]
        retrieved = {s.number for s in plan.similar_issues}  # lower bound; no override recorded
        d = decide_component(
            plan.predicted_component,
            top3,
            plan.declared_attribution,
            issue["title"],
            issue["body"],
            retrieved,
        )
        gold = norm(issue["gold_component"])
        top1 = top3[0]["label"]
        rows.append(
            {
                "id": issue["id"],
                "repo": issue["repo"],
                "gold": issue["gold_component"],
                "llm": plan.predicted_component,
                "classifier_top1": top1,
                "served_on": d.component,
                "decided_by": d.decided_by,
                "llm_status": meta.get("llm_status"),
                "llm_departed_from_top1": norm(plan.predicted_component) != norm(top1),
                "correct_off": norm(plan.predicted_component) == gold,
                "correct_on": norm(d.component) == gold,
                "correct_classifier_top1": norm(top1) == gold,
            }
        )
    return rows


def _verify_manifest(models_dir: Path) -> None:
    """Refuse to replay with artifacts that are not the manifest's: the prompt embeds the
    classifier top-3, so a stale classifier misses every cassette key (found 2026-10-20)."""
    import hashlib

    for line in (ROOT / "data" / "models" / "MANIFEST.sha256").read_text("utf-8").splitlines():
        digest, _, rel = line.strip().partition(" ")
        rel = rel.lstrip("*").strip()
        if rel.startswith("data/models/component_classifier_"):
            got = hashlib.sha256((models_dir / Path(rel).name).read_bytes()).hexdigest()
            if got != digest:
                raise SystemExit(f"{rel}: sha256 {got} != MANIFEST {digest}; pass --models-dir")


def main(argv: list[str]) -> None:
    import run_eval

    out = OUT
    while argv:
        flag, value, argv = argv[0], argv[1], argv[2:]
        if flag == "--models-dir":
            run_eval.MODELS_DIR = Path(value)
        elif flag == "--out":
            out = Path(value)
        else:
            raise SystemExit(f"unknown option {flag}")
    _verify_manifest(run_eval.MODELS_DIR)
    rows = replay(*COMMITTED) + replay(*EXPANSION)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "an issue appears in both recorded sets"
    k8s = [r for r in rows if r["repo"] == "kubernetes/kubernetes"]
    vscode = [r for r in rows if r["repo"] == "microsoft/vscode"]
    report = {
        "generated_by": "scripts/qp3a_component_decision_eval.py",
        "sources": {
            "committed": [p.relative_to(ROOT).as_posix() for p in COMMITTED],
            "expansion": [p.relative_to(ROOT).as_posix() for p in EXPANSION],
        },
        "note": "Legacy recordings carry no declared_attribution, so flag ON == classifier top-1.",
        "pooled": summarize(rows),
        "kubernetes/kubernetes": summarize(k8s),
        "microsoft/vscode": summarize(vscode),
        "rows": rows,
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("pooled", "kubernetes/kubernetes", "microsoft/vscode")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
