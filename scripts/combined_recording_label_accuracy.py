"""One-off: compute predicted_component == gold_component label accuracy for the combined
(retrieval + resolution) cassette recording, same methodology as ADR-0037 ("computed by
replaying this run's actual cassette through eval/run_eval.py's own pipeline").

Usage:
    python scripts/combined_recording_label_accuracy.py <path-to-downloaded-cassette.json>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import pandas as pd  # noqa: E402

from cassette import CassettePlayer  # noqa: E402
from frozen_retriever import build_frozen_retrievers  # noqa: E402
from run_eval import CI_API_KEY, EVAL_SET_PATH, REPO_MAP, _load_eval_set, _load_models  # noqa: E402


def main() -> None:
    cassette_path = Path(sys.argv[1])
    cassette = CassettePlayer(cassette_path, strict=True)
    issues = _load_eval_set(EVAL_SET_PATH)
    frozen_retrievers = build_frozen_retrievers(EVAL_SET_PATH)

    models: dict[str, dict] = {}
    for repo, slug in REPO_MAP.items():
        models[repo] = _load_models(repo, slug, cassette, frozen_retrievers)

    correct: dict[str, int] = {repo: 0 for repo in REPO_MAP}
    total: dict[str, int] = {repo: 0 for repo in REPO_MAP}

    for issue in issues:
        repo = issue["repo"]
        assistant = models[repo]["assistant"]
        row = pd.Series(
            {
                "title": issue["title"],
                "body_clean": issue["body"],
                "number": issue["number"],
                "created_at": (
                    pd.Timestamp(issue["created_at"])
                    if issue.get("created_at")
                    else pd.Timestamp("now", tz="UTC")
                ),
            }
        )
        plan, _meta = assistant.triage_with_metadata(row)
        gold = issue["gold_component"]
        total[repo] += 1
        if plan.predicted_component == gold:
            correct[repo] += 1

    overall_correct = sum(correct.values())
    overall_total = sum(total.values())
    print(f"overall: {overall_correct}/{overall_total} = {overall_correct / overall_total:.4f}")
    for repo in REPO_MAP:
        c, t = correct[repo], total[repo]
        print(f"  {repo}: {c}/{t} = {c / t:.4f}" if t else f"  {repo}: n=0")


if __name__ == "__main__":
    main()
