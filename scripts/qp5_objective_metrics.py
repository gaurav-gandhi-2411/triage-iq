"""QP5 objective-metrics CLI: score recorded triage plans against eval-set ground truth.

Zero tokens, zero network. The metric definitions are pre-registered in the docstring of
src/triage_iq/evaluation/objective_metrics.py. Two ways to supply plans:

  --eval-set EVAL.jsonl --plans PLANS.jsonl
        PLANS.jsonl: one {"id", "plan": TriagePlan dump, "metadata": {...}} per line (what
        --dump-plans writes; the Q8 re-record's plans should be exported in this shape).
  --replay CASSETTE EVAL.jsonl   (repeatable)
        Strict cassette replay through the real TriageAssistant (no network, nothing written to the
        cassette), as scripts/qp3a_component_decision_eval.py does. Needs data/models (or
        --models-dir) matching MANIFEST.sha256 and data/processed.

Examples (repo root):
  python scripts/qp5_objective_metrics.py \\
      --replay eval/cassettes/eval_cassette.json eval/eval_set.jsonl \\
      --replay reports/vscode_expansion_2026-10-12/eval_cassette.json \\
               eval/expansions/vscode_2026-10-12/eval_set.jsonl \\
      --label "baseline on the OLD recorded 127" --out reports/<name>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from triage_iq.evaluation import objective_metrics as om  # noqa: E402

DEFAULT_EVIDENCE = {
    "kubernetes/kubernetes": Path(
        r"D:\triageiq-data\models\candidates\resolution_evidence_kubernetes_kubernetes_v1.parquet"
    )
}
# Metadata kept per replayed plan: enough for scoring, small enough to dump.
_META_KEYS = ("llm_status", "resolution_point_days", "resolution_point_source")


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file (report provenance)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    """Eval set JSONL -> rows."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def replay_records(cassette: Path, eval_set: Path) -> tuple[list[dict], dict[str, dict]]:
    """Strict replay of one recorded set: (eval rows, {id: record}). No network, no writes."""
    from checkpoint_validation import build_replay_context, issue_to_row
    from qp3a_component_decision_eval import _use_snapshot_conformal_store

    _use_snapshot_conformal_store()
    issues, _cassette, assistants, _ = build_replay_context(cassette, eval_set, need_judge=False)
    records: dict[str, dict] = {}
    for issue in issues:
        plan, meta = assistants[issue["repo"]].triage_with_metadata(issue_to_row(issue))
        records[issue["id"]] = {
            "id": issue["id"],
            "plan": plan.model_dump(),
            "metadata": {k: meta.get(k) for k in _META_KEYS},
        }
    return issues, records


def _kv(items: list[str], cast: type) -> dict[str, Any]:
    out = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"expected REPO=VALUE, got {item!r}")
        out[key] = cast(val)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval-set", type=Path)
    ap.add_argument("--plans", type=Path)
    ap.add_argument("--replay", nargs=2, metavar=("CASSETTE", "EVAL_SET"), action="append")
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--evidence", action="append", default=[], metavar="REPO=PARQUET")
    ap.add_argument("--no-evidence", action="store_true", help="score without any evidence lookup")
    ap.add_argument("--naive-days", action="append", default=[], metavar="REPO=DAYS")
    ap.add_argument("--k", type=int, default=None, help="truncate retrieved lists to the top k")
    ap.add_argument("--label", default="", help="free-text label stored in the report")
    ap.add_argument("--note", action="append", default=[], help="note stored in the report")
    ap.add_argument("--dump-plans", type=Path, help="write the replayed plans as plans JSONL")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if bool(args.replay) == bool(args.eval_set or args.plans):
        raise SystemExit("give either --replay (repeatable) or --eval-set with --plans")
    rows: list[dict[str, Any]] = []
    records: dict[str, dict] = {}
    sources: dict[str, Any] = {}
    if args.replay:
        import run_eval

        if args.models_dir:
            run_eval.MODELS_DIR = args.models_dir
        from qp3a_component_decision_eval import _verify_manifest

        _verify_manifest(run_eval.MODELS_DIR)
        sources["models_dir"] = str(run_eval.MODELS_DIR)
        sources["replayed_sets"] = []
        for cassette, eval_set in args.replay:
            r, rec = replay_records(Path(cassette), Path(eval_set))
            if set(rec) & set(records):
                raise SystemExit("an issue id appears in two replayed sets")
            rows += r
            records.update(rec)
            sources["replayed_sets"].append(
                {
                    "cassette": cassette,
                    "cassette_sha256": sha256(Path(cassette)),
                    "eval_set": eval_set,
                    "eval_set_sha256": sha256(Path(eval_set)),
                    "n": len(r),
                }
            )
    else:
        if not (args.eval_set and args.plans):
            raise SystemExit("--eval-set and --plans go together")
        rows, records = load_eval_rows(args.eval_set), om.load_records(args.plans)
        sources["eval_set"] = {"path": str(args.eval_set), "sha256": sha256(args.eval_set)}
        sources["plans"] = {"path": str(args.plans), "sha256": sha256(args.plans)}
    if args.dump_plans:
        args.dump_plans.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values()),
            encoding="utf-8",
        )

    evidence_paths = {} if args.no_evidence else dict(DEFAULT_EVIDENCE)
    evidence_paths.update({k: Path(v) for k, v in _kv(args.evidence, str).items()})
    evidence = {}
    sources["evidence"] = {}
    for repo, path in evidence_paths.items():
        if not path.exists():
            sources["evidence"][repo] = {"path": str(path), "status": "MISSING: repo unscorable"}
            continue
        evidence[repo] = om.load_evidence_lookup(path)
        sources["evidence"][repo] = {"path": str(path), "sha256": sha256(path)}
    naive = {**om.NAIVE_DAYS_DEFAULT, **_kv(args.naive_days, float)}

    pairs = om.join_records(rows, records)
    scores = om.score_all(pairs, evidence, naive, args.k)
    report = {
        "generated_by": "scripts/qp5_objective_metrics.py",
        "label": args.label,
        "notes": args.note,
        "sources": sources,
        "naive_days_input": naive,
        **scores,
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pooled = scores["pooled"]
    print(json.dumps({k: pooled[k] for k in ("n_rows", "n_with_plan", "n_degraded_plans")}))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
