"""vscode monthly retrain: assemble -> train -> evaluate -> gate -> stage (ADR-0072).

Every stage reads/writes one --out directory. Nothing is uploaded or deployed; `stage` only writes
candidate files plus a manifest diff into <out>/staging. Exit codes: 0 accepted / stage done,
10 gate refused (a normal outcome: champion kept), 2 error (also fail-closed: no staging).

  python scripts/vscode_retrain.py all --derived-dir D --champion MODELS_DIR_OR_BUNDLE \
      --as-of 2026-10-22 --out OUT [--manifest data/models/MANIFEST.sha256]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_vscode_lib as vl  # noqa: E402
import vscode_retrain_lib as rl  # noqa: E402

REFUSED, ERROR = 10, 2


def _ids(d: pd.DataFrame, split: str) -> list[int]:
    return sorted(int(n) for n in d.loc[d["split"] == split, "number"])


def stage_assemble(a: argparse.Namespace) -> dict[str, Any]:
    w = rl.make_windows(pd.Timestamp(a.as_of), a.window_months)
    d = rl.assemble(a.derived_dir, w)
    a.out.mkdir(parents=True, exist_ok=True)
    flat = d.drop(columns=["title", "body", "t72", "earliest72"])  # parquet cannot hold frozensets
    flat["final"] = flat["final"].map(sorted)
    flat.to_parquet(a.out / "window.parquet", index=False)
    meta = {
        "as_of": str(pd.Timestamp(a.as_of).date()),
        "windows": {k: str(v) for k, v in vars(w).items()},
        "counts": d["split"].value_counts().to_dict(),
        "input_sha256": {
            f: rl.file_sha256(a.derived_dir / f) for f in ("issues.parquet", "label_events.parquet")
        },
        "split_number_sha256": {
            s: hashlib.sha256(json.dumps(_ids(d, s)).encode()).hexdigest()
            for s in ("train", "calibration", "holdout")
        },
    }
    (a.out / "window_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _load(out: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        meta = json.loads((out / "window_meta.json").read_text(encoding="utf-8"))
        d = pd.read_parquet(out / "window.parquet")
        d["final"] = d["final"].map(frozenset)
        return d, meta
    except (OSError, ValueError) as e:
        raise RuntimeError(f"assemble output missing/unreadable in {out}: {e}") from e


def stage_train(a: argparse.Namespace) -> dict[str, Any]:
    d, _ = _load(a.out)
    tr, ca = d[d["split"] == "train"], d[d["split"] == "calibration"]
    classes = rl.regenerate_classes(tr)
    bundle, grid = rl.train_challenger(
        tr.reset_index(drop=True), ca.reset_index(drop=True), classes
    )
    joblib.dump(bundle, a.out / "challenger.joblib")
    info = {"classes": classes, "classes_sha256": vl.classes_sha256(classes[:-1]),
            "n_train": len(tr), "n_calibration": len(ca), "grid": grid, "tau": bundle["tau"],
            "conformal": bundle["conformal"], "challenger_sha256": rl.file_sha256(a.out / "challenger.joblib")}  # fmt: skip
    (a.out / "train_info.json").write_text(
        json.dumps(info, indent=1, default=str), encoding="utf-8"
    )
    return info


def stage_evaluate(a: argparse.Namespace) -> dict[str, Any]:
    d, _ = _load(a.out)
    info = json.loads((a.out / "train_info.json").read_text(encoding="utf-8"))
    classes = info["classes"]
    ho = d[d["split"] == "holdout"].reset_index(drop=True)
    ho = ho.merge(  # served champion needs raw title/body
        pd.read_parquet(a.derived_dir / "issues.parquet", columns=["number", "title", "body"]),
        on="number",
    )
    bundle = joblib.load(a.out / "challenger.joblib")
    p_ch = rl.bundle_proba(bundle, ho["text"], classes)
    p_champ, champ_sha = rl.champion_proba(a.champion, ho, classes)
    tr = d[d["split"] == "train"].reset_index(drop=True)
    p_naive, _ = vl.majority_proba(vl.gold(tr, classes)[0], classes, len(ho))
    m = rl.evaluate_holdout(ho, classes, p_ch, bundle["tau"], p_champ, p_naive)
    m["champion"]["sha256"] = champ_sha
    (a.out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def stage_gate(a: argparse.Namespace) -> dict[str, Any]:
    """Always writes gate_decision.json; any unreadable input is a refusal, never a pass."""
    try:
        m = json.loads((a.out / "metrics.json").read_text(encoding="utf-8"))
        info = json.loads((a.out / "train_info.json").read_text(encoding="utf-8"))
        meta = json.loads((a.out / "window_meta.json").read_text(encoding="utf-8"))
        res = rl.gate_decision(m, info["n_train"], len(info["classes"]))
        res |= {"metrics": m, "classes_sha256": info["classes_sha256"], "window": meta,
                "candidate_sha256": info["challenger_sha256"], "seed": rl.SEED}  # fmt: skip
    except (OSError, ValueError, KeyError) as e:
        res = {"decision": "refuse", "reasons": [f"input_error: {e!r}"], "checks": []}
    (a.out / "gate_decision.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8"
    )
    return res


def stage_stage(a: argparse.Namespace) -> dict[str, Any]:
    """Publish-prep into <out>/staging only; re-verifies the gate JSON against the file hash."""
    g = json.loads((a.out / "gate_decision.json").read_text(encoding="utf-8"))
    cand = a.out / "challenger.joblib"
    if g.get("decision") != "accept" or g.get("candidate_sha256") != rl.file_sha256(cand):
        raise RuntimeError("gate did not accept this exact candidate; nothing staged")
    stg = a.out / "staging"
    stg.mkdir(exist_ok=True)
    name = f"component_classifier_microsoft_vscode_family_v2_{g['window']['as_of']}.joblib"
    shutil.copy2(cand, stg / name)
    old = a.manifest.read_text(encoding="utf-8").splitlines(keepends=True)
    new = [*old, f"{g['candidate_sha256']}  data/models/{name}\n"]  # champion line kept: rollback
    diff = "".join(difflib.unified_diff(old, new, "MANIFEST.sha256", "MANIFEST.sha256.candidate"))
    (stg / "MANIFEST.sha256.candidate").write_text("".join(new), encoding="utf-8")
    (stg / "manifest.diff").write_text(diff, encoding="utf-8")
    m = g["metrics"]
    (stg / "candidate_metadata.json").write_text(json.dumps({
        "artifact": name, "classes_sha256": g["classes_sha256"], "window": g["window"],
        "holdout_strict_top3": m["challenger"]["strict_top3"],
        "naive_holdout_strict_top3": m["naive"]["strict_top3"],
        "champion_holdout_strict_top3": m["champion"]["strict_top3"],
        "model_beats_naive": True, "gate_decision_sha256": rl.file_sha256(a.out / "gate_decision.json"),
    }, indent=1), encoding="utf-8")  # fmt: skip
    return {"staged": str(stg), "artifact": name}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["assemble", "train", "evaluate", "gate", "stage", "all"])
    ap.add_argument("--derived-dir", type=Path)
    ap.add_argument("--champion", type=Path, help="served models dir, or a retrain bundle .joblib")
    ap.add_argument("--as-of", help="run date YYYY-MM-DD (data cutoff = as-of minus 30 days)")
    ap.add_argument("--window-months", type=int, default=rl.WINDOW_MONTHS)
    ap.add_argument("--manifest", type=Path, default=Path("data/models/MANIFEST.sha256"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    todo = ["assemble", "train", "evaluate", "gate", "stage"] if a.stage == "all" else [a.stage]
    need = {"assemble": ("derived_dir", "as_of"), "evaluate": ("derived_dir", "champion")}
    miss = [f"--{k.replace('_', '-')}" for s in todo for k in need.get(s, ()) if getattr(a, k) is None]  # fmt: skip
    if miss:
        ap.error(f"required for {a.stage}: {', '.join(sorted(set(miss)))}")
    np.random.seed(rl.SEED)
    try:
        with threadpool_limits(limits=4):
            for s in todo:
                if s == "gate":
                    res = stage_gate(a)
                    print(json.dumps({"decision": res["decision"], "reasons": res["reasons"]}))
                    if res["decision"] != "accept":
                        return REFUSED
                elif s == "stage" and a.stage == "all":
                    print(json.dumps(stage_stage(a)))
                else:
                    r = {"assemble": stage_assemble, "train": stage_train, "evaluate": stage_evaluate,
                         "stage": stage_stage}[s](a)  # fmt: skip
                    print(s, "ok", json.dumps(r, default=str)[:300])
    except Exception as e:  # noqa: BLE001 - fail closed: any error is exit 2 with nothing staged
        print(f"ERROR in {s}: {e!r}", file=sys.stderr)
        return ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())
