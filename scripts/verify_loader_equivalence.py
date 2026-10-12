"""Prove two versions of the model loader serve bit-identical signals.

Loads the ModelStore from `--data-dir` with whichever `triage_iq` is first on PYTHONPATH, runs the
serving signal path (`TriageAssistant._collect_signals`, the code /triage runs before the LLM) over the
first N rows of each repo's train parquet, and writes a canonical JSON of every returned value
(floats via repr, numpy as lists, DataFrames as records) plus its sha256.

Usage (run once per code version, then diff the two hashes):
    PYTHONPATH=<tree>/src python scripts/verify_loader_equivalence.py --data-dir data --out a.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def canon(x):  # noqa: ANN001, ANN201
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, (np.integer, int, bool, np.bool_)):
        return int(x) if not isinstance(x, (bool, np.bool_)) else bool(x)
    if isinstance(x, (str, type(None))):
        return x
    if isinstance(x, np.ndarray):
        return [canon(v) for v in x.tolist()]
    if isinstance(x, pd.DataFrame):
        return {"__df__": [canon(r) for r in x.to_dict("records")]}
    if isinstance(x, pd.Series):
        return {str(k): canon(v) for k, v in x.items()}
    if isinstance(x, dict):
        # `_t_*` keys are wall-clock timers of each stage (they differ on every run by construction).
        return {
            str(k): canon(v)
            for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
            if not str(k).startswith("_t_")
        }
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    return f"<{type(x).__name__}>"  # objects without a stable value (no memory addresses in the output)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()

    import triage_iq
    from triage_iq.api.loader import ModelStore

    t0 = time.perf_counter()
    store = ModelStore.load_all(data_dir=Path(a.data_dir), groq_api_key="dummy-never-used")
    load_s = time.perf_counter() - t0
    out: dict = {"triage_iq_file": str(Path(triage_iq.__file__).resolve()), "repos": {}}
    for repo, slug in (("microsoft/vscode", "microsoft_vscode"), ("kubernetes/kubernetes", "kubernetes_kubernetes")):
        asst = store.get(repo).assistant
        df = pd.read_parquet(Path(a.data_dir) / "processed" / f"{slug}_temporal_train.parquet").head(a.n)
        rows = []
        for r in df.itertuples(index=False):
            issue = pd.Series(
                {"number": -1, "title": r.title, "body_clean": r.body_clean, "created_at": r.created_at}
            )
            rows.append(canon(asst._collect_signals(issue)))
        out["repos"][repo] = rows
    body = json.dumps(out["repos"], sort_keys=True)
    out["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out["n_rows"] = {k: len(v) for k, v in out["repos"].items()}
    out["load_all_seconds_local_warm"] = round(load_s, 2)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("sha256", out["sha256"], "rows", out["n_rows"], "load_s", out["load_all_seconds_local_warm"])
    print("module", out["triage_iq_file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
