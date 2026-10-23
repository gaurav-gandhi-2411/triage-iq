"""Single-row CPU latency probe for an ONNX classifier (inference only; numpy + onnxruntime + tokenizers).

Runs in a throwaway venv, never in the project environment. Prints one JSON line:
end-to-end (tokenise + session.run + renormalised sigmoid) and session-only p50/p95 in ms, for
all cores and for 2 intra-op threads (the 2-vCPU container shape), after a 5-row warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def pct(ts: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(ts, 50)),
        "p95_ms": float(np.percentile(ts, 95)),
        "n": len(ts),
    }


def probe(
    model: Path, tok_path: Path, texts: list[str], max_len: int, threads: int
) -> dict[str, dict]:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=max_len)
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}

    def feed(t: str) -> dict[str, np.ndarray]:
        enc = tok.encode(t)
        ids = np.array([enc.ids], dtype=np.int64)
        f = {"input_ids": ids, "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        return {k: v for k, v in f.items() if k in names}

    for t in texts[:5]:
        sess.run(None, feed(t))
    e2e, run_only = [], []
    for t in texts:
        t0 = time.perf_counter()
        f = feed(t)
        t1 = time.perf_counter()
        lg = sess.run(None, f)[0]
        s = 1 / (1 + np.exp(-lg))
        s /= s.sum(axis=1, keepdims=True)
        t2 = time.perf_counter()
        e2e.append((t2 - t0) * 1000)
        run_only.append((t2 - t1) * 1000)
    return {"end_to_end": pct(e2e), "session_only": pct(run_only)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=384)
    a = ap.parse_args()
    texts = json.loads(a.texts.read_text(encoding="utf-8"))
    res = {
        f"threads_{label}": probe(a.model, a.tokenizer, texts, a.max_len, n)
        for label, n in (("all", 0), ("2", 2))
    }
    out = {
        "cpu_count": os.cpu_count(), "n_texts": len(texts),
        "end_to_end": {k: v["end_to_end"] for k, v in res.items()},
        "session_only": {k: v["session_only"] for k, v in res.items()},
    }  # fmt: skip
    print(json.dumps(out))


if __name__ == "__main__":
    main()
