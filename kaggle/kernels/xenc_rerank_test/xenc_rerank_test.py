"""Kaggle kernel (R9 test stage): SCORING ONLY with the frozen X4 cross-encoder.

Design: docs/Q5B_XENC_RERANK_PREREG_2026-10.md; freeze: reports/r9_xenc_rerank_frozen_2026-10.json.
Inputs under /kaggle/input: the issue texts, ``test_candidates.parquet`` (query, cand, ret_rank;
NO labels), ``freeze.json`` (a copy of the committed freeze) and the X4 weights. No training, no
label of any kind. Refuses to score unless the weights' sha256 equals the frozen one. Writes
``X4_test_scores.parquet`` (query, cand, score) and ``run_info.json``. Same scoring recipe as
the validation kernel (text cut, 512 tokens, length-sorted batches of 64, fp16 autocast).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN, SIDE_CHARS, SCORE_BATCH = 512, 3000, 64
IN, OUT = Path("/kaggle/input"), Path("/kaggle/working")


def find(name: str) -> Path:
    hits = sorted(IN.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {IN}, found {hits}")
    return hits[0]


def side_text(title: str, body_clean: str) -> str:
    """Production-recipe text (``title. clean_body``), cut to SIDE_CHARS characters per side."""
    return f"{title.strip()}. {body_clean.strip()}"[:SIDE_CHARS]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_weights(freeze: dict, weights_dir: Path) -> dict[str, str]:
    """Every pinned X4 file must exist with the frozen sha256; returns the observed shas."""
    seen = {}
    for name, want in freeze["x4_files_sha256"].items():
        seen[name] = file_sha(weights_dir / name)
        if seen[name] != want:
            raise RuntimeError(f"frozen weight file {name} differs: refusing to score")
    return seen


@torch.inference_mode()
def score_pairs(model, tok, a: list[str], b: list[str]) -> np.ndarray:  # noqa: ANN001
    """Logits for all pairs in input order (length-sorted batches cut padding)."""
    model.eval()
    order = sorted(range(len(a)), key=lambda i: len(a[i]) + len(b[i]))
    out = np.zeros(len(a), dtype=np.float32)
    for s in range(0, len(order), SCORE_BATCH):
        idx = order[s : s + SCORE_BATCH]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True,
                  truncation="longest_first", max_length=MAX_LEN, return_tensors="pt")  # fmt: skip
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc.to("cuda")).logits
        out[idx] = logits.float().squeeze(-1).cpu().numpy()
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: refusing to run (fail loud)")
    t0 = time.time()
    freeze = json.loads(find("freeze.json").read_text(encoding="utf-8"))
    weights_dir = find("model.safetensors").parent
    seen = check_weights(freeze, weights_dir)
    issues = pd.read_parquet(find("microsoft__vscode__issues.parquet")).reset_index(drop=True)
    cand = pd.read_parquet(find("test_candidates.parquet"))
    if set(cand.columns) != {"query", "cand", "ret_rank"}:
        raise RuntimeError(f"unexpected candidate columns {list(cand.columns)}")
    text_of = {
        int(n): side_text(t, b)
        for n, t, b in zip(
            issues["number"],
            issues["title"].fillna(""),
            issues["body_clean"].fillna(""),
            strict=True,
        )  # fmt: skip
    }
    missing = (set(cand["query"]) | set(cand["cand"])) - set(text_of)
    if missing:
        raise RuntimeError(f"{len(missing)} issues without text, e.g. {sorted(missing)[:5]}")
    tok = AutoTokenizer.from_pretrained(weights_dir)
    model = AutoModelForSequenceClassification.from_pretrained(weights_dir).float().cuda()
    if model.config.num_labels != 1:
        raise RuntimeError("expected a 1-logit head")
    sc = score_pairs(model, tok, [text_of[int(q)] for q in cand["query"]],
                     [text_of[int(c)] for c in cand["cand"]])  # fmt: skip
    pd.DataFrame({"query": cand["query"], "cand": cand["cand"], "score": sc}).to_parquet(
        OUT / "X4_test_scores.parquet", index=False
    )
    info = {"torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "n_pairs": len(cand), "weights_sha256": seen,
            "selected": freeze["selected"], "seconds_total": round(time.time() - t0)}  # fmt: skip
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
    print(info, flush=True)


if __name__ == "__main__":
    main()
