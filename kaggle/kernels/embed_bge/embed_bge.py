"""Kaggle kernel: BAAI/bge-base-en-v1.5 document embeddings (CLS pooling + L2 norm).

Reads every *.parquet under /kaggle/input with columns (repo, kind, number, text) and writes, to
/kaggle/working, per input file `<name>_emb.npy` (float16, n x 768), `<name>_ids.parquet`
(repo, kind, number, row order == embedding order) and one `run_info.json`.

Documents are encoded WITHOUT the BGE query instruction. At search time, queries must use
"Represent this sentence for searching relevant passages: " (see similar_issues.py).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-base-en-v1.5"
# Pinned so reruns are bit-for-bit comparable; bump deliberately and re-embed everything.
REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64  # fits T4 (16 GB) / P100 at 512 tokens in fp16 with large margin
SEED = 42
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")
# Text prepended to every document (e.g. the BGE query instruction). Kaggle runs ONLY this file,
# so `qk_kaggle.py push-kernel --text-prefix` rewrites this exact line in the staged copy.
TEXT_PREFIX = ""


def production_text(title: str, body_clean: str, tokenizer) -> str:  # noqa: ANN001
    """Replicates similar_issues._build_text's tokenizer path (Kaggle can't import triage_iq).

    Body is cut by TOKEN count so title + body fit MAX_SEQ_LENGTH (-2 for [CLS]/[SEP]), exactly
    as serving does with max_tokens=model.max_seq_length. Keep in sync; a unit test compares the
    two with a fake tokenizer.
    """
    prefix = f"{title.strip()}. "
    n_prefix = len(tokenizer.encode(prefix, add_special_tokens=False))
    budget = max(MAX_SEQ_LENGTH - n_prefix - 2, 0)
    ids = tokenizer.encode(
        body_clean.strip(), add_special_tokens=False, truncation=True, max_length=budget
    )
    return prefix + tokenizer.decode(ids, skip_special_tokens=True)


def encode(texts: list[str], tokenizer, model, device: str) -> np.ndarray:  # noqa: ANN001
    """Return float16 (n, 768) L2-normalised CLS embeddings in input order."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sort cuts padding
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            batch = tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(device)
            cls = model(**batch).last_hidden_state[:, 0].float()
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out[idx] = cls.cpu().numpy().astype(np.float16)
    return out


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).eval().to(device)
    if device == "cuda":
        model = model.half()
    prefix = TEXT_PREFIX
    info: dict = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "pooling": "cls+l2",
        "doc_instruction": prefix or None,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "files": {},
    }
    t_all = time.time()
    for path in sorted(INPUT_ROOT.rglob("*.parquet")):
        df = pd.read_parquet(path).reset_index(drop=True)
        t0 = time.time()
        if {"title", "body_clean"} <= set(df.columns):  # text_recipe=production issue dataset
            texts = [
                production_text(t, b, tokenizer)
                for t, b in zip(df["title"].fillna(""), df["body_clean"].fillna(""), strict=True)
            ]
        else:  # legacy issues / PR datasets: `text` is already final
            texts = df["text"].fillna("").astype(str).tolist()
        emb = encode([prefix + t for t in texts], tokenizer, model, device)
        np.save(OUTPUT_ROOT / f"{path.stem}_emb.npy", emb)
        df[["repo", "kind", "number"]].to_parquet(OUTPUT_ROOT / f"{path.stem}_ids.parquet")
        info["files"][path.stem] = {"n": len(df), "seconds": round(time.time() - t0, 1)}
        print(path.stem, len(df), info["files"][path.stem]["seconds"], flush=True)
    info["seconds_total"] = round(time.time() - t_all, 1)
    (OUTPUT_ROOT / "run_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
