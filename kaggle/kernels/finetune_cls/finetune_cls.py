"""Kaggle kernel: fine-tune a small encoder for the kubernetes sig/* task (multi-label BCE).

Pre-registered design lives in scripts/qp1_finetuned_encoder.py (docstring); this file only
executes it. Reads /kaggle/input/**/{train,validation,test_texts}.parquet + classes.json,
trains the grid {lr 2e-5, 5e-5} x {epochs 3, 5} on TRAIN only, scores each config on VALIDATION,
keeps the best (validation top-3, tie macro-F1), then writes into /kaggle/working:
  grid.json, selected_model/ (HF), validation_probs.parquet, test_probs.parquet (fp32, ids
  aligned: column `number` + one column per class), model_fp32.onnx, model_int8.onnx,
  validation_probs_int8.parquet, test_probs_int8.parquet, tokenizer files, run_info.json.
Probabilities = sigmoid scores renormalised to sum 1 per row. Test LABELS are not in the input.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# onnx / onnxruntime / sentencepiece are not guaranteed on the Kaggle image (internet is on).
subprocess.run(  # noqa: S603
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "onnx",
        "onnxruntime",
        "sentencepiece",
        "protobuf",
    ],
    check=False,
)

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

CANDIDATES = [  # (model id, pinned revision); the second is used only if the first fails to load
    ("microsoft/deberta-v3-small", "a36c739020e01763fe789b4b85e2df55d6180012"),
    ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
]
MAX_LEN, BATCH, WARMUP, SEED = 384, 16, 0.10, 42
GRID = [(lr, ep) for lr in (2e-5, 5e-5) for ep in (3, 5)]
IN, OUT = Path("/kaggle/input"), Path("/kaggle/working")


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def find(name: str) -> Path:
    hits = sorted(IN.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {IN}, found {hits}")
    return hits[0]


def renorm_sigmoid(logits: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return s / s.sum(axis=1, keepdims=True)


def metrics(p: np.ndarray, y_primary: np.ndarray) -> dict[str, float]:
    """Top-1, top-3 and macro-F1 (over classes present) against the primary class index."""
    order = np.argsort(-p, axis=1, kind="stable")
    top1 = float((order[:, 0] == y_primary).mean())
    top3 = float((order[:, :3] == y_primary[:, None]).any(axis=1).mean())
    f1 = []
    for c in np.unique(y_primary):
        tp = float(((order[:, 0] == c) & (y_primary == c)).sum())
        denom = float((y_primary == c).sum() + (order[:, 0] == c).sum())
        f1.append(2 * tp / denom)
    return {"top1": top1, "top3": top3, "macro_f1": float(np.mean(f1))}


def encode(tok, texts: list[str]) -> list[list[int]]:  # noqa: ANN001
    return tok(texts, truncation=True, max_length=MAX_LEN)["input_ids"]


def pad(tok, seqs: list[list[int]], device: str):  # noqa: ANN001, ANN201
    b = tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt")
    return b["input_ids"].to(device), b["attention_mask"].to(device)


@torch.inference_mode()
def predict_logits(model, tok, ids: list[list[int]], device: str, bs: int = 64) -> np.ndarray:  # noqa: ANN001
    model.eval()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # length-sort cuts padding
    out = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        x, m = pad(tok, [ids[i] for i in idx], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out[idx] = model(input_ids=x, attention_mask=m).logits.float().cpu().numpy()
    return out


def train_one(mid_rev, tok, tr_ids, tr_y, va_ids, lr, epochs, n_cls, device):  # noqa: ANN001, ANN201
    """Train one grid config on TRAIN only; return (model, validation probabilities, seconds)."""
    seed_all()
    model = AutoModelForSequenceClassification.from_pretrained(
        mid_rev[0], revision=mid_rev[1], num_labels=n_cls, problem_type="multi_label_classification"
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(tr_ids) + BATCH - 1) // BATCH)
    warm = int(WARMUP * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(SEED)
    y_t = torch.tensor(tr_y, dtype=torch.float32)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_ids))
        for s in range(0, len(perm), BATCH):
            idx = perm[s : s + BATCH]
            x, m = pad(tok, [tr_ids[i] for i in idx], device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), y_t[idx].to(device)
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(
            f"lr={lr} ep={ep + 1}/{epochs} last_loss={loss.item():.4f} {time.time() - t0:.0f}s",
            flush=True,
        )
    p = renorm_sigmoid(predict_logits(model, tok, va_ids, device))
    return model, p, time.time() - t0


def export_onnx(model, tok, path: Path) -> None:  # noqa: ANN001
    """Export logits-only ONNX with dynamic batch/sequence axes (legacy exporter)."""

    class Wrap(torch.nn.Module):
        def __init__(self, m) -> None:  # noqa: ANN001
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):  # noqa: ANN001, ANN201
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    model = model.float().cpu().eval()
    x = tok(["export probe text"] * 2, padding="max_length", max_length=64, return_tensors="pt")
    axes = {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}}
    kw = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "opset_version": 17,
        "dynamic_axes": axes,
    }  # noqa: E501
    args = (x["input_ids"], x["attention_mask"])
    try:
        torch.onnx.export(Wrap(model), args, str(path), dynamo=False, **kw)
    except TypeError:  # older torch has no `dynamo` kwarg (the legacy exporter is the only one)
        torch.onnx.export(Wrap(model), args, str(path), **kw)


def onnx_logits(path: Path, tok, ids: list[list[int]], bs: int = 16) -> np.ndarray:  # noqa: ANN001
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = None
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        b = tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="np")
        feed = {k: b[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
        lg = sess.run(None, feed)[0]
        if out is None:
            out = np.zeros((len(ids), lg.shape[1]), dtype=np.float32)
        out[idx] = lg
    return out


def save_probs(path: Path, numbers: np.ndarray, p: np.ndarray, classes: list[str]) -> None:
    df = pd.DataFrame(p, columns=classes)
    df.insert(0, "number", numbers.astype("int64"))
    df.to_parquet(path, index=False)


def top3(p: np.ndarray) -> np.ndarray:
    return np.argsort(-p, axis=1, kind="stable")[:, :3]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    info: dict = {"device": device, "torch": torch.__version__, "gpu": gpu}
    t_start = time.time()
    classes = json.loads(find("classes.json").read_text())["classes"]
    n_cls, pos = len(classes), {c: i for i, c in enumerate(classes)}
    tr, va, te = (
        pd.read_parquet(find(f))
        for f in ("train.parquet", "validation.parquet", "test_texts.parquet")
    )
    assert "primary" not in te.columns and "labels" not in te.columns, "test labels present"  # noqa: S101
    mid_rev, tok = None, None
    for cand in CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(cand[0], revision=cand[1])
            AutoModelForSequenceClassification.from_pretrained(
                cand[0], revision=cand[1], num_labels=n_cls
            )
            mid_rev = cand
            break
        except Exception as e:  # noqa: BLE001
            info.setdefault("load_failures", []).append({"model": cand[0], "error": repr(e)[:500]})
            print("load failed", cand[0], repr(e)[:300], flush=True)
    if mid_rev is None:
        (OUT / "run_info.json").write_text(json.dumps(info, indent=1))
        raise SystemExit("no candidate model could be loaded")
    info["model"], info["revision"] = mid_rev
    tr_ids, va_ids, te_ids = (encode(tok, d["text"].tolist()) for d in (tr, va, te))
    tr_y = np.zeros((len(tr), n_cls), dtype=np.float32)
    for i, labs in enumerate(tr["labels"]):
        for c in json.loads(labs):
            tr_y[i, pos[c]] = 1.0
    va_prim = np.array([pos[c] for c in va["primary"]])
    rows, best = [], None
    for lr, ep in GRID:
        model, p, secs = train_one(mid_rev, tok, tr_ids, tr_y, va_ids, lr, ep, n_cls, device)
        row = {"lr": lr, "epochs": ep, "train_seconds": secs, **metrics(p, va_prim)}
        rows.append(row)
        print(row, flush=True)
        key = (row["top3"], row["macro_f1"])
        if best is None or key > best[0]:
            best = (key, row)
            model.save_pretrained(OUT / "selected_model")
        del model
        torch.cuda.empty_cache()
        (OUT / "grid.json").write_text(json.dumps({"grid": rows, "selected": best[1]}, indent=1))
    tok.save_pretrained(OUT / "selected_model")
    model = AutoModelForSequenceClassification.from_pretrained(OUT / "selected_model").to(device)
    pv = renorm_sigmoid(predict_logits(model, tok, va_ids, device))
    pt = renorm_sigmoid(predict_logits(model, tok, te_ids, device))
    info["selected_validation_recheck"] = metrics(pv, va_prim)
    save_probs(OUT / "validation_probs.parquet", va["number"].to_numpy(), pv, classes)
    save_probs(OUT / "test_probs.parquet", te["number"].to_numpy(), pt, classes)
    info["train_and_predict_seconds"] = time.time() - t_start
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1))
    try:  # ONNX last: a failed export must not lose the probabilities written above
        from onnxruntime.quantization import QuantType, quantize_dynamic

        export_onnx(model, tok, OUT / "model_fp32.onnx")
        quantize_dynamic(
            str(OUT / "model_fp32.onnx"), str(OUT / "model_int8.onnx"), weight_type=QuantType.QInt8
        )
        p32 = renorm_sigmoid(onnx_logits(OUT / "model_fp32.onnx", tok, va_ids))
        p8 = renorm_sigmoid(onnx_logits(OUT / "model_int8.onnx", tok, va_ids))
        o32, o8 = top3(p32), top3(p8)
        info["onnx_parity_validation"] = {
            "fp32_onnx_vs_torch_max_abs_diff": float(np.abs(p32 - pv).max()),
            "int8_vs_fp32_top1_agreement": float((o32[:, 0] == o8[:, 0]).mean()),
            "int8_vs_fp32_top3_set_agreement": float(
                np.mean([set(a) == set(b) for a, b in zip(o32, o8, strict=True)])
            ),
            "fp32_onnx_metrics": metrics(p32, va_prim),
            "int8_metrics": metrics(p8, va_prim),
        }
        save_probs(OUT / "validation_probs_int8.parquet", va["number"].to_numpy(), p8, classes)
        p8t = renorm_sigmoid(onnx_logits(OUT / "model_int8.onnx", tok, te_ids))
        save_probs(OUT / "test_probs_int8.parquet", te["number"].to_numpy(), p8t, classes)
        info["onnx_sizes_bytes"] = {
            f: os.path.getsize(OUT / f) for f in ("model_fp32.onnx", "model_int8.onnx")
        }
    except Exception as e:  # noqa: BLE001
        info["onnx_error"] = repr(e)[:1500]
    info["total_seconds"] = time.time() - t_start
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
