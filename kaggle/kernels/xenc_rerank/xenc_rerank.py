"""Kaggle kernel (R9): fine-tune cross-encoder rerankers on vscode TRAIN duplicate pairs, score
the validation candidate lists of the frozen R8 retriever.

Design: docs/Q5B_XENC_RERANK_PREREG_2026-10.md. Inputs under /kaggle/input: the issue texts
(columns number, title, body_clean), train_groups.parquet (anchor, pos, neg0..neg6), splits.parquet
(number, split) and val_candidates.parquet (query, cand, ret_rank; NO labels). No test query and no
validation / test duplicate label is in the input. Per config X1..X4 (model x lr; 2 epochs, 4 groups
of 1 positive + 7 hard negatives per step, seed 42) it writes ``<id>_model/`` (fp32 HF) and
``<id>_val_scores.parquet`` (query, cand, score); the untouched pretrained models are scored as
``Z-M`` / ``Z-B`` (diagnostic). ``run_info.json`` records timings and the skipped units.
"""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "M": ("cross-encoder/ms-marco-MiniLM-L12-v2", "7b0235231ca2674cb8ca8f022859a6eba2b1c968"),
    "B": ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"),
}
CONFIGS = {"X1": ("M", 2e-5), "X2": ("M", 6e-5), "X3": ("B", 1e-5), "X4": ("B", 3e-5)}
# Run order: cheap units first; a zero-shot diagnostic precedes each model class.
UNITS = ["Z-M", "X1", "X2", "Z-B", "X3", "X4"]
MAX_LEN, SIDE_CHARS, SCORE_BATCH = 512, 3000, 64
GROUPS, N_NEG, EPOCHS, WARMUP, WD, SEED = 4, 7, 2, 0.10, 0.01, 42
TIME_BUDGET_S = 4 * 3600  # prereg: no new unit is started after 4 h
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


def side_text(title: str, body_clean: str) -> str:
    """Production-recipe text (``title. clean_body``), cut to SIDE_CHARS characters per side."""
    return f"{title.strip()}. {body_clean.strip()}"[:SIDE_CHARS]


def assert_no_heldout(groups: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Fail loudly if any anchor / positive / negative is not a train-split issue."""
    train = set(splits.loc[splits["split"] == "train", "number"].astype(int))
    neg_cols = [c for c in groups.columns if c.startswith("neg")]
    used = set(groups["anchor"]) | set(groups["pos"])
    for c in neg_cols:
        used |= set(groups[c].astype(int))
    bad = sorted(used - train)
    if bad:
        raise RuntimeError(f"{len(bad)} non-train issues in training data, e.g. {bad[:5]}")


def group_loss(logits: torch.Tensor) -> torch.Tensor:
    """Listwise cross-entropy over (G, 1 + N_NEG) logits; the positive is column 0."""
    return F.cross_entropy(
        logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    )


def group_batches(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled batches of up to GROUPS group indices (the last one may be smaller)."""
    perm = rng.permutation(n)
    return [perm[s : s + GROUPS] for s in range(0, n, GROUPS)]


def pair_logits(model, tok, a: list[str], b: list[str], device: str) -> torch.Tensor:  # noqa: ANN001
    """One logit per (a, b) pair, fp32 (autocast fp16 forward); grads flow if model.train()."""
    enc = tok(a, b, padding=True, truncation="longest_first", max_length=MAX_LEN,
              return_tensors="pt")  # fmt: skip
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        out = model(**enc.to(device)).logits
    return out.float().squeeze(-1)


@torch.inference_mode()
def score_pairs(model, tok, a: list[str], b: list[str], device: str) -> np.ndarray:  # noqa: ANN001
    """Logits for all pairs in input order (length-sorted batches cut padding)."""
    model.eval()
    order = sorted(range(len(a)), key=lambda i: len(a[i]) + len(b[i]))
    out = np.zeros(len(a), dtype=np.float32)
    for s in range(0, len(order), SCORE_BATCH):
        idx = order[s : s + SCORE_BATCH]
        out[idx] = (
            pair_logits(model, tok, [a[i] for i in idx], [b[i] for i in idx], device).cpu().numpy()
        )
    return out


def load_model(key: str, device: str):  # noqa: ANN201
    name, rev = MODELS[key]
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev).float()
    if model.config.num_labels != 1:
        raise RuntimeError(f"{name}: expected a 1-logit head, got {model.config.num_labels}")
    return model.to(device)


def train_one(key: str, lr: float, groups: pd.DataFrame, text_of: dict, tok, device: str):  # noqa: ANN001, ANN201
    """Fine-tune on the train groups; fp32 master weights, fp16 autocast forward."""
    seed_all()
    model = load_model(key, device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    rng = np.random.default_rng(SEED)
    steps = EPOCHS * len(group_batches(len(groups), np.random.default_rng(SEED)))
    warm = int(WARMUP * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    cands = groups[["pos", *[f"neg{i}" for i in range(N_NEG)]]].to_numpy()
    anchors = groups["anchor"].to_numpy()
    losses = []
    for ep in range(EPOCHS):
        model.train()
        ep_loss = []
        for idx in group_batches(len(groups), rng):
            a = [text_of[int(anchors[g])] for g in idx for _ in range(1 + N_NEG)]
            b = [text_of[int(n)] for g in idx for n in cands[g]]
            loss = group_loss(pair_logits(model, tok, a, b, device).view(len(idx), 1 + N_NEG))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ep_loss.append(float(loss))
        losses.append(float(np.mean(ep_loss)))
        print("epoch", ep, "loss", losses[-1], flush=True)
    return model, losses


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: refusing to run (fail loud)")
    device, t_all = "cuda", time.time()
    issues = pd.read_parquet(find("microsoft__vscode__issues.parquet")).reset_index(drop=True)
    groups, splits = (
        pd.read_parquet(find("train_groups.parquet")),
        pd.read_parquet(find("splits.parquet")),
    )
    cand = pd.read_parquet(find("val_candidates.parquet"))
    assert_no_heldout(groups, splits)
    text_of = {
        int(n): side_text(t, b)
        for n, t, b in zip(
            issues["number"],
            issues["title"].fillna(""),
            issues["body_clean"].fillna(""),
            strict=True,
        )
    }
    missing = (set(cand["query"]) | set(cand["cand"])) - set(text_of)
    if missing:
        raise RuntimeError(
            f"{len(missing)} candidate issues without text, e.g. {sorted(missing)[:5]}"
        )
    a_txt = [text_of[int(q)] for q in cand["query"]]
    b_txt = [text_of[int(c)] for c in cand["cand"]]
    info: dict = {
        "models": MODELS, "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "n_groups": len(groups),
        "n_val_pairs": len(cand), "seed": SEED, "units": {},
    }  # fmt: skip
    toks: dict = {}
    last = 0.0
    for unit in UNITS:
        if time.time() - t_all + last > TIME_BUDGET_S:
            info["units"][unit] = {"skipped": "time budget"}
            continue
        t0 = time.time()
        zero = unit.startswith("Z-")
        key = unit[2:] if zero else CONFIGS[unit][0]
        if key not in toks:
            toks[key] = AutoTokenizer.from_pretrained(MODELS[key][0], revision=MODELS[key][1])
        rec: dict = {"model": key}
        if zero:
            model = load_model(key, device)
        else:
            model, rec["epoch_loss"] = train_one(
                key, CONFIGS[unit][1], groups, text_of, toks[key], device
            )
            rec["lr"] = CONFIGS[unit][1]
            rec["train_seconds"] = round(time.time() - t0)
            model.save_pretrained(OUT / f"{unit}_model")
            toks[key].save_pretrained(OUT / f"{unit}_model")
        sc = score_pairs(model, toks[key], a_txt, b_txt, device)
        pd.DataFrame({"query": cand["query"], "cand": cand["cand"], "score": sc}).to_parquet(
            OUT / f"{unit}_val_scores.parquet", index=False
        )
        last = time.time() - t0
        rec["total_seconds"] = round(last)
        info["units"][unit] = rec
        (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
        print(unit, rec, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    info["seconds_total"] = round(time.time() - t_all)
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
