"""Kaggle kernel (R8): fine-tune bge-base-en-v1.5 on vscode TRAIN duplicate pairs, embed all issues.

Design: docs/Q5A_FINETUNE_PREREG_2026-10.md. Inputs under /kaggle/input: the production-recipe issue
texts (columns number, title, body_clean), train_pairs.parquet (anchor, pos, neg, comp, neg_comp)
and splits.parquet (number, split). No duplicate label of validation / test is in the input.
Per config F1..F4 (lr x hard negative; 3 epochs, batch 32, seed 42) it writes to /kaggle/working
``<id>_model/`` (fp32 HF), ``<id>_emb.npy`` (float16, n x 768, cls + l2, dataset row order), plus
``ids.parquet``, ``parity_base_emb.npy`` / ``parity_idx.npy`` (untouched model, 2,000 issues, to
compare with the #345 vectors) and ``run_info.json``.
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
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-base-en-v1.5"
REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"  # production revision, as embed_bge.py
MAX_SEQ_LENGTH, EMBED_BATCH = 512, 64
BATCH, EPOCHS, WARMUP, WD, SCALE, SEED = 32, 3, 0.10, 0.01, 20.0, 42
CONFIGS = {"F1": (1e-5, False), "F2": (3e-5, False), "F3": (1e-5, True), "F4": (3e-5, True)}
PARITY_N = 2000
TIME_BUDGET_S = 4 * 3600  # prereg: no new config is started after 4 h
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


def production_text(title: str, body_clean: str, tokenizer) -> str:  # noqa: ANN001
    """Same token-based cut as embed_bge.production_text / similar_issues._build_text."""
    prefix = f"{title.strip()}. "
    n_prefix = len(tokenizer.encode(prefix, add_special_tokens=False))
    budget = max(MAX_SEQ_LENGTH - n_prefix - 2, 0)
    ids = tokenizer.encode(
        body_clean.strip(), add_special_tokens=False, truncation=True, max_length=budget
    )
    return prefix + tokenizer.decode(ids, skip_special_tokens=True)


def assert_no_heldout(pairs: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Fail loudly if any anchor / positive / negative is not a train-split issue."""
    train = set(splits.loc[splits["split"] == "train", "number"].astype(int))
    used = set(pairs["anchor"]) | set(pairs["pos"]) | set(pairs["neg"].dropna().astype(int))
    bad = sorted(used - train)
    if bad:
        raise RuntimeError(f"{len(bad)} non-train issues in training data, e.g. {bad[:5]}")


def mnrl_loss(
    a: torch.Tensor, cand: torch.Tensor, comp_a: torch.Tensor, comp_c: torch.Tensor, scale: float
) -> torch.Tensor:
    """In-batch softmax loss. ``cand`` = the B positives, then (optionally) B hard negatives;
    candidate j is the target of anchor j. A candidate in the same duplicate group as the anchor
    (``comp`` equal) other than its own positive is masked out (false negative)."""
    b = a.shape[0]
    logits = (a @ cand.T) * scale
    mask = comp_c[None, :] == comp_a[:, None]
    mask[torch.arange(b), torch.arange(b)] = False
    logits = logits.masked_fill(mask, float("-inf"))
    return F.cross_entropy(logits, torch.arange(b, device=a.device))


def batches(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled index batches of BATCH; a trailing batch of < 2 is dropped (needs a negative)."""
    perm = rng.permutation(n)
    out = [perm[s : s + BATCH] for s in range(0, n, BATCH)]
    return [b for b in out if len(b) >= 2]


def cls_l2(model, tok, texts: list[str], device: str) -> torch.Tensor:  # noqa: ANN001
    """CLS embedding, L2-normalised, fp32 (autocast fp16 forward); grads flow if model.train()."""
    b = tok(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        h = model(**b.to(device)).last_hidden_state[:, 0]
    return F.normalize(h.float(), p=2, dim=1)


@torch.inference_mode()
def embed_all(model, tok, texts: list[str], device: str) -> np.ndarray:  # noqa: ANN001
    """float16 (n, 768) embeddings in input order (length-sorted batches cut padding)."""
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    for s in range(0, len(order), EMBED_BATCH):
        idx = order[s : s + EMBED_BATCH]
        out[idx] = (
            cls_l2(model, tok, [texts[i] for i in idx], device).cpu().numpy().astype(np.float16)
        )
    return out


def train_one(lr: float, hard: bool, pairs: pd.DataFrame, text_of: dict, tok, device: str):  # noqa: ANN001, ANN201
    """Fine-tune on the train pairs; fp32 master weights (R11 lesson), fp16 autocast forward."""
    seed_all()
    model = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).float().to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    rng = np.random.default_rng(SEED)
    steps = EPOCHS * len(batches(len(pairs), np.random.default_rng(SEED)))
    warm = int(WARMUP * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    comp, ncomp = pairs["comp"].to_numpy(), pairs["neg_comp"].to_numpy()
    losses = []
    for ep in range(EPOCHS):
        model.train()
        ep_loss = []
        for idx in batches(len(pairs), rng):
            rows = pairs.iloc[idx]
            texts = [text_of[int(n)] for n in rows["anchor"]] + [
                text_of[int(n)] for n in rows["pos"]
            ]
            if hard:
                texts += [text_of[int(n)] for n in rows["neg"]]
            e = cls_l2(model, tok, texts, device)
            b = len(idx)
            comp_c = comp[idx] if not hard else np.concatenate([comp[idx], ncomp[idx]])
            loss = mnrl_loss(
                e[:b], e[b:], torch.as_tensor(comp[idx], device=device),
                torch.as_tensor(comp_c, device=device), SCALE,
            )  # fmt: skip
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
    device = "cuda"
    t_all = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    issues = pd.read_parquet(find("microsoft__vscode__issues.parquet")).reset_index(drop=True)
    pairs, splits = (
        pd.read_parquet(find("train_pairs.parquet")),
        pd.read_parquet(find("splits.parquet")),
    )
    assert_no_heldout(pairs, splits)
    texts = [
        production_text(t, b, tok)
        for t, b in zip(issues["title"].fillna(""), issues["body_clean"].fillna(""), strict=True)
    ]
    text_of = dict(zip(issues["number"].astype(int), texts, strict=True))
    issues[["repo", "kind", "number"]].to_parquet(OUT / "ids.parquet")
    info: dict = {
        "model_id": MODEL_ID, "revision": REVISION, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "n_pairs": len(pairs), "n_issues": len(issues), "seed": SEED, "configs": {},
    }  # fmt: skip
    base = AutoModel.from_pretrained(MODEL_ID, revision=REVISION).float().to(device)
    pidx = np.sort(np.random.default_rng(SEED).choice(len(texts), PARITY_N, replace=False))
    np.save(OUT / "parity_idx.npy", pidx)
    np.save(OUT / "parity_base_emb.npy", embed_all(base, tok, [texts[i] for i in pidx], device))
    del base
    gc.collect()
    torch.cuda.empty_cache()
    last = 0.0
    for cid, (lr, hard) in CONFIGS.items():
        if time.time() - t_all + last > TIME_BUDGET_S:
            info["configs"][cid] = {"skipped": "time budget"}
            continue
        t0 = time.time()
        model, losses = train_one(lr, hard, pairs, text_of, tok, device)
        t_train = time.time() - t0
        model.save_pretrained(OUT / f"{cid}_model")
        np.save(OUT / f"{cid}_emb.npy", embed_all(model, tok, texts, device))
        last = time.time() - t0
        info["configs"][cid] = {"lr": lr, "hard_negative": hard, "epoch_loss": losses,
                                "train_seconds": round(t_train), "total_seconds": round(last)}  # fmt: skip
        (OUT / "run_info.json").write_text(json.dumps(info, indent=2))
        print(cid, info["configs"][cid], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    tok.save_pretrained(OUT / "tokenizer")
    info["seconds_total"] = round(time.time() - t_all)
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
