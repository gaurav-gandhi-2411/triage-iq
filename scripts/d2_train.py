"""D2: fine-tune BAAI/bge-base-en-v1.5 on D1's clean, disjoint-asserted training pools.

MEASURE-FIRST (spec.md): run this on vscode_duplicate ALONE with default hyperparameters first.
ESCALATE that single-run result before running any sweep or the k8s_related attempt. Only sweep
if the single run shows real signal on the held-out eval (scripts/d2_eval_finetuned.py).

Leakage guard: re-asserts train/eval issue-level disjointness (scripts/d2_assert_leakage_guard.py)
as a hard pre-flight gate before training starts -- non-negotiable, fails hard on violation.

Training method: direct HuggingFace loop (not sentence-transformers .fit()) with
MultipleNegativesRankingLoss, CLS-token pooling (matching BGE's native config) + L2-normalize,
saved in sentence-transformers format so scripts/d2_eval_finetuned.py / SimilarIssueRetriever
can load it directly.

CORRECTED (see the ADR superseding ADR-0031/0033/0034): the first D2 run used mean pooling
(w3_t4_train.py's method) and max_seq_length=128, both of which diverge from
BAAI/bge-base-en-v1.5's own config (CLS pooling, 512 max length) -- two mechanistic confounds
that alone could explain the regression that run measured, independent of data/approach. Both
fixed here: CLS pooling (cls_pool(), matching the saved 1_Pooling/config.json) and MAX_LEN=512.

Anchor/positive text is looked up from the processed corpus (title + body_clean[:512]) by issue
number, NOT from the train-pool JSON's title-only columns -- this matches exactly how
SimilarIssueRetriever.build_index() constructs index text, so the fine-tuned embedder's training
distribution matches its eval-time (and serving-time) input distribution.

Usage:
  python scripts/d2_train.py --task vscode_duplicate                    # measure-first default run
  python scripts/d2_train.py --task k8s_related --epochs 8 --lr 1e-5    # thin-data run, more regularization
  python scripts/d2_train.py --task vscode_duplicate --lr 1e-5 --epochs 3 --run-name lr1e-5_ep3   # sweep leg

Reads:  reports/d1_train_pool_{task}.json
        data/d2_hard_negatives_{task}.parquet   (scripts/d2_mine_train_negatives.py)
        data/processed/issues_{repo}.parquet
Writes: data/models/d2_finetuned_{task}[_{run_name}]/   (sentence-transformers format)
        reports/d2_train_{task}[_{run_name}].json        (loss curve, hyperparams, provenance)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from d2_assert_leakage_guard import TASKS, assert_task_disjoint
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
BASE_MODEL = "BAAI/bge-base-en-v1.5"
# Corrected (see the ADR superseding ADR-0031/0033/0034): was 128, a train/inference length
# mismatch -- the original run truncated 65.73% of anchor/positive/negative examples at 128
# tokens. Measured the real BGE-tokenizer length distribution over vscode_duplicate's training
# triplets (25,980 texts): p50=146, p90=217, p95=230, p99=272, max=314. 512 (BGE-base's native
# max) round-trips through O(L^2) attention on mostly-padding for no benefit -- at seq_len=512
# with batch=16 the RTX 3070 (8GB) sat at 92% VRAM and epoch 1 alone took 3h25m (~47x slower per
# step than the original run), a VRAM-pressure/paging effect, not pure compute scaling. 256 is
# the smallest power of 2 covering p95, truncating only 2.26% (586/25980) of examples -- prod/eval
# still embed at BGE's own default (untouched), this only bounds what TRAINING pads/truncates to.
MAX_LEN = 256
MAX_BODY = 512
TEMPERATURE = 0.05  # standard for bge MNRL fine-tunes (matches w3_t4_train.py)
GRAD_ACCUM_STEPS = 2  # per-device batch = batch_size // GRAD_ACCUM_STEPS, same effective batch

REPO_BY_TASK = {
    "vscode_duplicate": "microsoft_vscode",
    "k8s_related": "kubernetes_kubernetes",
}
REPORTS = Path("reports")
DATA = Path("data")
MODELS_DIR = DATA / "models"

# Measure-first defaults. vscode_duplicate: 1734 pairs is a normal-sized fine-tune set for this
# model/loss combo (matches w3_t4_train.py's defaults, which trained on similar per-repo volumes).
# k8s_related: 264 pairs is THIN -- fewer epochs + higher weight decay to fight overfit, per
# spec.md's pre-registered "may not generalize" expectation.
DEFAULTS = {
    "vscode_duplicate": {"lr": 2e-5, "epochs": 5, "batch_size": 16, "weight_decay": 0.01},
    "k8s_related": {"lr": 2e-5, "epochs": 3, "batch_size": 16, "weight_decay": 0.05},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(title: object, body: object) -> str:
    t = (str(title) if title is not None else "").strip()
    b = (str(body) if body is not None else "").strip()[:MAX_BODY]
    return f"{t}. {b}"


class TripletDataset(torch.utils.data.Dataset):
    def __init__(
        self, anchors: list[str], positives: list[str], negatives: list[str], tokenizer: AutoTokenizer
    ) -> None:
        assert len(anchors) == len(positives) == len(negatives)
        self.enc_a = tokenizer(
            anchors, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
        )
        self.enc_p = tokenizer(
            positives, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
        )
        self.enc_n = tokenizer(
            negatives, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
        )

    def __len__(self) -> int:
        return self.enc_a["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "a_input_ids": self.enc_a["input_ids"][idx],
            "a_attention_mask": self.enc_a["attention_mask"][idx],
            "p_input_ids": self.enc_p["input_ids"][idx],
            "p_attention_mask": self.enc_p["attention_mask"][idx],
            "n_input_ids": self.enc_n["input_ids"][idx],
            "n_attention_mask": self.enc_n["attention_mask"][idx],
        }


def load_triplets(task: str) -> tuple[list[str], list[str], list[str]]:
    train_file, _ = TASKS[task]
    repo = REPO_BY_TASK[task]
    train_pairs = json.loads((REPORTS / train_file).read_text(encoding="utf-8"))
    hard_negs = pd.read_parquet(DATA / f"d2_hard_negatives_{task}.parquet")

    corpus = pd.read_parquet(DATA / "processed" / f"issues_{repo}.parquet")
    text_by_num = dict(
        zip(corpus["number"].astype(int), corpus.apply(lambda r: build_text(r["title"], r["body_clean"]), axis=1), strict=True)
    )

    negs_by_pair: dict[tuple[int, int], list[str]] = {}
    for _, row in hard_negs.iterrows():
        key = (int(row["query_number"]), int(row["original_number"]))
        negs_by_pair.setdefault(key, []).append(str(row["neg_text"]))

    anchors, positives, negatives = [], [], []
    skipped_no_neg = 0
    skipped_no_text = 0
    for p in train_pairs:
        q, o = int(p["query_number"]), int(p["original_number"])
        if q not in text_by_num or o not in text_by_num:
            skipped_no_text += 1
            continue
        negs = negs_by_pair.get((q, o), [])
        if not negs:
            skipped_no_neg += 1
            continue
        for neg_text in negs:
            anchors.append(text_by_num[q])
            positives.append(text_by_num[o])
            negatives.append(neg_text)

    logger.info(
        "[%s] built %d anchor/pos/neg triplets from %d train pairs (%d skipped: no hard neg, "
        "%d skipped: issue missing from corpus)",
        task, len(anchors), len(train_pairs), skipped_no_neg, skipped_no_text,
    )
    return anchors, positives, negatives


def cls_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    # BGE's native pooling (BAAI/bge-base-en-v1.5's own 1_Pooling/config.json:
    # pooling_mode_cls_token=true, pooling_mode_mean_tokens=false) -- corrected from the prior
    # mean_pool(), which trained against the grain of what the checkpoint was pretrained with.
    # See the ADR superseding ADR-0031/0033/0034.
    return hidden[:, 0, :]


def _save_st_model(model: AutoModel, tokenizer: AutoTokenizer, out_dir: Path) -> None:
    transformer_dir = out_dir / "0_Transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(transformer_dir))
    tokenizer.save_pretrained(str(transformer_dir))
    word_embedding_model = st_models.Transformer(str(transformer_dir), max_seq_length=MAX_LEN).cpu()
    # CLS-token pooling to match BGE's native config -- must agree with cls_pool() above and
    # with the base model's own 1_Pooling/config.json (asserted before training, see verify step).
    pooling_model = st_models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
    )
    normalize_model = st_models.Normalize()
    st = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_model])
    st.save(str(out_dir))
    del st, word_embedding_model, pooling_model, normalize_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(task: str, lr: float, epochs: int, batch_size: int, weight_decay: float, out_dir: Path) -> dict:
    assert_task_disjoint(task)  # hard pre-flight gate, non-negotiable

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s] device=%s", task, device)

    anchors, positives, negatives = load_triplets(task)
    if not anchors:
        raise SystemExit(f"[{task}] zero usable triplets -- check hard-negative mining output")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = TripletDataset(anchors, positives, negatives, tokenizer)
    # Gradient accumulation to hold the spec's effective batch under VRAM constraints (the
    # original max_seq_length=512 run sat at 92% VRAM and epoch 1 alone took 3h25m -- a
    # paging/memory-pressure effect, not pure compute scaling; see the ADR superseding
    # ADR-0031/0033/0034). NOTE: MNRL's loss uses IN-BATCH negatives, so this is NOT strictly
    # gradient-identical to one true batch of `batch_size` -- each per-device micro-batch only
    # sees its own (smaller) in-batch candidate pool (2x per_device_batch candidates, not
    # 2x batch_size). Standard practice for VRAM-constrained contrastive fine-tuning, but
    # documented precisely rather than claimed as exactly equivalent.
    per_device_batch = max(1, batch_size // GRAD_ACCUM_STEPS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )

    model = AutoModel.from_pretrained(BASE_MODEL).to(device)
    optimizer_steps_per_epoch = -(-len(loader) // GRAD_ACCUM_STEPS)  # ceil division
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(0.10 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))

    logger.info(
        "[%s] %d triplets, effective_batch=%d, per_device_batch=%d, grad_accum=%d, epochs=%d, "
        "micro_batches/epoch=%d, optimizer_steps/epoch=%d, total_optimizer_steps=%d, lr=%g, wd=%g",
        task, len(anchors), batch_size, per_device_batch, GRAD_ACCUM_STEPS, epochs,
        len(loader), optimizer_steps_per_epoch, total_steps, lr, weight_decay,
    )

    loss_history: list[float] = []
    t0 = time.perf_counter()
    step_t0 = t0
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_micro_steps = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            a_ids, a_mask = batch["a_input_ids"].to(device), batch["a_attention_mask"].to(device)
            p_ids, p_mask = batch["p_input_ids"].to(device), batch["p_attention_mask"].to(device)
            n_ids, n_mask = batch["n_input_ids"].to(device), batch["n_attention_mask"].to(device)

            emb_a = F.normalize(cls_pool(model(a_ids, a_mask).last_hidden_state, a_mask), p=2, dim=1)
            emb_p = F.normalize(cls_pool(model(p_ids, p_mask).last_hidden_state, p_mask), p=2, dim=1)
            emb_n = F.normalize(cls_pool(model(n_ids, n_mask).last_hidden_state, n_mask), p=2, dim=1)

            candidates = torch.cat([emb_p, emb_n], dim=0)
            scores = (emb_a @ candidates.T) / TEMPERATURE
            labels = torch.arange(len(emb_a), device=device)
            loss = F.cross_entropy(scores, labels) / GRAD_ACCUM_STEPS
            loss.backward()

            is_last_micro_batch = (i + 1) == len(loader)
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or is_last_micro_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS  # undo the accumulation scaling for logging
            epoch_micro_steps += 1

            # Pace sanity-check instrumentation (not needed for correctness): every 20 micro-steps,
            # log elapsed time + s/step + VRAM so a bad config is caught in minutes, not hours.
            if epoch == 0 and (i + 1) % 20 == 0:
                elapsed_steps = time.perf_counter() - step_t0
                mem_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                mem_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                logger.info(
                    "[%s] pace check: %d/%d micro-steps, %.3fs/step, VRAM allocated=%.0fMB reserved=%.0fMB",
                    task, i + 1, len(loader), elapsed_steps / (i + 1), mem_mb, mem_reserved_mb,
                )

        avg_loss = epoch_loss / max(1, epoch_micro_steps)
        loss_history.append(avg_loss)
        logger.info("[%s] epoch %d/%d  avg_loss=%.4f", task, epoch + 1, epochs, avg_loss)

    elapsed = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_st_model(model, tokenizer, out_dir)
    logger.info("[%s] training complete in %.1fs, saved -> %s", task, elapsed, out_dir)

    return {
        "task": task,
        "base_model": BASE_MODEL,
        "n_triplets": len(anchors),
        "hyperparams": {
            "lr": lr, "epochs": epochs, "batch_size": batch_size, "weight_decay": weight_decay,
            "temperature": TEMPERATURE, "max_len": MAX_LEN, "seed": SEED,
            "per_device_batch": per_device_batch, "grad_accum_steps": GRAD_ACCUM_STEPS,
        },
        "loss_by_epoch": loss_history,
        "train_seconds": elapsed,
        "device": str(device),
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--run-name", type=str, default=None, help="suffix for sweep legs, e.g. lr1e-5_ep3")
    args = ap.parse_args()

    defaults = DEFAULTS[args.task]
    lr = args.lr if args.lr is not None else defaults["lr"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    weight_decay = args.weight_decay if args.weight_decay is not None else defaults["weight_decay"]

    suffix = f"_{args.run_name}" if args.run_name else ""
    out_dir = MODELS_DIR / f"d2_finetuned_{args.task}{suffix}"
    report_path = REPORTS / f"d2_train_{args.task}{suffix}.json"

    result = train(args.task, lr, epochs, batch_size, weight_decay, out_dir)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
