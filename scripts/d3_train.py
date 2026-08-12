"""D3: fine-tune BAAI/bge-base-en-v1.5 on the expanded mining-precision training pools -- the D2
retry D2 never actually got (D2 run 1 was confounded by mean/CLS pooling mismatch + 128-token
truncation, withdrawn by the ADR superseding ADR-0031/0033/0034; D2 run 2, corrected, was
underpowered at 1,734 pairs / n=200 test, NO SIGNAL). Same training method as
scripts/d2_train.py (CLS-token pooling matching BGE's native config, MAX_LEN=256, verified
BEFORE this run via scripts/d3_verify_config.py -- all three checks PASS, see
reports/d3_config_verification.json), pointed at the larger, precision-corrected pools instead:
k8s_related grows from D2's 264 (NO-GO'd as thin) to 448 pairs -- no longer thin, run for the
first time. vscode_duplicate grows from D2's 1,734 to 1,958 pairs.

Reads:  reports/mining_precision_train_pool_{task}.json
        data/d3_hard_negatives_{task}.parquet   (scripts/d3_mine_train_negatives.py)
        data/processed/issues_{repo}.parquet
Writes: data/models/d3_finetuned_{task}/   (sentence-transformers format)
        reports/d3_train_{task}.json
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
from d3_assert_leakage_guard import TASKS, assert_task_disjoint
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
BASE_MODEL = "BAAI/bge-base-en-v1.5"
MAX_LEN = 256  # verified against measured p95=223 on this pool, scripts/d3_verify_config.py
MAX_BODY = 512
TEMPERATURE = 0.05
GRAD_ACCUM_STEPS = 2

REPO_BY_TASK = {
    "vscode_duplicate": "microsoft_vscode",
    "k8s_related": "kubernetes_kubernetes",
    "k8s_related_valid_subset": "kubernetes_kubernetes",
    "k8s_related_fullcorpus_negs": "kubernetes_kubernetes",
}
REPORTS = Path("reports")
DATA = Path("data")
MODELS_DIR = DATA / "models"

# k8s_related is no longer thin (448 vs D2's 264) but still smaller than vscode_duplicate --
# keep D2's own per-task regularization split (more weight decay, fewer epochs for k8s) rather
# than introducing a fresh hyperparameter sweep; this is a measure-first single default run.
DEFAULTS = {
    "vscode_duplicate": {"lr": 2e-5, "epochs": 5, "batch_size": 16, "weight_decay": 0.01},
    "k8s_related": {"lr": 2e-5, "epochs": 4, "batch_size": 16, "weight_decay": 0.05},
    # D3a candidate-A test (ADR-0048 follow-up): same hyperparams as k8s_related -- this run's
    # purpose is isolating the effect of pool precision, not re-sweeping hyperparameters on top.
    "k8s_related_valid_subset": {"lr": 2e-5, "epochs": 4, "batch_size": 16, "weight_decay": 0.05},
    # D3a candidate-B test: same hyperparams and same 448-pair pool as k8s_related -- only the
    # hard-negative candidate space differs (full corpus vs. training-pool-restricted).
    "k8s_related_fullcorpus_negs": {"lr": 2e-5, "epochs": 4, "batch_size": 16, "weight_decay": 0.05},
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
    hard_negs = pd.read_parquet(DATA / f"d3_hard_negatives_{task}.parquet")

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
    return hidden[:, 0, :]


def _save_st_model(model: AutoModel, tokenizer: AutoTokenizer, out_dir: Path) -> None:
    transformer_dir = out_dir / "0_Transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(transformer_dir))
    tokenizer.save_pretrained(str(transformer_dir))
    word_embedding_model = st_models.Transformer(str(transformer_dir), max_seq_length=MAX_LEN).cpu()
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
    assert_task_disjoint(task)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s] device=%s", task, device)

    anchors, positives, negatives = load_triplets(task)
    if not anchors:
        raise SystemExit(f"[{task}] zero usable triplets -- check hard-negative mining output")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = TripletDataset(anchors, positives, negatives, tokenizer)
    per_device_batch = max(1, batch_size // GRAD_ACCUM_STEPS)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")
    )

    model = AutoModel.from_pretrained(BASE_MODEL).to(device)
    optimizer_steps_per_epoch = -(-len(loader) // GRAD_ACCUM_STEPS)
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

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            epoch_micro_steps += 1

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
    args = ap.parse_args()

    defaults = DEFAULTS[args.task]
    lr = args.lr if args.lr is not None else defaults["lr"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    weight_decay = args.weight_decay if args.weight_decay is not None else defaults["weight_decay"]

    out_dir = MODELS_DIR / f"d3_finetuned_{args.task}"
    report_path = REPORTS / f"d3_train_{args.task}.json"

    result = train(args.task, lr, epochs, batch_size, weight_decay, out_dir)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
