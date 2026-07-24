"""Phase B (classifier improvement): DeBERTa-v3-base component classifier.

ARM 1 (--arm single): softmax over the single collapsed `component` label -- same supervision
as the current TF-IDF+LR baseline. Isolates the architecture effect cleanly.

ARM 2 (--arm multi): sigmoid/BCE over ALL valid component labels per issue (multi-hot,
built from `labels_raw` via classifier_eval.py::all_matching_component_labels -- the same
function that measured the 30.4%/8.0% multi-true-label collapse in Phase A). Tests whether
fixing the supervision defect (normalize_labels() discarding valid labels) matters more than
the architecture change.

Leakage guard: scripts/classifier_assert_leakage_guard.py asserted as a hard pre-flight gate.

Config, measured not assumed (Phase A/D2 lesson):
  - Tokenizer/checkpoint match by construction (single BASE_MODEL constant for both).
  - max_seq_length=512 (DeBERTa's native ceiling): measured token-length p95 far exceeds even
    256 (1100 vscode / 429 k8s) -- 512 truncates only 8.33%/3.42%, the best achievable without
    altering the base architecture.
  - Classification head/pooling: HF's native AutoModelForSequenceClassification for the
    deberta-v2 architecture family (v3-base uses this class) attaches the library's own tested
    ContextPooler + classifier head -- not hand-rolled, so there is no scope for the D2-style
    mean-vs-CLS divergence. Both arms share the identical base class and pooling head, differing
    only in problem_type (single_label_classification vs multi_label_classification) and loss
    (CrossEntropy vs BCEWithLogits), both handled internally by HF's Trainer/model classes.

Usage:
  python scripts/deberta_train.py --arm single --repo microsoft_vscode
  python scripts/deberta_train.py --arm multi --repo microsoft_vscode
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, "src")
from triage_iq.evaluation.classifier_eval import all_matching_component_labels  # noqa: E402
from triage_iq.models.component_classifier import _build_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 512
PER_DEVICE_BATCH = 8
GRAD_ACCUM_STEPS = 2  # effective batch 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS = Path("reports")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_leakage_guard_passed() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/classifier_assert_leakage_guard.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("Leakage guard FAILED -- refusing to train.")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray, multi_label: bool) -> None:
        self.encodings = encodings
        self.labels = labels
        self.multi_label = multi_label

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.multi_label:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_multi_hot(repo: str, labels_raw_series: pd.Series, classes: list[str]) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(labels_raw_series), len(classes)), dtype=np.float32)
    for i, labels_raw in enumerate(labels_raw_series):
        for m in all_matching_component_labels(repo, labels_raw):
            if m in class_to_idx:
                Y[i, class_to_idx[m]] = 1.0
    return Y


def train(
    arm: str, repo: str,
    max_len: int = MAX_LEN,
    per_device_batch: int = PER_DEVICE_BATCH,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
) -> dict:
    assert_leakage_guard_passed()
    set_seed(SEED)

    train_df = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_val.parquet")

    X_train = _build_text(train_df["title"], train_df["body_clean"]).tolist()
    X_val = _build_text(val_df["title"], val_df["body_clean"]).tolist()

    le = LabelEncoder()
    le.fit(train_df["component"])
    classes = list(le.classes_)
    n_classes = len(classes)

    multi_label = arm == "multi"
    if multi_label:
        y_train = build_multi_hot(repo, train_df["labels_raw"], classes)
        y_val = build_multi_hot(repo, val_df["labels_raw"], classes)
        problem_type = "multi_label_classification"
    else:
        y_train = le.transform(train_df["component"])
        y_val = le.transform(val_df["component"])
        problem_type = "single_label_classification"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s/%s] device=%s, n_classes=%d, train=%d, val=%d",
                arm, repo, device, n_classes, len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    enc_train = tokenizer(X_train, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    enc_val = tokenizer(X_val, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")

    ds_train = TextDataset(enc_train, y_train, multi_label)
    ds_val = TextDataset(enc_val, y_val, multi_label)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=n_classes, problem_type=problem_type,
    ).to(device)

    out_dir = MODELS_DIR / f"deberta_{arm}_{repo}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=20,  # pace-check granularity
        seed=SEED,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val)

    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    import joblib
    joblib.dump(le, out_dir / "label_encoder.pkl")

    logger.info("[%s/%s] training complete in %.1fs, saved -> %s", arm, repo, elapsed, out_dir)

    log_history = [
        {k: v for k, v in entry.items() if k in ("epoch", "loss", "eval_loss")}
        for entry in trainer.state.log_history
        if "loss" in entry or "eval_loss" in entry
    ]

    result = {
        "arm": arm,
        "repo": repo,
        "base_model": BASE_MODEL,
        "problem_type": problem_type,
        "n_classes": n_classes,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "hyperparams": {
            "lr": LR, "epochs": EPOCHS, "per_device_batch": per_device_batch,
            "grad_accum_steps": grad_accum_steps, "effective_batch": per_device_batch * grad_accum_steps,
            "max_len": max_len, "weight_decay": WEIGHT_DECAY, "seed": SEED,
        },
        "log_history": log_history,
        "train_seconds": elapsed,
        "device": str(device),
        "out_dir": str(out_dir),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["single", "multi"])
    ap.add_argument("--repo", required=True, choices=["microsoft_vscode", "kubernetes_kubernetes"])
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--per-device-batch", type=int, default=PER_DEVICE_BATCH)
    ap.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    result = train(args.arm, args.repo, args.max_len, args.per_device_batch, args.grad_accum_steps)
    suffix = f"_{args.run_name}" if args.run_name else ""
    report_path = REPORTS / f"deberta_train_{args.arm}_{args.repo}{suffix}.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
