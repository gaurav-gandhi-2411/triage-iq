"""DistilBERT fine-tuned classification head for component labeling.

Trained per-repo since label vocabularies differ across projects.
Training uses GPU if available; latency benchmarks run on CPU.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256


class _IssueDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _make_compute_metrics(label_encoder):
    from sklearn.metrics import f1_score, accuracy_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, preds),
        }

    return compute_metrics


class _WeightedLossTrainer(Trainer):
    """Trainer with class-balanced cross-entropy loss (mirrors TF-IDF class_weight='balanced')."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self._class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


class DistilBERTComponentClassifier:
    """DistilBERT base + classification head, fine-tuned per repo."""

    def __init__(self, repo: str, num_labels: int) -> None:
        self.repo = repo
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._save_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.Series,
        y_train: pd.Series,
        X_val: pd.Series,
        y_val: pd.Series,
        epochs: int = 8,
        output_dir: str = "data/models/distilbert_tmp",
    ) -> "DistilBERTComponentClassifier":
        self._save_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        from sklearn.utils.class_weight import compute_class_weight

        self.label_encoder = LabelEncoder()
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)

        cw = compute_class_weight("balanced", classes=np.arange(self.num_labels), y=y_train_enc)
        class_weights = torch.tensor(cw, dtype=torch.float)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=self.num_labels
        )

        train_enc = self.tokenizer(
            list(X_train), truncation=True, padding=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        val_enc = self.tokenizer(
            list(X_val), truncation=True, padding=True, max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        train_dataset = _IssueDataset(train_enc, y_train_enc)
        val_dataset = _IssueDataset(val_enc, y_val_enc)

        use_cuda = torch.cuda.is_available()
        train_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=50,
            fp16=use_cuda,
            dataloader_num_workers=0,
            report_to="none",
        )

        trainer = _WeightedLossTrainer(
            class_weights=class_weights,
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_make_compute_metrics(self.label_encoder),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("Training %s on %d examples (%s)",
                    self.repo, len(y_train), "GPU" if use_cuda else "CPU")
        trainer.train()

        # Best model is already loaded by load_best_model_at_end=True
        logger.info("Training complete for %s", self.repo)

        # Save best model weights + tokenizer + label encoder
        self._persist(output_dir)
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.label_encoder.inverse_transform(proba.argmax(axis=1))

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        assert self.model is not None and self.label_encoder is not None
        device = next(self.model.parameters()).device
        self.model.eval()

        all_probs = []
        batch_size = 32
        texts = list(X)

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=MAX_LENGTH, return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def classes_(self) -> np.ndarray:
        assert self.label_encoder is not None
        return self.label_encoder.classes_

    def to_cpu(self) -> "DistilBERTComponentClassifier":
        """Move model to CPU (for latency benchmark)."""
        if self.model is not None:
            self.model.cpu()
        return self

    def to_gpu(self) -> "DistilBERTComponentClassifier":
        if self.model is not None and torch.cuda.is_available():
            self.model.cuda()
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, output_dir: str) -> None:
        import joblib
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        joblib.dump(self.label_encoder, Path(output_dir) / "label_encoder.pkl")
        logger.info("Saved DistilBERT model to %s", output_dir)

    def save(self, output_dir: str) -> None:
        self._persist(output_dir)

    @classmethod
    def load(cls, output_dir: str, repo: str) -> "DistilBERTComponentClassifier":
        import joblib
        le: LabelEncoder = joblib.load(Path(output_dir) / "label_encoder.pkl")
        num_labels = len(le.classes_)
        obj = cls(repo=repo, num_labels=num_labels)
        obj.label_encoder = le
        obj.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        obj._save_dir = output_dir
        return obj
