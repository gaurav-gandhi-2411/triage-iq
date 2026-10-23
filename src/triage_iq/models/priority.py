"""Text-only priority model (Phase 3b flag 1, docs/QP3B_DESIGN_2026-10.md).

Serves the P4 candidate (scripts/qp3b_priority_p4.py): TF-IDF blocks over the title and the
first 4000 body characters + class-balanced multinomial LR. The training-time `Featurizer`
lives in scripts/ and cannot be imported from src, so scripts/qp3b_package_priority_model.py
re-saves the fitted pieces as plain sklearn objects + data; this module reproduces the
transform exactly. Pure inference, no I/O except `load`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack

PRIORITY_CLASSES = ("low", "medium", "high")  # P4's fixed class order
BODY_CHARS = 4000  # P4 trained on title + "\n" + body[:4000]
FORMAT_VERSION = 1
_FIELDS = ("title", "body", "full")


def build_text(title: str, body: str) -> str:
    """The exact text P4 was trained on (qp3b_priority.load_split)."""
    return f"{title}\n{body[:BODY_CHARS]}"


def _field(name: str, text: str) -> str:
    """Mirror of qp1_tfidf_features.field_title / field_body / field_full for one text."""
    if name == "full":
        return text
    head, _, rest = text.partition("\n")
    return head if name == "title" else rest


@dataclass(frozen=True)
class PriorityPrediction:
    """label: argmax class. proba: probability per class (PRIORITY_CLASSES keys)."""

    label: str
    proba: dict[str, float]


class PriorityModel:
    """Packaged P4: `blocks` is [(fitted TfidfVectorizer, field name)], `lr` a fitted LR."""

    def __init__(self, blocks: list[tuple[Any, str]], lr: Any) -> None:
        bad = [f for _, f in blocks if f not in _FIELDS]
        if bad:
            raise ValueError(f"unknown priority feature field(s): {bad}")
        self.blocks, self.lr = blocks, lr

    @classmethod
    def load(cls, path: Path) -> PriorityModel:
        """Load a packaged artifact; raises ValueError on an unknown format."""
        d = joblib.load(path)
        if not isinstance(d, dict) or d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: not a priority-model artifact v{FORMAT_VERSION}")
        if tuple(d["classes"]) != PRIORITY_CLASSES:
            raise ValueError(f"{path}: unexpected classes {d['classes']}")
        return cls([(b["vectorizer"], b["field"]) for b in d["blocks"]], d["lr"])

    def proba_texts(self, texts: list[str]) -> np.ndarray:
        """(n, 3) probabilities aligned to PRIORITY_CLASSES (an unseen class gets 0)."""
        x = hstack([v.transform([_field(f, t) for t in texts]) for v, f in self.blocks]).tocsr()
        out = np.zeros((len(texts), len(PRIORITY_CLASSES)))
        out[:, [PRIORITY_CLASSES.index(c) for c in self.lr.classes_]] = self.lr.predict_proba(x)
        return out

    def predict(self, title: str, body: str) -> PriorityPrediction:
        """Priority for one issue from the serve-time inputs (title, body)."""
        p = self.proba_texts([build_text(title, body)])[0]
        return PriorityPrediction(
            PRIORITY_CLASSES[int(p.argmax())],
            {c: float(v) for c, v in zip(PRIORITY_CLASSES, p, strict=True)},
        )


def model_rationale(served: str, proba: dict[str, float], llm_proposal: str) -> str:
    """Templated rationale citing the model probability (used when the LLM's text contradicts)."""
    note = "" if llm_proposal.lower() == served else f" (the LLM proposed {llm_proposal})"
    return (
        f"Text-only priority model: {served} (p={proba[served]:.2f}; "
        + ", ".join(f"{c} {proba[c]:.2f}" for c in PRIORITY_CLASSES)
        + f"){note}."
    )


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file (artifact provenance)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
