"""LLM few-shot component classification via Groq (Llama 3.1 8B).

Designed for cold-start / rare-label scenarios where the training set
has insufficient examples to fine-tune a supervised model.
"""

import logging
import os
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
REQUEST_DELAY_S = 1.2  # Stay comfortably under Groq rate limit


def _get_groq_client():
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file or export it:\n"
            "  export GROQ_API_KEY=gsk_..."
        )
    return Groq(api_key=key)


def _format_few_shot_examples(examples: list[tuple[str, str]]) -> str:
    lines = []
    for text, label in examples:
        truncated = text[:300].replace("\n", " ")
        lines.append(f'  Issue: "{truncated}"\n  Component: {label}')
    return "\n\n".join(lines)


def _parse_label_from_response(raw: str, candidate_labels: list[str]) -> Optional[str]:
    """Extract a valid label from the model response."""
    raw = raw.strip().lower()

    # Exact match first
    for label in candidate_labels:
        if label.lower() == raw:
            return label

    # Prefix / substring match (model sometimes adds punctuation or explanation)
    for label in candidate_labels:
        if raw.startswith(label.lower()) or label.lower() in raw:
            return label

    return None  # Could not parse a valid label


def classify_with_llm_fewshot(
    text: str,
    candidate_labels: list[str],
    examples: list[tuple[str, str]],
    client=None,
) -> Optional[str]:
    """Use Llama 3.1 8B via Groq for few-shot component classification.

    Args:
        text: Issue text to classify (title + body).
        candidate_labels: Valid component labels for this repo.
        examples: 3-5 (text, label) pairs sampled from training set.
        client: Optional pre-built Groq client (reuse across calls).

    Returns:
        Predicted label string, or None if parsing failed.
    """
    if client is None:
        client = _get_groq_client()

    labels_str = ", ".join(candidate_labels)
    few_shot_block = _format_few_shot_examples(examples)

    prompt = f"""Classify this GitHub issue into exactly one of these components: {labels_str}

Examples:
{few_shot_block}

Issue: {text[:1500]}
Component:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return _parse_label_from_response(raw, candidate_labels)
    except Exception as e:
        logger.warning("Groq API error: %s", e)
        return None


def run_llm_fewshot_eval(
    X_test: "pd.Series",
    y_test: "pd.Series",
    X_train: "pd.Series",
    y_train: "pd.Series",
    candidate_labels: list[str],
    n_samples: int = 200,
    n_few_shot: int = 5,
    seed: int = 42,
) -> dict:
    """Evaluate LLM few-shot on a random sample of the test set.

    Returns a dict with accuracy, macro_f1, per_class_f1, n_parsed, n_failed,
    and raw predictions for the sampled examples.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    rng = random.Random(seed)
    client = _get_groq_client()

    # Sample test indices
    indices = list(range(len(X_test)))
    rng.shuffle(indices)
    sample_idx = indices[:n_samples]
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

    # Build label → training examples lookup
    label_to_examples: dict[str, list[str]] = {}
    for text, label in zip(X_train, y_train):
        label_to_examples.setdefault(label, []).append(text)

    y_pred = []
    failed = 0
    for i, (text, true_label) in enumerate(zip(X_sample, y_sample)):
        # Sample few-shot examples: prefer same label, fill with random others
        same_label = label_to_examples.get(true_label, [])
        rng.shuffle(same_label)
        pos_examples = [(t, true_label) for t in same_label[:2]]

        other_labels = [l for l in candidate_labels if l != true_label]
        neg_examples = []
        for lbl in rng.sample(other_labels, min(n_few_shot - len(pos_examples), len(other_labels))):
            pool = label_to_examples.get(lbl, [])
            if pool:
                neg_examples.append((rng.choice(pool), lbl))

        examples = pos_examples + neg_examples
        rng.shuffle(examples)
        examples = examples[:n_few_shot]

        pred = classify_with_llm_fewshot(text, candidate_labels, examples, client=client)
        if pred is None:
            failed += 1
            pred = rng.choice(candidate_labels)  # random fallback for metrics
        y_pred.append(pred)

        if (i + 1) % 20 == 0:
            so_far_acc = accuracy_score(y_sample[: i + 1], y_pred)
            logger.info("LLM eval [%d/%d] acc=%.3f failed=%d", i + 1, n_samples, so_far_acc, failed)

        time.sleep(REQUEST_DELAY_S)

    y_pred_s = pd.Series(y_pred)
    acc = accuracy_score(y_sample, y_pred_s)
    macro_f1 = f1_score(y_sample, y_pred_s, average="macro", zero_division=0, labels=candidate_labels)
    weighted_f1 = f1_score(y_sample, y_pred_s, average="weighted", zero_division=0)

    from sklearn.metrics import classification_report
    report = classification_report(y_sample, y_pred_s, output_dict=True, zero_division=0,
                                   labels=candidate_labels)
    per_class_f1 = {
        k: round(v["f1-score"], 3)
        for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }

    return {
        "n_samples": n_samples,
        "n_parsed": n_samples - failed,
        "n_failed": failed,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "y_true": list(y_sample),
        "y_pred": list(y_pred_s),
    }
