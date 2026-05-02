"""Evaluation utilities for the component classifier."""

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_classifier(model, X_test: pd.Series, y_test: pd.Series) -> dict:
    """Comprehensive classifier evaluation."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_()

    # Encode y_test to integers for confusion matrix
    le = model.label_encoder
    y_test_enc = le.transform(y_test)
    y_pred_enc = le.transform(y_pred)

    cm = confusion_matrix(y_test_enc, y_pred_enc)

    # Top confused pairs (off-diagonal, sorted by count)
    rows, cols = np.where(cm > 0)
    confusions = [
        (classes[r], classes[c], int(cm[r, c]))
        for r, c in zip(rows, cols, strict=False)
        if r != c
    ]
    confusions.sort(key=lambda x: -x[2])

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "per_class_metrics": report,
        "confusion_matrix": cm,
        "classes": classes,
        "top_confusions": confusions[:20],
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def latency_benchmark(model, X_sample: pd.Series, n_iters: int = 200) -> dict:
    """Measure single-sample prediction latency (p50, p95, p99, throughput)."""
    # Warm-up
    for _ in range(10):
        model.predict(X_sample.iloc[:1])

    latencies_ms = []
    for i in range(n_iters):
        idx = i % len(X_sample)
        t0 = time.perf_counter()
        model.predict(X_sample.iloc[idx : idx + 1])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "predict_per_sec": float(1000 / arr.mean()),
    }


def batch_latency_benchmark(model, X_sample: pd.Series, batch_size: int = 100) -> dict:
    """Throughput benchmark for batch prediction."""
    n_batches = max(10, 200 // batch_size)
    times = []
    for _ in range(n_batches):
        batch = X_sample.iloc[:batch_size]
        t0 = time.perf_counter()
        model.predict(batch)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "batch_size": batch_size,
        "batch_p50_ms": float(np.percentile(arr, 50)),
        "batch_p95_ms": float(np.percentile(arr, 95)),
        "per_sample_ms": float(arr.mean() / batch_size),
        "samples_per_sec": float(batch_size * 1000 / arr.mean()),
    }


def calibration_analysis(y_test: pd.Series, y_proba: np.ndarray, classes: np.ndarray,
                          label_encoder, n_bins: int = 10) -> dict:
    """Reliability analysis: predicted confidence vs actual accuracy."""

    y_test_enc = label_encoder.transform(y_test)

    # One-vs-rest reliability: max predicted probability vs correctness
    max_proba = y_proba.max(axis=1)
    y_pred_enc = y_proba.argmax(axis=1)
    correct = (y_pred_enc == y_test_enc).astype(int)

    # Bin into confidence buckets
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (max_proba >= lo) & (max_proba < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(correct[mask].mean())
        bin_confs.append(max_proba[mask].mean())
        bin_sizes.append(int(mask.sum()))

    ece = float(sum(
        s * abs(a - c)
        for a, c, s in zip(bin_accs, bin_confs, bin_sizes, strict=False)
    ) / max(sum(bin_sizes), 1))

    mean_conf = float(max_proba.mean())
    mean_acc = float(correct.mean())

    return {
        "ece": ece,
        "mean_confidence": mean_conf,
        "mean_accuracy": mean_acc,
        "overconfident": mean_conf > mean_acc,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_sizes": bin_sizes,
    }


def plot_confusion_heatmap(
    cm: np.ndarray,
    classes: np.ndarray,
    repo: str,
    out_path: str,
    top_n: int = 20,
) -> None:
    """Plot top-N confusion matrix heatmap (by total activity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Select top_n classes by total test count
    totals = cm.sum(axis=1)
    top_idx = np.argsort(totals)[-top_n:][::-1]
    cm_sub = cm[np.ix_(top_idx, top_idx)]
    labels_sub = classes[top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_sub, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels_sub)),
        yticks=range(len(labels_sub)),
        xticklabels=labels_sub,
        yticklabels=labels_sub,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion Matrix (top {top_n} classes) — {repo.replace('_', '/')}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion heatmap to %s", out_path)


def plot_per_class_f1(
    per_class_metrics: dict,
    repo: str,
    out_path: str,
) -> None:
    """Bar chart of F1 per component, sorted descending."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = [k for k in per_class_metrics if k not in ("accuracy", "macro avg", "weighted avg")]
    f1s = [per_class_metrics[c]["f1-score"] for c in classes]
    supports = [per_class_metrics[c]["support"] for c in classes]

    order = np.argsort(f1s)
    classes = [classes[i] for i in order]
    f1s = [f1s[i] for i in order]
    supports = [supports[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(5, len(classes) * 0.3)))
    ax.barh(range(len(classes)), f1s, color="steelblue")
    ax.set(
        yticks=range(len(classes)),
        yticklabels=[f"{c} (n={s})" for c, s in zip(classes, supports, strict=False)],
        xlabel="F1 score",
        title=f"Per-class F1 — {repo.replace('_', '/')}",
        xlim=(0, 1),
    )
    ax.axvline(np.mean(f1s), color="red", linestyle="--", alpha=0.6, label=f"macro avg={np.mean(f1s):.3f}")
    ax.legend(fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-class F1 chart to %s", out_path)


def plot_calibration(
    cal: dict,
    repo: str,
    out_path: str,
) -> None:
    """Reliability diagram: confidence vs accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(cal["bin_confidences"], cal["bin_accuracies"], "o-", color="steelblue",
            label=f"Model (ECE={cal['ece']:.3f})")
    ax.set(
        xlabel="Mean predicted confidence",
        ylabel="Fraction correct",
        title=f"Calibration — {repo.replace('_', '/')}",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    ax.text(
        0.05, 0.90,
        f"Mean conf: {cal['mean_confidence']:.3f}\nMean acc: {cal['mean_accuracy']:.3f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration chart to %s", out_path)
