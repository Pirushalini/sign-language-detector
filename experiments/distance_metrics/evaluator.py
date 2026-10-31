"""Evaluation engine for distance metrics comparison.

Calculates:
  1. Overall classification accuracy
  2. Macro F1-score, Precision, and Recall
  3. Per-letter recall breakdown
  4. Confusion matrix
  5. Average inference latency per query (milliseconds)
  6. Class scores and top-2 margin (separation between best and second-best candidate)
  7. Probabilistic calibration metrics: Brier score and Expected Calibration Error (ECE)

All metrics are evaluated under strictly identical conditions across all distance functions.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .matcher import DistanceTemplateMatcher


@dataclass
class EvaluationResult:
    """Complete evaluation statistics for one distance metric."""

    metric_name: str
    is_synthetic: bool
    num_train_samples: int
    num_test_samples: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    avg_latency_ms: float
    avg_margin: float
    brier_score: float
    ece: float
    correct_count: int
    incorrect_count: int
    per_letter_recall: Dict[str, float] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)
    confusion_matrix: Optional[np.ndarray] = None
    sample_scores: Optional[np.ndarray] = None  # (N_test, n_classes)
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    confidences: Optional[np.ndarray] = None
    margins: Optional[np.ndarray] = None


def compute_ece(confidences: np.ndarray, correct_mask: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins.

    Formulation:
        ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|

    ECE measures whether a model's predicted confidence corresponds to its empirical accuracy
    (e.g., if predictions with ~80% confidence are actually correct ~80% of the time).
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct_mask, dtype=bool)
    n_samples = len(conf)
    if n_samples == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(conf, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(corr[mask])
            bin_conf = np.mean(conf[mask])
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(probabilities: np.ndarray, y_true_indices: np.ndarray) -> float:
    """Compute multi-class Brier score.

    Formulation:
        BS = (1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{i, c} - y_{i, c}^*)^2
    where y_{i, c}^* is 1 if class c is true, 0 otherwise.
    Lower Brier scores indicate superior probabilistic accuracy.
    """
    P = np.asarray(probabilities, dtype=np.float64)
    N, C = P.shape
    if N == 0:
        return 0.0

    one_hot = np.zeros((N, C), dtype=np.float64)
    one_hot[np.arange(N), y_true_indices] = 1.0

    return float(np.mean(np.sum((P - one_hot) ** 2, axis=1)))


def evaluate_single_metric(
    metric_name: str,
    X_train: np.ndarray,
    y_train: Sequence[str],
    src_train: Sequence[int],
    X_test: np.ndarray,
    y_test: Sequence[str],
    is_synthetic: bool = False,
    k: int = 3,
    temperature: float = 0.12,
    cov_reg: float = 1e-3,
    var_floor: float = 1e-6,
) -> EvaluationResult:
    """Train and evaluate DistanceTemplateMatcher on a fixed split for one metric.

    Strict Data Leakage Protection:
        Variance and covariance statistics are fitted purely on X_train.
        X_test is used strictly for evaluation.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    # Initialize matcher with specified metric
    matcher = DistanceTemplateMatcher(
        metric=metric_name,
        k=k,
        temperature=temperature,
        cov_reg=cov_reg,
        var_floor=var_floor,
    )
    matcher.add(X_train, y_train, src_train)

    # Force statistical preparation on training data
    matcher._prepare()

    classes = matcher.classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test_arr = np.asarray(y_test, dtype="<U8")
    y_test_idx = np.array([class_to_idx[c] for c in y_test_arr], dtype=int)
    N_test = len(y_test_arr)

    # Benchmark inference latency
    t0 = time.perf_counter()
    scores = matcher.class_scores(X_test)  # (N_test, n_classes)
    t1 = time.perf_counter()
    avg_latency_ms = ((t1 - t0) / max(N_test, 1)) * 1000.0

    # Probabilities via softmax on negative scores
    shifted = -(scores - np.min(scores, axis=1, keepdims=True)) / max(temperature, 1e-6)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.maximum(np.sum(exp_shifted, axis=1, keepdims=True), 1e-12)

    # Predictions and confidence
    pred_indices = np.argmin(scores, axis=1)
    y_pred = np.array([classes[i] for i in pred_indices], dtype="<U8")
    confidences = np.max(probs, axis=1)

    # Margins: difference between top-1 and top-2 probabilities
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    if sorted_probs.shape[1] > 1:
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    else:
        margins = np.ones(N_test, dtype=np.float32)

    # Classification accuracy and counts
    correct_mask = y_pred == y_test_arr
    correct_count = int(np.sum(correct_mask))
    incorrect_count = N_test - correct_count
    accuracy = float(correct_count / max(N_test, 1))

    # Precision, Recall, Macro F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_arr, y_pred, labels=classes, average="macro", zero_division=0
    )

    # Per-letter recall
    cm = confusion_matrix(y_test_arr, y_pred, labels=classes)
    per_letter_recall = {}
    for i, c in enumerate(classes):
        row_sum = np.sum(cm[i, :])
        per_letter_recall[c] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0

    # Calibration metrics
    brier = compute_brier_score(probs, y_test_idx)
    ece = compute_ece(confidences, correct_mask)

    return EvaluationResult(
        metric_name=metric_name,
        is_synthetic=is_synthetic,
        num_train_samples=len(X_train),
        num_test_samples=N_test,
        accuracy=accuracy,
        macro_f1=float(f1),
        macro_precision=float(prec),
        macro_recall=float(rec),
        avg_latency_ms=avg_latency_ms,
        avg_margin=float(np.mean(margins)),
        brier_score=brier,
        ece=ece,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        per_letter_recall=per_letter_recall,
        classes=classes,
        confusion_matrix=cm,
        sample_scores=scores,
        y_true=y_test_arr,
        y_pred=y_pred,
        confidences=confidences,
        margins=margins,
    )


def plot_confusion_matrix(cm: np.ndarray, classes: Sequence[str], title: str, output_path: Path) -> None:
    """Save high-resolution confusion matrix plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted Letter", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Letter", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, f"{val}",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def plot_metrics_comparison(results: List[EvaluationResult], output_path: Path) -> None:
    """Save comparative bar charts for Accuracy, Macro F1, Margin, and Latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.metric_name.capitalize() for r in results]
    accs = [r.accuracy * 100.0 for r in results]
    f1s = [r.macro_f1 * 100.0 for r in results]
    margins = [r.avg_margin for r in results]
    latencies = [r.avg_latency_ms for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    palette = ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6"]

    # 1. Accuracy
    axes[0, 0].bar(names, accs, color=palette, width=0.55)
    axes[0, 0].set_ylabel("Accuracy (%)", fontweight="bold")
    axes[0, 0].set_title("Overall Accuracy", fontweight="bold")
    axes[0, 0].set_ylim(0, 105)
    for i, v in enumerate(accs):
        axes[0, 0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # 2. Macro F1
    axes[0, 1].bar(names, f1s, color=palette, width=0.55)
    axes[0, 1].set_ylabel("Macro F1 (%)", fontweight="bold")
    axes[0, 1].set_title("Macro F1-Score", fontweight="bold")
    axes[0, 1].set_ylim(0, 105)
    for i, v in enumerate(f1s):
        axes[0, 1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # 3. Top-2 Score Margin
    axes[1, 0].bar(names, margins, color=palette, width=0.55)
    axes[1, 0].set_ylabel("Margin (0 to 1)", fontweight="bold")
    axes[1, 0].set_title("Top-2 Class Probability Margin", fontweight="bold")
    axes[1, 0].set_ylim(0, 1.1)
    for i, v in enumerate(margins):
        axes[1, 0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    # 4. Latency
    axes[1, 1].bar(names, latencies, color=palette, width=0.55)
    axes[1, 1].set_ylabel("Latency (ms / query)", fontweight="bold")
    axes[1, 1].set_title("Inference Latency", fontweight="bold")
    max_lat = max(latencies) if latencies else 1.0
    axes[1, 1].set_ylim(0, max_lat * 1.25)
    for i, v in enumerate(latencies):
        axes[1, 1].text(i, v + (max_lat * 0.02), f"{v:.3f} ms", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Distance Metrics Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
