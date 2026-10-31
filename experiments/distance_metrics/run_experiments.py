"""Experiment runner for comparing the four alternative distance metrics.

Runs Euclidean, Cosine, Standardized Euclidean, and Mahalanobis metrics
under identical train/test conditions and outputs structured results:
  - metrics_comparison.csv
  - per_letter_recall.csv
  - prediction_scores.csv
  - confusion_matrix_<metric>.png
  - comparison_barchart.png
  - DISTANCE_METRICS_REPORT.md

Usage:
    python -m experiments.distance_metrics.run_experiments
    python -m experiments.distance_metrics.run_experiments --synthetic
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from matcher import SRC_EXTERNAL, SRC_USER, load_templates
from utils import EXTERNAL_TEMPLATES, LETTERS, USER_TEMPLATES

from .evaluator import (
    EvaluationResult,
    evaluate_single_metric,
    plot_confusion_matrix,
    plot_metrics_comparison,
)

METRICS = ["euclidean", "cosine", "seuclidean", "mahalanobis"]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def stratified_split(y: np.ndarray, test_fraction: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/test split ensuring identical class proportions across splits."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_fraction))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def load_evaluation_data(force_synthetic: bool = False, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Load dataset for evaluation: either real templates or synthetic fallback.

    Returns:
        (X, y, src, is_synthetic)
    """
    has_real = EXTERNAL_TEMPLATES.exists() or USER_TEMPLATES.exists()

    if has_real and not force_synthetic:
        Xe, ye = np.zeros((0, 132), np.float32), np.array([], dtype="<U8")
        Xu, yu = np.zeros((0, 132), np.float32), np.array([], dtype="<U8")
        if EXTERNAL_TEMPLATES.exists():
            Xe, ye, _, _ = load_templates(EXTERNAL_TEMPLATES)
        if USER_TEMPLATES.exists():
            Xu, yu, _, _ = load_templates(USER_TEMPLATES)
        X = np.vstack([Xe, Xu])
        y = np.concatenate([ye, yu])
        src = np.concatenate([np.full(len(Xe), SRC_EXTERNAL, np.int8), np.full(len(Xu), SRC_USER, np.int8)])
        print(f"Loaded REAL templates: {len(X)} samples across {len(np.unique(y))} classes.")
        return X, y, src, False

    print("\n" + "=" * 78)
    print("  [NOTICE] Real template files (data/templates_external.npz) not found.")
    print("  Generating synthetic kinematic dataset for pipeline verification ONLY.")
    print("  Synthetic results verify the mathematical pipeline but DO NOT reflect")
    print("  real-world signing accuracy.")
    print("=" * 78 + "\n")

    from experiments.synthetic import make_letter_dataset
    # Generate 50 external + 25 user synthetic samples per letter (all 26 letters)
    Xe, ye, Xu, yu = make_letter_dataset(per_letter_ext=50, per_letter_user=25, seed=seed)
    X = np.vstack([Xe, Xu])
    y = np.concatenate([ye, yu])
    src = np.concatenate([np.full(len(Xe), SRC_EXTERNAL, np.int8), np.full(len(Xu), SRC_USER, np.int8)])
    return X, y, src, True


def run_all_evaluations(
    out_dir: Path = DEFAULT_RESULTS_DIR,
    force_synthetic: bool = False,
    test_split: float = 0.2,
    seed: int = 42,
) -> List[EvaluationResult]:
    """Run full evaluation suite across all 4 distance metrics on identical data splits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, src, is_synthetic = load_evaluation_data(force_synthetic=force_synthetic, seed=seed)

    train_idx, test_idx = stratified_split(y, test_fraction=test_split, seed=seed)
    X_train, y_train, src_train = X[train_idx], y[train_idx], src[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Dataset split: {len(train_idx)} train samples, {len(test_idx)} test samples.")
    print(f"Testing {len(METRICS)} distance metrics under identical evaluation splits...\n")

    results: List[EvaluationResult] = []
    for m in METRICS:
        print(f"  Evaluating: {m.upper()}...", end="", flush=True)
        res = evaluate_single_metric(
            metric_name=m,
            X_train=X_train,
            y_train=y_train,
            src_train=src_train,
            X_test=X_test,
            y_test=y_test,
            is_synthetic=is_synthetic,
            k=3,
        )
        results.append(res)
        print(f" Done. (Acc: {res.accuracy * 100:.2f}%, F1: {res.macro_f1 * 100:.2f}%, Latency: {res.avg_latency_ms:.3f} ms)")

    # 1. Save metrics_comparison.csv
    csv_path = out_dir / "metrics_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Metric", "Is_Synthetic", "Train_Samples", "Test_Samples",
            "Accuracy", "Macro_F1", "Macro_Precision", "Macro_Recall",
            "Avg_Latency_ms", "Avg_Margin", "Brier_Score", "ECE",
            "Correct_Count", "Incorrect_Count"
        ])
        for r in results:
            writer.writerow([
                r.metric_name, r.is_synthetic, r.num_train_samples, r.num_test_samples,
                f"{r.accuracy:.4f}", f"{r.macro_f1:.4f}", f"{r.macro_precision:.4f}", f"{r.macro_recall:.4f}",
                f"{r.avg_latency_ms:.4f}", f"{r.avg_margin:.4f}", f"{r.brier_score:.4f}", f"{r.ece:.4f}",
                r.correct_count, r.incorrect_count
            ])
    print(f"\nSaved metrics comparison to {csv_path.name}")

    # 2. Save per_letter_recall.csv
    recall_path = out_dir / "per_letter_recall.csv"
    all_classes = sorted(set(y_test))
    with open(recall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Letter"] + [r.metric_name for r in results])
        for c in all_classes:
            row = [c] + [f"{r.per_letter_recall.get(c, 0.0):.4f}" for r in results]
            writer.writerow(row)
    print(f"Saved per-letter recall to {recall_path.name}")

    # 3. Save prediction_scores.csv (detailed query-level predictions)
    scores_path = out_dir / "prediction_scores.csv"
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Query_Index", "True_Letter"]
        for r in results:
            m = r.metric_name
            header.extend([f"{m}_pred", f"{m}_conf", f"{m}_margin"])
        writer.writerow(header)
        for i in range(len(X_test)):
            row = [i, y_test[i]]
            for r in results:
                row.extend([
                    r.y_pred[i],
                    f"{r.confidences[i]:.4f}",
                    f"{r.margins[i]:.4f}"
                ])
            writer.writerow(row)
    print(f"Saved query predictions to {scores_path.name}")

    # 4. Generate confusion matrix plots
    for r in results:
        cm_path = out_dir / f"confusion_matrix_{r.metric_name}.png"
        status_label = "(Synthetic Pipeline Smoke-Test)" if r.is_synthetic else "(Real Dataset)"
        title = f"Confusion Matrix: {r.metric_name.capitalize()} {status_label}\nAcc: {r.accuracy * 100:.1f}%, F1: {r.macro_f1 * 100:.1f}%"
        plot_confusion_matrix(r.confusion_matrix, r.classes, title, cm_path)
    print("Saved confusion matrix plots.")

    # 5. Generate comparison bar chart
    bar_path = out_dir / "comparison_barchart.png"
    plot_metrics_comparison(results, bar_path)
    print(f"Saved summary comparison chart to {bar_path.name}")

    # 6. Generate detailed DISTANCE_METRICS_REPORT.md
    generate_markdown_report(results, out_dir / "DISTANCE_METRICS_REPORT.md")
    print(f"Generated comprehensive report at DISTANCE_METRICS_REPORT.md")

    return results


def generate_markdown_report(results: List[EvaluationResult], output_path: Path) -> None:
    """Write comprehensive markdown evaluation report."""
    is_synth = results[0].is_synthetic if results else False
    train_n = results[0].num_train_samples if results else 0
    test_n = results[0].num_test_samples if results else 0

    best_acc_res = max(results, key=lambda r: r.accuracy)
    best_f1_res = max(results, key=lambda r: r.macro_f1)
    fastest_res = min(results, key=lambda r: r.avg_latency_ms)

    md = []
    md.append("# Task 2: Comparative Analysis of Alternative Distance Metrics")
    md.append("## Sign Language Detector (Two-Handed Fingerspelling A–Z)\n")

    if is_synth:
        md.append("> [!WARNING]")
        md.append("> **SYNTHETIC PIPELINE SMOKE-TEST ONLY**")
        md.append("> Real dataset templates (`data/templates_external.npz`) were not present in the workspace.")
        md.append("> The figures below were computed using a kinematic synthetic hand model (`experiments.synthetic`)")
        md.append("> to verify the numerical stability, zero-leakage pipeline, and metrics calculation.")
        md.append("> **These numbers do not represent real-world human signing performance.**\n")
    else:
        md.append("> [!NOTE]")
        md.append("> Evaluation performed on real hand landmark template datasets under a stratified 80/20 split.\n")

    md.append("### 1. Executive Summary\n")
    md.append(f"- **Evaluated Metrics**: Euclidean, Cosine, Standardized Euclidean, Mahalanobis.")
    md.append(f"- **Experimental Dataset**: {train_n} training samples, {test_n} test queries.")
    md.append(f"- **Highest Accuracy**: **{best_acc_res.metric_name.capitalize()}** ({best_acc_res.accuracy * 100:.2f}%)")
    md.append(f"- **Highest Macro F1**: **{best_f1_res.metric_name.capitalize()}** ({best_f1_res.macro_f1 * 100:.2f}%)")
    md.append(f"- **Fastest Inference**: **{fastest_res.metric_name.capitalize()}** ({fastest_res.avg_latency_ms:.3f} ms / prediction)\n")

    md.append("### 2. Comprehensive Performance Comparison\n")
    md.append("| Metric | Accuracy | Macro F1 | Macro Precision | Macro Recall | Latency (ms) | Top-2 Margin | Brier Score | ECE |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        md.append(
            f"| **{r.metric_name.capitalize()}** | {r.accuracy * 100:.2f}% | {r.macro_f1 * 100:.2f}% | "
            f"{r.macro_precision * 100:.2f}% | {r.macro_recall * 100:.2f}% | {r.avg_latency_ms:.3f} ms | "
            f"{r.avg_margin:.3f} | {r.brier_score:.4f} | {r.ece:.4f} |"
        )
    md.append("")

    md.append("### 3. Detailed Metric Formulation & Mathematical Analysis\n")
    md.append("#### 1. Euclidean Distance (Baseline)")
    md.append("- **Formula**: $d_E(\\mathbf{q}, \\mathbf{x}) = \\sqrt{\\sum_{i=1}^{132} (q_i - x_i)^2}$")
    md.append("- **Behavior**: Standard straight-line distance. Assumes isotropic, uncorrelated landmark noise across all 132 features.")
    md.append("- **Observation**: High-variance fingertips contribute heavily to the distance, which can cause confusion on letters with identical hand shapes but subtle palm/wrist rotations.\n")

    md.append("#### 2. Cosine Distance")
    md.append("- **Formula**: $d_C(\\mathbf{q}, \\mathbf{x}) = 1 - \\frac{\\mathbf{q} \\cdot \\mathbf{x}}{\\|\\mathbf{q}\\|_2 \\|\\mathbf{x}\\|_2}$")
    md.append("- **Behavior**: Measures directional angular alignment of the feature vectors, discarding magnitude.")
    md.append("- **Safeguards**: Zero vectors (e.g. absent Hand 2) are safely guarded against division by zero ($0/0$).\n")

    md.append("#### 3. Standardized Euclidean Distance")
    md.append("- **Formula**: $d_{SE}(\\mathbf{q}, \\mathbf{x}) = \\sqrt{\\sum_{i=1}^{132} \\frac{(q_i - x_i)^2}{\\sigma_i^2}}$")
    md.append("- **Behavior**: Normalizes each feature axis by its sample variance $\\sigma_i^2$ computed strictly from training templates.")
    md.append("- **Safeguards**: Enforces a variance floor $\\sigma_i^2 \\ge 10^{-6}$ for fixed coordinates (such as the centered wrist origin $(0, 0, 0)$).\n")

    md.append("#### 4. Mahalanobis Distance")
    md.append("- **Formula**: $d_M(\\mathbf{q}, \\mathbf{x}) = \\sqrt{(\\mathbf{q} - \\mathbf{x})^T \\mathbf{\\Sigma}_{\\text{reg}}^{-1} (\\mathbf{q} - \\mathbf{x})}$")
    md.append("- **Behavior**: Decorrelates anatomical landmark features (e.g., finger joints moving in unison).")
    md.append("- **Safeguards**: Employs Tikhonov ridge and shrinkage regularization $\\mathbf{\\Sigma}_{\\text{reg}} = (1 - \\alpha)\\mathbf{\\Sigma} + \\alpha \\mathbf{I} + \\lambda \\mathbf{I}$ with Cholesky whitening factorization.\n")

    md.append("### 4. Per-Letter Accuracy Highlights\n")
    md.append("Per-letter breakdown summary (stored in full in `per_letter_recall.csv`):")
    # Show weakest letters across metrics
    for r in results:
        sorted_recall = sorted(r.per_letter_recall.items(), key=lambda kv: kv[1])
        weakest = [f"{c} ({rec*100:.1f}%)" for c, rec in sorted_recall[:3]]
        md.append(f"- **{r.metric_name.capitalize()}** lowest recall letters: {', '.join(weakest)}")
    md.append("")

    md.append("### 5. Generated Artifacts in `experiments/distance_metrics/results/`\n")
    md.append("- `metrics_comparison.csv`: Summary metrics table across all 4 distance formulations.")
    md.append("- `per_letter_recall.csv`: Matrix of recall values for all letters A–Z.")
    md.append("- `prediction_scores.csv`: Per-query classification decisions, confidences, and top-2 margins.")
    md.append("- `comparison_barchart.png`: 4-panel visual comparison (Accuracy, F1, Margin, Latency).")
    md.append("- `confusion_matrix_<metric>.png`: Full $26 \\times 26$ confusion matrices for each metric.")
    md.append("")

    output_path.write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic kinematic data")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR, help="Results output directory")
    args = parser.parse_args()

    run_all_evaluations(out_dir=args.out, force_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
