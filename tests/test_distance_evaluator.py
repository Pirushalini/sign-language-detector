"""Unit tests for the distance metrics evaluation engine and experiment runner.

Tests:
  - Metric computation correctness (Accuracy, F1, ECE, Brier score, Margin).
  - Perfect and worst-case calibration edge cases.
  - EvaluationResult structure and confusion matrix consistency.
  - Identical train/test conditions across all 4 distance metrics.
  - CSV export, confusion matrix plotting, and report generation in run_experiments.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.distance_metrics.evaluator import (
    EvaluationResult,
    compute_brier_score,
    compute_ece,
    evaluate_single_metric,
)
from experiments.distance_metrics.run_experiments import run_all_evaluations, stratified_split

RNG = np.random.default_rng(123)


def make_mini_dataset(n_classes: int = 4, samples_per_class: int = 15):
    """Generate minimal synthetic dataset with 132-dim features for testing."""
    classes = [chr(ord("A") + i) for i in range(n_classes)]
    X, y = [], []
    for c in classes:
        proto = RNG.normal(size=132).astype(np.float32)
        for _ in range(samples_per_class):
            X.append(proto + RNG.normal(scale=0.08, size=132).astype(np.float32))
            y.append(c)
    return np.stack(X), np.array(y), np.zeros(len(y), dtype=np.int8)


# ------------------------------------------------------------ 1. Calibration math
def test_compute_ece_perfect_and_worst():
    # Perfect calibration: 100% confident and 100% correct
    conf_perfect = np.ones(20)
    corr_perfect = np.ones(20, dtype=bool)
    assert np.isclose(compute_ece(conf_perfect, corr_perfect), 0.0)

    # Worst calibration: 100% confident and 0% correct
    corr_wrong = np.zeros(20, dtype=bool)
    assert np.isclose(compute_ece(conf_perfect, corr_wrong), 1.0)


def test_compute_brier_score_perfect():
    probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    y_true = np.array([0, 1, 2])
    assert np.isclose(compute_brier_score(probs, y_true), 0.0)


# ------------------------------------------------------------ 2. Single metric evaluation
@pytest.mark.parametrize("metric", ["euclidean", "cosine", "seuclidean", "mahalanobis"])
def test_evaluate_single_metric_structure(metric):
    X, y, src = make_mini_dataset(n_classes=3, samples_per_class=12)
    tr_idx, te_idx = stratified_split(y, test_fraction=0.33, seed=42)

    res = evaluate_single_metric(
        metric_name=metric,
        X_train=X[tr_idx],
        y_train=y[tr_idx],
        src_train=src[tr_idx],
        X_test=X[te_idx],
        y_test=y[te_idx],
        k=2,
    )

    assert isinstance(res, EvaluationResult)
    assert res.metric_name == metric
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.macro_f1 <= 1.0
    assert 0.0 <= res.macro_precision <= 1.0
    assert 0.0 <= res.macro_recall <= 1.0
    assert res.avg_latency_ms > 0.0
    assert 0.0 <= res.avg_margin <= 1.0
    assert 0.0 <= res.brier_score <= 2.0
    assert 0.0 <= res.ece <= 1.0
    assert res.correct_count + res.incorrect_count == len(te_idx)
    assert res.confusion_matrix.shape == (3, 3)
    assert np.sum(res.confusion_matrix) == len(te_idx)
    assert len(res.per_letter_recall) == 3


# ------------------------------------------------------------ 3. Split parity across all 4 metrics
def test_all_metrics_use_identical_test_split():
    X, y, src = make_mini_dataset(n_classes=3, samples_per_class=10)
    tr_idx, te_idx = stratified_split(y, test_fraction=0.3, seed=99)

    results = []
    for m in ["euclidean", "cosine", "seuclidean", "mahalanobis"]:
        r = evaluate_single_metric(
            metric_name=m,
            X_train=X[tr_idx],
            y_train=y[tr_idx],
            src_train=src[tr_idx],
            X_test=X[te_idx],
            y_test=y[te_idx],
            k=2,
        )
        results.append(r)

    # Verify identical sample counts and identical ground-truth targets
    for r in results[1:]:
        assert r.num_train_samples == results[0].num_train_samples
        assert r.num_test_samples == results[0].num_test_samples
        assert np.array_equal(r.y_true, results[0].y_true)


# ------------------------------------------------------------ 4. End-to-end runner test
def test_run_all_evaluations_pipeline(tmp_path):
    """Test full evaluation runner execution and artifact generation."""
    out_dir = tmp_path / "eval_results"
    results = run_all_evaluations(out_dir=out_dir, force_synthetic=True, seed=42)

    assert len(results) == 4
    assert (out_dir / "metrics_comparison.csv").exists()
    assert (out_dir / "per_letter_recall.csv").exists()
    assert (out_dir / "prediction_scores.csv").exists()
    assert (out_dir / "comparison_barchart.png").exists()
    assert (out_dir / "DISTANCE_METRICS_REPORT.md").exists()

    for m in ["euclidean", "cosine", "seuclidean", "mahalanobis"]:
        assert (out_dir / f"confusion_matrix_{m}.png").exists()

    # Verify report contains synthetic disclaimer
    report_text = (out_dir / "DISTANCE_METRICS_REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC PIPELINE SMOKE-TEST ONLY" in report_text
    assert "Euclidean" in report_text
    assert "Mahalanobis" in report_text
