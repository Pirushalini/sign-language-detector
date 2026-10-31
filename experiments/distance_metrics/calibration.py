"""Metric-specific confidence calibration via Temperature Scaling.

Features:
  - Optimizes temperature parameter tau separately for each distance metric.
  - Fits tau strictly on a held-out validation set (10%) to prevent data leakage.
  - Compares before-and-after calibration metrics on an unseen test set (20%):
      - Expected Calibration Error (ECE)
      - Brier Score
      - Mean Confidence
      - Top-Two Probability Margin
      - Accuracy (preserved invariant)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .evaluator import compute_brier_score, compute_ece
from .matcher import DistanceTemplateMatcher


@dataclass
class CalibrationComparison:
    """Comparison metrics before and after temperature scaling on test data."""

    metric_name: str
    optimal_temperature: float
    baseline_temperature: float
    accuracy: float
    # Before calibration (baseline tau)
    ece_before: float
    brier_before: float
    mean_conf_before: float
    mean_margin_before: float
    # After calibration (optimal tau)
    ece_after: float
    brier_after: float
    mean_conf_after: float
    mean_margin_after: float


def compute_softmax_probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Compute normalized softmax probabilities from raw class distance scores."""
    temp = max(float(temperature), 1e-5)
    # Negative distance scaling
    shifted = -(scores - np.min(scores, axis=1, keepdims=True)) / temp
    # Exponentiate with numerical safety
    exp_shifted = np.exp(np.clip(shifted, -50.0, 50.0))
    probs = exp_shifted / np.maximum(np.sum(exp_shifted, axis=1, keepdims=True), 1e-12)
    return probs


def optimize_temperature(
    val_scores: np.ndarray,
    val_y_idx: np.ndarray,
    search_range: Tuple[float, float] = (1e-4, 5.0),
) -> float:
    """Find the optimal temperature tau that minimizes Negative Log-Likelihood (NLL) on validation data.

    Formulation:
        tau* = argmin_{tau} - (1/N_val) * sum_i log( p_i(y_i; tau) )

    Strict Data Leakage Protection:
        This function must be called ONLY with validation data (val_scores, val_y_idx).
        Test data must NEVER be passed to this function.
    """
    scores = np.asarray(val_scores, dtype=np.float64)
    y_idx = np.asarray(val_y_idx, dtype=int)
    n_samples = len(y_idx)

    def nll_objective(t: float) -> float:
        if t <= 1e-5:
            return 1e9
        probs = compute_softmax_probabilities(scores, t)
        p_true = probs[np.arange(n_samples), y_idx]
        loss = -np.mean(np.log(np.maximum(p_true, 1e-12)))
        return float(loss)

    res = minimize_scalar(nll_objective, bounds=search_range, method="bounded")
    return float(res.x)


def evaluate_calibration(
    metric_name: str,
    matcher: DistanceTemplateMatcher,
    X_val: np.ndarray,
    y_val: Sequence[str],
    X_test: np.ndarray,
    y_test: Sequence[str],
    baseline_temp: float = 0.12,
) -> CalibrationComparison:
    """Perform metric-specific calibration on validation fold and evaluate on test fold."""
    classes = matcher.classes
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_val_arr = np.asarray(y_val, dtype="<U8")
    y_val_idx = np.array([class_to_idx[c] for c in y_val_arr], dtype=int)

    y_test_arr = np.asarray(y_test, dtype="<U8")
    y_test_idx = np.array([class_to_idx[c] for c in y_test_arr], dtype=int)

    # 1. Validation phase: compute validation scores and optimize temperature
    val_scores = matcher.class_scores(X_val)
    optimal_temp = optimize_temperature(val_scores, y_val_idx)

    # 2. Test phase: compute test scores (evaluated at baseline and optimal temperature)
    test_scores = matcher.class_scores(X_test)
    test_pred_idx = np.argmin(test_scores, axis=1)
    y_pred = np.array([classes[i] for i in test_pred_idx], dtype="<U8")
    correct_mask = y_pred == y_test_arr
    accuracy = float(np.mean(correct_mask))

    # --- Baseline Evaluation (tau = baseline_temp) ---
    probs_base = compute_softmax_probabilities(test_scores, baseline_temp)
    conf_base = np.max(probs_base, axis=1)
    sorted_probs_base = np.sort(probs_base, axis=1)[:, ::-1]
    margins_base = (
        sorted_probs_base[:, 0] - sorted_probs_base[:, 1]
        if sorted_probs_base.shape[1] > 1
        else np.ones(len(X_test))
    )
    ece_base = compute_ece(conf_base, correct_mask)
    brier_base = compute_brier_score(probs_base, y_test_idx)

    # --- Calibrated Evaluation (tau = optimal_temp) ---
    probs_cal = compute_softmax_probabilities(test_scores, optimal_temp)
    conf_cal = np.max(probs_cal, axis=1)
    sorted_probs_cal = np.sort(probs_cal, axis=1)[:, ::-1]
    margins_cal = (
        sorted_probs_cal[:, 0] - sorted_probs_cal[:, 1]
        if sorted_probs_cal.shape[1] > 1
        else np.ones(len(X_test))
    )
    ece_cal = compute_ece(conf_cal, correct_mask)
    brier_cal = compute_brier_score(probs_cal, y_test_idx)

    return CalibrationComparison(
        metric_name=metric_name,
        optimal_temperature=optimal_temp,
        baseline_temperature=baseline_temp,
        accuracy=accuracy,
        ece_before=ece_base,
        brier_before=brier_base,
        mean_conf_before=float(np.mean(conf_base)),
        mean_margin_before=float(np.mean(margins_base)),
        ece_after=ece_cal,
        brier_after=brier_cal,
        mean_conf_after=float(np.mean(conf_cal)),
        mean_margin_after=float(np.mean(margins_cal)),
    )
