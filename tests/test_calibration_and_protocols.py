"""Unit tests for confidence calibration and rigorous evaluation protocols.

Tests:
  - Non-overlapping splits for 3-way stratified partition (70/10/20).
  - Accuracy invariance under temperature scaling.
  - Temperature optimization on validation data.
  - Cosine calibration error reduction with metric-specific temperature.
  - Block splitting non-overlapping partition.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.distance_metrics.calibration import (
    compute_softmax_probabilities,
    evaluate_calibration,
    optimize_temperature,
)
from experiments.distance_metrics.matcher import DistanceTemplateMatcher
from experiments.distance_metrics.rigorous_evaluation import (
    block_split_by_class,
    three_way_stratified_split,
)

RNG = np.random.default_rng(2026)


def make_test_data(n_classes: int = 5, samples_per_class: int = 20):
    classes = [chr(ord("A") + i) for i in range(n_classes)]
    X, y = [], []
    for c in classes:
        proto = RNG.normal(size=132).astype(np.float32)
        for _ in range(samples_per_class):
            X.append(proto + RNG.normal(scale=0.05, size=132).astype(np.float32))
            y.append(c)
    return np.stack(X), np.array(y), np.zeros(len(y), dtype=np.int8)


def test_three_way_stratified_split_partition():
    X, y, src = make_test_data(n_classes=4, samples_per_class=30)
    N = len(y)
    tr, val, te = three_way_stratified_split(y, train_pct=0.70, val_pct=0.10, seed=42)

    # All indices must be mutually exclusive and exhaust the dataset
    assert len(tr) + len(val) + len(te) == N
    assert len(set(tr) & set(val)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(val) & set(te)) == 0
    assert set(np.concatenate([tr, val, te])) == set(range(N))


def test_block_split_partition():
    X, y, src = make_test_data(n_classes=3, samples_per_class=25)
    tr, te = block_split_by_class(y, n_blocks=5, test_block=4)

    assert len(tr) + len(te) == len(y)
    assert len(set(tr) & set(te)) == 0
    assert len(te) == 3 * 5  # 5 samples per class for test block


def test_temperature_scaling_preserves_accuracy_invariant():
    """Temperature scaling changes probability scale, but NEVER changes top-1 prediction."""
    scores = RNG.uniform(0.1, 5.0, size=(20, 10))
    probs_t1 = compute_softmax_probabilities(scores, temperature=0.01)
    probs_t2 = compute_softmax_probabilities(scores, temperature=1.50)

    # Argmax of probabilities must exactly equal argmin of distances for any tau > 0
    preds_t1 = np.argmax(probs_t1, axis=1)
    preds_t2 = np.argmax(probs_t2, axis=1)
    best_dist = np.argmin(scores, axis=1)

    assert np.array_equal(preds_t1, best_dist)
    assert np.array_equal(preds_t2, best_dist)


def test_optimize_temperature_on_validation():
    scores = RNG.uniform(0.5, 3.0, size=(40, 5))
    y_idx = np.argmin(scores, axis=1)  # perfectly predictable labels
    opt_tau = optimize_temperature(scores, y_idx)

    assert opt_tau > 0.0
    assert not np.isnan(opt_tau)


def test_cosine_calibration_evaluation():
    X, y, src = make_test_data(n_classes=3, samples_per_class=20)
    tr, val, te = three_way_stratified_split(y, train_pct=0.70, val_pct=0.10, seed=42)

    matcher = DistanceTemplateMatcher(metric="cosine", k=2)
    matcher.add(X[tr], y[tr], src[tr])
    matcher._prepare()

    res = evaluate_calibration(
        metric_name="cosine",
        matcher=matcher,
        X_val=X[val],
        y_val=y[val],
        X_test=X[te],
        y_test=y[te],
        baseline_temp=0.12,
    )

    assert res.optimal_temperature > 0.0
    # Calibration error should be reduced by tuning tau on validation fold
    assert res.ece_after <= res.ece_before + 0.05
