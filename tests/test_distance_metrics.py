"""Unit tests for the four alternative distance metrics.

Tests:
  - Non-negativity and correct shapes.
  - Identity of indiscernibles (d(x, x) == 0) and symmetry (d(x, y) == d(y, x)).
  - Zero-vector handling for cosine distance (avoiding 0/0 NaN).
  - Variance floor for standardized Euclidean distance (handling zero-variance features).
  - Covariance regularization for Mahalanobis distance (handling singular / N < D matrices).
  - Equivalence checks (SEuclidean with unit variance == Euclidean; Mahalanobis with I == Euclidean).
  - Data leakage check (variance and covariance fitted strictly on training data).
  - End-to-end classification through DistanceTemplateMatcher for all 4 metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.distance_metrics.metrics import (
    cosine_distance,
    euclidean_distance,
    fit_covariance,
    fit_feature_variance,
    mahalanobis_distance,
    standardized_euclidean_distance,
)
from experiments.distance_metrics.matcher import DistanceTemplateMatcher
from matcher import SRC_EXTERNAL, SRC_USER

RNG = np.random.default_rng(42)


# ------------------------------------------------------------ 1. Basic metric properties
@pytest.mark.parametrize("metric_fn", [euclidean_distance, cosine_distance])
def test_identity_and_symmetry(metric_fn):
    x = RNG.normal(size=(5, 132)).astype(np.float32)
    y = RNG.normal(size=(5, 132)).astype(np.float32)

    # Identity of indiscernibles: d(x, x) == 0
    d_self = metric_fn(x, x)
    assert np.allclose(np.diag(d_self), 0.0, atol=1e-5)

    # Symmetry: d(x, y) == d(y, x).T
    d_xy = metric_fn(x, y)
    d_yx = metric_fn(y, x)
    assert np.allclose(d_xy, d_yx.T, atol=1e-5)
    assert np.all(d_xy >= -1e-6)


def test_standardized_euclidean_identity_and_symmetry():
    X_train = RNG.normal(size=(50, 132)).astype(np.float32)
    var = fit_feature_variance(X_train)
    x = X_train[:5]
    y = X_train[5:10]

    d_self = standardized_euclidean_distance(x, x, variance=var)
    assert np.allclose(np.diag(d_self), 0.0, atol=1e-5)

    d_xy = standardized_euclidean_distance(x, y, variance=var)
    d_yx = standardized_euclidean_distance(y, x, variance=var)
    assert np.allclose(d_xy, d_yx.T, atol=1e-5)
    assert np.all(d_xy >= 0.0)


def test_mahalanobis_identity_and_symmetry():
    X_train = RNG.normal(size=(50, 132)).astype(np.float32)
    _, inv_cov, W = fit_covariance(X_train)
    x = X_train[:5]
    y = X_train[5:10]

    d_self = mahalanobis_distance(x, x, whitening_matrix=W)
    assert np.allclose(np.diag(d_self), 0.0, atol=1e-4)

    d_xy = mahalanobis_distance(x, y, whitening_matrix=W)
    d_yx = mahalanobis_distance(y, x, whitening_matrix=W)
    assert np.allclose(d_xy, d_yx.T, atol=1e-4)
    assert np.all(d_xy >= 0.0)


# ------------------------------------------------------------ 2. Numerical safeguards
def test_cosine_zero_vector_handling():
    """Verify that zero vectors do not cause division by zero or NaN in cosine distance."""
    zero_vec = np.zeros((1, 132), dtype=np.float32)
    non_zero = RNG.normal(size=(1, 132)).astype(np.float32)

    # 1. Zero vs Zero -> distance must be 0.0
    d_zz = cosine_distance(zero_vec, zero_vec)
    assert not np.isnan(d_zz).any()
    assert np.isclose(d_zz[0, 0], 0.0)

    # 2. Zero vs Non-zero -> distance must be 1.0 (orthogonal)
    d_zn = cosine_distance(zero_vec, non_zero)
    assert not np.isnan(d_zn).any()
    assert np.isclose(d_zn[0, 0], 1.0)

    d_nz = cosine_distance(non_zero, zero_vec)
    assert not np.isnan(d_nz).any()
    assert np.isclose(d_nz[0, 0], 1.0)

    # 3. Parallel vectors -> distance must be 0.0
    d_par = cosine_distance(non_zero, 3.5 * non_zero)
    assert np.isclose(d_par[0, 0], 0.0, atol=1e-5)

    # 4. Opposite vectors -> distance must be 2.0
    d_opp = cosine_distance(non_zero, -1.0 * non_zero)
    assert np.isclose(d_opp[0, 0], 2.0, atol=1e-5)


def test_standardized_euclidean_variance_floor():
    """Verify that constant features (variance == 0) do not cause division by zero or NaN."""
    X_train = RNG.normal(size=(40, 132)).astype(np.float32)
    # Simulate wrist landmark permanently centered at 0.0 (zero variance)
    X_train[:, :3] = 0.0

    var = fit_feature_variance(X_train, var_floor=1e-6)
    assert np.all(var[:3] >= 1e-6)
    assert not np.isnan(var).any()

    # Query with zero-variance features
    q = np.zeros((2, 132), dtype=np.float32)
    d = standardized_euclidean_distance(q, X_train[:5], variance=var)
    assert not np.isnan(d).any()
    assert not np.isinf(d).any()
    assert np.all(d >= 0.0)


def test_mahalanobis_singular_covariance_regularization():
    """Verify that singular covariance (N < D and collinear features) is properly regularized."""
    # Only 10 samples for 132 dimensions -> heavily singular matrix (rank <= 9)
    X_small = RNG.normal(size=(10, 132)).astype(np.float32)
    # Inject exact collinearity (feature 1 == feature 2)
    X_small[:, 1] = X_small[:, 0]
    # Inject exact constant zero feature
    X_small[:, 2] = 0.0

    cov_reg, inv_cov, W = fit_covariance(X_small, reg_factor=1e-3)

    assert not np.isnan(cov_reg).any()
    assert not np.isnan(inv_cov).any()
    assert not np.isnan(W).any()

    # Regularized covariance must be strictly positive definite
    eigvals = np.linalg.eigvalsh(cov_reg)
    assert np.all(eigvals > 0.0)

    # Compute distances on rank-deficient data
    q = RNG.normal(size=(3, 132)).astype(np.float32)
    d = mahalanobis_distance(q, X_small, whitening_matrix=W)
    assert not np.isnan(d).any()
    assert not np.isinf(d).any()
    assert d.shape == (3, 10)
    assert np.all(d >= 0.0)


# ------------------------------------------------------------ 3. Theoretical equivalences
def test_seuclidean_unit_variance_equals_euclidean():
    """When variance is identically 1.0 for all dimensions, SEuclidean must equal Euclidean."""
    q = RNG.normal(size=(4, 132)).astype(np.float32)
    x = RNG.normal(size=(7, 132)).astype(np.float32)
    unit_var = np.ones(132, dtype=np.float32)

    d_se = standardized_euclidean_distance(q, x, variance=unit_var)
    d_e = euclidean_distance(q, x)
    assert np.allclose(d_se, d_e, atol=1e-5)


def test_mahalanobis_identity_covariance_equals_euclidean():
    """When covariance is identically the identity matrix, Mahalanobis must equal Euclidean."""
    q = RNG.normal(size=(4, 132)).astype(np.float32)
    x = RNG.normal(size=(7, 132)).astype(np.float32)
    eye = np.eye(132, dtype=np.float32)

    d_m = mahalanobis_distance(q, x, whitening_matrix=eye)
    d_e = euclidean_distance(q, x)
    assert np.allclose(d_m, d_e, atol=1e-5)


# ------------------------------------------------------------ 4. Data leakage check
def test_no_data_leakage_in_matcher():
    """Verify variance and covariance are computed purely from training templates, not queries."""
    matcher = DistanceTemplateMatcher(metric="seuclidean")
    X_train = RNG.normal(loc=5.0, scale=2.0, size=(30, 132)).astype(np.float32)
    y_train = ["A"] * 30
    matcher.add(X_train, y_train, [SRC_EXTERNAL] * 30)

    # Force prepare
    matcher._prepare()
    fitted_var = matcher._variance.copy()

    # Query with radically different distribution (loc=100.0, scale=50.0)
    X_test = RNG.normal(loc=100.0, scale=50.0, size=(10, 132)).astype(np.float32)
    _ = matcher.class_scores(X_test)

    # Fitted variance must remain completely unchanged
    assert np.array_equal(matcher._variance, fitted_var)


# ------------------------------------------------------------ 5. End-to-end DistanceTemplateMatcher
@pytest.mark.parametrize("metric", ["euclidean", "cosine", "seuclidean", "mahalanobis"])
def test_distance_matcher_end_to_end(metric):
    """Verify that DistanceTemplateMatcher successfully classifies queries for all 4 metrics."""
    matcher = DistanceTemplateMatcher(metric=metric, k=3, reject_distance=3.0)

    # Generate synthetic letter prototypes for A, B, C
    n_features = 132
    protos = {
        "A": RNG.normal(loc=0.0, scale=1.0, size=n_features).astype(np.float32),
        "B": RNG.normal(loc=3.0, scale=1.0, size=n_features).astype(np.float32),
        "C": RNG.normal(loc=-3.0, scale=1.0, size=n_features).astype(np.float32),
    }

    X_train, y_train = [], []
    for letter, proto in protos.items():
        for _ in range(25):
            noise = RNG.normal(scale=0.05, size=n_features).astype(np.float32)
            X_train.append(proto + noise)
            y_train.append(letter)

    matcher.add(np.stack(X_train), y_train, [SRC_EXTERNAL] * len(y_train))
    assert matcher.available
    assert matcher.classes == ["A", "B", "C"]

    # Test single prediction
    test_sample = protos["B"] + RNG.normal(scale=0.05, size=n_features).astype(np.float32)
    pred = matcher.predict(test_sample)
    assert pred.label == "B"
    assert pred.confidence > 0.3
    assert len(pred.top3) == 3

    # Test batch prediction
    batch_queries = [protos["A"] + 0.02, protos["B"] + 0.02, protos["C"] + 0.02]
    labels, distances = matcher.predict_labels(np.stack(batch_queries))
    assert list(labels) == ["A", "B", "C"]
    assert len(distances) == 3


def test_distance_matcher_user_weight():
    """Verify user template weighting behaves identically across distance metrics."""
    p = RNG.normal(size=132).astype(np.float32)
    m = DistanceTemplateMatcher(metric="euclidean", user_weight=0.5)
    # Add identical points with different source
    m.add(np.stack([p, p]), ["A", "B"], [SRC_EXTERNAL, SRC_USER])
    query = p + 0.05
    # Since user_weight is 0.5, user template 'B' distance is halved and wins
    assert m.predict(query).label == "B"
