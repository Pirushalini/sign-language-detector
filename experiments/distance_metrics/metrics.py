"""Four distance metric formulations with numerical safeguards.

Features:
  1. Euclidean Distance: standard straight-line metric.
  2. Cosine Distance: angle-based metric with robust zero-vector handling.
  3. Standardized Euclidean: feature-variance-weighted metric with variance floor.
  4. Mahalanobis Distance: covariance-decorrelated metric with Tikhonov/shrinkage
     regularization and Cholesky transformation.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


# ------------------------------------------------------------ statistical estimators
def fit_feature_variance(X_train: np.ndarray, var_floor: float = 1e-6) -> np.ndarray:
    """Compute per-feature sample variance on training data with a variance floor.

    Safeguard:
        Features with zero or near-zero variance (e.g. wrist landmark permanently fixed
        at (0, 0, 0) by centering) are bounded from below by var_floor to prevent division
        by zero or NaN distances.

    Parameters:
        X_train: shape (N, D) training landmark vectors.
        var_floor: minimum variance threshold (default 1e-6).

    Returns:
        variance: shape (D,) array with variance per feature >= var_floor.
    """
    X = np.atleast_2d(np.asarray(X_train, dtype=np.float64))
    if X.shape[0] < 2:
        # Fallback to ones if single sample or empty
        return np.ones(X.shape[1], dtype=np.float32)

    var = np.var(X, axis=0, ddof=1)
    var = np.maximum(var, var_floor)
    return np.asarray(var, dtype=np.float32)


def fit_covariance(
    X_train: np.ndarray,
    reg_factor: float = 1e-3,
    shrinkage: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a regularized covariance matrix and its Cholesky whitening transform.

    Safeguards:
        In 132-D space, raw empirical covariance matrices are often ill-conditioned or
        singular due to collinear joint geometry, fixed wrist origins, or small sample
        sizes (N < D).
        We regularize using a convex blend with the identity matrix (shrinkage) plus
        a diagonal Tikhonov ridge (reg_factor * mean_variance * I):
            Sigma_reg = (1 - alpha) * Sigma + alpha * (Tr(Sigma)/D) * I + reg * I
        The whitening matrix W = L^{-T} is computed via Cholesky decomposition
        Sigma_reg = L @ L.T, allowing Mahalanobis distance to be computed as
        Euclidean distance in whitened space: ||(q - x) @ W||_2.

    Parameters:
        X_train: shape (N, D) training feature vectors.
        reg_factor: diagonal ridge loading factor relative to trace (default 1e-3).
        shrinkage: optional manual shrinkage intensity in [0, 1]. If None, Ledoit-Wolf
                   or heuristic trace shrinkage is used.

    Returns:
        (cov_reg, inv_cov, W):
            cov_reg: regularized covariance matrix, shape (D, D)
            inv_cov: inverse covariance matrix, shape (D, D)
            W: whitening transformation matrix, shape (D, D)
    """
    X = np.atleast_2d(np.asarray(X_train, dtype=np.float64))
    N, D = X.shape

    if N < 2:
        eye = np.eye(D, dtype=np.float32)
        return eye, eye, eye

    # Center training data
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    raw_cov = (X_centered.T @ X_centered) / max(N - 1, 1)

    # Base variance scale
    trace_val = float(np.trace(raw_cov))
    mean_var = max(trace_val / max(D, 1), 1e-6)

    # Determine shrinkage alpha
    if shrinkage is not None:
        alpha = float(np.clip(shrinkage, 0.0, 1.0))
    else:
        # Heuristic shrinkage: higher shrinkage when N is close to or smaller than D
        alpha = float(np.clip(D / (N + D), 0.05, 0.50))

    # Regularized covariance: blend with scaled identity and add ridge
    cov_reg = (1.0 - alpha) * raw_cov + alpha * mean_var * np.eye(D)
    ridge = max(reg_factor * mean_var, 1e-5)
    cov_reg += ridge * np.eye(D)

    # Symmetrize
    cov_reg = 0.5 * (cov_reg + cov_reg.T)

    # Cholesky factorization for stable whitening
    try:
        L = np.linalg.cholesky(cov_reg)
        # Whitening matrix: W = L^{-T} such that W.T @ cov_reg @ W = I
        L_inv = np.linalg.inv(L)
        W = L_inv.T
        inv_cov = L_inv.T @ L_inv
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse / eigh if Cholesky fails on extreme inputs
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 1e-5)
        cov_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        inv_cov = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

    return (
        np.asarray(cov_reg, dtype=np.float32),
        np.asarray(inv_cov, dtype=np.float32),
        np.asarray(W, dtype=np.float32),
    )


# ------------------------------------------------------------ distance functions
def euclidean_distance(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between query vectors and templates.

    Formulation:
        d_E(q, x) = sqrt(||q - x||_2^2) = sqrt(sum_i (q_i - x_i)^2)

    Vectorized Identity:
        ||q - x||^2 = ||q||^2 - 2(q . x^T) + ||x||^2

    Parameters:
        Q: shape (M, D) or (D,) query landmark vector(s).
        X: shape (N, D) template landmark vectors.

    Returns:
        D: shape (M, N) matrix of pairwise distances.
    """
    Q_mat = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    X_mat = np.atleast_2d(np.asarray(X, dtype=np.float64))

    if Q_mat.shape[1] != X_mat.shape[1]:
        raise ValueError(f"Feature dimension mismatch: Q has {Q_mat.shape[1]}, X has {X_mat.shape[1]}")

    q_sq = np.sum(Q_mat * Q_mat, axis=1, keepdims=True)   # (M, 1)
    x_sq = np.sum(X_mat * X_mat, axis=1, keepdims=True).T # (1, N)
    d2 = q_sq - 2.0 * (Q_mat @ X_mat.T) + x_sq
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)


def cosine_distance(Q: np.ndarray, X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Pairwise Cosine distance between query vectors and templates.

    Formulation:
        d_C(q, x) = 1.0 - (q . x) / (||q||_2 * ||x||_2)

    Numerical Safeguards:
        1. Vectors with norm < eps (e.g. empty hands where coordinates are all zero)
           are detected before division to avoid 0 / 0 NaN.
        2. If both q and x are zero vectors, distance is 0.0 (identical empty hand).
        3. If exactly one is a zero vector, distance is 1.0 (orthogonal mismatch).
        4. Distances are strictly bounded to [0.0, 2.0].

    Parameters:
        Q: shape (M, D) or (D,) query vector(s).
        X: shape (N, D) template vectors.
        eps: zero-norm threshold.

    Returns:
        D: shape (M, N) pairwise cosine distances.
    """
    Q_mat = np.atleast_2d(np.asarray(Q, dtype=np.float32))
    X_mat = np.atleast_2d(np.asarray(X, dtype=np.float32))

    if Q_mat.shape[1] != X_mat.shape[1]:
        raise ValueError(f"Feature dimension mismatch: Q has {Q_mat.shape[1]}, X has {X_mat.shape[1]}")

    q_norm = np.linalg.norm(Q_mat, axis=1, keepdims=True)  # (M, 1)
    x_norm = np.linalg.norm(X_mat, axis=1, keepdims=True)  # (N, 1)

    q_zero = (q_norm < eps).ravel()  # (M,)
    x_zero = (x_norm < eps).ravel()  # (N,)

    # Safe unit-norm normalization
    Q_unit = np.where(q_norm < eps, 0.0, Q_mat / np.maximum(q_norm, eps))
    X_unit = np.where(x_norm < eps, 0.0, X_mat / np.maximum(x_norm, eps))

    similarity = Q_unit @ X_unit.T  # (M, N)
    D = 1.0 - similarity
    D = np.clip(D, 0.0, 2.0).astype(np.float32)

    # Apply zero-vector semantics
    if np.any(q_zero) or np.any(x_zero):
        both_zero = np.outer(q_zero, x_zero)
        one_zero = np.logical_xor(q_zero[:, None], x_zero[None, :])
        D[both_zero] = 0.0
        D[one_zero] = 1.0

    return D


def standardized_euclidean_distance(
    Q: np.ndarray,
    X: np.ndarray,
    variance: np.ndarray,
    var_floor: float = 1e-6,
) -> np.ndarray:
    """Pairwise Standardized Euclidean distance (variance-normalized).

    Formulation:
        d_SE(q, x) = sqrt( sum_{k=1}^D (q_k - x_k)^2 / sigma_k^2 )

    Equivalence:
        Standard Euclidean distance computed on vectors scaled by 1 / sqrt(sigma_k^2).

    Safeguards:
        variance is element-wise clamped to var_floor to ensure numerical stability.

    Parameters:
        Q: shape (M, D) or (D,) query vector(s).
        X: shape (N, D) template vectors.
        variance: shape (D,) per-feature variance fitted strictly on training data.
        var_floor: minimum variance floor.

    Returns:
        D: shape (M, N) standardized Euclidean distances.
    """
    Q_mat = np.atleast_2d(np.asarray(Q, dtype=np.float32))
    X_mat = np.atleast_2d(np.asarray(X, dtype=np.float32))
    var = np.asarray(variance, dtype=np.float32).ravel()

    if var.shape[0] != Q_mat.shape[1]:
        raise ValueError(f"Variance vector length {var.shape[0]} does not match feature dimension {Q_mat.shape[1]}")

    var_safe = np.maximum(var, var_floor)
    scale = 1.0 / np.sqrt(var_safe)  # (D,)

    # Transform both sets to standard-deviation normalized space
    Q_scaled = Q_mat * scale[None, :]
    X_scaled = X_mat * scale[None, :]

    return euclidean_distance(Q_scaled, X_scaled)


def mahalanobis_distance(
    Q: np.ndarray,
    X: np.ndarray,
    inv_cov: Optional[np.ndarray] = None,
    whitening_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pairwise Mahalanobis distance.

    Formulation:
        d_M(q, x) = sqrt( (q - x)^T Sigma^{-1} (q - x) )

    High-Performance Formulation via Whitening Transform:
        Given Sigma = L @ L^T, let W = L^{-T}.
        Then d_M(q, x) = ||(q - x) @ W||_2 = ||q_white - x_white||_2.
        This transforms pairwise Mahalanobis calculation from O(M * N * D^2)
        into a linear projection O((M + N) * D^2) followed by fast vectorized
        Euclidean distance O(M * N * D).

    Parameters:
        Q: shape (M, D) or (D,) query vector(s).
        X: shape (N, D) template vectors.
        inv_cov: shape (D, D) inverse covariance matrix (used if whitening_matrix is None).
        whitening_matrix: shape (D, D) matrix W such that W.T @ Sigma @ W = I.

    Returns:
        D: shape (M, N) Mahalanobis distances.
    """
    Q_mat = np.atleast_2d(np.asarray(Q, dtype=np.float32))
    X_mat = np.atleast_2d(np.asarray(X, dtype=np.float32))
    D_dim = Q_mat.shape[1]

    if whitening_matrix is not None:
        W = np.asarray(whitening_matrix, dtype=np.float32)
        if W.shape != (D_dim, D_dim):
            raise ValueError(f"whitening_matrix shape {W.shape} expected ({D_dim}, {D_dim})")
        Q_white = Q_mat @ W
        X_white = X_mat @ W
        return euclidean_distance(Q_white, X_white)

    if inv_cov is not None:
        inv_C = np.asarray(inv_cov, dtype=np.float32)
        if inv_C.shape != (D_dim, D_dim):
            raise ValueError(f"inv_cov shape {inv_C.shape} expected ({D_dim}, {D_dim})")
        # Direct calculation for smaller sets or explicit inverse
        try:
            # Attempt Cholesky on inv_cov: inv_C = L_inv.T @ L_inv
            L_inv = np.linalg.cholesky(inv_C).T
            Q_white = Q_mat @ L_inv.T
            X_white = X_mat @ L_inv.T
            return euclidean_distance(Q_white, X_white)
        except np.linalg.LinAlgError:
            # Fallback to pairwise delta quadratic form
            M_q, N_t = Q_mat.shape[0], X_mat.shape[0]
            D_out = np.empty((M_q, N_t), dtype=np.float32)
            for i in range(M_q):
                diff = X_mat - Q_mat[i:i+1, :]  # (N_t, D)
                quad = np.sum((diff @ inv_C) * diff, axis=1)
                D_out[i, :] = np.sqrt(np.maximum(quad, 0.0))
            return D_out

    raise ValueError("Either whitening_matrix or inv_cov must be provided.")
