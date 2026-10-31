"""Template Matcher with pluggable alternative distance metrics.

Supports:
  - "euclidean": standard weighted Euclidean distance
  - "cosine": angle-based cosine distance with zero-vector handling
  - "seuclidean": standardized Euclidean (variance-normalized)
  - "mahalanobis": regularized Mahalanobis distance with covariance whitening

All statistics (variances and covariance matrices) are fitted EXCLUSIVELY
on training templates, ensuring zero data leakage.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np

from matcher import Prediction, SRC_EXTERNAL, SRC_USER, load_templates
from utils import EXTERNAL_TEMPLATES, FEATURE_LEN, FLAGS_START, INTER_START, USER_TEMPLATES
from .metrics import (
    euclidean_distance,
    cosine_distance,
    standardized_euclidean_distance,
    mahalanobis_distance,
    fit_feature_variance,
    fit_covariance,
)

SUPPORTED_METRICS = ("euclidean", "cosine", "seuclidean", "mahalanobis")


class DistanceTemplateMatcher:
    """k-NN template matcher supporting alternative distance metrics.

    Parameters:
        k: how many nearest templates of each letter to average (default 3).
        metric: distance metric ("euclidean", "cosine", "seuclidean", "mahalanobis").
        inter_weight: weight of wrist-to-wrist features (default 2.0).
        flag_weight: weight of hand-present flags (default 3.0).
        user_weight: multiplier for personal templates (< 1 prefers user templates).
        temperature: softmax scaling factor for probability output.
        reject_distance: distance threshold above which predictions are rejected.
        feature_weights: optional explicit 132-D weight vector.
        n_features: dimensionality of features (default 132).
        var_floor: minimum variance floor for standardized Euclidean.
        cov_reg: diagonal ridge regularization factor for Mahalanobis.
        shrinkage: optional shrinkage factor for Mahalanobis covariance.
    """

    def __init__(
        self,
        k: int = 3,
        metric: str = "euclidean",
        inter_weight: float = 2.0,
        flag_weight: float = 3.0,
        user_weight: float = 0.85,
        temperature: float = 0.12,
        reject_distance: float = 1.5,
        feature_weights: Optional[np.ndarray] = None,
        n_features: int = FEATURE_LEN,
        var_floor: float = 1e-6,
        cov_reg: float = 1e-3,
        shrinkage: Optional[float] = None,
    ):
        if metric.lower() not in SUPPORTED_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Supported metrics: {SUPPORTED_METRICS}")

        self.k = k
        self.metric = metric.lower()
        self.inter_weight = inter_weight
        self.flag_weight = flag_weight
        self.user_weight = user_weight
        self.temperature = temperature
        self.reject_distance = reject_distance
        self.feature_weights = feature_weights
        self.n_features = n_features
        self.var_floor = var_floor
        self.cov_reg = cov_reg
        self.shrinkage = shrinkage

        self.X = np.zeros((0, n_features), dtype=np.float32)
        self.y = np.array([], dtype="<U8")
        self.src = np.array([], dtype=np.int8)

        # Fitted statistics (computed strictly from self.X in _prepare)
        self._variance: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._whitening_matrix: Optional[np.ndarray] = None

        self._sqrt_w: Optional[np.ndarray] = None
        self._classes: List[str] = []
        self._class_idx: List[np.ndarray] = []
        self._scale: Optional[np.ndarray] = None
        self._dirty = True

    # ----------------------------------------------------------- template management
    @classmethod
    def from_files(cls, paths: Optional[Iterable[Path]] = None, **kwargs) -> "DistanceTemplateMatcher":
        """Instantiate and load templates from disk."""
        m = cls(**kwargs)
        for p in (paths if paths is not None else [EXTERNAL_TEMPLATES, USER_TEMPLATES]):
            p = Path(p)
            if p.exists():
                X, y, src, _ = load_templates(p)
                m.add(X, y, src)
        return m

    def add(self, X: np.ndarray, y: Sequence[str], src: Sequence[int]) -> None:
        """Add reference templates."""
        X_arr = np.asarray(X, dtype=np.float32)
        if len(X_arr) == 0:
            return
        if X_arr.shape[1] != self.n_features:
            raise ValueError(f"Feature width {X_arr.shape[1]} does not match expected {self.n_features}")

        self.X = np.vstack([self.X, X_arr])
        self.y = np.concatenate([self.y, np.asarray(y, dtype="<U8")])
        self.src = np.concatenate([self.src, np.asarray(src, dtype=np.int8)])
        self._dirty = True

    @property
    def available(self) -> bool:
        return len(self.y) > 0

    @property
    def classes(self) -> List[str]:
        return sorted(set(self.y.tolist()))

    def configure(self, **kwargs) -> None:
        """Update matcher parameters and mark caches as dirty."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Invalid matcher attribute: {key}")
            if key == "metric":
                value = value.lower()
                if value not in SUPPORTED_METRICS:
                    raise ValueError(f"Unknown metric '{value}'. Supported: {SUPPORTED_METRICS}")
            setattr(self, key, value)
        self._dirty = True

    # ----------------------------------------------------------- fitting & cache
    def _prepare(self) -> None:
        """Fit statistics and precompute transformed templates strictly on training data."""
        if not self.available:
            self._classes = []
            self._class_idx = []
            self._dirty = False
            return

        n = self.X.shape[1]
        if self.feature_weights is not None:
            w = np.asarray(self.feature_weights, dtype=np.float32)
            if w.shape != (n,):
                raise ValueError(f"feature_weights has shape {w.shape}, expected ({n},)")
        else:
            w = np.ones(n, dtype=np.float32)
            w[INTER_START:FLAGS_START] = self.inter_weight
            w[FLAGS_START:FEATURE_LEN] = self.flag_weight

        self._sqrt_w = np.sqrt(w).astype(np.float32)
        self._scale = np.where(self.src == SRC_USER, self.user_weight, 1.0).astype(np.float32)

        # Base weighted templates
        X_weighted = self.X * self._sqrt_w[None, :]

        # Fit metric-specific parameters strictly on training templates
        if self.metric == "seuclidean":
            # Feature variance fitted on weighted templates
            self._variance = fit_feature_variance(X_weighted, var_floor=self.var_floor)
        elif self.metric == "mahalanobis":
            # Regularized covariance fitted on weighted templates
            self._cov, self._inv_cov, self._whitening_matrix = fit_covariance(
                X_weighted, reg_factor=self.cov_reg, shrinkage=self.shrinkage
            )
        else:
            self._variance = None
            self._cov = None
            self._inv_cov = None
            self._whitening_matrix = None

        self._classes = self.classes
        self._class_idx = [np.flatnonzero(self.y == c) for c in self._classes]
        self._dirty = False

    # ----------------------------------------------------------- distance computation
    def _distances(self, Q: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between queries Q (M, D) and templates X (N, D)."""
        Q_mat = np.atleast_2d(np.asarray(Q, dtype=np.float32))
        Qw = Q_mat * self._sqrt_w[None, :]
        Xw = self.X * self._sqrt_w[None, :]

        if self.metric == "euclidean":
            D = euclidean_distance(Qw, Xw)
        elif self.metric == "cosine":
            D = cosine_distance(Qw, Xw)
        elif self.metric == "seuclidean":
            D = standardized_euclidean_distance(Qw, Xw, variance=self._variance, var_floor=self.var_floor)
        elif self.metric == "mahalanobis":
            D = mahalanobis_distance(
                Qw, Xw, inv_cov=self._inv_cov, whitening_matrix=self._whitening_matrix
            )
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Scale distances to user templates to prioritize personal calibration
        return D * self._scale[None, :]

    def class_scores(self, Q: np.ndarray) -> np.ndarray:
        """Compute mean distance to top-k nearest templates for every class.

        Output shape: (M, len(classes))
        """
        if self._dirty:
            self._prepare()
        if not self.available:
            return np.full((len(Q), 0), np.inf, dtype=np.float32)

        Q_mat = np.atleast_2d(np.asarray(Q, dtype=np.float32))
        D = self._distances(Q_mat)
        scores = np.empty((Q_mat.shape[0], len(self._classes)), dtype=np.float32)

        for j, idx in enumerate(self._class_idx):
            sub = D[:, idx]
            kk = min(self.k, sub.shape[1])
            # Partition to extract k-smallest without full sorting
            scores[:, j] = np.mean(np.partition(sub, kk - 1, axis=1)[:, :kk], axis=1)

        return scores

    # ----------------------------------------------------------- prediction
    def predict(self, feat: np.ndarray) -> Prediction:
        """Predict the letter for a single feature vector."""
        if not self.available:
            return Prediction(None, 0.0)
        if self._dirty:
            self._prepare()

        s = self.class_scores(feat)[0]
        if len(s) == 0:
            return Prediction(None, 0.0)

        # Softmax probabilities with temperature scaling
        shifted = -(s - s.min()) / max(self.temperature, 1e-6)
        p = np.exp(shifted - shifted.max())
        p /= np.maximum(p.sum(), 1e-12)

        order = np.argsort(-p)[:3]
        top3 = [(self._classes[i], float(p[i])) for i in order]
        best = int(order[0])
        distance = float(s[best])

        if distance > self.reject_distance:
            return Prediction(None, 0.0, top3, distance)
        return Prediction(self._classes[best], float(p[best]), top3, distance)

    def predict_labels(self, Q: np.ndarray, chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels and best scores for a batch of queries."""
        if self._dirty:
            self._prepare()
        if not self.available:
            return np.array([]), np.array([], dtype=np.float32)

        labels: List[str] = []
        best: List[float] = []

        for start in range(0, len(Q), chunk):
            s = self.class_scores(Q[start : start + chunk])
            j = np.argmin(s, axis=1)
            labels.extend(self._classes[i] for i in j)
            best.extend(s[np.arange(len(j)), j])

        return np.array(labels, dtype="<U8"), np.array(best, dtype=np.float32)
