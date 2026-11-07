"""Template matcher: classification by distance to stored reference samples.

There is no training step and no fitted model. Reference feature vectors
(from the imported dataset and from your own calibration) are stored as they
are, and a live frame is classified by its distance to the nearest stored
examples of each letter. This is instance-based (non-parametric) classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils import (EXTERNAL_TEMPLATES, FEATURE_LEN, FLAGS_START, HAND_LEN,
                   INTER_START, ONE_HAND_LETTERS, USER_TEMPLATES)

SRC_EXTERNAL = 0
SRC_USER = 1


@dataclass
class Prediction:
    """Result for one frame. label is None when nothing matches well enough."""

    label: Optional[str]
    confidence: float
    top3: List[Tuple[str, float]] = field(default_factory=list)
    distance: float = float("inf")
    rejection_reason: Optional[str] = None


TIP_IDX = [4, 8, 12, 16, 20]      # thumb, index, middle, ring, pinky tips


def relation_features(X: np.ndarray) -> np.ndarray:
    """Fingertip-to-fingertip distances between the two hands, 25 values per row.

    Recovered from the stored 132 features, so no template needs re-recording.
    Only x and y are used: MediaPipe's z is measured from each hand's own wrist
    and cannot be compared between hands. Rows without two hands get zeros.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    a = X[:, :HAND_LEN].reshape(-1, 21, 3)[:, TIP_IDX, :2]                  # hand 1 tips
    b = X[:, HAND_LEN:2 * HAND_LEN].reshape(-1, 21, 3)[:, TIP_IDX, :2]      # hand 2 tips
    delta = X[:, INTER_START:INTER_START + 2]                               # wrist2 - wrist1
    a_in_b = a - delta[:, None, :]                                          # hand 1 tips, seen from hand 2's wrist
    d = np.linalg.norm(a_in_b[:, :, None, :] - b[:, None, :, :], axis=-1).reshape(len(X), -1)
    two = (X[:, FLAGS_START] > 0.5) & (X[:, FLAGS_START + 1] > 0.5)
    d[~two] = 0.0
    return d.astype(np.float32)


def swap_two_hands(X: np.ndarray) -> np.ndarray:
    """Swap Hand 1 and Hand 2 features for two-handed feature vectors.

    Swaps hand landmarks [0:63] with [63:126], negates the inter-wrist vector [126:129],
    keeps wrist distance [129] identical, and swaps presence flags [130] and [131].
    Single-hand vectors are left unchanged.
    """
    X = np.asarray(X, dtype=np.float32)
    was_1d = (X.ndim == 1)
    out = np.atleast_2d(X).copy()
    two = (out[:, FLAGS_START] > 0.5) & (out[:, FLAGS_START + 1] > 0.5)
    if np.any(two):
        h1 = out[two, :HAND_LEN].copy()
        h2 = out[two, HAND_LEN:2 * HAND_LEN].copy()
        out[two, :HAND_LEN] = h2
        out[two, HAND_LEN:2 * HAND_LEN] = h1
        out[two, INTER_START:INTER_START + 3] = -out[two, INTER_START:INTER_START + 3]
        f1 = out[two, FLAGS_START].copy()
        f2 = out[two, FLAGS_START + 1].copy()
        out[two, FLAGS_START] = f2
        out[two, FLAGS_START + 1] = f1
    return out[0] if was_1d else out


# ------------------------------------------------------------ file helpers
def save_templates(path: Path, X: np.ndarray, y: Sequence[str], src: Sequence[int],
                   sessions: Optional[Sequence[str]] = None) -> None:
    """Save reference templates to a compressed .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(y)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype="<U8"),
        src=np.asarray(src, dtype=np.int8),
        session=np.asarray(sessions if sessions is not None else [""] * n, dtype="<U32"),
    )


def load_templates(path: Path):
    """Load templates saved by save_templates. Returns (X, y, src, session)."""
    with np.load(path, allow_pickle=False) as d:
        return d["X"], d["y"], d["src"], d["session"]


# ----------------------------------------------------------------- matcher
class TemplateMatcher:
    """k-nearest-template matcher with per-letter scores.

    k:             how many nearest templates of a letter are averaged
    metric:        "euclidean" (default) or "cosine"
    inter_weight:  weight of the wrist-to-wrist features (how the hands sit)
    flag_weight:   weight of the hand-present flags, so a frame with one hand
                   is pushed away from two-hand templates without being excluded
    user_weight:   distances to your own templates are multiplied by this
                   (below 1 means your own samples are preferred)
    temperature:   turns letter scores into probabilities (smaller = sharper)
    reject_distance: if the best score is above this, nothing is returned
    feature_weights: optional per-feature weight vector (one value per column of X). When
                   given it replaces inter_weight / flag_weight completely. A weight w
                   scales the squared difference of that feature, so w = 0 switches a
                   feature off. Used by the weighting experiments (experiments/).
    n_features:    width of the feature vectors (default 132). Experiments may append
                   extra columns (for example palm-side values).
    """

    def __init__(self, k: int = 3, metric: str = "euclidean", inter_weight: float = 2.0,
                 rel_weight: float = 0.0,
                 flag_weight: float = 3.0, user_weight: float = 0.85,
                 temperature: float = 0.12, reject_distance: float = 1.5,
                 dual_order: bool = True,
                 feature_weights: Optional[np.ndarray] = None, n_features: int = FEATURE_LEN):
        self.k = k
        self.metric = metric
        self.inter_weight = inter_weight
        self.rel_weight = rel_weight
        self.flag_weight = flag_weight
        self.user_weight = user_weight
        self.temperature = temperature
        self.reject_distance = reject_distance
        self.dual_order = dual_order
        self.feature_weights = feature_weights
        self.X = np.zeros((0, n_features), dtype=np.float32)
        self.y = np.array([], dtype="<U8")
        self.src = np.array([], dtype=np.int8)
        self._dirty = True

    # ----------------------------------------------------------- loading
    @classmethod
    def from_files(cls, paths: Optional[Iterable[Path]] = None,
                   require_two_hands: bool = True, **kwargs) -> "TemplateMatcher":
        """Load every template file that exists (external dataset and/or your own)."""
        m = cls(**kwargs)
        for p in (paths if paths is not None else [EXTERNAL_TEMPLATES, USER_TEMPLATES]):
            p = Path(p)
            if p.exists():
                X, y, src, _ = load_templates(p)
                if X.shape[1] != FEATURE_LEN:
                    raise ValueError(f"{p.name} has {X.shape[1]} features, expected {FEATURE_LEN}. "
                                     "Re-run import_dataset.py / calibrate.py.")
                if require_two_hands:
                    n_hands = (X[:, FLAGS_START] > 0.5).astype(int) + (
                        X[:, FLAGS_START + 1] > 0.5
                    )
                    is_one = np.isin(y, list(ONE_HAND_LETTERS))
                    keep = np.where(is_one, n_hands == 1, n_hands == 2)
                    X, y, src = X[keep], y[keep], src[keep]
                m.add(X, y, src)
        return m

    def add(self, X: np.ndarray, y: Sequence[str], src: Sequence[int]) -> None:
        self.X = np.vstack([self.X, np.asarray(X, dtype=np.float32)])
        self.y = np.concatenate([self.y, np.asarray(y, dtype="<U8")])
        self.src = np.concatenate([self.src, np.asarray(src, dtype=np.int8)])
        self._dirty = True

    @property
    def available(self) -> bool:
        return len(self.y) > 0

    @property
    def classes(self) -> List[str]:
        return sorted(set(self.y.tolist()))

    def counts(self) -> Dict[str, Tuple[int, int]]:
        """Per letter: (external templates, user templates)."""
        out = {}
        for c in self.classes:
            mask = self.y == c
            out[c] = (int(np.sum(mask & (self.src == SRC_EXTERNAL))), int(np.sum(mask & (self.src == SRC_USER))))
        return out

    # ----------------------------------------------------------- internals
    def configure(self, **kwargs) -> None:
        """Change matching options (k, metric, weights, ...) and rebuild caches."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(key)
            setattr(self, key, value)
        self._dirty = True

    def _prepare(self) -> None:
        n = self.X.shape[1]
        if self.feature_weights is not None:
            w = np.asarray(self.feature_weights, dtype=np.float32)
            if w.shape != (n,):
                raise ValueError(f"feature_weights has shape {w.shape}, expected ({n},)")
        else:
            w = np.ones(n, dtype=np.float32)
            w[INTER_START:FLAGS_START] = self.inter_weight
            w[FLAGS_START:FEATURE_LEN] = self.flag_weight
        self._sqrt_w = np.sqrt(w)
        Xw = self._embed(self.X)
        if self.metric == "cosine":
            norms = np.linalg.norm(Xw, axis=1, keepdims=True)
            Xw = Xw / np.maximum(norms, 1e-9)
        self._Xw = Xw
        self._sq_norm = np.sum(Xw * Xw, axis=1)
        self._scale = np.where(self.src == SRC_USER, self.user_weight, 1.0).astype(np.float32)
        self._classes = self.classes
        self._class_idx = [np.flatnonzero(self.y == c) for c in self._classes]
        self._dirty = False

    def _embed(self, Q: np.ndarray) -> np.ndarray:
        """Weighted feature vectors (plus the hand-to-hand distances when rel_weight > 0)."""
        Qw = Q * self._sqrt_w
        if self.rel_weight > 0:
            Qw = np.hstack([Qw, relation_features(Q) * np.float32(np.sqrt(self.rel_weight))])
        return Qw

    def _distances(self, Q: np.ndarray) -> np.ndarray:
        """Distances from each query row to every template, shape (M, N)."""
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float32))
        Qw = self._embed(Q)
        if self.metric == "cosine":
            Qw = Qw / np.maximum(np.linalg.norm(Qw, axis=1, keepdims=True), 1e-9)
            D = 1.0 - Qw @ self._Xw.T
        else:
            d2 = np.sum(Qw * Qw, axis=1, keepdims=True) - 2.0 * (Qw @ self._Xw.T) + self._sq_norm[None, :]
            D = np.sqrt(np.maximum(d2, 0.0))

        if self.dual_order:
            two = (Q[:, FLAGS_START] > 0.5) & (Q[:, FLAGS_START + 1] > 0.5)
            if np.any(two):
                Q_swap = swap_two_hands(Q[two])
                Qw_swap = self._embed(Q_swap)
                if self.metric == "cosine":
                    Qw_swap = Qw_swap / np.maximum(np.linalg.norm(Qw_swap, axis=1, keepdims=True), 1e-9)
                    D_swap = 1.0 - Qw_swap @ self._Xw.T
                else:
                    d2_swap = np.sum(Qw_swap * Qw_swap, axis=1, keepdims=True) - 2.0 * (Qw_swap @ self._Xw.T) + self._sq_norm[None, :]
                    D_swap = np.sqrt(np.maximum(d2_swap, 0.0))
                D[two] = np.minimum(D[two], D_swap)

        return D * self._scale[None, :]

    def class_scores(self, Q: np.ndarray) -> np.ndarray:
        """Per-letter score (mean of the k smallest distances), shape (M, n_classes)."""
        if self._dirty:
            self._prepare()
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float32))
        D = self._distances(Q)
        scores = np.empty((Q.shape[0], len(self._classes)), dtype=np.float32)
        for j, idx in enumerate(self._class_idx):
            sub = D[:, idx]
            kk = min(self.k, sub.shape[1])
            scores[:, j] = np.mean(np.partition(sub, kk - 1, axis=1)[:, :kk], axis=1)
        return scores

    # ---------------------------------------------------------- prediction
    def predict(self, feat: np.ndarray) -> Prediction:
        """Classify one feature vector."""
        if not self.available:
            return Prediction(None, 0.0, rejection_reason="No templates available")
        if self._dirty:
            self._prepare()
        s = self.class_scores(feat)[0]
        shifted = -(s - s.min()) / max(self.temperature, 1e-6)
        p = np.exp(shifted - shifted.max())
        p /= p.sum()
        order = np.argsort(-p)[:3]
        top3 = [(self._classes[i], float(p[i])) for i in order]
        best = int(order[0])
        distance = float(s[best])
        if distance > self.reject_distance:
            return Prediction(None, 0.0, top3, distance,
                              rejection_reason=f"Distance {distance:.2f} > max {self.reject_distance:.2f}")
        return Prediction(self._classes[best], float(p[best]), top3, distance, rejection_reason=None)

    def predict_labels(self, Q: np.ndarray, chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Best letter and its score for many queries (used by evaluate.py)."""
        if self._dirty:
            self._prepare()
        labels, best = [], []
        for start in range(0, len(Q), chunk):
            s = self.class_scores(Q[start:start + chunk])
            j = np.argmin(s, axis=1)
            labels.extend(self._classes[i] for i in j)
            best.extend(s[np.arange(len(j)), j])
        return np.array(labels), np.array(best, dtype=np.float32)