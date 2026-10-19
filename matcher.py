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

from utils import (EXTERNAL_TEMPLATES, FEATURE_LEN, FLAGS_START, INTER_START,
                   USER_TEMPLATES)

SRC_EXTERNAL = 0
SRC_USER = 1


@dataclass
class Prediction:
    """Result for one frame. label is None when nothing matches well enough."""

    label: Optional[str]
    confidence: float
    top3: List[Tuple[str, float]] = field(default_factory=list)
    distance: float = float("inf")


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
    """

    def __init__(self, k: int = 3, metric: str = "euclidean", inter_weight: float = 2.0,
                 flag_weight: float = 3.0, user_weight: float = 0.85,
                 temperature: float = 0.12, reject_distance: float = 1.5):
        self.k = k
        self.metric = metric
        self.inter_weight = inter_weight
        self.flag_weight = flag_weight
        self.user_weight = user_weight
        self.temperature = temperature
        self.reject_distance = reject_distance
        self.X = np.zeros((0, FEATURE_LEN), dtype=np.float32)
        self.y = np.array([], dtype="<U8")
        self.src = np.array([], dtype=np.int8)
        self._dirty = True

    # ----------------------------------------------------------- loading
    @classmethod
    def from_files(cls, paths: Optional[Iterable[Path]] = None, **kwargs) -> "TemplateMatcher":
        """Load every template file that exists (external dataset and/or your own)."""
        m = cls(**kwargs)
        for p in (paths if paths is not None else [EXTERNAL_TEMPLATES, USER_TEMPLATES]):
            p = Path(p)
            if p.exists():
                X, y, src, _ = load_templates(p)
                if X.shape[1] != FEATURE_LEN:
                    raise ValueError(f"{p.name} has {X.shape[1]} features, expected {FEATURE_LEN}. "
                                     "Re-run import_dataset.py / calibrate.py.")
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
        w = np.ones(FEATURE_LEN, dtype=np.float32)
        w[INTER_START:FLAGS_START] = self.inter_weight
        w[FLAGS_START:] = self.flag_weight
        self._sqrt_w = np.sqrt(w)
        Xw = self.X * self._sqrt_w
        if self.metric == "cosine":
            norms = np.linalg.norm(Xw, axis=1, keepdims=True)
            Xw = Xw / np.maximum(norms, 1e-9)
        self._Xw = Xw
        self._sq_norm = np.sum(Xw * Xw, axis=1)
        self._scale = np.where(self.src == SRC_USER, self.user_weight, 1.0).astype(np.float32)
        self._classes = self.classes
        self._class_idx = [np.flatnonzero(self.y == c) for c in self._classes]
        self._dirty = False

    def _distances(self, Q: np.ndarray) -> np.ndarray:
        """Distances from each query row to every template, shape (M, N)."""
        Qw = Q * self._sqrt_w
        if self.metric == "cosine":
            Qw = Qw / np.maximum(np.linalg.norm(Qw, axis=1, keepdims=True), 1e-9)
            D = 1.0 - Qw @ self._Xw.T
        else:
            d2 = np.sum(Qw * Qw, axis=1, keepdims=True) - 2.0 * (Qw @ self._Xw.T) + self._sq_norm[None, :]
            D = np.sqrt(np.maximum(d2, 0.0))
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
            return Prediction(None, 0.0)
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
            return Prediction(None, 0.0, top3, distance)
        return Prediction(self._classes[best], float(p[best]), top3, distance)

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
