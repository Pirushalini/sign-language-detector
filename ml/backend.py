"""ml/backend.py — ML recognition backend for the live application.

Provides a unified interface so the live app can switch between
  (a) the existing TemplateMatcher
  (b) any trained ML pipeline (KNN / SVM / RF / MLP)

The interface is designed to expose the same information the existing
Prediction dataclass carries, plus extra score information for the teammate
implementing confidence estimation (Task 4).

Usage (from app.py or logic.py)
--------------------------------
    from ml.backend import MLBackend, BACKEND_TEMPLATE, BACKEND_ML

    backend = MLBackend(mode=BACKEND_ML, model_name="svm_ext")
    backend.load()
    result = backend.predict(feature_vector)
    # result.label, result.confidence, result.top3, result.scores, result.margin

Loading once
------------
Call backend.load() once when the application starts (or when the user switches
mode in settings).  Do NOT call load() on every frame.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import FEATURE_LEN

# ---------------------------------------------------------------- mode constants
BACKEND_TEMPLATE = "template"
BACKEND_ML = "ml"

VALID_MODES = (BACKEND_TEMPLATE, BACKEND_ML)


# ---------------------------------------------------------------- result type
@dataclass
class MLPrediction:
    """Extended prediction result from the ML backend.

    Attributes
    ----------
    label       : best predicted letter, or None when nothing matches
    confidence  : calibrated probability (or softmax for non-prob models)
    top3        : list of (letter, score) for the top 3 candidates
    scores      : full dict {letter: score} for all classes
    margin      : gap between the top-1 and top-2 scores (useful for Task 4)
    latency_ms  : inference time in milliseconds
    mode        : which backend produced this prediction
    """

    label: Optional[str]
    confidence: float
    top3: List[Tuple[str, float]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    latency_ms: float = 0.0
    mode: str = BACKEND_TEMPLATE
    distance: float = 0.0


# ---------------------------------------------------------------- backend class
class MLBackend:
    """Recognition backend that wraps either the TemplateMatcher or an ML pipeline.

    Parameters
    ----------
    mode       : BACKEND_TEMPLATE or BACKEND_ML
    model_name : name of the saved ML model to load (e.g. "svm_ext")
    confidence_threshold : predictions below this are returned as label=None
    """

    def __init__(
        self,
        mode: str = BACKEND_TEMPLATE,
        model_name: str = "svm_ext",
        confidence_threshold: float = 0.0,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES!r}, got {mode!r}")
        self.mode = mode
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.reject_distance: float = 1.5
        self._loaded = False
        self._pipe = None       # sklearn Pipeline (ML mode)
        self._tm = None         # TemplateMatcher (template mode)
        self._classes: List[str] = []
        self._has_proba = False

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        """Load the model. Call once before calling predict()."""
        if self.mode == BACKEND_TEMPLATE:
            self._load_template_matcher()
        else:
            self._load_ml_pipeline()
        self._loaded = True

    def _load_template_matcher(self) -> None:
        from matcher import TemplateMatcher
        self._tm = TemplateMatcher.from_files()
        self._classes = self._tm.classes
        self._has_proba = False

    def _load_ml_pipeline(self) -> None:
        from ml.train import load_model
        self._pipe, meta = load_model(self.model_name)
        self._classes = meta.get("classes", [])
        self._has_proba = hasattr(self._pipe, "predict_proba")

    # ---------------------------------------------------------------- prediction
    def predict(self, feat: np.ndarray) -> MLPrediction:
        """Classify one 132-dim feature vector.

        Parameters
        ----------
        feat : (132,) float32 array from build_features()

        Returns
        -------
        MLPrediction with label, confidence, top3, scores, margin, latency_ms
        """
        if not self._loaded:
            raise RuntimeError("Call backend.load() before predict().")

        feat = np.asarray(feat, dtype=np.float32)
        if feat.shape != (FEATURE_LEN,):
            return MLPrediction(None, 0.0, mode=self.mode)

        t0 = time.perf_counter()

        if self.mode == BACKEND_TEMPLATE:
            result = self._predict_template(feat)
        else:
            result = self._predict_ml(feat)

        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def _predict_template(self, feat: np.ndarray) -> MLPrediction:
        pred = self._tm.predict(feat)
        scores = {label: float(conf) for label, conf in pred.top3}
        label = pred.label
        conf = pred.confidence
        if conf < self.confidence_threshold:
            label = None
        margin = 0.0
        if len(pred.top3) >= 2:
            margin = pred.top3[0][1] - pred.top3[1][1]
        return MLPrediction(
            label=label,
            confidence=conf,
            top3=pred.top3,
            scores=scores,
            margin=float(margin),
            mode=BACKEND_TEMPLATE,
        )

    def _predict_ml(self, feat: np.ndarray) -> MLPrediction:
        x2d = feat.reshape(1, -1)

        if self._has_proba:
            proba = self._pipe.predict_proba(x2d)[0]  # (n_classes,)
            scores_full = {c: float(p) for c, p in zip(self._classes, proba)}
            label = self._classes[int(np.argmax(proba))]
            confidence = float(proba.max())
            sorted_idx = np.argsort(-proba)
            top3 = [(self._classes[i], float(proba[i])) for i in sorted_idx[:3]]
            margin = float(proba[sorted_idx[0]] - proba[sorted_idx[1]]) if len(sorted_idx) >= 2 else 0.0
        else:
            # Non-probabilistic: use raw decision function if available
            label_pred = self._pipe.predict(x2d)[0]
            label = str(label_pred)
            confidence = 1.0  # no calibrated probability
            scores_full = {label: 1.0}
            top3 = [(label, 1.0)]
            margin = 1.0

        if confidence < self.confidence_threshold:
            label = None

        return MLPrediction(
            label=label,
            confidence=confidence,
            top3=top3,
            scores=scores_full,
            margin=margin,
            mode=BACKEND_ML,
        )

    # ---------------------------------------------------------------- properties
    @property
    def available(self) -> bool:
        """True if the backend has at least one class to predict."""
        if not self._loaded:
            return False
        if self.mode == BACKEND_TEMPLATE:
            return self._tm is not None and self._tm.available
        return self._pipe is not None and len(self._classes) > 0

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (f"MLBackend(mode={self.mode!r}, model={self.model_name!r}, "
                f"classes={len(self._classes)}, {status})")

