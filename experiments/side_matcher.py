"""Template matcher that knows the palm side of each hand (Experiment 3).

Two ways to use the side information, both optional:

  side_weight   the two palm-side values (one per hand slot, in [-1, 1]) are appended to the feature
                vector as two extra columns and take part in the distance like any other feature.
  side_penalty  a template whose DECIDED side disagrees with the query's DECIDED side gets a fixed
                extra distance per disagreeing hand. 'Decided' means |value| >= decide_at.

With both at 0 this behaves exactly like TemplateMatcher (checked in the tests).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from matcher import TemplateMatcher
from utils import FEATURE_LEN

from .palm_side import TAU, side_values_for_templates


def augment_with_side(X: np.ndarray, tau: float = TAU) -> np.ndarray:
    """(N, 132) features -> (N, 134): the two palm-side values (slot rule) are appended."""
    return np.hstack([X, side_values_for_templates(X, tau)]).astype(np.float32)


class SideAwareMatcher(TemplateMatcher):
    def __init__(self, base_weights: np.ndarray, side_weight: float = 0.0, side_penalty: float = 0.0,
                 decide_at: float = 0.5, **kwargs):
        weights = np.concatenate([np.asarray(base_weights, dtype=np.float32),
                                  np.full(2, side_weight, dtype=np.float32)])
        super().__init__(feature_weights=weights, n_features=FEATURE_LEN + 2, **kwargs)
        self.side_penalty = side_penalty
        self.decide_at = decide_at

    def _distances(self, Q: np.ndarray) -> np.ndarray:
        D = super()._distances(Q)
        if self.side_penalty <= 0:
            return D
        qs = Q[:, FEATURE_LEN:FEATURE_LEN + 2]
        ts = self.X[:, FEATURE_LEN:FEATURE_LEN + 2]
        penalty = np.zeros_like(D)
        for h in range(2):
            both = (np.abs(qs[:, h:h + 1]) >= self.decide_at) & (np.abs(ts[None, :, h]) >= self.decide_at)
            clash = np.sign(qs[:, h:h + 1]) != np.sign(ts[None, :, h])
            penalty += (both & clash)
        return D + self.side_penalty * penalty.astype(D.dtype)
