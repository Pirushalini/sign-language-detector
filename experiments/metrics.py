"""Metrics for accuracy, confidence quality and paired comparisons.

Confidence = the softmax probability the matcher shows in the app, computed exactly like
TemplateMatcher.predict: p = softmax(-(score - min score) / temperature).

Why several confidence metrics?
    The raw confidence depends on the SCALE of the distances. Add a palm-side penalty and every
    distance grows, so the same temperature gives a sharper softmax and a higher confidence even if
    the system is not any better. Therefore we also report scale-free measures:
      AUROC           how well confidence separates correct from wrong answers (0.5 = useless, 1 = perfect)
      risk-coverage   accuracy when only the most confident x % of answers are accepted
      ECE after fitting one temperature per configuration (isolates 'is the ranking good' from 'is the scale right')
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- probabilities
def softmax_probs(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Per-class probabilities from per-class distances (rows = queries). inf = class not available."""
    s = np.asarray(scores, dtype=np.float64)
    shifted = -(s - s.min(axis=1, keepdims=True)) / max(temperature, 1e-6)
    e = np.exp(shifted - shifted.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def summarise_predictions(scores: np.ndarray, y_idx: np.ndarray, temperature: float) -> Dict[str, np.ndarray]:
    """Everything per query: predicted class, confidence, margin to the 2nd class, correctness, top-3 hit."""
    p = softmax_probs(scores, temperature)
    order = np.argsort(-p, axis=1)
    pred = order[:, 0]
    rows = np.arange(len(p))
    return {
        "pred": pred,
        "conf": p[rows, pred],
        "margin": p[rows, order[:, 0]] - p[rows, order[:, 1]],
        "correct": pred == y_idx,
        "top3": np.any(order[:, :3] == y_idx[:, None], axis=1),
        "p_true": p[rows, y_idx],
        "best_dist": scores[rows, np.argmin(scores, axis=1)],
    }


# ---------------------------------------------------------------- accuracy-like
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------- calibration
def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: weighted gap between confidence and accuracy per confidence bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    """(mean confidence, accuracy, count) per bin, for the reliability diagram."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out


def nll(scores: np.ndarray, y_idx: np.ndarray, temperature: float) -> float:
    p = softmax_probs(scores, temperature)
    return float(-np.mean(np.log(np.maximum(p[np.arange(len(p)), y_idx], 1e-12))))


def top1_nll(scores: np.ndarray, y_idx: np.ndarray, temperature: float) -> float:
    """Cross-entropy of 'was the top answer right?' against the top answer's confidence."""
    r = summarise_predictions(scores, y_idx, temperature)
    c = np.clip(r["conf"], 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(r["correct"], np.log(c), np.log(1 - c))))


def fit_temperature(scores: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature that makes the TOP confidence best match how often the top answer is right.

    One number per configuration, grid search on the top-1 cross-entropy (the quantity ECE measures).
    """
    grid = np.logspace(-2.5, 0.7, 60)
    return float(grid[int(np.argmin([top1_nll(scores, y_idx, t) for t in grid]))])


def brier(scores: np.ndarray, y_idx: np.ndarray, temperature: float) -> float:
    p = softmax_probs(scores, temperature)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(p)), y_idx] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Probability that a random correct answer has a higher confidence than a random wrong one."""
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def selective(conf: np.ndarray, correct: np.ndarray, threshold: float) -> Tuple[float, float]:
    """(coverage, accuracy of the accepted answers) when only conf >= threshold is accepted."""
    m = conf >= threshold
    if not m.any():
        return 0.0, float("nan")
    return float(m.mean()), float(correct[m].mean())


def risk_coverage(conf: np.ndarray, correct: np.ndarray, points: int = 21):
    """Accuracy of the most confident fraction c of answers, for c from small to 1."""
    order = np.argsort(-conf, kind="stable")
    c_sorted = correct[order]
    cum = np.cumsum(c_sorted) / np.arange(1, len(c_sorted) + 1)
    covs = np.linspace(0.05, 1.0, points)
    return [(float(c), float(cum[max(0, int(np.ceil(c * len(cum))) - 1)])) for c in covs]


# ----------------------------------------------------------- paired statistics
def cluster_bootstrap_diff(stat: Callable[[np.ndarray], float], groups: np.ndarray,
                           n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """Estimate and 95 % interval of stat(indices) when whole groups are resampled.

    Neighbouring frames of the same sign are near-identical, so resampling single frames would give
    intervals that are far too narrow. Groups (letter x recording block) are resampled instead.
    stat receives an index array into the query list and returns a number (usually a difference).
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    point = stat(np.arange(len(groups)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots[b] = stat(np.concatenate([members[g] for g in pick]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mcnemar_p(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Exact McNemar p-value on the queries where exactly one of the two systems is right.

    Treats queries as independent, which they are not (neighbouring frames); read it as optimistic.
    The cluster bootstrap above is the more honest interval.
    """
    from scipy.stats import binomtest
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)
