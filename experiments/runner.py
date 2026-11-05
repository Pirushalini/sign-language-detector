"""Core of the experiments: build a matcher from a Config, score folds, tune weights, summarise.

All numbers come from TemplateMatcher.class_scores (the same code path as the app), so a
configuration that wins here is directly usable in the app.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from matcher import TemplateMatcher
from utils import LETTERS

from . import metrics as M
from .data import Dataset, Fold, LETTER_INDEX, groups_for, inner_folds
from .regions import GRID_VALUES, PROFILES, feature_weights
from .side_matcher import SideAwareMatcher, augment_with_side

DEFAULT_TEMPERATURE = 0.12    # TemplateMatcher default: what the app shows
APP_THRESHOLD = 0.80          # Settings.threshold default


@dataclass
class Config:
    name: str
    profile: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 2.0)   # tip, joint, palm, inter
    z: float = 1.0
    side_weight: float = 0.0
    side_penalty: float = 0.0
    k: int = 3
    complexity: int = 0           # used to prefer the simpler of two equally good configurations

    @property
    def uses_side(self) -> bool:
        return self.side_weight > 0 or self.side_penalty > 0

    def weights(self) -> np.ndarray:
        t, j, p, i = self.profile
        return feature_weights(t, j, p, i, z=self.z)


def build_matcher(cfg: Config, X: np.ndarray, y: np.ndarray, src: np.ndarray) -> TemplateMatcher:
    if cfg.uses_side:
        m = SideAwareMatcher(cfg.weights(), side_weight=cfg.side_weight, side_penalty=cfg.side_penalty,
                             k=cfg.k, temperature=DEFAULT_TEMPERATURE)
        m.add(augment_with_side(X), y, src)
    else:
        m = TemplateMatcher(feature_weights=cfg.weights(), k=cfg.k, temperature=DEFAULT_TEMPERATURE)
        m.add(X, y, src)
    return m


def score_queries(m: TemplateMatcher, Q: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Per-letter scores for all 26 letters; letters without templates get +inf."""
    m._prepare() if m._dirty else None
    out = np.full((len(Q), len(LETTERS)), np.inf, dtype=np.float32)
    cols = [LETTER_INDEX[c] for c in m._classes]
    for s in range(0, len(Q), chunk):
        out[s:s + chunk][:, cols] = m.class_scores(Q[s:s + chunk])
    return out


@dataclass
class Result:
    name: str
    scores: np.ndarray            # (M, 26)
    y_idx: np.ndarray
    groups: np.ndarray
    ms_per_query: float
    chosen_profiles: List[Tuple[float, ...]] = field(default_factory=list)
    query_index: np.ndarray = field(default_factory=lambda: np.zeros(0, int))

    def summary(self, temperature: float = DEFAULT_TEMPERATURE) -> Dict[str, float]:
        r = M.summarise_predictions(self.scores, self.y_idx, temperature)
        pred_letters = np.array(LETTERS)[r["pred"]]
        true_letters = np.array(LETTERS)[self.y_idx]
        correct, conf = r["correct"], r["conf"]
        T_fit = M.fit_temperature(self.scores, self.y_idx)
        rf = M.summarise_predictions(self.scores, self.y_idx, T_fit)
        cov, prec = M.selective(conf, correct, APP_THRESHOLD)
        return {
            "n_queries": int(len(correct)),
            "accuracy": float(correct.mean()),
            "macro_f1": M.macro_f1(true_letters, pred_letters),
            "top3_accuracy": float(r["top3"].mean()),
            "mean_confidence": float(conf.mean()),
            "mean_conf_correct": float(conf[correct].mean()) if correct.any() else float("nan"),
            "mean_conf_wrong": float(conf[~correct].mean()) if (~correct).any() else float("nan"),
            "mean_margin": float(r["margin"].mean()),
            "auroc": M.auroc(conf, correct),
            "ece_default_T": M.ece(conf, correct),
            "fitted_T": T_fit,
            "ece_fitted_T": M.ece(rf["conf"], rf["correct"]),
            "brier_default_T": M.brier(self.scores, self.y_idx, temperature),
            "nll_default_T": M.nll(self.scores, self.y_idx, temperature),
            "coverage_at_app_threshold": cov,
            "precision_at_app_threshold": prec,
            "median_best_distance": float(np.median(r["best_dist"])),
            "p95_best_distance": float(np.percentile(r["best_dist"], 95)),
            "ms_per_query": self.ms_per_query,
        }


def evaluate(cfg_for_fold: Callable[[Fold], Config], ds: Dataset, folds: Sequence[Fold],
             name: Optional[str] = None) -> Result:
    """Score every fold's queries with a matcher built from that fold's training templates."""
    y_idx_all = ds.y_idx
    S, Y, G, Q_IDX, chosen = [], [], [], [], []
    t_total, n_total = 0.0, 0
    cfg = None
    for f in folds:
        cfg = cfg_for_fold(f)
        chosen.append(cfg.profile)
        m = build_matcher(cfg, ds.X[f.train], ds.y[f.train], ds.src[f.train])
        Q = augment_with_side(ds.X[f.test]) if cfg.uses_side else ds.X[f.test]
        m._prepare()
        t0 = time.perf_counter()
        S.append(score_queries(m, Q))
        t_total += time.perf_counter() - t0
        n_total += len(f.test)
        Y.append(y_idx_all[f.test])
        G.append(groups_for(ds, f.test))
        Q_IDX.append(f.test)
    return Result(name or cfg.name, np.vstack(S), np.concatenate(Y), np.concatenate(G),
                  1000.0 * t_total / max(n_total, 1), chosen, np.concatenate(Q_IDX))


def single_query_latency(cfg: Config, ds: Dataset, fold: Fold, n: int = 100) -> float:
    """Milliseconds for one matcher.predict() call, as in the live app (one frame at a time)."""
    m = build_matcher(cfg, ds.X[fold.train], ds.y[fold.train], ds.src[fold.train])
    Q = augment_with_side(ds.X[fold.test[:n]]) if cfg.uses_side else ds.X[fold.test[:n]]
    m.predict(Q[0])                                    # warm up (builds the caches)
    t0 = time.perf_counter()
    for q in Q:
        m.predict(q)
    return 1000.0 * (time.perf_counter() - t0) / len(Q)


# -------------------------------------------------------------------- tuning
def tune_weights(ds: Dataset, train: np.ndarray, grid: Sequence[float] = GRID_VALUES, ext_cap: int = 100,
                 base: Optional[Config] = None):
    """Grid search of the four region weights with validation folds inside the training data.

    Returns (best (tip, joint, palm, inter), table [(profile, inner accuracy)]). Ties go to the profile
    closest to all-ones, so an arbitrary choice is never made between equals.
    """
    base = base or Config("tune")
    folds = inner_folds(ds, train, ext_cap)
    table = []
    for prof in itertools.product(grid, repeat=4):
        cfg = Config("t", profile=prof, z=base.z, k=base.k)
        r = evaluate(lambda f: cfg, ds, folds)
        table.append((prof, float(np.mean(r.scores.argmin(axis=1) == r.y_idx))))
    best_acc = max(a for _, a in table)
    best = min((p for p, a in table if a >= best_acc - 1e-12), key=lambda p: sum(abs(v - 1.0) for v in p))
    return best, table


def evaluate_tuned(ds: Dataset, folds: Sequence[Fold], name: str = "region_tuned", **kw) -> Result:
    """Per fold, tune the weights on that fold's training data only, then score the fold's queries."""
    cache: Dict[int, Tuple[float, ...]] = {}

    def cfg_for(f: Fold) -> Config:
        key = hash(f.train.tobytes())
        if key not in cache:
            cache[key], _ = tune_weights(ds, f.train, **kw)
        return Config(name, profile=cache[key], complexity=2)

    return evaluate(cfg_for, ds, folds, name)


def sweep_regions(ds: Dataset, folds: Sequence[Fold], values: Sequence[float]) -> Dict[str, Dict[float, float]]:
    """One-factor-at-a-time: scale one region (others at 1) and report accuracy."""
    out: Dict[str, Dict[float, float]] = {}
    for ri, region in enumerate(("fingertips", "finger_joints", "palm_wrist", "inter_hand")):
        out[region] = {}
        for v in values:
            prof = [1.0, 1.0, 1.0, 1.0]
            prof[ri] = v
            r = evaluate(lambda f: Config("s", profile=tuple(prof)), ds, folds)
            out[region][float(v)] = float(np.mean(r.scores.argmin(axis=1) == r.y_idx))
    # depth coordinate
    out["z_coordinate"] = {}
    for v in values:
        r = evaluate(lambda f: Config("s", profile=(1.0, 1.0, 1.0, 1.0), z=v), ds, folds)
        out["z_coordinate"][float(v)] = float(np.mean(r.scores.argmin(axis=1) == r.y_idx))
    return out


# ------------------------------------------------------------ paired compare
def paired_comparison(a: Result, b: Result, n_boot: int = 1000, seed: int = 0) -> Dict[str, Tuple[float, float, float]]:
    """b minus a (point, low, high) for accuracy, mean confidence and AUROC, with a cluster bootstrap."""
    assert np.array_equal(a.query_index, b.query_index), "results must be on the same queries"
    ra = M.summarise_predictions(a.scores, a.y_idx, DEFAULT_TEMPERATURE)
    rb = M.summarise_predictions(b.scores, b.y_idx, DEFAULT_TEMPERATURE)

    def d_acc(i): return rb["correct"][i].mean() - ra["correct"][i].mean()
    def d_conf(i): return rb["conf"][i].mean() - ra["conf"][i].mean()
    def d_auc(i): return M.auroc(rb["conf"][i], rb["correct"][i]) - M.auroc(ra["conf"][i], ra["correct"][i])

    out = {
        "delta_accuracy": M.cluster_bootstrap_diff(d_acc, a.groups, n_boot, seed),
        "delta_mean_confidence": M.cluster_bootstrap_diff(d_conf, a.groups, n_boot, seed),
        "delta_auroc": M.cluster_bootstrap_diff(d_auc, a.groups, n_boot, seed),
    }
    out["mcnemar_p"] = (M.mcnemar_p(rb["correct"], ra["correct"]),) * 3
    return out
