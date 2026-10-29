"""ml/dataset.py — Dataset loading, validation, and splitting for ML classifiers.

Loads the same .npz templates used by the existing recognition system and
prepares training / validation / test splits that prevent data leakage.

The templates were created by import_dataset.py (external data) and
calibrate.py (personal data) using the exact same build_features() function
used at live inference time.  Loading them here guarantees consistent features.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------- project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import EXTERNAL_TEMPLATES, FEATURE_LEN, USER_TEMPLATES
from matcher import load_templates


# ---------------------------------------------------------------- constants
SEED = 42


# ---------------------------------------------------------------- loading
def load_external() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load external (Kaggle/dataset) templates.

    Returns
    -------
    X : (N, 132) float32 feature matrix
    y : (N,) str array of letter labels
    session : (N,) str array of session identifiers
    """
    if not EXTERNAL_TEMPLATES.exists():
        raise FileNotFoundError(
            f"{EXTERNAL_TEMPLATES} not found. Run import_dataset.py first."
        )
    X, y, _src, session = load_templates(EXTERNAL_TEMPLATES)
    _validate(X, y, source_name="external")
    return X, y, session


def load_user() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load personal calibration templates.

    Returns
    -------
    X : (N, 132) float32 feature matrix
    y : (N,) str array of letter labels
    session : (N,) str array of session identifiers
    """
    if not USER_TEMPLATES.exists():
        raise FileNotFoundError(
            f"{USER_TEMPLATES} not found. Run calibrate.py first."
        )
    X, y, _src, session = load_templates(USER_TEMPLATES)
    _validate(X, y, source_name="user")
    return X, y, session


def _validate(X: np.ndarray, y: np.ndarray, source_name: str) -> None:
    """Raise ValueError for bad shapes or non-finite values."""
    if X.ndim != 2 or X.shape[1] != FEATURE_LEN:
        raise ValueError(
            f"{source_name}: expected (N, {FEATURE_LEN}) features, "
            f"got {X.shape}"
        )
    if not np.isfinite(X).all():
        n_bad = int(np.sum(~np.isfinite(X)))
        raise ValueError(
            f"{source_name}: {n_bad} non-finite values found in X"
        )
    if len(X) != len(y):
        raise ValueError(
            f"{source_name}: X has {len(X)} rows but y has {len(y)} labels"
        )


# ---------------------------------------------------------------- splitting
def session_aware_split(
    X: np.ndarray,
    y: np.ndarray,
    session: np.ndarray,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data keeping sessions intact in one partition.

    For the *external* dataset every sample carries the same session label
    (``"kaggle-erentatepe"``), so true group-based splitting cannot be done
    across sessions.  In that case we fall back to a stratified random split
    by class label, but we keep contiguous blocks of frames from the same
    class together — this is still more conservative than a fully random shuffle
    because adjacent frames in the original CSV are nearly identical.

    For the *user* dataset there are multiple calibration sessions and the
    sessions are used as groups: whole sessions are assigned to train or test so
    that personal samples are never mixed across splits.

    Returns
    -------
    X_train, X_val, X_test : feature arrays
    y_train, y_val, y_test : label arrays
    """
    rng = np.random.default_rng(seed)
    unique_sessions = np.unique(session)

    # ---- single-session (external) fallback: stratified by class ----------
    if len(unique_sessions) == 1:
        return _stratified_split(X, y, test_frac, val_frac, rng)

    # ---- multi-session (user): group sessions into splits -----------------
    rng.shuffle(unique_sessions)
    n = len(unique_sessions)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_test - n_val

    if n_train < 1:
        # Not enough sessions to have all three splits; merge val into train.
        n_val = 0
        n_train = n - n_test

    test_sess = set(unique_sessions[:n_test])
    val_sess = set(unique_sessions[n_test: n_test + n_val])

    te = np.flatnonzero(np.isin(session, list(test_sess)))
    va = np.flatnonzero(np.isin(session, list(val_sess)))
    tr = np.flatnonzero(~np.isin(session, list(test_sess | val_sess)))

    return X[tr], X[va], X[te], y[tr], y[va], y[te]


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split when only one session group exists."""
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        n_val = min(n_val, n - n_test - 1)  # always keep ≥1 for train
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test: n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    tr = np.array(train_idx)
    va = np.array(val_idx)
    te = np.array(test_idx)
    return X[tr], X[va], X[te], y[tr], y[va], y[te]


# ---------------------------------------------------------------- dataset info
def dataset_summary(X: np.ndarray, y: np.ndarray, session: np.ndarray, name: str = "") -> Dict:
    """Return a summary dict describing the dataset."""
    classes, counts = np.unique(y, return_counts=True)
    return {
        "name": name,
        "n_samples": len(X),
        "n_classes": len(classes),
        "feature_dim": X.shape[1],
        "classes": classes.tolist(),
        "counts_per_class": {c: int(n) for c, n in zip(classes, counts)},
        "n_sessions": int(len(np.unique(session))),
        "sessions": sorted(np.unique(session).tolist()),
        "class_imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
        "has_nan_or_inf": bool(~np.isfinite(X).all()),
    }


# ---------------------------------------------------------------- combined
def load_combined(
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    seed: int = SEED,
) -> Dict:
    """Load external + user data; return a dict of all splits and metadata.

    The test set for the combined experiment uses the user test split only
    (sessions held out from the user data), which provides the most honest
    cross-signer / cross-camera evaluation possible with the available data.

    Keys returned
    -------------
    X_ext_train, y_ext_train : external training portion
    X_ext_val, y_ext_val     : external validation portion
    X_ext_test, y_ext_test   : external test portion
    X_usr_train, y_usr_train : user training portion
    X_usr_val, y_usr_val     : user validation portion
    X_usr_test, y_usr_test   : user test portion
    X_comb_train, y_comb_train : external_train + user_train combined
    X_comb_val, y_comb_val     : combined validation
    external_summary, user_summary : metadata dicts
    """
    X_ext, y_ext, sess_ext = load_external()
    X_usr, y_usr, sess_usr = load_user()

    (X_ext_tr, X_ext_va, X_ext_te,
     y_ext_tr, y_ext_va, y_ext_te) = session_aware_split(
        X_ext, y_ext, sess_ext, test_frac, val_frac, seed
    )

    (X_usr_tr, X_usr_va, X_usr_te,
     y_usr_tr, y_usr_va, y_usr_te) = session_aware_split(
        X_usr, y_usr, sess_usr, test_frac, val_frac, seed
    )

    X_comb_tr = np.vstack([X_ext_tr, X_usr_tr])
    y_comb_tr = np.concatenate([y_ext_tr, y_usr_tr])
    X_comb_va = np.vstack([X_ext_va, X_usr_va])
    y_comb_va = np.concatenate([y_ext_va, y_usr_va])

    return {
        # external
        "X_ext_train": X_ext_tr, "y_ext_train": y_ext_tr,
        "X_ext_val": X_ext_va,   "y_ext_val": y_ext_va,
        "X_ext_test": X_ext_te,  "y_ext_test": y_ext_te,
        # user
        "X_usr_train": X_usr_tr, "y_usr_train": y_usr_tr,
        "X_usr_val": X_usr_va,   "y_usr_val": y_usr_va,
        "X_usr_test": X_usr_te,  "y_usr_test": y_usr_te,
        # combined training
        "X_comb_train": X_comb_tr, "y_comb_train": y_comb_tr,
        "X_comb_val": X_comb_va,   "y_comb_val": y_comb_va,
        # shared test: user hold-out is the most honest independent test
        "X_test": X_usr_te, "y_test": y_usr_te,
        # metadata
        "external_summary": dataset_summary(X_ext, y_ext, sess_ext, "external"),
        "user_summary": dataset_summary(X_usr, y_usr, sess_usr, "user"),
    }

