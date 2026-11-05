"""Datasets and evaluation protocols for Experiments 1 and 3.

Three protocols (they answer different questions, so they are reported separately):

  A  external held-out   random 80/20 split of the imported templates (same as evaluate.py).
                         OPTIMISTIC: neighbouring frames of one sign are near-identical.
  B  external -> user    templates = all imported data, queries = your own recordings.
                         The honest 'new signer, new camera' test. Letters that are missing from
                         the import (H, J, Y) cannot be tested here.
  C  combined, block CV  templates = imported data + your own recordings, minus one held-out BLOCK of your
                         recordings; queries = that block. This is how the app really runs (it uses both
                         template files) and the primary protocol. Blocks are consecutive chunks of each
                         letter's recording (a letter's ~60 samples are cut into K parts) so near-identical
                         neighbouring frames never sit on both sides of the split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from matcher import SRC_EXTERNAL, SRC_USER, load_templates
from utils import EXTERNAL_TEMPLATES, LETTERS, USER_TEMPLATES

LETTER_INDEX = {c: i for i, c in enumerate(LETTERS)}


@dataclass
class Dataset:
    X: np.ndarray            # (N, 132)
    y: np.ndarray            # (N,) letters
    src: np.ndarray          # (N,) SRC_EXTERNAL / SRC_USER
    block: np.ndarray        # (N,) block id of user samples, -1 for imported ones
    description: str = ""
    synthetic: bool = False

    @property
    def y_idx(self) -> np.ndarray:
        return np.array([LETTER_INDEX[c] for c in self.y])

    @property
    def ext(self) -> np.ndarray:
        return np.flatnonzero(self.src == SRC_EXTERNAL)

    @property
    def user(self) -> np.ndarray:
        return np.flatnonzero(self.src == SRC_USER)


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray


def assign_blocks(y_user: Sequence[str], n_blocks: int) -> np.ndarray:
    """Cut each letter's samples (in recording order) into n_blocks consecutive parts."""
    y_user = np.asarray(y_user)
    block = np.zeros(len(y_user), dtype=np.int32)
    for c in np.unique(y_user):
        idx = np.flatnonzero(y_user == c)
        block[idx] = np.minimum((np.arange(len(idx)) * n_blocks) // len(idx), n_blocks - 1)
    return block


def from_arrays(X_ext, y_ext, X_user, y_user, n_blocks: int = 5, description: str = "",
                synthetic: bool = False) -> Dataset:
    X_ext = np.asarray(X_ext, np.float32).reshape(-1, 132) if len(X_ext) else np.zeros((0, 132), np.float32)
    X_user = np.asarray(X_user, np.float32).reshape(-1, 132) if len(X_user) else np.zeros((0, 132), np.float32)
    X = np.vstack([X_ext, X_user])
    y = np.concatenate([np.asarray(y_ext, dtype="<U8"), np.asarray(y_user, dtype="<U8")])
    src = np.concatenate([np.full(len(X_ext), SRC_EXTERNAL, np.int8), np.full(len(X_user), SRC_USER, np.int8)])
    block = np.concatenate([np.full(len(X_ext), -1, np.int32), assign_blocks(y_user, n_blocks)])
    return Dataset(X, y, src, block, description, synthetic)


def load_real(n_blocks: int = 5) -> Dataset:
    """Load data/templates_external.npz and data/templates_user.npz (either may be missing)."""
    Xe, ye = np.zeros((0, 132), np.float32), np.array([], dtype="<U8")
    Xu, yu = np.zeros((0, 132), np.float32), np.array([], dtype="<U8")
    parts = []
    if EXTERNAL_TEMPLATES.exists():
        Xe, ye, _, _ = load_templates(EXTERNAL_TEMPLATES)
        parts.append(f"imported {len(ye)}")
    if USER_TEMPLATES.exists():
        Xu, yu, _, _ = load_templates(USER_TEMPLATES)
        parts.append(f"own recordings {len(yu)}")
    if not parts:
        raise SystemExit("No template files found in data/. Run import_dataset.py and/or calibrate.py first, "
                         "or use --synthetic for a pipeline check.")
    return from_arrays(Xe, ye, Xu, yu, n_blocks, description=", ".join(parts))


# ------------------------------------------------------------------ protocols
def stratified_split(y: np.ndarray, idx: np.ndarray, test_fraction: float, seed: int):
    """Same logic as evaluate.split_indices but on a subset of indices."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c].copy()
        rng.shuffle(members)
        n_test = max(1, int(len(members) * test_fraction))
        test.extend(members[:n_test])
        train.extend(members[n_test:])
    return np.array(train, dtype=int), np.array(test, dtype=int)


def protocol_a(ds: Dataset, seed: int = 0) -> List[Fold]:
    if len(ds.ext) == 0:
        return []
    tr, te = stratified_split(ds.y, ds.ext, 0.2, seed)
    return [Fold("A", tr, te)]


def protocol_b(ds: Dataset) -> List[Fold]:
    if len(ds.ext) == 0 or len(ds.user) == 0:
        return []
    shared = np.isin(ds.y[ds.user], np.unique(ds.y[ds.ext]))
    return [Fold("B", ds.ext, ds.user[shared])]


def protocol_c(ds: Dataset) -> List[Fold]:
    if len(ds.user) == 0:
        return []
    blocks = np.unique(ds.block[ds.user])
    if len(blocks) < 2:
        return []
    folds = []
    for b in blocks:
        test = ds.user[ds.block[ds.user] == b]
        train = np.concatenate([ds.ext, ds.user[ds.block[ds.user] != b]])
        folds.append(Fold(f"C{b}", train, test))
    return folds


def all_protocols(ds: Dataset) -> Dict[str, List[Fold]]:
    out = {"A": protocol_a(ds), "B": protocol_b(ds), "C": protocol_c(ds)}
    return {k: v for k, v in out.items() if v}


PROTOCOL_TITLES = {
    "A": "A: imported data, random held-out split (optimistic)",
    "B": "B: imported data -> your own recordings (new signer)",
    "C": "C: imported + own recordings, block cross-validation (deployment-like)",
}


def primary_protocol(protocols: Dict[str, List[Fold]]) -> str:
    for k in ("C", "B", "A"):
        if k in protocols:
            return k
    raise SystemExit("No usable protocol: need imported and/or own templates.")


def groups_for(ds: Dataset, idx: np.ndarray) -> np.ndarray:
    """Resampling group of each query: letter x block for own recordings, the sample itself otherwise."""
    g = np.empty(len(idx), dtype=np.int64)
    for n, i in enumerate(idx):
        g[n] = LETTER_INDEX[ds.y[i]] * 1000 + ds.block[i] if ds.block[i] >= 0 else 10_000_000 + int(i)
    return g


# ---------------------------------------------------------- inner folds for tuning
def cap_per_letter(ds: Dataset, idx: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Keep at most `cap` imported templates per letter (own recordings are always kept)."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(ds.y[idx]):
        members = idx[(ds.y[idx] == c) & (ds.src[idx] == SRC_EXTERNAL)]
        if len(members) > cap:
            members = rng.choice(members, cap, replace=False)
        keep.extend(members)
    keep.extend(idx[ds.src[idx] == SRC_USER])
    return np.array(sorted(keep), dtype=int)


def inner_folds(ds: Dataset, train: np.ndarray, ext_cap: int = 100) -> List[Fold]:
    """Validation folds INSIDE a training set, used to pick weights without touching the real test queries."""
    train = cap_per_letter(ds, train, ext_cap)
    user_train = train[ds.src[train] == SRC_USER]
    blocks = np.unique(ds.block[user_train]) if len(user_train) else []
    if len(blocks) >= 2:
        return [Fold(f"inner{b}", train[~np.isin(train, user_train[ds.block[user_train] == b])],
                     user_train[ds.block[user_train] == b]) for b in blocks]
    ext_train = train[ds.src[train] == SRC_EXTERNAL]
    folds = []
    for seed in (11, 12):
        tr, te = stratified_split(ds.y, ext_train, 0.2, seed)
        folds.append(Fold(f"inner_r{seed}", tr, te))
    return folds
