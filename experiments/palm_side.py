"""Palm-side identification (Experiment 2).

Question: is the PALM or the BACK of each hand turned towards the camera?

Geometry (the key idea)
    Look at the wrist and the knuckles of the index finger (landmark 5) and the pinky (17).
    In the picture, a hand shows a signed area
            s = cross2(p5 - p0, p17 - p0) / (|p5 - p0| * |p17 - p0|)        (range -1 to 1)
    Its sign depends on TWO things at once: which hand it is (right- or left-shaped) and which
    side faces the camera. Turn the hand over, or swap right for left, and the sign flips.
    From the synthetic hand model (tests/test_experiments.py checks this):
            palm towards the camera   <=>   s * chirality < 0
    with chirality = +1 for a right-shaped hand and -1 for a left-shaped hand (y axis pointing down).
    So the side can be read from s once the chirality is known. The methods differ in HOW they get it:

    G1  label      chirality from MediaPipe's own Left/Right label (needs the label)
    G2  slot       chirality from the hand's position: the hand on the left of the picture is taken as
                   the signer's right hand (needs two hands in a natural arrangement; no label)
    G3  3D         chirality from the 3D landmarks themselves: the thumb sits on the palm side of the
                   palm plane, so sign((p5 x p17) . thumb) tells right from left (no label, no assumption
                   about arrangement, but relies on MediaPipe's z values)
    G4  kNN        nearest recorded examples with known side vote (instance-based, like the letter matcher,
                   but needs labelled recordings)

All methods return a value v in [-1, 1] per hand: +1 = palm towards the camera, -1 = back towards the
camera, near 0 = unsure (for example a hand seen edge-on). A hand is "decided" when |v| >= delta.

Everything here works on wrist-relative hands of shape (N, 2, 21, 3), as produced by
`hands_from_features` (from the stored 132-value templates) or `hands_from_landmarks` (raw recordings).
Slot 0 is the hand on the left of the picture, slot 1 the right one, like utils.build_features.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from utils import FLAGS_START, HAND_LEN, N_LANDMARKS

TAU = 0.35            # |s| at which the value saturates to +-1
SLOT_CHIRALITY = (+1, -1)   # slot 0 = signer's right hand (right-shaped), slot 1 = left hand


# ------------------------------------------------------------------ hands in
def hands_from_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Wrist-relative hands (N, 2, 21, 3) and presence flags (N, 2) from 132-value feature vectors."""
    X = np.atleast_2d(X)
    hands = X[:, :2 * HAND_LEN].reshape(len(X), 2, N_LANDMARKS, 3).astype(np.float32)
    present = X[:, FLAGS_START:FLAGS_START + 2] > 0.5
    return hands, present


def hands_from_landmarks(landmarks: np.ndarray, aspect: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Same, from raw landmarks (N, 2, 21, 3) with NaN for a missing hand (the recording format).

    Uses the same steps as utils.build_features: x and z times the aspect, move the wrist to the origin,
    divide by the largest wrist distance in the image plane.
    """
    L = np.asarray(landmarks, dtype=np.float32).copy()
    present = ~np.isnan(L[:, :, 0, 0])
    L = np.nan_to_num(L, nan=0.0)
    L[..., 0] *= aspect
    L[..., 2] *= aspect
    rel = L - L[:, :, :1, :]
    scale = np.maximum(np.max(np.linalg.norm(rel[..., :2], axis=-1), axis=-1), 1e-6)
    return rel / scale[..., None, None], present


# ------------------------------------------------------------ raw measurements
def signed_area(hands: np.ndarray) -> np.ndarray:
    """Normalised signed area s of wrist, index knuckle and pinky knuckle in the picture, shape (N, 2)."""
    a = hands[:, :, 5, :2]
    b = hands[:, :, 17, :2]
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return cross / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)


def thumb_triple_product(hands: np.ndarray) -> np.ndarray:
    """Cosine between the palm normal (index knuckle x pinky knuckle) and the thumb tip, shape (N, 2).

    Computed in the maths frame (x right, y up, z towards the viewer; MediaPipe has y down and z towards
    the camera negative). It does not depend on how the hand is rotated, only on whether it is a right-
    or a left-shaped hand: > 0 right-shaped, < 0 left-shaped, about 0 when the thumb lies in the palm plane.
    """
    h = hands * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    n = np.cross(h[:, :, 5, :], h[:, :, 17, :])
    t = h[:, :, 4, :]
    num = np.sum(n * t, axis=-1)
    den = np.maximum(np.linalg.norm(n, axis=-1) * np.linalg.norm(t, axis=-1), 1e-9)
    return num / den


def _value(s: np.ndarray, chirality: np.ndarray, present: np.ndarray, tau: float) -> np.ndarray:
    v = np.clip(-chirality * s / tau, -1.0, 1.0)
    return np.where(present, v, 0.0).astype(np.float32)


# ------------------------------------------------------------------- methods
def g1_label(hands, present, mp_label, tau: float = TAU) -> np.ndarray:
    """G1: chirality from MediaPipe's label. mp_label is +1 for 'Right' and -1 for 'Left' (0 = unknown).

    MediaPipe expects a mirrored picture, so on a normal picture its 'Left' means a right-SHAPED hand.
    Hence chirality = -mp_label. (Experiment 2 checks this on your own recordings.)
    """
    chir = -np.asarray(mp_label, dtype=np.float32)
    return _value(signed_area(hands), chir, present & (chir != 0), tau)


def g2_slot(hands, present, tau: float = TAU, slot_chirality=SLOT_CHIRALITY) -> np.ndarray:
    """G2: chirality from the position of the hand in the picture (slot 0 = left in the image)."""
    chir = np.broadcast_to(np.asarray(slot_chirality, dtype=np.float32), present.shape)
    return _value(signed_area(hands), chir, present, tau)


def g3_intrinsic(hands, present, tau: float = TAU, min_cos: float = 0.05) -> np.ndarray:
    """G3: chirality from the 3D landmarks (thumb relative to the palm plane)."""
    c = thumb_triple_product(hands)
    chir = np.sign(c)
    v = _value(signed_area(hands), chir, present & (np.abs(c) >= 1e-9), tau)
    return v * np.minimum(1.0, np.abs(c) / min_cos)         # a thumb in the palm plane gives little trust


class KNNSide:
    """G4: instance-based side classifier. Examples of the same hand slot vote (k nearest)."""

    def __init__(self, k: int = 7):
        self.k = k
        self.F = [np.zeros((0, HAND_LEN), np.float32)] * 2
        self.side = [np.zeros(0, np.float32)] * 2

    def fit(self, hands, present, side) -> "KNNSide":
        for slot in range(2):
            m = present[:, slot] & (side[:, slot] != 0)
            self.F[slot] = hands[m, slot].reshape(m.sum(), -1).astype(np.float32)
            self.side[slot] = side[m, slot].astype(np.float32)
        return self

    def score(self, hands, present) -> np.ndarray:
        v = np.zeros(present.shape, dtype=np.float32)
        for slot in range(2):
            m = np.flatnonzero(present[:, slot])
            if len(m) == 0 or len(self.side[slot]) == 0:
                continue
            Q = hands[m, slot].reshape(len(m), -1)
            R = self.F[slot]
            d2 = (Q * Q).sum(1, keepdims=True) - 2 * Q @ R.T + (R * R).sum(1)[None, :]
            kk = min(self.k, R.shape[0])
            idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
            v[m, slot] = self.side[slot][idx].mean(axis=1)
        return v


# ---------------------------------------------------------------- evaluation
def decide(v: np.ndarray, delta: float) -> np.ndarray:
    """+1 palm, -1 back, 0 unsure."""
    return np.where(np.abs(v) >= delta, np.sign(v), 0).astype(np.int8)


def evaluate_side(v: np.ndarray, side_true: np.ndarray, present: np.ndarray, delta: float = 0.15,
                  mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Accuracy of one method over all labelled, present hands (every hand counts once).

    coverage           share of hands the method dared to decide
    accuracy_decided   correct among decided hands
    accuracy_all       correct among ALL hands (unsure counts as wrong): the strict number
    recall_palm/back   share of palm / back hands found
    precision_palm/back  how often a 'palm' / 'back' answer is right
    """
    m = present & (side_true != 0)
    if mask is not None:
        m = m & mask
    truth = side_true[m]
    pred = decide(v, delta)[m]
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    decided = pred != 0
    correct = pred == truth

    def rate(a, b):
        return float(a / b) if b else float("nan")

    out = {
        "n": n,
        "coverage": rate(decided.sum(), n),
        "accuracy_decided": rate((correct & decided).sum(), decided.sum()),
        "accuracy_all": rate(correct.sum(), n),
        "recall_palm": rate(((pred == 1) & (truth == 1)).sum(), (truth == 1).sum()),
        "recall_back": rate(((pred == -1) & (truth == -1)).sum(), (truth == -1).sum()),
        "precision_palm": rate(((pred == 1) & (truth == 1)).sum(), (pred == 1).sum()),
        "precision_back": rate(((pred == -1) & (truth == -1)).sum(), (pred == -1).sum()),
    }
    # confusion: rows = true (palm, back), columns = predicted (palm, back, unsure)
    out["confusion"] = [[int(((truth == t) & (pred == p)).sum()) for p in (1, -1, 0)] for t in (1, -1)]
    return out


def chirality_accuracy(hands, present, mp_label, true_hand) -> Dict[str, float]:
    """How often is each chirality source right? Uses frames where the physical hand is known
    (single-hand recordings). True shape chirality of a physical hand on a normal picture is +1 right, -1 left.
    """
    m = present & (true_hand != 0)
    if not m.any():
        return {}
    out = {"n": int(m.sum())}
    out["label_correct"] = float(np.mean((-np.asarray(mp_label)[m]) == true_hand[m]))
    out["label_correct_if_swapped"] = float(np.mean((np.asarray(mp_label)[m]) == true_hand[m]))
    slot_c = np.broadcast_to(np.asarray(SLOT_CHIRALITY), present.shape)
    out["slot_correct"] = float(np.mean(slot_c[m] == true_hand[m]))
    out["intrinsic_correct"] = float(np.mean(np.sign(thumb_triple_product(hands))[m] == true_hand[m]))
    return out


def side_values_for_templates(X: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Palm-side values (N, 2) for stored 132-value features, using the slot rule (G2).

    Template files carry no MediaPipe label, so Experiment 3 uses the slot rule for templates and
    for queries alike. Hands that are not present get 0.
    """
    hands, present = hands_from_features(X)
    return g2_slot(hands, present, tau)
