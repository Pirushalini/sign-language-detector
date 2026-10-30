"""Format of the labelled palm-side recordings (data/palm_side_labels.npz) and helpers.

Arrays (N frames, hand slots 0 = left of the picture, 1 = right of the picture):
    landmarks   (N, 2, 21, 3) float32   raw MediaPipe coordinates; NaN where a hand is absent
    mp_label    (N, 2) int8             MediaPipe label: +1 'Right', -1 'Left', 0 absent
    mp_score    (N, 2) float32          MediaPipe's confidence for that label
    side        (N, 2) int8             GROUND TRUTH: +1 palm towards the camera, -1 back towards the camera, 0 absent
    true_hand   (N, 2) int8             GROUND TRUTH physical hand for single-hand clips: +1 right, -1 left, 0 unknown/absent
    pose        (N,)   str              pose style that was asked for
    condition   (N,)   str              e.g. 'right_palm', 'both_backs'
    block       (N,)   int32            one id per recording run (used to split train / test without leaks)
    aspect      ()     float32          image width / height

The recording tool (record_palm_side.py) writes this file; experiments/run_experiments.py reads it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# condition name -> list of (physical hand: +1 right / -1 left, side: +1 palm / -1 back). One entry = one hand in the picture.
CONDITIONS: Dict[str, List[Tuple[int, int]]] = {
    "right_palm": [(+1, +1)],
    "right_back": [(+1, -1)],
    "left_palm": [(-1, +1)],
    "left_back": [(-1, -1)],
    "both_palms": [(+1, +1), (-1, +1)],
    "both_backs": [(+1, -1), (-1, -1)],
}
CONDITION_TEXT = {
    "right_palm": "RIGHT hand only, PALM facing the camera",
    "right_back": "RIGHT hand only, BACK of the hand facing the camera",
    "left_palm": "LEFT hand only, PALM facing the camera",
    "left_back": "LEFT hand only, BACK of the hand facing the camera",
    "both_palms": "BOTH hands, both PALMS facing the camera",
    "both_backs": "BOTH hands, both BACKS facing the camera",
}
POSES = {
    "open": "open flat hand, fingers apart",
    "fist": "closed fist",
    "point": "index finger pointing, other fingers folded",
    "free": "any finger-spelling letters, change shape and tilt a little",
}


def hand_items(hands: Sequence[np.ndarray], result) -> List[Tuple[np.ndarray, int, float]]:
    """(landmarks, mp_label, mp_score) for every detected hand of a MediaPipe result."""
    items = []
    labels = getattr(result, "multi_handedness", None) or []
    for i, h in enumerate(hands):
        if i < len(labels):
            c = labels[i].classification[0]
            items.append((np.asarray(h, np.float32), +1 if c.label == "Right" else -1, float(c.score)))
        else:
            items.append((np.asarray(h, np.float32), 0, 0.0))
    return items


def pack_frame(items: Sequence[Tuple[np.ndarray, int, float]]):
    """Order up to two hands left to right (like utils.build_features) and pad to two slots."""
    items = sorted(items[:2], key=lambda t: float(t[0][0, 0]))
    L = np.full((2, 21, 3), np.nan, np.float32)
    M = np.zeros(2, np.int8)
    S = np.zeros(2, np.float32)
    for slot, (h, lab, score) in enumerate(items):
        L[slot], M[slot], S[slot] = h, lab, score
    return L, M, S


def labels_for(condition: str, n_hands_seen: int):
    """Ground-truth (side, true_hand) per slot for a frame of this condition, or None if the hand count is wrong.

    Two-hand conditions give the same side to both hands; the physical hand is not known per slot (0).
    """
    spec = CONDITIONS[condition]
    if n_hands_seen != len(spec):
        return None
    side = np.zeros(2, np.int8)
    true_hand = np.zeros(2, np.int8)
    for k, (hand, s) in enumerate(spec):
        side[k] = s
        if len(spec) == 1:
            true_hand[k] = hand
    return side, true_hand


def save_labels(path: Path, rec: Dict[str, list], aspect: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        landmarks=np.stack(rec["landmarks"]).astype(np.float32), mp_label=np.stack(rec["mp_label"]).astype(np.int8),
        mp_score=np.stack(rec["mp_score"]).astype(np.float32), side=np.stack(rec["side"]).astype(np.int8),
        true_hand=np.stack(rec["true_hand"]).astype(np.int8), pose=np.array(rec["pose"], dtype="<U16"),
        condition=np.array(rec["condition"], dtype="<U32"), block=np.array(rec["block"], dtype=np.int32),
        aspect=np.array(aspect, dtype=np.float32))


def load_labels(path: Path) -> Tuple[Dict[str, list], float]:
    """Read an existing file back into the list form used while recording (so new clips can be appended)."""
    with np.load(path, allow_pickle=False) as d:
        rec = {k: list(d[k]) for k in ("landmarks", "mp_label", "mp_score", "side", "true_hand", "pose", "condition", "block")}
        return rec, float(d["aspect"])


def empty_record() -> Dict[str, list]:
    return {k: [] for k in ("landmarks", "mp_label", "mp_score", "side", "true_hand", "pose", "condition", "block")}
