"""A small kinematic hand model that produces MediaPipe-style landmarks.

WHY THIS EXISTS
    The project's real data (the Kaggle templates and your own recordings) is not shipped
    with the code. This module is used for
      * unit tests of the palm-side geometry (is the sign convention right?), and
      * a smoke test of the experiment runner (`run_experiments.py --synthetic`).
    Numbers computed on synthetic hands say NOTHING about real accuracy. The runner stamps
    every synthetic report with a warning for that reason.

Coordinate conventions
    Model (maths) frame: X to the right, Y up, Z towards the viewer. A canonical RIGHT hand has
    its palm facing +Z (towards the viewer) with the fingers pointing up, so the thumb is at +X.
    A left hand is the mirror image (X -> -X).
    Output (MediaPipe) frame: x to the right, y DOWN, z = -Z (smaller z means closer to the
    camera), all relative to the wrist, then placed in the normalised image.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import LETTERS

# Knuckle (MCP) positions of index, middle, ring, pinky in the palm plane (Z = 0).
_MCP = np.array([[0.30, 0.95], [0.08, 1.00], [-0.12, 0.95], [-0.32, 0.85]])
_SEG = np.array([[0.45, 0.28, 0.22], [0.50, 0.30, 0.23], [0.45, 0.28, 0.22], [0.35, 0.20, 0.18]])
_SPLAY = np.array([0.08, 0.0, -0.08, -0.20])                     # sideways spread of each finger
_THUMB_CMC = np.array([0.18, 0.12, 0.05])
_THUMB_SEG = (0.35, 0.30, 0.25)

FINGER_LANDMARKS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def canonical_right_hand(flex: Sequence[float], thumb: float = 0.3) -> np.ndarray:
    """21 points (maths frame) of a right hand with the palm facing +Z.

    flex:  4 finger flexion values (index..pinky), 0 = straight, about 1.5 = fist.
           Fingers bend towards the palm side (+Z), as real fingers do.
    thumb: 0 = thumb out to the side, 1 = thumb folded across the palm.
    """
    pts = np.zeros((21, 3))
    for f in range(4):
        base = np.array([_MCP[f, 0], _MCP[f, 1], 0.0])
        pts[FINGER_LANDMARKS[f][0]] = base
        theta = 0.0
        pos = base.copy()
        for j in range(3):
            theta += flex[f] * (0.45 if j == 0 else 0.55)        # cumulative bend of this joint
            d = np.array([_SPLAY[f], np.cos(theta), np.sin(theta)])
            d /= np.linalg.norm(d)
            pos = pos + _SEG[f, j] * d
            pts[FINGER_LANDMARKS[f][j + 1]] = pos
    pos = _THUMB_CMC.copy()
    pts[1] = pos
    for j in range(3):
        curl = thumb * (0.4 + 0.3 * j)
        d = np.array([0.6 * (1 - curl) - 0.5 * curl, 0.75, 0.15 + 0.8 * curl])
        d /= np.linalg.norm(d)
        pos = pos + _THUMB_SEG[j] * d
        pts[2 + j] = pos
    return pts


def place_hand(model: np.ndarray, chirality: int, facing: int, yaw: float = 0.0, pitch: float = 0.0,
               roll: float = 0.0, scale: float = 0.25, center: Tuple[float, float] = (0.5, 0.5),
               aspect: float = 1.0, noise: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Turn a canonical model into MediaPipe-style landmarks.

    chirality: +1 right hand, -1 left hand
    facing:    +1 palm towards the camera, -1 back of the hand towards the camera
    yaw / pitch / roll: extra rotation in radians (yaw about the vertical axis etc.)
    scale:     hand size as a fraction of the image height
    aspect:    image width / height (x and z are divided by it, like MediaPipe's normalisation)
    """
    pts = model.copy()
    if chirality < 0:
        pts[:, 0] *= -1.0                                         # mirror image = left hand
    if facing < 0:
        pts = pts @ rot_y(np.pi).T                                # turn the hand over
    pts = pts @ (rot_z(roll) @ rot_x(pitch) @ rot_y(yaw)).T
    out = np.empty_like(pts)
    out[:, 0] = center[0] + pts[:, 0] * scale / aspect
    out[:, 1] = center[1] - pts[:, 1] * scale
    out[:, 2] = -pts[:, 2] * scale / aspect                       # wrist-relative depth
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        out[1:] += rng.normal(0, noise, size=out[1:].shape)
    return out.astype(np.float32)


# ------------------------------------------------------------- pose sampling
POSE_STYLES = ("open", "fist", "point", "free")


def sample_flex(style: str, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    if style == "open":
        return rng.uniform(0.0, 0.25, 4), rng.uniform(0.0, 0.4)
    if style == "fist":
        return rng.uniform(1.1, 1.6, 4), rng.uniform(0.5, 0.9)
    if style == "point":
        f = rng.uniform(1.1, 1.6, 4)
        f[0] = rng.uniform(0.0, 0.2)
        return f, rng.uniform(0.4, 0.9)
    return rng.uniform(0.0, 1.6, 4), rng.uniform(0.0, 1.0)


def make_palm_side_dataset(n_per_condition: int = 40, seed: int = 0, aspect: float = 16 / 9,
                           noise: float = 0.004, tilt_deg: float = 35.0) -> Dict[str, np.ndarray]:
    """Synthetic stand-in for data/palm_side_labels.npz (same arrays and meaning).

    The hands are noise-free apart from `noise` and a random tilt, so the geometric methods are
    expected to score close to 100 %. A real recording will be harder.
    """
    rng = np.random.default_rng(seed)
    conds = [("right_palm", [(+1, +1)]), ("right_back", [(+1, -1)]), ("left_palm", [(-1, +1)]),
             ("left_back", [(-1, -1)]), ("both_palms", [(+1, +1), (-1, +1)]),
             ("both_backs", [(+1, -1), (-1, -1)])]
    lm, mp, side, true_hand, pose, cond, block = [], [], [], [], [], [], []
    block_id = 0
    for cname, hands_spec in conds:
        for style in POSE_STYLES:
            for _ in range(n_per_condition):
                items = []
                for k, (chir, facing) in enumerate(hands_spec):
                    flex, thumb = sample_flex(style, rng)
                    a = np.deg2rad(tilt_deg)
                    h = place_hand(canonical_right_hand(flex, thumb), chir, facing,
                                   yaw=rng.uniform(-a, a), pitch=rng.uniform(-a, a), roll=rng.uniform(-a, a),
                                   scale=rng.uniform(0.2, 0.3),
                                   center=(0.5 + (0.0 if len(hands_spec) == 1 else (-0.2 if chir > 0 else 0.2)), 0.55),
                                   aspect=aspect, noise=noise, rng=rng)
                    items.append((h, chir, facing))
                items.sort(key=lambda t: float(t[0][0, 0]))        # left-most hand first (slot 0)
                L = np.full((2, 21, 3), np.nan, np.float32)
                M = np.zeros(2, np.int8); S = np.zeros(2, np.int8); T = np.zeros(2, np.int8)
                for slot, (h, chir, facing) in enumerate(items):
                    L[slot] = h
                    # MediaPipe label is "Right" for a LEFT-shaped hand in the image (it expects a mirrored picture).
                    M[slot] = -chir
                    S[slot] = facing
                    T[slot] = chir if len(hands_spec) == 1 else 0
                lm.append(L); mp.append(M); side.append(S); true_hand.append(T)
                pose.append(style); cond.append(cname); block.append(block_id)
            block_id += 1
    n = len(lm)
    return {
        "landmarks": np.stack(lm), "mp_label": np.stack(mp), "mp_score": np.ones((n, 2), np.float32),
        "side": np.stack(side), "true_hand": np.stack(true_hand),
        "pose": np.array(pose, dtype="<U16"), "condition": np.array(cond, dtype="<U32"),
        "block": np.array(block, dtype=np.int32), "aspect": np.array(aspect, dtype=np.float32),
    }


# --------------------------------------------------------- letter-like classes
def make_letter_dataset(per_letter_ext: int = 120, per_letter_user: int = 60, seed: int = 0,
                        aspect: float = 16 / 9, hard: float = 1.0):
    """Synthetic two-hand 'letters' for smoke-testing Experiments 1 and 3.

    Returns (X_ext, y_ext, X_user, y_user). Each class has a fixed pose for both hands (flexion,
    orientation, palm or back to the camera, hand spacing). The 'user' domain uses a different
    orientation bias and more noise to mimic a new signer and camera.
    """
    from utils import build_features
    rng = np.random.default_rng(seed)
    letters = [c for c in LETTERS]
    classes = {}
    for c in letters:
        classes[c] = [dict(flex=rng.uniform(0, 1.6, 4), thumb=rng.uniform(0, 1),
                           facing=int(rng.choice([-1, 1])),
                           yaw=rng.uniform(-0.5, 0.5), pitch=rng.uniform(-0.5, 0.5), roll=rng.uniform(-0.5, 0.5))
                      for _ in range(2)] + [dict(gap=rng.uniform(0.15, 0.45), dy=rng.uniform(-0.1, 0.1))]

    def sample(c, domain):
        h1, h2, rel = classes[c]
        noise = 0.004 if domain == "ext" else 0.008
        jitter = (0.10 if domain == "ext" else 0.16) * hard
        bias = 0.0 if domain == "ext" else 0.12
        hands = []
        for k, spec in enumerate((h1, h2)):
            flex = np.clip(spec["flex"] + rng.normal(0, 0.12 * hard, 4), 0, 1.7)
            hands.append(place_hand(
                canonical_right_hand(flex, np.clip(spec["thumb"] + rng.normal(0, 0.1 * hard), 0, 1)),
                chirality=+1 if k == 0 else -1, facing=spec["facing"],
                yaw=spec["yaw"] + bias + rng.normal(0, jitter), pitch=spec["pitch"] + rng.normal(0, jitter),
                roll=spec["roll"] + rng.normal(0, jitter), scale=rng.uniform(0.2, 0.28),
                center=(0.5 + (-1 if k == 0 else 1) * rel["gap"] / 2, 0.5 + (0 if k == 0 else rel["dy"])),
                aspect=aspect, noise=noise, rng=rng))
        return build_features(hands, aspect=aspect)

    X_e, y_e, X_u, y_u = [], [], [], []
    for c in letters:
        for _ in range(per_letter_ext):
            X_e.append(sample(c, "ext")); y_e.append(c)
        for _ in range(per_letter_user):
            X_u.append(sample(c, "user")); y_u.append(c)
    return np.stack(X_e), np.array(y_e), np.stack(X_u), np.array(y_u)
