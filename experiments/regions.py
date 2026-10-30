"""Landmark regions and per-feature weight profiles (Experiment 1).

MediaPipe numbers the 21 landmarks of a hand like this:

    0            wrist
    1  2  3  4   thumb  (CMC, MCP, IP, TIP)
    5  6  7  8   index  (MCP, PIP, DIP, TIP)
    9 10 11 12   middle
   13 14 15 16   ring
   17 18 19 20   pinky

Regions used in the experiment
    fingertips      4, 8, 12, 16, 20
    finger joints   2, 3, 6, 7, 10, 11, 14, 15, 18, 19   (every non-tip joint of the fingers)
    palm and wrist  0, 1, 5, 9, 13, 17                   (wrist, thumb base, the four knuckles)
    inter-hand      the 4 wrist-to-wrist values (dx, dy, dz, distance)

The 132-value feature vector (see utils.build_features) is wrist-relative, so the
wrist itself is always (0, 0, 0). It therefore carries no information, and the
"palm and wrist" region really means "the five palm-base points".

The two presence flags (columns 130, 131) are not a region of the experiment. They only stop
a one-hand frame from matching a two-hand template, so they keep the same weight in every strategy.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from utils import FEATURE_LEN, FLAGS_START, HAND_LEN, INTER_START, N_LANDMARKS

FINGERTIPS: Tuple[int, ...] = (4, 8, 12, 16, 20)
FINGER_JOINTS: Tuple[int, ...] = (2, 3, 6, 7, 10, 11, 14, 15, 18, 19)
PALM_WRIST: Tuple[int, ...] = (0, 1, 5, 9, 13, 17)

LANDMARK_REGIONS: Dict[str, Tuple[int, ...]] = {
    "fingertips": FINGERTIPS,
    "finger_joints": FINGER_JOINTS,
    "palm_wrist": PALM_WRIST,
}
REGION_NAMES = ("fingertips", "finger_joints", "palm_wrist", "inter_hand")

# Number of landmarks that actually vary (the wrist is always 0).
EFFECTIVE_COUNT = {"fingertips": 5, "finger_joints": 10, "palm_wrist": 5}

DEFAULT_FLAG_WEIGHT = 3.0   # same as TemplateMatcher.flag_weight


def _check_partition() -> None:
    covered = sorted(FINGERTIPS + FINGER_JOINTS + PALM_WRIST)
    assert covered == list(range(N_LANDMARKS)), "regions must cover all 21 landmarks exactly once"


_check_partition()


def region_columns(region: str) -> np.ndarray:
    """Column indices (in the 132-vector) that belong to a region, for both hands."""
    if region == "inter_hand":
        return np.arange(INTER_START, FLAGS_START)
    cols = []
    for hand in range(2):
        for lm in LANDMARK_REGIONS[region]:
            base = hand * HAND_LEN + lm * 3
            cols.extend([base, base + 1, base + 2])
    return np.array(sorted(cols))


def feature_weights(tip: float = 1.0, joint: float = 1.0, palm: float = 1.0, inter: float = 1.0,
                    z: float = 1.0, flags: float = DEFAULT_FLAG_WEIGHT) -> np.ndarray:
    """Build a 132-long weight vector from four region weights.

    z multiplies the depth (z) columns on top of the region weight. MediaPipe's z is
    the least reliable coordinate, so z < 1 is a natural thing to test.
    """
    w = np.ones(FEATURE_LEN, dtype=np.float32)
    for name, value in (("fingertips", tip), ("finger_joints", joint), ("palm_wrist", palm)):
        w[region_columns(name)] = value
    w[INTER_START:FLAGS_START] = inter
    w[FLAGS_START:] = flags
    if z != 1.0:
        for hand in range(2):
            w[hand * HAND_LEN + 2: (hand + 1) * HAND_LEN: 3] *= z   # every third value is a z
        w[INTER_START + 2] *= z                                      # dz between the wrists
    return w


def balanced_profile() -> Tuple[float, float, float, float]:
    """Weights that give every landmark region the same total influence.

    Plain uniform weighting lets the 10 joint points count twice as much as the 5 fingertips.
    Here each region weight is 1 / (number of varying landmarks), scaled so that the average
    landmark weight is 1. The inter-hand weight is kept at 1.
    """
    inv = {r: 1.0 / n for r, n in EFFECTIVE_COUNT.items()}
    total_landmarks = sum(EFFECTIVE_COUNT.values())
    norm = total_landmarks / sum(inv[r] * EFFECTIVE_COUNT[r] for r in inv)
    return inv["fingertips"] * norm, inv["finger_joints"] * norm, inv["palm_wrist"] * norm, 1.0


# Strategies compared in Experiment 1: name -> (tip, joint, palm, inter).
# They are fixed BEFORE looking at any results (the 'tuned' one is found by search).
PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "uniform": (1.0, 1.0, 1.0, 1.0),
    "repo_default": (1.0, 1.0, 1.0, 2.0),          # what matcher.py does today
    "region_balanced": balanced_profile(),
    "tips_emphasis": (2.0, 1.0, 0.5, 2.0),
    "palm_emphasis": (0.5, 1.0, 2.0, 2.0),
    "inter_emphasis": (1.0, 1.0, 1.0, 4.0),
}

# One-factor-at-a-time sweep: each region alone is scaled while the others stay at 1.
SWEEP_VALUES = (0.0, 0.5, 1.0, 2.0, 4.0)
# Grid for the cross-validated search.
GRID_VALUES = (0.5, 1.0, 2.0)
