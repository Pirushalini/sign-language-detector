"""Shared helpers: paths, hand tracking, feature building and chart cropping.

The importer, the calibration tool and the live app all call build_features(),
so the same preprocessing is used everywhere. If preprocessing differed between
them, matching would fail on live frames.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "assets"
CHART_PATH = ASSETS_DIR / "alphabet_chart.png"
SETTINGS_PATH = ROOT / "settings.json"
EXTERNAL_TEMPLATES = DATA_DIR / "templates_external.npz"
USER_TEMPLATES = DATA_DIR / "templates_user.npz"

# -------------------------------------------------------------- classes
LETTERS: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
IDLE = "idle"

# -------------------------------------------------------------- features
N_LANDMARKS = 21
HAND_LEN = N_LANDMARKS * 3          # x, y, z for 21 points = 63
INTER_START = 2 * HAND_LEN          # 126: wrist-to-wrist dx, dy, dz, distance
FLAGS_START = INTER_START + 4       # 130: hand 1 present, hand 2 present
FEATURE_LEN = FLAGS_START + 2       # 132

# Assumed width/height of the dataset images (a webcam or phone in landscape).
# The original project does not record the image size, so this is a setting.
DATASET_ASPECT = 16 / 9


def build_features(hands: Sequence[np.ndarray], aspect: float = 1.0) -> np.ndarray:
    """Turn up to two hands into one fixed-length feature vector.

    hands:  list of (21, 3) arrays in MediaPipe normalised coordinates.
    aspect: image width / image height. x and z are multiplied by it so that
            x and y use the same unit (otherwise hand shapes look stretched
            on non-square frames).

    Layout (132 values):
        [0:63]    hand 1 (left-most in the image), wrist-relative, scaled
        [63:126]  hand 2, same treatment (zeros if missing)
        [126:130] wrist-to-wrist vector (dx, dy, dz) and its length, divided by
                  the average hand size. This keeps how the two hands sit
                  relative to each other, which matters for touching hands.
        [130:132] presence flags for hand 1 and hand 2
    """
    feats = np.zeros(FEATURE_LEN, dtype=np.float32)
    prepared = []
    for h in list(hands)[:2]:
        arr = np.asarray(h, dtype=np.float32).reshape(N_LANDMARKS, 3).copy()
        arr[:, 0] *= aspect
        arr[:, 2] *= aspect
        prepared.append(arr)
    if not prepared:
        return feats

    # Order hands left to right. MediaPipe's own order and Left/Right labels
    # are not stable, so the wrist x-position is used instead.
    prepared.sort(key=lambda a: float(a[0, 0]))

    scales, wrists = [], []
    for i, arr in enumerate(prepared):
        wrist = arr[0].copy()
        rel = arr - wrist
        scale = max(float(np.max(np.linalg.norm(rel[:, :2], axis=1))), 1e-6)
        feats[i * HAND_LEN:(i + 1) * HAND_LEN] = (rel / scale).reshape(-1)
        feats[FLAGS_START + i] = 1.0
        scales.append(scale)
        wrists.append(wrist)

    if len(prepared) == 2:
        delta = (wrists[1] - wrists[0]) / float(np.mean(scales))
        feats[INTER_START:INTER_START + 3] = delta
        feats[INTER_START + 3] = float(np.linalg.norm(delta))
    return feats


def mirror_hands(hands: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Flip hands left-right (x -> 1 - x). Used to also match left-handed signers."""
    out = []
    for h in hands:
        arr = np.asarray(h, dtype=np.float32).reshape(N_LANDMARKS, 3).copy()
        arr[:, 0] = 1.0 - arr[:, 0]
        out.append(arr)
    return out


# ---------------------------------------------------------- hand tracking
class HandTracker:
    """Thin wrapper around MediaPipe Hands (up to two hands)."""

    def __init__(self, min_detection: float = 0.6, min_tracking: float = 0.6):
        import mediapipe as mp  # imported here so tests of pure maths stay light

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )

    def process(self, frame_bgr: np.ndarray):
        """Return (list of (21, 3) arrays, raw MediaPipe result) for a BGR frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        hands = []
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                hands.append(np.array([[p.x, p.y, p.z] for p in hl.landmark], dtype=np.float32))
        return hands, result

    def draw(self, frame_bgr: np.ndarray, result) -> None:
        """Draw hand landmarks on the frame in place."""
        if not result.multi_hand_landmarks:
            return
        mp = self._mp
        for hl in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame_bgr,
                hl,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(37, 99, 235), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

    def close(self) -> None:
        self._hands.close()


# ----------------------------------------------------------------- camera
def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera. DirectShow starts much faster on Windows."""
    if sys.platform.startswith("win"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def list_cameras(max_index: int = 4) -> List[int]:
    """Return indices of cameras that can be opened (checks 0 to max_index)."""
    found = []
    for i in range(max_index + 1):
        cap = open_camera(i)
        if cap is not None and cap.isOpened():
            found.append(i)
        if cap is not None:
            cap.release()
    return found


# ------------------------------------------------------------------ chart
# The chart is a photo of a 7 x 4 grid (A to G on the first row, and so on).
# These fractions are only used to highlight the current letter, so a small
# error from the photo's perspective does not matter.
CHART_COLS, CHART_ROWS = 7, 4
CHART_BOX = (0.048, 0.190, 0.951, 0.951)  # left, top, right, bottom (fits the british-sign.co.uk chart, 3535 x 2500)


def chart_cell(letter: str, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """Pixel rectangle (x0, y0, x1, y1) of a letter's cell, or None for non-letters."""
    letter = letter.upper()
    if letter not in LETTERS:
        return None
    row, col = divmod(LETTERS.index(letter), CHART_COLS)
    left, top, right, bottom = CHART_BOX
    cell_w = (right - left) * width / CHART_COLS
    cell_h = (bottom - top) * height / CHART_ROWS
    x0 = left * width + col * cell_w
    y0 = top * height + row * cell_h
    return int(x0), int(y0), int(x0 + cell_w), int(y0 + cell_h)
