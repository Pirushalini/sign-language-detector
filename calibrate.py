"""Guided calibration: record your own reference samples, one letter at a time.

A reference chart window highlights the current letter, so you copy the sign
from the chart. About 40 samples per letter take around 5 seconds. Samples are
saved after every letter, so you can quit and continue later.

Usage:
    python calibrate.py                       # all 26 letters
    python calibrate.py --letters H J Y       # only the letters missing from the dataset
    python calibrate.py --samples 60 --session me_evening

Keys:  SPACE record   N next   P previous   Q or ESC quit
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np

from matcher import SRC_USER, load_templates, save_templates
from utils import (CHART_PATH, FEATURE_LEN, LETTERS, USER_TEMPLATES, HandTracker,
                   build_features, chart_cell, open_camera)

WINDOW = "Calibration"
CHART_WINDOW = "Reference chart"
COUNTDOWN_S = 3
NO_HANDS_TIMEOUT_S = 12


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def load_user():
    if USER_TEMPLATES.exists():
        X, y, _, sess = load_templates(USER_TEMPLATES)
        return list(X), list(y), list(sess)
    return [], [], []


def save_user(X: List[np.ndarray], y: List[str], sessions: List[str]) -> None:
    save_templates(USER_TEMPLATES, np.stack(X) if X else np.zeros((0, FEATURE_LEN), np.float32),
                   y, [SRC_USER] * len(y), sessions)


def chart_image(letter: str):
    """Reference chart with the current letter's cell highlighted (or None)."""
    if not CHART_PATH.exists():
        return None
    img = cv2.imread(str(CHART_PATH))
    if img is None:
        return None
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    rect = chart_cell(letter, img.shape[1], img.shape[0])
    if rect:
        cv2.rectangle(img, rect[:2], rect[2:], (235, 99, 37), 4)
    return img


def draw_strip(frame, letters: List[str], current: int, done: Dict[str, int]) -> None:
    """Row of small boxes at the bottom: done, current, or still missing."""
    h, w = frame.shape[:2]
    box = min(34, (w - 20) // len(letters))
    x0 = (w - box * len(letters)) // 2
    for i, c in enumerate(letters):
        x = x0 + i * box
        color = (80, 80, 80)
        if done.get(c, 0) > 0:
            color = (60, 160, 60)
        if i == current:
            color = (235, 99, 37)
        cv2.rectangle(frame, (x, h - box - 8), (x + box - 3, h - 8), color, -1)
        cv2.putText(frame, c, (x + box // 2 - 8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--letters", nargs="+", default=LETTERS, help="letters to record (default: A to Z)")
    ap.add_argument("--samples", type=int, default=40, help="samples per letter (default 40)")
    ap.add_argument("--interval", type=float, default=0.12, help="seconds between samples")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--session", default=datetime.now().strftime("session_%Y%m%d_%H%M"))
    ap.add_argument("--append", action="store_true",
                    help="keep earlier samples of these letters instead of replacing them")
    args = ap.parse_args()

    letters = [c.upper() for c in args.letters if c.upper() in LETTERS]
    if not letters:
        raise SystemExit("No valid letters given.")

    print("Sign each letter exactly as shown in the reference chart.")
    print("Use the same hand for the same role every time (for example, the same hand does the pointing).")
    print("Keys: SPACE record, N next, P previous, Q or ESC quit.\n")

    X, y, sessions = load_user()
    done: Dict[str, int] = {c: y.count(c) for c in LETTERS}

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}. Close other apps that use the webcam, "
                         "or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = HandTracker()

    if CHART_PATH.exists():
        cv2.namedWindow(CHART_WINDOW, cv2.WINDOW_NORMAL)
    else:
        print(f"Note: {CHART_PATH} not found, so no reference chart window is shown.")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    idx = 0
    state = "ready"          # ready -> countdown -> recording -> ready
    t_state = time.time()
    t_last_sample = 0.0
    buffer: List[np.ndarray] = []
    message = ""
    shown_letter = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break
            h, w = frame.shape[:2]
            hands, result = tracker.process(frame)
            tracker.draw(frame, result)
            view = cv2.flip(frame, 1)               # mirror for display only
            letter = letters[idx]

            if letter != shown_letter:
                img = chart_image(letter)
                if img is not None:
                    cv2.imshow(CHART_WINDOW, img)
                shown_letter = letter

            now = time.time()
            if state == "countdown":
                left = COUNTDOWN_S - int(now - t_state)
                if left <= 0:
                    state, t_state, buffer = "recording", now, []
                else:
                    cv2.putText(view, str(left), (w // 2 - 40, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
                                5, (255, 255, 255), 10, cv2.LINE_AA)
            elif state == "recording":
                if hands and now - t_last_sample >= args.interval:
                    buffer.append(build_features(hands, aspect=w / h))
                    t_last_sample = now
                if len(buffer) >= args.samples:
                    # Replace this letter's older samples, then save straight away.
                    if not args.append:
                        keep = [i for i, c in enumerate(y) if c != letter]
                        X, y, sessions = [X[i] for i in keep], [y[i] for i in keep], [sessions[i] for i in keep]
                    X += buffer
                    y += [letter] * len(buffer)
                    sessions += [args.session] * len(buffer)
                    save_user(X, y, sessions)
                    done[letter] = y.count(letter)
                    message = f"Saved {len(buffer)} samples for {letter}"
                    state, buffer = "ready", []
                    if idx < len(letters) - 1:
                        idx += 1
                elif now - t_state > NO_HANDS_TIMEOUT_S and len(buffer) < args.samples // 2:
                    message = "Not enough hand frames. Show your hands clearly and try again."
                    state, buffer = "ready", []

            # ---------- overlay
            put_text(view, f"Letter {letter}", (20, 60), 2.0, (255, 255, 255), 4)
            put_text(view, f"{idx + 1} of {len(letters)}   hands detected: {len(hands)}", (20, 100), 0.7)
            if state == "ready":
                put_text(view, "SPACE: record   N: next   P: previous   Q: quit", (20, 135), 0.65, (200, 230, 255))
            elif state == "recording":
                put_text(view, f"Recording {len(buffer)}/{args.samples}  (move your hands slightly)",
                         (20, 135), 0.7, (120, 255, 120))
            if message:
                put_text(view, message, (20, 170), 0.65, (120, 220, 255))
            draw_strip(view, letters, idx, done)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and state == "ready":
                state, t_state, message = "countdown", now, ""
            elif key in (ord("n"), ord(".")) and state == "ready":
                idx, message = min(idx + 1, len(letters) - 1), ""
            elif key in (ord("p"), ord(",")) and state == "ready":
                idx, message = max(idx - 1, 0), ""
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    total = {c: y.count(c) for c in LETTERS if y.count(c)}
    print(f"\nSaved to {USER_TEMPLATES}")
    print("Your samples per letter:", total)
    print("Next: python evaluate.py   then   python app.py")


if __name__ == "__main__":
    main()
