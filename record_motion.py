"""Dynamic gesture recording tool for British Sign Language (BSL) fingerspelling.

Specifically designed for movement-based letters (H and J).
Records complete multi-frame movement sequences, normalizes palm-relative
trajectories, allows instant preview/accept/discard, and stores them in
data/motion_templates.npz.

Usage:
    python record_motion.py                  # record both H and J (25 takes each)
    python record_motion.py --letters H      # record only H
    python record_motion.py --letters J      # record only J
    python record_motion.py --samples 30 --camera 0

Keys:
    In READY:      SPACE start countdown   N next letter   P prev letter   Q quit
    In RECORDING:  SPACE stop recording (or let it auto-stop when movement finishes)
    In REVIEW:     A accept take   D discard/retry   SPACE accept & next
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from motion import (DYNAMIC_LETTERS, FEATURE_DIM, MOTION_TEMPLATES, N_RESAMPLE_POINTS,
                    compute_palm_frame, extract_motion_features, identify_hands,
                    load_motion_templates, resample_trajectory, save_motion_templates)
from utils import HandTracker, open_camera

WINDOW = "BSL Dynamic Gesture Recorder"
COUNTDOWN_SECONDS = 3

INSTRUCTIONS = {
    "H": (
        "Letter H (Sweep across palm):",
        "1. Open non-dominant hand flat, palm facing up.",
        "2. Place flat dominant hand across the palm near the wrist.",
        "3. Sweep forward across the palm and off the fingertips.",
    ),
    "J": (
        "Letter J (Trace middle finger to thumb):",
        "1. Open non-dominant hand flat, palm facing up.",
        "2. Touch dominant index finger to tip of middle finger.",
        "3. Trace down the middle finger into palm and curve toward thumb.",
    ),
}


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.7,
             color: Tuple[int, int, int] = (255, 255, 255), thick: int = 2) -> None:
    """Draw anti-aliased text with a dark drop shadow for high visibility."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, letter: str, state: str, sample_idx: int, total_samples: int,
             hands_count: int, message: str, trail: List[Tuple[int, int]]) -> None:
    """Draw recording status, instructions, and trajectory trail on the frame."""
    h, w = frame.shape[:2]

    # Draw moving fingertip trail
    if len(trail) > 1:
        for i in range(1, len(trail)):
            color = (0, int(255 * (i / len(trail))), 255)
            thickness = max(2, int(4 * (i / len(trail))))
            cv2.line(frame, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)

    # Top banner card
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 160), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Letter title and progress
    put_text(frame, f"BSL Movement Letter: {letter}", (24, 46), 1.1, (255, 255, 255), 3)
    put_text(frame, f"Sample: {sample_idx + 1} of {total_samples}", (420, 46), 0.8, (120, 220, 255), 2)
    hands_col = (100, 255, 100) if hands_count == 2 else (0, 165, 255)
    put_text(frame, f"Hands: {hands_count}/2", (640, 46), 0.8, hands_col, 2)

    # Instructions for current letter
    inst = INSTRUCTIONS.get(letter, ("", "", ""))
    put_text(frame, inst[1], (24, 82), 0.65, (220, 225, 230), 1)
    put_text(frame, inst[2], (24, 110), 0.65, (220, 225, 230), 1)

    # State status prompt
    if state == "READY":
        put_text(frame, "Press SPACE to start countdown   |   N: next letter   P: prev letter   Q: quit",
                 (24, 144), 0.6, (140, 230, 255), 1)
    elif state == "COUNTDOWN":
        put_text(frame, "Get ready! Position hands in starting pose...", (24, 144), 0.65, (255, 200, 0), 2)
    elif state == "RECORDING":
        put_text(frame, "RECORDING: Perform movement now! (SPACE to finish early)", (24, 144), 0.7, (0, 255, 100), 2)
    elif state == "REVIEW":
        put_text(frame, "Review: [A / SPACE] Accept & Save    [D] Discard & retry", (24, 144), 0.7, (0, 255, 255), 2)

    # Bottom status message
    if message:
        cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), (30, 30, 30), -1)
        put_text(frame, message, (24, h - 22), 0.7, (255, 255, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--letters", nargs="+", default=DYNAMIC_LETTERS,
                        help="letters to record (default: H J)")
    parser.add_argument("--samples", type=int, default=25,
                        help="target recordings per letter (default: 25)")
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam index (default: 0)")
    parser.add_argument("--append", action="store_true",
                        help="keep existing recorded samples instead of replacing")
    args = parser.parse_args()

    letters = [c.upper() for c in args.letters if c.upper() in DYNAMIC_LETTERS]
    if not letters:
        letters = DYNAMIC_LETTERS

    # Load existing recordings
    existing_trajs, existing_labels = load_motion_templates(MOTION_TEMPLATES)
    saved_data: Dict[str, List[np.ndarray]] = {c: [] for c in DYNAMIC_LETTERS}
    for t, l in zip(existing_trajs, existing_labels):
        if l in saved_data:
            saved_data[l].append(t)

    print(f"\nBSL Dynamic Movement Recorder")
    print(f"Target letters: {letters}")
    print(f"Samples per letter: {args.samples}")
    print(f"Current saved samples: { {c: len(saved_data[c]) for c in DYNAMIC_LETTERS} }\n")

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}. Check webcam connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = HandTracker()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    curr_idx = 0
    state = "READY"  # READY -> COUNTDOWN -> RECORDING -> REVIEW
    t_state_start = time.monotonic()
    message = "Position both hands in view to start."

    recorded_features: List[np.ndarray] = []
    trail_points: List[Tuple[int, int]] = []
    review_candidate: Optional[np.ndarray] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Webcam frame read failed.")
                break

            h, w = frame.shape[:2]
            hands, result = tracker.process(frame)
            tracker.draw(frame, result)
            view = cv2.flip(frame, 1)  # mirror for natural viewing

            letter = letters[curr_idx]
            now = time.monotonic()

            # Fingertip position for live HUD trail
            fingertip_px: Optional[Tuple[int, int]] = None
            if len(hands) >= 2:
                try:
                    moving, supporting, _, _ = identify_hands(hands)
                    # Convert moving index tip to mirrored pixel coordinates
                    idx_pt = moving[8]
                    px = int((1.0 - idx_pt[0]) * w)
                    py = int(idx_pt[1] * h)
                    fingertip_px = (px, py)
                except Exception:
                    pass

            # State Machine
            if state == "COUNTDOWN":
                left = COUNTDOWN_SECONDS - int(now - t_state_start)
                if left <= 0:
                    state = "RECORDING"
                    t_state_start = now
                    recorded_features.clear()
                    trail_points.clear()
                    message = "Recording! Perform movement now."
                else:
                    cv2.putText(view, str(left), (w // 2 - 50, h // 2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 215, 255), 8, cv2.LINE_AA)

            elif state == "RECORDING":
                if fingertip_px is not None:
                    trail_points.append(fingertip_px)
                if len(hands) >= 2:
                    try:
                        moving, supporting, _, _ = identify_hands(hands)
                        feat = extract_motion_features(moving, supporting)
                        recorded_features.append(feat)
                    except Exception:
                        pass

                duration = now - t_state_start
                # Auto-stop after 1.6s of recording
                if duration >= 1.6 or len(recorded_features) >= 45:
                    if len(recorded_features) >= 8:
                        review_candidate = resample_trajectory(np.array(recorded_features, dtype=np.float32))
                        state = "REVIEW"
                        message = f"Movement captured ({len(recorded_features)} frames). Press [A] to Accept, [D] to Discard."
                    else:
                        state = "READY"
                        message = "Movement too short or hands lost. Try again."

            elif state == "REVIEW":
                pass

            # Draw interface
            sample_count = len(saved_data[letter])
            draw_hud(view, letter, state, sample_count, args.samples, len(hands), message, trail_points)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if state == "READY":
                if key == 32:  # SPACE
                    if len(hands) < 2:
                        message = "Please show both hands clearly before starting."
                    else:
                        state = "COUNTDOWN"
                        t_state_start = now
                        trail_points.clear()
                        message = "Starting in 3 seconds..."
                elif key in (ord("n"), ord(".")):
                    curr_idx = min(curr_idx + 1, len(letters) - 1)
                    trail_points.clear()
                    message = f"Selected letter {letters[curr_idx]}"
                elif key in (ord("p"), ord(",")):
                    curr_idx = max(curr_idx - 1, 0)
                    trail_points.clear()
                    message = f"Selected letter {letters[curr_idx]}"

            elif state == "RECORDING":
                if key == 32:  # SPACE early finish
                    if len(recorded_features) >= 8:
                        review_candidate = resample_trajectory(np.array(recorded_features, dtype=np.float32))
                        state = "REVIEW"
                        message = f"Captured {len(recorded_features)} frames. Press [A] to Accept, [D] to Discard."
                    else:
                        state = "READY"
                        message = "Recording cancelled (too few frames)."

            elif state == "REVIEW":
                if key in (ord("a"), ord("A"), 32):  # Accept
                    if review_candidate is not None:
                        if not args.append and sample_count == 0:
                            saved_data[letter] = []
                        saved_data[letter].append(review_candidate)

                        # Save to disk immediately
                        all_t, all_l = [], []
                        for l_key, trajs in saved_data.items():
                            for tr in trajs:
                                all_t.append(tr)
                                all_l.append(l_key)
                        save_motion_templates(MOTION_TEMPLATES, all_t, all_l)

                        sample_count = len(saved_data[letter])
                        message = f"Saved take #{sample_count} for {letter}!"
                        trail_points.clear()
                        review_candidate = None

                        if sample_count >= args.samples and curr_idx < len(letters) - 1:
                            curr_idx += 1
                            message += f" Finished {letter}! Moving to {letters[curr_idx]}."
                        state = "READY"

                elif key in (ord("d"), ord("D")):  # Discard
                    trail_points.clear()
                    review_candidate = None
                    state = "READY"
                    message = f"Take discarded for {letter}. Ready to try again."

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    total_counts = {c: len(saved_data[c]) for c in DYNAMIC_LETTERS}
    print(f"\nRecording session finished. Motion templates stored in: {MOTION_TEMPLATES}")
    print(f"Recorded samples per letter: {total_counts}")


if __name__ == "__main__":
    main()
