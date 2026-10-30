"""Record labelled palm / back-of-hand clips (ground truth for Experiment 2).

For every combination of condition (which hand, which side faces the camera) and pose style,
the window tells you what to show. Press SPACE, wait for the countdown, then hold the pose and move
a little for about 5 seconds. Only frames that contain the expected number of hands are kept.

6 conditions x 4 poses x 40 frames = 960 frames, around 6 minutes. Samples are saved after every
clip, so you can quit with Q and continue later (new clips are appended).

Usage:
    python record_palm_side.py
    python record_palm_side.py --samples 60 --poses open fist
    python record_palm_side.py --conditions right_palm right_back --camera 1

Keys:  SPACE record   N next   P previous   Q or ESC quit
Then:  python -m experiments.run_experiments --exp 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.palm_labels import (CONDITION_TEXT, CONDITIONS, POSES, empty_record, hand_items, labels_for,
                                     load_labels, pack_frame, save_labels)
from utils import DATA_DIR, HandTracker, open_camera

OUT_PATH = DATA_DIR / "palm_side_labels.npz"
COUNTDOWN_S = 3
CLIP_TIMEOUT_S = 15


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--poses", nargs="+", default=list(POSES), choices=list(POSES))
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--interval", type=float, default=0.12)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    clips = [(c, p) for c in args.conditions for p in args.poses]
    out_path = Path(args.out)
    rec, aspect_saved = load_labels(out_path) if out_path.exists() else (empty_record(), None)
    next_block = (int(max(rec["block"])) + 1) if rec["block"] else 0
    done = {}
    for c, p in zip(rec["condition"], rec["pose"]):
        done[(str(c), str(p))] = done.get((str(c), str(p)), 0) + 1

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}. Close other apps that use the webcam, or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = HandTracker()
    cv2.namedWindow("Palm side recording", cv2.WINDOW_NORMAL)

    idx, state, t_state, t_last, message = 0, "ready", time.time(), 0.0, ""
    buffer = []
    aspect = aspect_saved
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break
            h, w = frame.shape[:2]
            if aspect is None:
                aspect = w / h
            elif abs(aspect - w / h) > 1e-3:
                raise SystemExit(f"Camera aspect {w / h:.3f} differs from the saved file ({aspect:.3f}). "
                                 "Use the same camera and resolution, or record to another file with --out.")
            hands, result = tracker.process(frame)
            tracker.draw(frame, result)
            view = cv2.flip(frame, 1)                       # mirror for display only
            cond, pose = clips[idx]
            expected = len(CONDITIONS[cond])
            now = time.time()

            if state == "countdown":
                left = COUNTDOWN_S - int(now - t_state)
                if left <= 0:
                    state, t_state, buffer = "recording", now, []
                else:
                    cv2.putText(view, str(left), (w // 2 - 40, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 5,
                                (255, 255, 255), 10, cv2.LINE_AA)
            elif state == "recording":
                if len(hands) == expected and now - t_last >= args.interval:
                    L, M, S = pack_frame(hand_items(hands, result))
                    side, true_hand = labels_for(cond, len(hands))
                    buffer.append((L, M, S, side, true_hand))
                    t_last = now
                if len(buffer) >= args.samples:
                    for L, M, S, side, true_hand in buffer:
                        rec["landmarks"].append(L); rec["mp_label"].append(M); rec["mp_score"].append(S)
                        rec["side"].append(side); rec["true_hand"].append(true_hand)
                        rec["pose"].append(pose); rec["condition"].append(cond); rec["block"].append(next_block)
                    next_block += 1
                    save_labels(args.out, rec, aspect)
                    done[(cond, pose)] = done.get((cond, pose), 0) + len(buffer)
                    message = f"Saved {len(buffer)} frames for {cond} / {pose}"
                    state, buffer = "ready", []
                    idx = min(idx + 1, len(clips) - 1)
                elif now - t_state > CLIP_TIMEOUT_S and len(buffer) < args.samples // 2:
                    message = f"Too few frames with exactly {expected} hand(s). Try again."
                    state, buffer = "ready", []

            put_text(view, f"{idx + 1} of {len(clips)}:  {CONDITION_TEXT[cond]}", (20, 50), 0.8, (255, 255, 255), 2)
            put_text(view, f"Pose: {POSES[pose]}", (20, 90), 0.7, (200, 230, 255))
            put_text(view, f"hands detected: {len(hands)} (need {expected})   frames saved for this clip type: {done.get((cond, pose), 0)}",
                     (20, 125), 0.6)
            if state == "ready":
                put_text(view, "SPACE: record   N: next   P: previous   Q: quit", (20, 160), 0.6, (200, 230, 255))
            elif state == "recording":
                put_text(view, f"Recording {len(buffer)}/{args.samples}  (move a little)", (20, 160), 0.7, (120, 255, 120))
            if message:
                put_text(view, message, (20, 195), 0.6, (120, 220, 255))
            cv2.imshow("Palm side recording", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and state == "ready":
                state, t_state, message = "countdown", now, ""
            elif key in (ord("n"), ord(".")) and state == "ready":
                idx, message = min(idx + 1, len(clips) - 1), ""
            elif key in (ord("p"), ord(",")) and state == "ready":
                idx, message = max(idx - 1, 0), ""
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(f"\nSaved to {args.out}: {len(rec['side'])} frames in total.")
    print("Next: python -m experiments.run_experiments --exp 2")


if __name__ == "__main__":
    main()
