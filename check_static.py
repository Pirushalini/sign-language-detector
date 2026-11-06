"""Static template check.

    python check_static.py                      # current settings
    python check_static.py --inter 6            # try a heavier hand-to-hand weight
    python check_static.py --inter 6 --rel 10   # ... plus fingertip-to-fingertip distances

Part 1 lists the templates per letter. Part 2 tests your own recordings against the
dataset and against your other recordings, while hiding each sample's neighbours
(consecutive frames are near-copies, so the plain leave-one-out test says 100%).
"""
import argparse
from collections import Counter

import numpy as np

from matcher import SRC_USER, TemplateMatcher
from utils import FLAGS_START, HAND_LEN, LETTERS

ap = argparse.ArgumentParser()
ap.add_argument("--inter", type=float, default=2.0, help="weight of the wrist-to-wrist features")
ap.add_argument("--rel", type=float, default=0.0, help="weight of fingertip-to-fingertip distances (0 = off)")
ap.add_argument("--gap", type=int, default=15, help="hide this many neighbouring samples")
args = ap.parse_args()


def openness(X, slot):
    """Mean fingertip distance from the wrist: an open flat hand is high, a pointing hand lower."""
    tips = X[:, slot * HAND_LEN:(slot + 1) * HAND_LEN].reshape(-1, 21, 3)[:, [8, 12, 16, 20], :2]
    return np.linalg.norm(tips, axis=2).mean(axis=1)


raw = TemplateMatcher.from_files(require_two_hands=False)      # unfiltered, to see what is really stored
X0, y0, s0 = raw.X, raw.y, raw.src
two = (X0[:, FLAGS_START] > 0.5) & (X0[:, FLAGS_START + 1] > 0.5)
open_first = openness(X0, 0) > openness(X0, 1)                  # the open (base) hand is in slot 1 normally

print("letter  dataset  yours  one-hand  open-hand-first")
for c in LETTERS:
    mask = y0 == c
    n = int(mask.sum())
    if n == 0:
        print(f"{c}       none")
        continue
    mine = int((mask & (s0 == SRC_USER)).sum())
    sel = mask & two
    order = 100 * open_first[sel].mean() if sel.any() else float("nan")
    flag = "   <-- hands swap places" if sel.any() and 15 < order < 85 else ""
    print(f"{c}     {n - mine:7d} {mine:6d}   {100 * (1 - two[mask].mean()):5.0f}%      {order:5.0f}%{flag}")

# ---------------------------------------------------------------- part 2
m = TemplateMatcher.from_files(inter_weight=args.inter, rel_weight=args.rel)
X, y, src = m.X, m.y, m.src
user = np.flatnonzero(src == SRC_USER)
if len(user) == 0:
    raise SystemExit("\nNo user templates, so nothing to evaluate.")

pos = np.zeros(len(X), dtype=int)                               # order of each sample within its letter
for c in set(y[user].tolist()):
    idx = user[y[user] == c]
    pos[idx] = np.arange(len(idx))

m._prepare()
D = m._distances(X[user])
near = ((y[user][:, None] == y[None, :]) & (src[None, :] == SRC_USER)
        & (np.abs(pos[user][:, None] - pos[None, :]) < args.gap))
D[near] = np.inf                                                # hide the sample and its neighbours

scores = np.empty((len(user), len(m._classes)), dtype=np.float32)
for j, idx in enumerate(m._class_idx):
    sub = D[:, idx]
    kk = min(m.k, sub.shape[1])
    scores[:, j] = np.mean(np.partition(sub, kk - 1, axis=1)[:, :kk], axis=1)
pred = np.array(m._classes)[np.argmin(scores, axis=1)]

print(f"\nhold-out accuracy of your recordings   (inter={args.inter}, rel={args.rel}, gap={args.gap})")
total = 0.0
letters = [c for c in LETTERS if (y[user] == c).any()]
for c in letters:
    sel = y[user] == c
    acc = (pred[sel] == c).mean()
    total += acc
    wrong = Counter(pred[sel][pred[sel] != c].tolist())
    top = ", ".join(f"{k}x{v}" for k, v in wrong.most_common(2))
    print(f"{c}: {100 * acc:3.0f}%   {top}")
print(f"average over letters: {100 * total / len(letters):.1f}%")