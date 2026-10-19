"""Convert the public BSL landmark CSV files into reference templates.

Source: "BSL Numbers & Alphabet Hand Position For MediaPipe" (Kaggle, by
erentatepe; GitHub: datMaul/BSL_Numbers_and_Alphabet_Recognition). The CSV rows
are raw MediaPipe landmarks. They are converted here to the same 132-value
features that the live app uses, so they can be matched against webcam frames.

No model is trained. The converted rows are stored as templates.

Usage:
    python import_dataset.py
    python import_dataset.py --per-class 300 --mirror
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from matcher import SRC_EXTERNAL, save_templates
from utils import (DATASET_ASPECT, EXTERNAL_DIR, EXTERNAL_TEMPLATES, FEATURE_LEN,
                   LETTERS, N_LANDMARKS, build_features, mirror_hands)


def parse_label(raw: str):
    """'A - a' -> 'A'. Returns None for digits and anything that is not a letter."""
    first = raw.split("-")[0].strip().upper()
    return first if first in LETTERS else None


def read_rows(path: Path, n_hands: int):
    """Yield (label, list of (21, 3) arrays) for every usable row of a CSV file."""
    width = n_hands * N_LANDMARKS * 3
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) != width + 1:
                continue
            label = parse_label(row[-1])
            if label is None:
                continue
            values = np.asarray(row[:-1], dtype=np.float32).reshape(n_hands, N_LANDMARKS, 3)
            yield label, [values[i] for i in range(n_hands)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--two-hand", type=Path, default=EXTERNAL_DIR / "two_hand_dataset.csv")
    ap.add_argument("--one-hand", type=Path, default=EXTERNAL_DIR / "one_hand_dataset.csv")
    ap.add_argument("--per-class", type=int, default=400, help="max templates kept per letter (default 400)")
    ap.add_argument("--aspect", type=float, default=DATASET_ASPECT,
                    help="assumed width/height of the dataset images (default 16/9)")
    ap.add_argument("--mirror", action="store_true",
                    help="also add left-right mirrored copies, so left-handed signers match too")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_label = defaultdict(list)
    for path, n_hands in ((args.two_hand, 2), (args.one_hand, 1)):
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue
        count = 0
        for label, hands in read_rows(path, n_hands):
            by_label[label].append(hands)
            count += 1
        print(f"Read {count} letter rows from {path.name}")
    if not by_label:
        raise SystemExit("No data found. Put the two CSV files in data/external/ first.")

    rng = random.Random(args.seed)
    X, y = [], []
    for label in sorted(by_label):
        rows = by_label[label]
        if len(rows) > args.per_class:
            rows = rng.sample(rows, args.per_class)
        for hands in rows:
            X.append(build_features(hands, aspect=args.aspect))
            y.append(label)
            if args.mirror:
                X.append(build_features(mirror_hands(hands), aspect=args.aspect))
                y.append(label)

    X = np.stack(X)
    assert X.shape[1] == FEATURE_LEN
    save_templates(EXTERNAL_TEMPLATES, X, y, [SRC_EXTERNAL] * len(y), ["kaggle-erentatepe"] * len(y))

    print(f"\nSaved {len(y)} templates to {EXTERNAL_TEMPLATES}")
    print("Templates per letter:", {c: y.count(c) for c in sorted(set(y))})
    missing = [c for c in LETTERS if c not in set(y)]
    if missing:
        print(f"Letters not in the dataset: {' '.join(missing)}")
        print(f"Record them with:  python calibrate.py --letters {' '.join(missing)}")


if __name__ == "__main__":
    main()
