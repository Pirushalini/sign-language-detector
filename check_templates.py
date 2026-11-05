"""Find odd recorded motion clips.  python check_templates.py [--clean]"""
import shutil
import sys

import numpy as np

from motion import MOTION_TEMPLATES, dtw_distance, load_motion_templates, save_motion_templates

trajs, labels = load_motion_templates(MOTION_TEMPLATES)
labels = np.array(labels)
n = len(trajs)
if n == 0:
    raise SystemExit("No templates found.")

D = np.full((n, n), np.inf)
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = D[j, i] = dtw_distance(trajs[i], trajs[j])

bad = []
for letter in sorted(set(labels)):
    idx = np.flatnonzero(labels == letter)
    same = np.array([D[i, idx[idx != i]].min(initial=np.inf) for i in idx])
    other = np.array([D[i, labels != letter].min(initial=np.inf) for i in idx])
    med = float(np.median(same))
    print(f"{letter}: {len(idx)} clips, typical nearest-same distance {med:.3f}")
    for k, i in enumerate(idx):
        if same[k] > 2.5 * med or other[k] < same[k]:
            bad.append(int(i))
            print(f"   clip #{i}: nearest same {same[k]:.3f}, nearest other {other[k]:.3f}  <- suspicious")

if "--clean" in sys.argv and bad:
    shutil.copy(MOTION_TEMPLATES, MOTION_TEMPLATES.with_suffix(".bak.npz"))
    keep = [i for i in range(n) if i not in bad]
    save_motion_templates(MOTION_TEMPLATES, [trajs[i] for i in keep], [str(labels[i]) for i in keep])
    print(f"Removed {len(bad)} clips (backup saved next to the original).")