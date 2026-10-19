"""Evaluate the template matcher (no training involved) and run small ablations.

Two tests are reported:

1. Held-out split of the imported dataset. WARNING: the dataset has no signer or
   session information, and neighbouring frames are almost identical, so this
   number is optimistic. Use it to compare settings, not as proof of real accuracy.
2. Imported dataset -> your own calibration samples (only if you ran
   calibrate.py). This is the honest test, because it uses a different person,
   camera and room.

Usage:
    python evaluate.py
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from matcher import SRC_EXTERNAL, SRC_USER, TemplateMatcher, load_templates
from utils import EXTERNAL_TEMPLATES, REPORTS_DIR, USER_TEMPLATES


def split_indices(y: np.ndarray, test_fraction: float, seed: int):
    """Stratified random split. Returns (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_fraction))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.array(train), np.array(test)


def make_matcher(X, y, src=SRC_EXTERNAL, **kwargs) -> TemplateMatcher:
    m = TemplateMatcher(**kwargs)
    m.add(X, y, np.full(len(y), src, dtype=np.int8))
    return m


def accuracy(matcher: TemplateMatcher, Xq, yq) -> float:
    pred, _ = matcher.predict_labels(Xq)
    return float(np.mean(pred == yq))


def confusion_png(y_true, y_pred, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title("Confusion matrix (held-out templates)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    from sklearn.metrics import classification_report

    if not EXTERNAL_TEMPLATES.exists():
        raise SystemExit("data/templates_external.npz not found. Run import_dataset.py first.")
    X, y, _, _ = load_templates(EXTERNAL_TEMPLATES)
    REPORTS_DIR.mkdir(exist_ok=True)
    tr, te = split_indices(y, 0.2, seed=0)
    results: dict = {"note": "held-out split of imported data is optimistic (no signer information)"}

    base = make_matcher(X[tr], y[tr])
    pred, best = base.predict_labels(X[te])
    acc = float(np.mean(pred == y[te]))
    results["held_out_accuracy"] = acc
    print(f"\n1) Held-out split of imported data ({len(te)} samples, optimistic): accuracy {acc:.3f}")
    report = classification_report(y[te], pred, zero_division=0, output_dict=True)
    results["per_letter_f1"] = {c: round(report[c]["f1-score"], 3) for c in sorted(set(y))}
    worst = sorted(results["per_letter_f1"].items(), key=lambda kv: kv[1])[:5]
    print("   Weakest letters (F1):", ", ".join(f"{c} {v:.2f}" for c, v in worst))
    confused = Counter((t, p) for t, p in zip(y[te], pred) if t != p).most_common(5)
    print("   Most confused pairs (true -> predicted):", ", ".join(f"{t}->{p} x{n}" for (t, p), n in confused) or "none")
    q = np.percentile(best, [50, 95, 99])
    print(f"   Best-letter distance on held-out data: median {q[0]:.2f}, 95th {q[1]:.2f}, 99th {q[2]:.2f}")
    results["best_distance_percentiles"] = {"p50": float(q[0]), "p95": float(q[1]), "p99": float(q[2])}
    confusion_png(y[te], pred, REPORTS_DIR / "confusion_matrix.png")

    print("\n   Ablations (same held-out split):")
    print(f"   {'setting':<28}{'accuracy':>9}")
    ablations = {}
    for name, kw in [
        ("k=1", dict(k=1)), ("k=3 (default)", dict(k=3)), ("k=5", dict(k=5)), ("k=10", dict(k=10)),
        ("cosine distance", dict(metric="cosine")),
        ("inter-hand weight 0 (off)", dict(inter_weight=0.0)),
        ("inter-hand weight 1", dict(inter_weight=1.0)),
        ("inter-hand weight 4", dict(inter_weight=4.0)),
    ]:
        a = accuracy(make_matcher(X[tr], y[tr], **kw), X[te], y[te])
        ablations[name] = a
        print(f"   {name:<28}{a:>9.3f}")
    results["ablations"] = ablations

    # Templates per letter (how many are needed?)
    print("\n   Accuracy vs templates per letter:")
    curve = {}
    rng = np.random.default_rng(1)
    for n in (1, 5, 10, 25, 50, 100, 200):
        keep = []
        for c in np.unique(y[tr]):
            idx = tr[y[tr] == c]
            keep.extend(rng.choice(idx, size=min(n, len(idx)), replace=False))
        keep = np.array(keep)
        a = accuracy(make_matcher(X[keep], y[keep]), X[te], y[te])
        curve[str(n)] = a
        print(f"   {n:>4} per letter: {a:.3f}")
    results["templates_per_letter"] = curve

    # Honest test: imported data -> your own recordings.
    if USER_TEMPLATES.exists():
        Xu, yu, srcu, _ = load_templates(USER_TEMPLATES)
        shared = np.isin(yu, np.unique(y))
        if shared.any():
            m = make_matcher(X, y)
            pu, _ = m.predict_labels(Xu[shared])
            au = float(np.mean(pu == yu[shared]))
            results["external_to_user_accuracy"] = au
            print(f"\n2) Imported data -> your own recordings ({int(shared.sum())} samples): accuracy {au:.3f}")
            per = {c: float(np.mean(pu[yu[shared] == c] == c)) for c in sorted(set(yu[shared]))}
            print("   Per letter:", ", ".join(f"{c} {v:.2f}" for c, v in per.items()))
            results["external_to_user_per_letter"] = per
    else:
        print("\n2) No calibration samples yet. Run calibrate.py to get the honest test.")

    out = REPORTS_DIR / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out} and {REPORTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
