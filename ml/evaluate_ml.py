"""ml/evaluate_ml.py — Evaluate all four ML classifiers and the template matcher.

Runs all five classifiers on the same held-out test set and produces:
  - Overall accuracy and macro F1
  - Per-letter precision, recall, F1
  - Confusion matrices (absolute and normalised)
  - Inference timing statistics
  - Predicted probabilities / scores and confidence margin
  - ECE (expected calibration error) where probabilities are available
  - Saved reports to reports/ml/

Usage
-----
    python ml/evaluate_ml.py                 # evaluate all saved models
    python ml/evaluate_ml.py --skip-tm       # skip template matcher
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports" / "ml"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ml.dataset import load_combined, SEED
from ml.train import load_model
from matcher import SRC_EXTERNAL, TemplateMatcher, load_templates
from utils import EXTERNAL_TEMPLATES


# ================================================================ inference timing
def time_inference(predict_fn, X: np.ndarray, n_warmup: int = 5, repeats: int = 3) -> Dict:
    """Benchmark per-sample inference time.

    Parameters
    ----------
    predict_fn : callable that takes a (1, 132) array and returns a label
    X          : array of samples to use for benchmarking
    n_warmup   : warm-up predictions (not counted)
    repeats    : how many timing passes to average

    Returns
    -------
    Dict with mean_ms, median_ms, p95_ms (milliseconds per sample)
    """
    # Warm up
    for i in range(min(n_warmup, len(X))):
        predict_fn(X[i:i+1])

    times = []
    for _ in range(repeats):
        for i in range(len(X)):
            t0 = time.perf_counter()
            predict_fn(X[i:i+1])
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "n_samples_timed": len(times),
    }


# ================================================================ ECE
def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: List[str],
    n_bins: int = 10,
) -> float:
    """Multiclass ECE: average over all classes, micro-weighted."""
    y_idx = np.array([classes.index(c) for c in y_true])
    top_conf = proba.max(axis=1)
    top_pred = proba.argmax(axis=1)
    correct = (top_pred == y_idx).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (top_conf >= lo) & (top_conf < hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = top_conf[mask].mean()
        ece += mask.sum() * abs(acc_bin - conf_bin)
    return float(ece / len(y_true))


# ================================================================ confusion matrix plot
def save_confusion_matrix(
    y_true, y_pred, labels: List[str], title: str, path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, data, fmt, t in zip(
        axes,
        [cm, cm_norm],
        [".0f", ".2f"],
        ["Counts", "Normalised (recall)"],
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=45)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} — {t}")
        fig.colorbar(im, ax=ax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = data[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=5, color="white" if v > data.max() * 0.6 else "black")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_reliability_diagram(y_true, proba, classes: List[str], name: str, path: Path) -> None:
    """Plot confidence reliability diagram (calibration curve)."""
    y_idx = np.array([classes.index(c) for c in y_true])
    top_conf = proba.max(axis=1)
    top_pred = proba.argmax(axis=1)
    correct = (top_pred == y_idx).astype(float)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (top_conf >= lo) & (top_conf < hi)
        if mask.sum() == 0:
            bin_accs.append(float("nan"))
            bin_confs.append((lo + hi) / 2)
        else:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(top_conf[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(bin_edges[:-1], bin_accs, width=0.1, align="edge", alpha=0.7, color="#2563EB")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title(f"Reliability diagram — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ================================================================ evaluate one model
def evaluate_model(
    name: str,
    predict_fn,  # callable: (N,132) -> (N,) labels
    proba_fn,    # callable: (N,132) -> (N, C) probs, or None
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: List[str],
    timing_samples: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run evaluation and return a results dict."""
    print(f"\n--- Evaluating: {name} ---")

    # Predictions
    y_pred = predict_fn(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, labels=classes,
                                   zero_division=0, output_dict=True)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1_mac:.4f}")

    # Confusion matrix — use labels that appear in the test set
    test_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    cm_path = REPORTS_DIR / f"cm_{name}.png"
    save_confusion_matrix(y_test, y_pred, test_labels, name, cm_path)

    # Top confusion pairs
    cm = confusion_matrix(y_test, y_pred, labels=test_labels)
    np.fill_diagonal(cm, 0)
    flat = [(cm[i, j], test_labels[i], test_labels[j])
            for i in range(len(test_labels))
            for j in range(len(test_labels))
            if cm[i, j] > 0]
    flat.sort(reverse=True)
    top_confused = [(true, pred, int(n)) for n, true, pred in flat[:5]]
    print(f"  Top confused: {top_confused}")

    # Timing
    if timing_samples is None:
        timing_samples = X_test[:min(200, len(X_test))]
    timing = time_inference(predict_fn, timing_samples)
    print(f"  Inference   : mean {timing['mean_ms']:.3f} ms, "
          f"median {timing['median_ms']:.3f} ms, "
          f"p95 {timing['p95_ms']:.3f} ms")

    result = {
        "name": name,
        "accuracy": acc,
        "macro_f1": f1_mac,
        "per_letter": {
            c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
            for c in classes
            if c in report
        },
        "top_confused": top_confused,
        "timing": timing,
    }

    # Probability-based metrics
    if proba_fn is not None:
        try:
            proba = proba_fn(X_test)
            if proba.shape[1] != len(classes):
                proba = None
        except Exception as e:
            print(f"  WARNING: proba_fn failed: {e}")
            proba = None

        if proba is not None:
            ece = expected_calibration_error(y_test, proba, classes)
            margin = np.sort(proba, axis=1)[:, -1] - np.sort(proba, axis=1)[:, -2]
            result["ece"] = round(ece, 5)
            result["mean_margin"] = round(float(margin.mean()), 4)
            result["median_margin"] = round(float(np.median(margin)), 4)
            print(f"  ECE         : {ece:.5f}")
            print(f"  Mean margin : {margin.mean():.4f}")

            # Reliability diagram
            rel_path = REPORTS_DIR / f"reliability_{name}.png"
            save_reliability_diagram(y_test, proba, classes, name, rel_path)

    return result


# ================================================================ template matcher baseline
def evaluate_template_matcher(
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes: List[str],
) -> Dict[str, Any]:
    """Build a training-only template bank and evaluate it on the test set."""
    print("\n--- Evaluating: Template Matcher (baseline) ---")
    from matcher import SRC_EXTERNAL, TemplateMatcher

    m = TemplateMatcher()
    m.add(X_train, y_train, np.zeros(len(y_train), dtype=np.int8))

    # predict_labels returns (labels, distances)
    y_pred, distances = m.predict_labels(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, labels=classes,
                                   zero_division=0, output_dict=True)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1_mac:.4f}")

    test_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    cm_path = REPORTS_DIR / "cm_template_matcher.png"
    save_confusion_matrix(y_test, y_pred, test_labels, "Template Matcher", cm_path)

    # Timing: use predict() on single sample (the live code path)
    timing_X = X_test[:min(200, len(X_test))]

    def _predict_single(x):
        return np.array([m.predict(x[0]).label or "" ])

    timing = time_inference(_predict_single, timing_X)
    print(f"  Inference   : mean {timing['mean_ms']:.3f} ms, "
          f"median {timing['median_ms']:.3f} ms, "
          f"p95 {timing['p95_ms']:.3f} ms")

    cm = confusion_matrix(y_test, y_pred, labels=test_labels)
    np.fill_diagonal(cm, 0)
    flat = [(cm[i, j], test_labels[i], test_labels[j])
            for i in range(len(test_labels))
            for j in range(len(test_labels))
            if cm[i, j] > 0]
    flat.sort(reverse=True)
    top_confused = [(true, pred, int(n)) for n, true, pred in flat[:5]]

    return {
        "name": "Template Matcher",
        "accuracy": acc,
        "macro_f1": f1_mac,
        "per_letter": {
            c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
            for c in classes
            if c in report
        },
        "top_confused": top_confused,
        "timing": timing,
        "note": "Raw distances are not calibrated probabilities; ECE not computed.",
    }


# ================================================================ comparison table
def print_comparison_table(results: List[Dict]) -> None:
    header = f"{'Method':<22} {'Accuracy':>10} {'Macro F1':>10} {'Mean ms':>10} {'Median ms':>10} {'p95 ms':>10}"
    print("\n" + "=" * 75)
    print("  COMPARISON TABLE")
    print("=" * 75)
    print(header)
    print("-" * 75)
    for r in results:
        t = r.get("timing", {})
        print(
            f"{r['name']:<22} "
            f"{r['accuracy']:>10.4f} "
            f"{r['macro_f1']:>10.4f} "
            f"{t.get('mean_ms', float('nan')):>10.3f} "
            f"{t.get('median_ms', float('nan')):>10.3f} "
            f"{t.get('p95_ms', float('nan')):>10.3f}"
        )
    print("=" * 75)


def save_comparison_chart(results: List[Dict], path: Path) -> None:
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, accs, w, label="Accuracy", color="#2563EB", alpha=0.85)
    ax.bar(x + w / 2, f1s, w, label="Macro F1", color="#16A34A", alpha=0.85)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Task 3 — Classifier Comparison")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_latency_chart(results: List[Dict], path: Path) -> None:
    names = [r["name"] for r in results]
    means = [r.get("timing", {}).get("mean_ms", 0) for r in results]
    p95s = [r.get("timing", {}).get("p95_ms", 0) for r in results]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, means, w, label="Mean (ms)", color="#D97706", alpha=0.85)
    ax.bar(x + w / 2, p95s, w, label="p95 (ms)", color="#DC2626", alpha=0.85)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("Inference time (ms per sample)")
    ax.set_title("Task 3 — Inference Latency Comparison")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ================================================================ main
def run_evaluation(skip_tm: bool = False, data_source: str = "ext") -> Dict:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    skip_tm     : skip template matcher baseline
    data_source : which trained models to load ("ext", "usr", "combined")
    """
    splits = load_combined()

    # Choose the test set and training set for the baseline TM
    if data_source == "ext":
        X_test = splits["X_ext_test"]
        y_test = splits["y_ext_test"]
        X_train_tm = splits["X_ext_train"]
        y_train_tm = splits["y_ext_train"]
        suffix = "ext"
    elif data_source == "user":
        X_test = splits["X_usr_test"]
        y_test = splits["y_usr_test"]
        X_train_tm = splits["X_usr_train"]
        y_train_tm = splits["y_usr_train"]
        suffix = "usr"
    else:  # combined — use the user test set as the independent test
        X_test = splits["X_usr_test"]
        y_test = splits["y_usr_test"]
        X_train_tm = splits["X_comb_train"]
        y_train_tm = splits["y_comb_train"]
        suffix = "comb"

    classes = sorted(np.unique(y_test).tolist())
    print(f"\nTest set: {len(X_test)} samples, classes: {classes}")

    all_results = []

    # ---- Template Matcher baseline --------------------------------
    if not skip_tm:
        tm_res = evaluate_template_matcher(
            X_test, y_test, X_train_tm, y_train_tm, classes
        )
        all_results.append(tm_res)

    # ---- ML models ------------------------------------------------
    model_configs = [
        ("knn", "KNN"),
        ("svm", "SVM"),
        ("rf",  "Random Forest"),
        ("mlp", "MLP"),
    ]

    for mid, label in model_configs:
        model_name = f"{mid}_{suffix}"
        try:
            pipe, meta = load_model(model_name)
        except FileNotFoundError:
            print(f"\nModel {model_name} not found; skipping. Run ml/train.py first.")
            continue

        pipe_classes = meta.get("classes", classes)

        def make_predict_fn(p, pc):
            def _pred(X):
                return np.array(p.predict(X))
            return _pred

        def make_proba_fn(p, pc):
            if hasattr(p, "predict_proba"):
                def _proba(X):
                    return p.predict_proba(X)
                return _proba, pc
            return None, pc

        predict_fn = make_predict_fn(pipe, pipe_classes)
        proba_fn, proba_classes = make_proba_fn(pipe, pipe_classes)

        res = evaluate_model(
            label,
            predict_fn,
            proba_fn,
            X_test, y_test,
            pipe_classes,
        )
        all_results.append(res)

    # ---- Summary --------------------------------------------------
    print_comparison_table(all_results)
    save_comparison_chart(all_results, REPORTS_DIR / "comparison_accuracy.png")
    save_latency_chart(all_results, REPORTS_DIR / "comparison_latency.png")

    # Save JSON
    out_path = REPORTS_DIR / f"ml_results_{suffix}.json"
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "pipeline"}
        serializable.append(sr)
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {out_path}")

    return {"results": all_results}


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tm", action="store_true", help="skip template matcher baseline")
    ap.add_argument("--data", choices=["ext", "user", "combined"],
                    default="ext", help="which model set to evaluate")
    args = ap.parse_args()
    run_evaluation(skip_tm=args.skip_tm, data_source=args.data)

