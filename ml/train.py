"""ml/train.py — Train four scikit-learn classifiers on BSL landmark features.

Models
------
A. KNN  — K-Nearest Neighbours
B. SVM  — Support Vector Machine (RBF kernel, probability calibration)
C. RF   — Random Forest
D. MLP  — Multi-Layer Perceptron

Each model is wrapped in a scikit-learn Pipeline so that StandardScaler
preprocessing is applied before every model, preventing leakage (the scaler
is fitted only on training data).

Usage
-----
    python ml/train.py
    python ml/train.py --data ext      # train on external data only
    python ml/train.py --data user     # train on user data only
    python ml/train.py --data combined # train on external + user data
    python ml/train.py --cv            # enable cross-validation grid search
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.dataset import load_combined, SEED

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Fixed random seed for reproducibility
RNG = SEED


# ---------------------------------------------------------------- helpers
def _make_pipeline(clf) -> Pipeline:
    """Wrap classifier in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _grid_search(
    pipe: Pipeline,
    param_grid: List[Dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
) -> Tuple[Pipeline, Dict]:
    """Run GridSearchCV and return (best_estimator, best_params)."""
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RNG),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


# ================================================================ MODEL A — KNN
def build_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_cv: bool = False,
) -> Tuple[Pipeline, Dict]:
    """Train KNN classifier.

    When use_cv=True a grid search over k and weights is performed.
    When use_cv=False the default best configuration (k=5, distance-weighted)
    is used directly, which is fast and gives competitive results.
    """
    if use_cv:
        pipe = _make_pipeline(KNeighborsClassifier())
        grid = [
            {
                "clf__n_neighbors": [1, 3, 5, 7],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            }
        ]
        best, params = _grid_search(pipe, grid, X_train, y_train)
        return best, params
    else:
        pipe = _make_pipeline(KNeighborsClassifier(n_neighbors=5, weights="distance"))
        pipe.fit(X_train, y_train)
        return pipe, {"clf__n_neighbors": 5, "clf__weights": "distance"}


# ================================================================ MODEL B — SVM
def build_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_cv: bool = False,
) -> Tuple[Pipeline, Dict]:
    """Train SVM with probability calibration.

    SVM decision scores are not probabilities, so we use CalibratedClassifierCV
    with sigmoid calibration trained on the training fold only.
    When use_cv=True a grid search over C and gamma is used.
    """
    if use_cv:
        # Use a simpler non-probability SVM for grid search speed, then
        # calibrate the winner.
        pipe = _make_pipeline(SVC(kernel="rbf", random_state=RNG))
        grid = [
            {
                "clf__C": [1, 10, 100],
                "clf__gamma": ["scale", "auto"],
            },
            {
                "clf__kernel": ["linear"],
                "clf__C": [0.1, 1, 10],
            },
        ]
        best, params = _grid_search(pipe, grid, X_train, y_train)
        # Re-wrap the best SVM with probability calibration
        best_svm = best.named_steps["clf"]
        pipe_cal = Pipeline(
            [
                ("scaler", best.named_steps["scaler"]),
                (
                    "clf",
                    CalibratedClassifierCV(best_svm, cv=3, method="sigmoid"),
                ),
            ]
        )
        pipe_cal.fit(X_train, y_train)
        return pipe_cal, params
    else:
        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=RNG)
        cal_svm = CalibratedClassifierCV(svm, cv=3, method="sigmoid")
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", cal_svm)])
        pipe.fit(X_train, y_train)
        return pipe, {"clf__C": 10, "clf__gamma": "scale", "clf__kernel": "rbf"}


# ================================================================ MODEL C — Random Forest
def build_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_cv: bool = False,
) -> Tuple[Pipeline, Dict]:
    """Train Random Forest classifier."""
    if use_cv:
        pipe = _make_pipeline(
            RandomForestClassifier(random_state=RNG, class_weight="balanced")
        )
        grid = [
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 20, 40],
                "clf__min_samples_leaf": [1, 2, 4],
            }
        ]
        best, params = _grid_search(pipe, grid, X_train, y_train)
        return best, params
    else:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RNG,
            n_jobs=-1,
        )
        pipe = _make_pipeline(rf)
        pipe.fit(X_train, y_train)
        return pipe, {
            "clf__n_estimators": 200,
            "clf__max_depth": None,
            "clf__min_samples_leaf": 1,
        }


# ================================================================ MODEL D — MLP
def build_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_cv: bool = False,
) -> Tuple[Pipeline, Dict]:
    """Train MLP neural network.

    Layer sizes are intentionally modest — the dataset has at most ~8000
    training samples and 132 input features, so very deep networks overfit.
    Early stopping is always enabled to prevent overfitting.
    """
    if use_cv:
        pipe = _make_pipeline(
            MLPClassifier(
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RNG,
            )
        )
        grid = [
            {
                "clf__hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
                "clf__alpha": [1e-4, 1e-3, 1e-2],
                "clf__learning_rate_init": [1e-3, 5e-4],
            }
        ]
        best, params = _grid_search(pipe, grid, X_train, y_train)
        return best, params
    else:
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RNG,
        )
        pipe = _make_pipeline(mlp)
        pipe.fit(X_train, y_train)
        return pipe, {
            "clf__hidden_layer_sizes": (256, 128),
            "clf__alpha": 1e-3,
            "clf__learning_rate_init": 1e-3,
        }


# ================================================================ save / load
def save_model(pipe: Pipeline, name: str, params: Dict, classes: List[str],
               data_source: str) -> Path:
    """Pickle the pipeline and save metadata as JSON alongside it."""
    model_path = MODELS_DIR / f"{name}.pkl"
    meta_path = MODELS_DIR / f"{name}_meta.json"

    with open(model_path, "wb") as f:
        pickle.dump(pipe, f, protocol=4)

    meta = {
        "name": name,
        "data_source": data_source,
        "classes": classes,
        "hyperparameters": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in params.items()
        },
        "feature_dim": 132,
        "sklearn_pipeline_steps": [s for s, _ in pipe.steps],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return model_path


def load_model(name: str) -> Tuple[Pipeline, Dict]:
    """Load a saved pipeline and its metadata."""
    model_path = MODELS_DIR / f"{name}.pkl"
    meta_path = MODELS_DIR / f"{name}_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}")
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pipe, meta


# ================================================================ main training script
def train_all(data_source: str = "ext", use_cv: bool = False) -> Dict[str, Any]:
    """Train all four classifiers and save them.

    Parameters
    ----------
    data_source : "ext" | "user" | "combined"
    use_cv      : if True, run grid search; if False, use pre-tuned defaults.

    Returns
    -------
    A dict with trained pipelines, labels, and training times.
    """
    print(f"\n{'='*60}")
    print(f"  Task 3 — ML Classifier Training")
    print(f"  Data source : {data_source}")
    print(f"  Grid search : {use_cv}")
    print(f"{'='*60}\n")

    splits = load_combined()

    if data_source == "ext":
        X_train = splits["X_ext_train"]
        y_train = splits["y_ext_train"]
        X_val = splits["X_ext_val"]
        y_val = splits["y_ext_val"]
        suffix = "ext"
    elif data_source == "user":
        X_train = splits["X_usr_train"]
        y_train = splits["y_usr_train"]
        X_val = splits["X_usr_val"]
        y_val = splits["y_usr_val"]
        suffix = "usr"
    elif data_source == "combined":
        X_train = splits["X_comb_train"]
        y_train = splits["y_comb_train"]
        X_val = splits["X_comb_val"]
        y_val = splits["y_comb_val"]
        suffix = "comb"
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    # Merge val into train when val is empty
    if len(X_val) == 0:
        X_train_full = X_train
        y_train_full = y_train
    else:
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

    classes = sorted(np.unique(y_train_full).tolist())
    print(f"Training set  : {len(X_train_full)} samples, {len(classes)} classes")
    print(f"Classes       : {classes}\n")

    results = {}

    for model_id, builder, label in [
        ("knn", build_knn, "KNN"),
        ("svm", build_svm, "SVM"),
        ("rf",  build_rf,  "Random Forest"),
        ("mlp", build_mlp, "MLP"),
    ]:
        print(f"Training {label}...", end="", flush=True)
        t0 = time.monotonic()
        pipe, params = builder(X_train_full, y_train_full, use_cv=use_cv)
        elapsed = time.monotonic() - t0
        name = f"{model_id}_{suffix}"
        path = save_model(pipe, name, params, classes, data_source)
        print(f"  done in {elapsed:.1f}s  →  {path.name}")
        results[model_id] = {
            "pipeline": pipe,
            "params": params,
            "classes": classes,
            "train_time_s": elapsed,
            "model_name": name,
        }

    print("\nAll models trained and saved.\n")
    return results


# ---------------------------------------------------------------- CLI entry
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["ext", "user", "combined"],
                    default="ext", help="which data partition to train on")
    ap.add_argument("--cv", action="store_true",
                    help="use grid search cross-validation to find hyperparameters")
    args = ap.parse_args()
    train_all(data_source=args.data, use_cv=args.cv)

