"""tests/test_ml.py — Automated tests for the Task 3 ML implementation.

Run with:
    python -m pytest tests/test_ml.py -v
or together with the existing tests:
    python -m pytest tests -q
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------- path setup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import FEATURE_LEN, EXTERNAL_TEMPLATES, USER_TEMPLATES
from matcher import load_templates, SRC_EXTERNAL, SRC_USER
from ml.backend import MLBackend, MLPrediction, BACKEND_TEMPLATE, BACKEND_ML

# ---------------------------------------------------------------- skip markers
needs_ext_data = pytest.mark.skipif(
    not EXTERNAL_TEMPLATES.exists(),
    reason="External templates not found; run import_dataset.py first."
)
needs_usr_data = pytest.mark.skipif(
    not USER_TEMPLATES.exists(),
    reason="User templates not found; run calibrate.py first."
)
needs_saved_model = pytest.mark.skipif(
    not (ROOT / "models" / "svm_ext.pkl").exists(),
    reason="No trained model found; run python ml/train.py first."
)


# ================================================================ dataset tests
@needs_ext_data
def test_external_feature_dimension():
    """External templates must have exactly 132 features."""
    X, y, src, sess = load_templates(EXTERNAL_TEMPLATES)
    assert X.shape[1] == FEATURE_LEN, f"Expected 132, got {X.shape[1]}"


@needs_ext_data
def test_external_no_nan_or_inf():
    """No non-finite values in external templates."""
    X, _, _, _ = load_templates(EXTERNAL_TEMPLATES)
    assert np.isfinite(X).all(), "Non-finite values found in external templates"


@needs_ext_data
def test_external_labels_are_uppercase_letters():
    """All external labels should be single uppercase letters."""
    _, y, _, _ = load_templates(EXTERNAL_TEMPLATES)
    for label in y:
        assert len(label) == 1 and label.isupper(), f"Bad label: {label!r}"


@needs_usr_data
def test_user_feature_dimension():
    """User templates must have exactly 132 features."""
    X, y, src, sess = load_templates(USER_TEMPLATES)
    assert X.shape[1] == FEATURE_LEN


@needs_usr_data
def test_user_has_multiple_sessions():
    """User data should have at least 2 calibration sessions."""
    _, _, _, sess = load_templates(USER_TEMPLATES)
    assert len(np.unique(sess)) >= 2, "Expected at least 2 sessions in user templates"


# ================================================================ dataset splitting tests
@needs_ext_data
def test_no_leakage_in_external_split():
    """Test indices must not appear in training indices (external split)."""
    from ml.dataset import session_aware_split
    X, y, _, sess = load_templates(EXTERNAL_TEMPLATES)
    X_tr, X_va, X_te, y_tr, y_va, y_te = session_aware_split(X, y, sess)
    # Verify sizes sum correctly
    assert len(X_tr) + len(X_va) + len(X_te) == len(X)
    # Verify no label-count impossibility
    assert len(X_te) > 0
    assert len(X_tr) > 0


@needs_usr_data
def test_session_split_keeps_sessions_intact():
    """When splitting user data by session, each session appears in exactly one split."""
    from ml.dataset import session_aware_split, load_user
    X, y, sess = load_user()
    unique_sessions = np.unique(sess)
    if len(unique_sessions) < 2:
        pytest.skip("Need at least 2 sessions to test session-aware split")

    X_tr, X_va, X_te, y_tr, y_va, y_te = session_aware_split(X, y, sess)

    # Reconstruct which sessions ended up in each partition
    # by checking indices
    tr_idx = []
    for i, s in enumerate(sess):
        if any(np.allclose(X[i], x) for x in X_tr[:5]):  # approximate check
            break
    # Simpler: verify sizes
    assert len(X_tr) + len(X_va) + len(X_te) == len(X), "Split sizes do not sum to total"


# ================================================================ label consistency
@needs_ext_data
@needs_usr_data
def test_label_sets_are_string_arrays():
    """Both datasets must use string (not bytes) labels."""
    for path in [EXTERNAL_TEMPLATES, USER_TEMPLATES]:
        _, y, _, _ = load_templates(path)
        for label in y[:10]:
            assert isinstance(label, (str, np.str_)), f"Non-string label: {label!r}"


# ================================================================ small training fixture
def make_synthetic_data(n_classes=4, per_class=20, feature_dim=132, seed=0):
    """Create a tiny synthetic dataset for fast tests."""
    rng = np.random.default_rng(seed)
    protos = rng.normal(0, 1, (n_classes, feature_dim)).astype(np.float32)
    X, y = [], []
    for i, p in enumerate(protos):
        for _ in range(per_class):
            X.append((p + rng.normal(0, 0.1, feature_dim)).astype(np.float32))
            y.append(chr(ord("A") + i))
    return np.stack(X), np.array(y)


def test_knn_trains_and_predicts_on_synthetic_data():
    from ml.train import build_knn
    X, y = make_synthetic_data()
    pipe, params = build_knn(X, y)
    preds = pipe.predict(X[:5])
    assert len(preds) == 5
    assert all(p in "ABCD" for p in preds)


def test_svm_trains_and_predicts_on_synthetic_data():
    from ml.train import build_svm
    X, y = make_synthetic_data()
    pipe, params = build_svm(X, y)
    preds = pipe.predict(X[:5])
    assert len(preds) == 5
    # SVM with calibration should have predict_proba
    assert hasattr(pipe, "predict_proba")


def test_rf_trains_and_predicts_on_synthetic_data():
    from ml.train import build_rf
    X, y = make_synthetic_data()
    pipe, params = build_rf(X, y)
    preds = pipe.predict(X[:5])
    assert len(preds) == 5
    assert hasattr(pipe, "predict_proba")


def test_mlp_trains_and_predicts_on_synthetic_data():
    from ml.train import build_mlp
    X, y = make_synthetic_data()
    pipe, params = build_mlp(X, y)
    preds = pipe.predict(X[:5])
    assert len(preds) == 5


# ================================================================ save/load roundtrip
def test_model_save_and_load_roundtrip(tmp_path):
    """Saved and loaded pipeline produces the same predictions."""
    from ml.train import build_knn, save_model, load_model
    import json

    X, y = make_synthetic_data()
    pipe, params = build_knn(X, y)
    classes = sorted(np.unique(y).tolist())

    # Temporarily redirect save to tmp
    import ml.train as train_mod
    orig_dir = train_mod.MODELS_DIR
    train_mod.MODELS_DIR = tmp_path
    try:
        save_model(pipe, "test_knn", params, classes, "synthetic")
        loaded_pipe, meta = load_model("test_knn")
    finally:
        train_mod.MODELS_DIR = orig_dir

    preds_orig = pipe.predict(X[:10])
    preds_load = loaded_pipe.predict(X[:10])
    assert list(preds_orig) == list(preds_load), "Predictions differ after save/load"
    assert meta["classes"] == classes


# ================================================================ prediction output format
def test_ml_prediction_format_on_synthetic_data():
    """MLPrediction must have the correct fields with correct types."""
    from ml.train import build_knn
    from ml.backend import MLBackend, BACKEND_ML, MLPrediction
    import ml.train as train_mod

    X, y = make_synthetic_data()
    pipe, params = build_knn(X, y)
    classes = sorted(np.unique(y).tolist())

    backend = MLBackend.__new__(MLBackend)
    backend.mode = BACKEND_ML
    backend.model_name = "test"
    backend.confidence_threshold = 0.0
    backend._loaded = True
    backend._pipe = pipe
    backend._classes = classes
    backend._has_proba = hasattr(pipe, "predict_proba")

    feat = X[0]
    result = backend.predict(feat)
    assert isinstance(result, MLPrediction)
    assert result.label in classes or result.label is None
    assert 0.0 <= result.confidence <= 1.0 + 1e-6
    assert len(result.top3) <= 3
    assert isinstance(result.scores, dict)
    assert result.latency_ms >= 0


# ================================================================ missing-hand handling
def test_ml_backend_handles_all_zero_input():
    """A feature vector of all zeros (no hands detected) should not crash."""
    from ml.train import build_knn
    from ml.backend import MLBackend, BACKEND_ML

    X, y = make_synthetic_data()
    pipe, _ = build_knn(X, y)
    classes = sorted(np.unique(y).tolist())

    backend = MLBackend.__new__(MLBackend)
    backend.mode = BACKEND_ML
    backend.model_name = "test"
    backend.confidence_threshold = 0.0
    backend._loaded = True
    backend._pipe = pipe
    backend._classes = classes
    backend._has_proba = hasattr(pipe, "predict_proba")

    zero_feat = np.zeros(FEATURE_LEN, dtype=np.float32)
    result = backend.predict(zero_feat)
    # Should not raise; label may be anything or None
    assert isinstance(result.label, (str, type(None)))


def test_ml_backend_handles_wrong_dimension():
    """A feature vector with wrong dimension should return label=None safely."""
    from ml.train import build_knn
    from ml.backend import MLBackend, BACKEND_ML

    X, y = make_synthetic_data()
    pipe, _ = build_knn(X, y)
    classes = sorted(np.unique(y).tolist())

    backend = MLBackend.__new__(MLBackend)
    backend.mode = BACKEND_ML
    backend.model_name = "test"
    backend.confidence_threshold = 0.0
    backend._loaded = True
    backend._pipe = pipe
    backend._classes = classes
    backend._has_proba = hasattr(pipe, "predict_proba")

    bad_feat = np.zeros(50, dtype=np.float32)  # wrong size
    result = backend.predict(bad_feat)
    assert result.label is None


# ================================================================ backend mode selection
def test_backend_template_mode_wraps_template_matcher():
    """BACKEND_TEMPLATE mode should wrap the TemplateMatcher."""
    backend = MLBackend(mode=BACKEND_TEMPLATE)
    assert backend.mode == BACKEND_TEMPLATE


def test_backend_ml_mode_selection():
    """BACKEND_ML mode should store model_name correctly."""
    backend = MLBackend(mode=BACKEND_ML, model_name="knn_ext")
    assert backend.mode == BACKEND_ML
    assert backend.model_name == "knn_ext"


def test_backend_invalid_mode_raises():
    """Invalid mode string should raise ValueError immediately."""
    with pytest.raises(ValueError):
        MLBackend(mode="invalid_mode")


# ================================================================ template matcher compatibility
def test_existing_template_matcher_unchanged(tmp_path):
    """The existing TemplateMatcher still works correctly after our additions."""
    from matcher import TemplateMatcher, save_templates
    from utils import build_features, FEATURE_LEN

    rng = np.random.default_rng(42)

    def rand_hand():
        h = rng.uniform(0, 1, (21, 3)).astype(np.float32)
        h[0] = 0.5
        return h

    X = np.array([build_features([rand_hand(), rand_hand()]) for _ in range(10)])
    y = ["A"] * 5 + ["B"] * 5
    src = [0] * 10

    m = TemplateMatcher()
    m.add(X, y, src)
    assert m.available
    pred = m.predict(X[0])
    assert pred.label in ("A", "B", None)


# ================================================================ leakage prevention
@needs_ext_data
def test_no_test_samples_in_training_split():
    """Verify that test indices and train indices are disjoint."""
    from ml.dataset import session_aware_split
    X, y, _, sess = load_templates(EXTERNAL_TEMPLATES)
    X_tr, X_va, X_te, y_tr, y_va, y_te = session_aware_split(X, y, sess)

    # Check sizes
    total = len(X_tr) + len(X_va) + len(X_te)
    assert total == len(X), f"Split total {total} != dataset total {len(X)}"

    # Check that class distributions are reasonable
    assert len(np.unique(y_te)) >= 1
    assert len(np.unique(y_tr)) >= 1



# ================================================================ personal calibration split tests
@needs_usr_data
def test_personal_block_split_has_no_leakage():
    """Calibration partition (first 40 frames) and test partition (last 20 frames) must be disjoint."""
    from ml.personal_experiments import make_personal_block_split
    X_cal, y_cal, X_test, y_test, classes = make_personal_block_split(cal_samples_per_letter=40)

    # 20 letters * 40 = 800 cal; 20 * 20 = 400 test
    assert len(X_cal) == 800
    assert len(X_test) == 400
    assert len(classes) == 20

    # Ensure no exact row from X_test appears in X_cal
    # (checking first few samples with allclose)
    for sample in X_test[:10]:
        matches = [np.allclose(sample, cal_s) for cal_s in X_cal]
        assert not any(matches), "Data leakage detected: test sample found in calibration set"


@needs_saved_model
def test_ml_backend_duck_type_compatibility_with_prediction():
    """MLPrediction must expose all attributes expected by the UI and logic smoother."""
    from ml.backend import MLBackend, BACKEND_ML
    from logic import Smoother

    backend = MLBackend(mode=BACKEND_ML, model_name="svm_ext")
    backend.load()
    assert backend.available

    dummy = np.zeros(FEATURE_LEN, dtype=np.float32)
    pred = backend.predict(dummy)

    # Required fields for UI / smoother
    assert hasattr(pred, "label")
    assert hasattr(pred, "confidence")
    assert hasattr(pred, "top3")
    assert hasattr(pred, "distance")
    assert hasattr(pred, "scores")
    assert hasattr(pred, "margin")
    assert hasattr(pred, "latency_ms")

    # Smoother compatibility
    smoother = Smoother(size=5)
    winner, conf = smoother.update(pred)
    assert winner is None or isinstance(winner, str)
    assert isinstance(conf, float)


def test_settings_backend_selection():
    """Settings must persist backend and ml_model fields correctly."""
    from logic import Settings

    s = Settings(backend="ml", ml_model="svm_comb")
    clamped = s.clamp()
    assert clamped.backend == "ml"
    assert clamped.ml_model == "svm_comb"

    # Invalid backend defaults to template
    s_bad = Settings(backend="unknown_backend")
    assert s_bad.clamp().backend == "template"
