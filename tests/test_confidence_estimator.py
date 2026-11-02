"""Tests for confidence_estimator (Task 4)."""
import dataclasses
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import confidence_estimator as ce  # noqa: E402
from confidence_estimator import (  # noqa: E402
    ConfidenceEstimator,
    Decision,
    probabilities_to_scores,
)

DT = 0.04  # 25 fps

GOOD_A = {"A": 0.30, "B": 0.90, "C": 1.20}   # distance .30, margin .60
GOOD_B = {"A": 0.90, "B": 0.30, "C": 1.20}
WEAK_A = {"A": 1.00, "B": 1.20, "C": 1.50}   # distance 1.0, margin .20
AMBIGUOUS = {"A": 0.40, "B": 0.45, "C": 1.20}  # margin .05
FAR = {"A": 2.00, "B": 3.00, "C": 4.00}      # beyond reject distance


def feed(est, dists, n, t0=0.0, dt=DT):
    out = []
    for i in range(n):
        out.append(est.update(dists, now=t0 + i * dt))
    return out


# --------------------------------------------------------------- module
def test_module_imports_without_circular_import():
    assert hasattr(ce, "ConfidenceEstimator")
    assert ce.ConfidenceEstimator is ConfidenceEstimator


def test_decision_is_frozen():
    d = ConfidenceEstimator().update({"A": 0.1, "B": 1.0}, now=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.label = "Z"


# --------------------------------------------------------- constructor
@pytest.mark.parametrize(
    "kwargs",
    [
        {"strong_distance": 2.0, "reject_distance": 1.0},
        {"strong_distance": -0.1},
        {"minimum_margin": 0.5, "strong_margin": 0.3},
        {"minimum_margin": -0.1},
        {"window_size": 0},
        {"window_size": -3},
        {"window_size": 2.5},
        {"window_size": True},
        {"stable_ratio": 0},
        {"stable_ratio": 1.1},
        {"min_stable_seconds": -1},
        {"max_gap_seconds": 0},
        {"strong_distance": float("nan")},
        {"min_stable_seconds": float("nan")},
        {"reject_distance": float("inf")},
        {"strong_margin": "0.3"},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ConfidenceEstimator(**kwargs)


def test_boundary_configuration_accepted():
    ConfidenceEstimator(strong_distance=0.5, reject_distance=0.5,
                        strong_margin=0.2, minimum_margin=0.2,
                        window_size=1, stable_ratio=1, min_stable_seconds=0)


# ------------------------------------------------------------- no hand
def test_no_hand_state():
    d = ConfidenceEstimator().update(GOOD_A, hands_present=False, now=0.0)
    assert d.status == "no_hand"
    assert d.display == "No hand detected"
    assert d.label is None and not d.accepted


def test_no_hand_resets_stability():
    est = ConfidenceEstimator()
    assert feed(est, GOOD_A, 10)[-1].status == "strong"
    est.update(GOOD_A, hands_present=False, now=1.0)
    first = est.update(GOOD_A, now=1.04)
    assert first.status == "likely"
    assert est.history.maxlen == 7 and len(est.history) == 1


# -------------------------------------------------- invalid / odd input
@pytest.mark.parametrize("dists", [{}, {"A": 0.1}])
def test_fewer_than_two_classes_uncertain(dists):
    d = ConfidenceEstimator().update(dists, now=0.0)
    assert d.status == "uncertain" and d.display == "Uncertain gesture"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, "abc", None])
def test_invalid_distance_values_uncertain_and_reset(bad):
    est = ConfidenceEstimator()
    feed(est, GOOD_A, 5)
    d = est.update({"A": bad, "B": 1.0}, now=0.5)
    assert d.status == "uncertain"
    assert len(est.history) == 0 and est.since is None


@pytest.mark.parametrize("bad", [None, [0.1, 0.9], (0.1, 0.9), "AB", 5])
def test_non_mapping_input_raises(bad):
    with pytest.raises(TypeError):
        ConfidenceEstimator().update(bad, now=0.0)


@pytest.mark.parametrize("bad_now", [float("nan"), float("inf"), "1"])
def test_invalid_timestamp_raises(bad_now):
    with pytest.raises(ValueError):
        ConfidenceEstimator().update(GOOD_A, now=bad_now)


def test_non_string_labels_are_stringified():
    d = ConfidenceEstimator().update({1: 0.2, 2: 0.9}, now=0.0)
    assert d.label == "1" and d.display == "1?"


def test_numpy_scalars_accepted():
    np = pytest.importorskip("numpy")
    d = ConfidenceEstimator().update(
        {"A": np.float32(0.2), "B": np.float64(0.9)}, now=0.0)
    assert d.label == "A"


# ------------------------------------------------ ranking / margin math
def test_ranking_ignores_dict_order_and_smaller_is_better():
    d = ConfidenceEstimator().update({"C": 1.2, "A": 0.3, "B": 0.9}, now=0.0)
    assert d.label == "A"
    assert d.best_distance == pytest.approx(0.3)
    assert d.second_distance == pytest.approx(0.9)
    assert d.margin == pytest.approx(0.6)


def test_tie_gives_zero_margin_and_uncertain():
    d = ConfidenceEstimator().update({"B": 0.4, "A": 0.4, "C": 2.0}, now=0.0)
    assert d.status == "uncertain" and d.margin == 0.0


def test_margin_uses_only_top_two():
    d = ConfidenceEstimator().update({"A": 0.2, "B": 0.5, "C": 0.5}, now=0.0)
    assert d.margin == pytest.approx(0.3)


# -------------------------------------------------------- thresholds
def test_reject_distance_boundary():
    est = ConfidenceEstimator()
    at = est.update({"A": 1.5, "B": 3.0}, now=0.0)
    over = est.update({"A": 1.5001, "B": 3.0}, now=0.04)
    assert at.status == "likely"
    assert over.status == "uncertain"


def test_minimum_margin_boundary():
    est = ConfidenceEstimator()
    at = est.update({"A": 0.30, "B": 0.42}, now=0.0)   # margin 0.12
    under = est.update({"A": 0.30, "B": 0.419}, now=0.04)
    assert at.status == "likely"
    assert under.status == "uncertain"


def test_uncertain_reports_diagnostics_but_no_label():
    d = ConfidenceEstimator().update(AMBIGUOUS, now=0.0)
    assert d.label is None and d.accepted is False
    assert d.best_distance == pytest.approx(0.40)
    assert d.margin == pytest.approx(0.05)


def test_far_match_is_uncertain():
    assert ConfidenceEstimator().update(FAR, now=0.0).status == "uncertain"


# ------------------------------------------------------- four states
def test_strong_after_stable_window():
    res = feed(ConfidenceEstimator(), GOOD_A, 7)
    assert [r.status for r in res[:6]] == ["likely"] * 6
    assert res[6].status == "strong"
    assert res[6].display == "A" and res[6].accepted
    assert res[6].stability == 1.0


def test_likely_display_has_question_mark_and_not_accepted():
    d = ConfidenceEstimator().update(GOOD_A, now=0.0)
    assert d.status == "likely" and d.display == "A?" and not d.accepted


def test_stable_but_weak_distance_is_never_strong():
    res = feed(ConfidenceEstimator(), WEAK_A, 30)
    assert all(r.status == "likely" for r in res)


def test_stable_but_small_margin_is_never_strong():
    dists = {"A": 0.30, "B": 0.50, "C": 1.0}   # margin .20 (>min, <strong)
    res = feed(ConfidenceEstimator(), dists, 30)
    assert all(r.status == "likely" for r in res)


# -------------------------------------------------- stability behaviour
def test_alternating_labels_never_strong():
    est = ConfidenceEstimator()
    for i in range(40):
        d = est.update(GOOD_A if i % 2 == 0 else GOOD_B, now=i * DT)
        assert d.status != "strong"


def test_single_good_frame_after_weak_run_is_not_strong():
    est = ConfidenceEstimator()
    feed(est, WEAK_A, 6)
    d = est.update(GOOD_A, now=6 * DT)
    assert d.status == "likely"


def test_time_gate_enforced_at_high_frame_rate():
    est = ConfidenceEstimator()
    res = feed(est, GOOD_A, 12, dt=0.02)   # 50 fps, window full at 7th frame
    assert all(r.status == "likely" for r in res[:10])   # t < 0.2 s
    assert res[10].status == "strong"                    # t == 0.2 s


def test_label_change_requires_new_stability():
    est = ConfidenceEstimator()
    assert feed(est, GOOD_A, 10)[-1].status == "strong"
    res = feed(est, GOOD_B, 6, t0=10 * DT)
    assert all(r.status != "strong" for r in res[:5])
    assert res[5].status == "strong" and res[5].label == "B"


def test_single_rejected_frame_breaks_strong_and_timer_restarts():
    est = ConfidenceEstimator()
    feed(est, GOOD_A, 10)
    bad = est.update(AMBIGUOUS, now=10 * DT)
    assert bad.status == "uncertain"
    after = est.update(GOOD_A, now=11 * DT)
    assert after.status == "likely"
    later = feed(est, GOOD_A, 10, t0=12 * DT)
    assert later[-1].status == "strong"


def test_stability_ratio_uses_frames_seen_not_window_size():
    est = ConfidenceEstimator()
    res = feed(est, GOOD_A, 3)
    assert res[-1].stability == 1.0


def test_window_size_one_with_zero_time_gate():
    est = ConfidenceEstimator(window_size=1, min_stable_seconds=0)
    assert est.update(GOOD_A, now=0.0).status == "strong"


# --------------------------------------------------- timing robustness
def test_pause_in_updates_does_not_count_as_stability():
    est = ConfidenceEstimator()
    feed(est, GOOD_A, 6)                       # window not yet full
    d = est.update(GOOD_A, now=5.0)            # 5 s without updates
    assert d.status == "likely"
    assert len(est.history) == 1


def test_backwards_timestamp_resets_state():
    est = ConfidenceEstimator()
    feed(est, GOOD_A, 10, t0=100.0)
    d = est.update(GOOD_A, now=50.0)
    assert d.status == "likely" and len(est.history) == 1


def test_default_clock_is_monotonic(monkeypatch):
    t = iter(range(100))
    monkeypatch.setattr(ce.time, "monotonic", lambda: next(t) * 0.04)
    est = ConfidenceEstimator()
    res = [est.update(GOOD_A) for _ in range(7)]
    assert res[-1].status == "strong"


def test_reset_clears_everything():
    est = ConfidenceEstimator()
    feed(est, GOOD_A, 10)
    est.reset()
    assert len(est.history) == 0
    assert est.last_label is None and est.since is None
    assert est.update(GOOD_A, now=1.0).status == "likely"


# ---------------------------------------- probabilities vs distances
def test_probabilities_to_scores_orders_correctly():
    s = probabilities_to_scores({"A": 0.9, "B": 0.09, "C": 0.01})
    assert s["A"] < s["B"] < s["C"]
    assert s["B"] - s["A"] == pytest.approx(math.log(0.9 / 0.09))


def test_probability_zero_is_finite_and_large():
    s = probabilities_to_scores({"A": 1.0, "B": 0.0})
    assert s["A"] == 0.0 and math.isfinite(s["B"]) and s["B"] > 20


@pytest.mark.parametrize("bad", [float("nan"), -0.5, 1.5, "x", None])
def test_invalid_probability_becomes_nan_and_uncertain(bad):
    s = probabilities_to_scores({"A": 0.9, "B": bad})
    assert math.isnan(s["B"])
    assert ConfidenceEstimator().update(s, now=0.0).status == "uncertain"


def test_raw_probabilities_must_not_be_used_as_distances():
    # Documents the pitfall: probabilities are larger-is-better, so feeding
    # them directly picks the WRONG class.
    wrong = ConfidenceEstimator().update({"A": 0.9, "B": 0.1}, now=0.0)
    assert wrong.label == "B"
    ok = ConfidenceEstimator().update(
        probabilities_to_scores({"A": 0.9, "B": 0.1}), now=0.0)
    assert ok.label == "A"


def test_ml_pipeline_with_separate_calibration():
    est = ConfidenceEstimator(strong_distance=0.10, reject_distance=2.0,
                              strong_margin=2.0, minimum_margin=0.5)
    probs = {"A": 0.97, "B": 0.02, "C": 0.01}
    res = [est.update(probabilities_to_scores(probs), now=i * DT)
           for i in range(7)]
    assert res[-1].status == "strong" and res[-1].label == "A"
    unsure = est.update(probabilities_to_scores(
        {"A": 0.45, "B": 0.40, "C": 0.15}), now=7 * DT)
    assert unsure.status == "uncertain"


def test_decision_type():
    assert isinstance(ConfidenceEstimator().update(GOOD_A, now=0.0), Decision)
