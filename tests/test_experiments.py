"""Tests for the experiment code (no webcam, no window, no real data).

Run:  python -m pytest tests/test_experiments.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments import data as D
from experiments import metrics as M
from experiments import palm_side as PS
from experiments import runner as R
from experiments.palm_labels import hand_items, labels_for, pack_frame
from experiments.regions import (FINGER_JOINTS, FINGERTIPS, PALM_WRIST, balanced_profile, feature_weights,
                                 region_columns)
from experiments.side_matcher import SideAwareMatcher, augment_with_side
from experiments.synthetic import (canonical_right_hand, make_letter_dataset, make_palm_side_dataset, place_hand)
from matcher import SRC_EXTERNAL, TemplateMatcher
from utils import FEATURE_LEN, FLAGS_START, INTER_START, build_features


# ------------------------------------------------------------------ regions
def test_regions_partition_the_hand():
    assert sorted(FINGERTIPS + FINGER_JOINTS + PALM_WRIST) == list(range(21))


def test_region_columns_do_not_overlap_and_cover_all_but_flags():
    cols = [region_columns(r) for r in ("fingertips", "finger_joints", "palm_wrist", "inter_hand")]
    allc = np.concatenate(cols)
    assert len(allc) == len(set(allc.tolist())) == FLAGS_START


def test_feature_weights_layout():
    w = feature_weights(tip=2, joint=1, palm=0.5, inter=3, flags=3)
    assert w.shape == (FEATURE_LEN,)
    assert np.all(w[4 * 3:4 * 3 + 3] == 2)               # thumb tip, hand 1
    assert np.all(w[63 + 8 * 3:63 + 8 * 3 + 3] == 2)     # index tip, hand 2
    assert np.all(w[5 * 3:5 * 3 + 3] == 0.5)             # index knuckle = palm region
    assert np.all(w[INTER_START:FLAGS_START] == 3)
    assert np.all(w[FLAGS_START:] == 3)
    wz = feature_weights(z=0.5)
    assert wz[2] == 0.5 and wz[1] == 1.0 and wz[INTER_START + 2] == 0.5 and wz[INTER_START] == 1.0


def test_balanced_profile_gives_regions_equal_total_influence():
    t, j, p, _ = balanced_profile()
    assert np.isclose(5 * t, 10 * j) and np.isclose(5 * t, 5 * p)


# ------------------------------------------------------------ matcher change
def _templates(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, FEATURE_LEN)).astype(np.float32)
    y = np.array(list("ABC"))[rng.integers(0, 3, n)]
    return X, y, rng


def test_matcher_default_behaviour_unchanged_by_feature_weights_option():
    X, y, rng = _templates()
    a = TemplateMatcher(); a.add(X, y, np.zeros(len(y)))
    w = np.ones(FEATURE_LEN, np.float32); w[INTER_START:FLAGS_START] = a.inter_weight; w[FLAGS_START:] = a.flag_weight
    b = TemplateMatcher(feature_weights=w); b.add(X, y, np.zeros(len(y)))
    Q = rng.normal(size=(5, FEATURE_LEN)).astype(np.float32)
    assert np.allclose(a.class_scores(Q), b.class_scores(Q), atol=1e-5)


def test_matcher_rejects_wrong_weight_length():
    X, y, _ = _templates()
    m = TemplateMatcher(feature_weights=np.ones(10)); m.add(X, y, np.zeros(len(y)))
    with pytest.raises(ValueError):
        m.class_scores(X[:1])


def test_zero_weight_switches_a_feature_off():
    X, y, rng = _templates()
    w = np.ones(FEATURE_LEN, np.float32); w[0] = 0.0
    m = TemplateMatcher(feature_weights=w); m.add(X, y, np.zeros(len(y)))
    q = X[:1].copy(); q2 = q.copy(); q2[0, 0] += 100.0
    assert np.allclose(m.class_scores(q), m.class_scores(q2), atol=1e-4)


# ------------------------------------------------------------ palm side
@pytest.mark.parametrize("chir,facing", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_signed_area_sign_follows_chirality_times_side(chir, facing):
    """palm towards the camera  <=>  s * chirality < 0 (the rule used everywhere in palm_side.py)."""
    h = place_hand(canonical_right_hand([0, 0, 0, 0], 0.3), chir, facing)
    hands, present = PS.hands_from_landmarks(h[None, None].repeat(2, axis=1), 1.0)
    s = PS.signed_area(hands)[0, 0]
    assert (s * chir < 0) == (facing == 1)


def test_geometry_methods_on_rotated_synthetic_hands():
    d = make_palm_side_dataset(10, seed=3, tilt_deg=30)
    hands, present = PS.hands_from_landmarks(d["landmarks"], float(d["aspect"]))
    for v in (PS.g1_label(hands, present, d["mp_label"]), PS.g3_intrinsic(hands, present)):
        r = PS.evaluate_side(v, d["side"], present, 0.15)
        assert r["accuracy_all"] > 0.99 and r["n"] == present.sum()
    # the swapped label mapping must be (almost) always wrong
    r = PS.evaluate_side(PS.g1_label(hands, present, -d["mp_label"]), d["side"], present, 0.15)
    assert r["accuracy_all"] < 0.01


def test_slot_rule_fails_exactly_where_expected():
    """G2 is right for natural two-hand frames and wrong for a lone LEFT hand (it always sits in slot 0)."""
    d = make_palm_side_dataset(10, seed=4)
    hands, present = PS.hands_from_landmarks(d["landmarks"], float(d["aspect"]))
    v = PS.g2_slot(hands, present)
    two = (present.sum(axis=1) == 2)[:, None] & np.ones((1, 2), bool)
    left_alone = d["true_hand"] == -1
    assert PS.evaluate_side(v, d["side"], present, 0.15, mask=two)["accuracy_all"] > 0.99
    assert PS.evaluate_side(v, d["side"], present, 0.15, mask=left_alone)["accuracy_all"] < 0.01


def test_edge_on_hand_is_undecided():
    h = place_hand(canonical_right_hand([0, 0, 0, 0], 0.3), 1, 1, yaw=np.pi / 2)   # seen from the side
    hands, present = PS.hands_from_landmarks(h[None, None].repeat(2, axis=1), 1.0)
    v = PS.g2_slot(hands, present)
    assert abs(v[0, 0]) < 0.3


def test_hands_from_features_matches_hands_from_landmarks():
    d = make_palm_side_dataset(3, seed=5)
    n = 12
    aspect = float(d["aspect"])
    hands_raw, present = PS.hands_from_landmarks(d["landmarks"][:n], aspect)
    feats = np.stack([build_features([h for h in d["landmarks"][i] if not np.isnan(h[0, 0])], aspect) for i in range(n)])
    hands_f, present_f = PS.hands_from_features(feats)
    assert np.array_equal(present, present_f)
    assert np.allclose(hands_raw[present], hands_f[present], atol=1e-4)


def test_knn_side_learns_labels():
    d = make_palm_side_dataset(10, seed=6)
    hands, present = PS.hands_from_landmarks(d["landmarks"], float(d["aspect"]))
    knn = PS.KNNSide(5).fit(hands, present, d["side"])
    v = knn.score(hands, present)
    assert PS.evaluate_side(v, d["side"], present, 0.15)["accuracy_decided"] > 0.95


def test_evaluate_side_counts_unsure_as_wrong():
    v = np.array([[0.9, 0.05]]); side = np.array([[1, 1]], np.int8); present = np.array([[True, True]])
    r = PS.evaluate_side(v, side, present, 0.15)
    assert r["accuracy_all"] == 0.5 and r["accuracy_decided"] == 1.0 and r["coverage"] == 0.5


# ------------------------------------------------------------ recording format
def test_pack_frame_orders_left_to_right_and_pads():
    a = np.zeros((21, 3), np.float32); a[:, 0] = 0.7
    b = np.zeros((21, 3), np.float32); b[:, 0] = 0.2
    L, M_, S = pack_frame([(a, +1, 0.9), (b, -1, 0.8)])
    assert L[0, 0, 0] == pytest.approx(0.2) and M_[0] == -1 and M_[1] == 1
    L1, M1, _ = pack_frame([(a, +1, 0.9)])
    assert np.isnan(L1[1]).all() and M1[1] == 0


def test_labels_for_checks_hand_count():
    assert labels_for("right_palm", 2) is None
    side, true_hand = labels_for("right_back", 1)
    assert side[0] == -1 and true_hand[0] == 1
    side, true_hand = labels_for("both_palms", 2)
    assert list(side) == [1, 1] and list(true_hand) == [0, 0]


def test_hand_items_reads_mediapipe_labels():
    pb = pytest.importorskip("mediapipe.framework.formats.classification_pb2")

    class FakeResult:
        multi_handedness = [pb.ClassificationList(classification=[pb.Classification(label="Right", score=0.9)]),
                            pb.ClassificationList(classification=[pb.Classification(label="Left", score=0.8)])]

    h = np.zeros((21, 3), np.float32)
    items = hand_items([h, h], FakeResult())
    assert [i[1] for i in items] == [1, -1]

    class NoHands:
        multi_handedness = None
    assert hand_items([], NoHands()) == []


# --------------------------------------------------------------- metrics
def test_softmax_matches_matcher_predict():
    X, y, rng = _templates(80, seed=2)
    m = TemplateMatcher(reject_distance=1e9); m.add(X, y, np.zeros(len(y)))
    for _ in range(5):
        q = rng.normal(size=FEATURE_LEN).astype(np.float32)
        p = m.predict(q)
        r = M.summarise_predictions(m.class_scores(q), np.array([0]), m.temperature)
        assert p.label == m._classes[r["pred"][0]]
        assert p.confidence == pytest.approx(float(r["conf"][0]), abs=1e-5)


def test_ece_and_auroc_basics():
    conf = np.array([0.9] * 10); correct = np.array([True] * 9 + [False])
    assert M.ece(conf, correct) == pytest.approx(0.0, abs=1e-9)
    assert M.auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, True, False, False])) == 1.0
    assert M.auroc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    assert np.isnan(M.auroc(np.array([0.5, 0.6]), np.array([True, True])))


def test_fitted_temperature_does_not_worsen_ece_on_overconfident_scores():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 2, (400, 26)); y = rng.integers(0, 26, 400); s[np.arange(400), y] -= 0.8
    r0 = M.summarise_predictions(s, y, 0.12)
    T = M.fit_temperature(s, y)
    r1 = M.summarise_predictions(s, y, T)
    assert M.ece(r1["conf"], r1["correct"]) <= M.ece(r0["conf"], r0["correct"]) + 1e-9


def test_cluster_bootstrap_is_wider_when_values_are_correlated_within_clusters():
    """Near-identical neighbouring frames: resampling whole blocks must not claim more certainty than resampling frames."""
    rng = np.random.default_rng(0)
    g_big = np.repeat(np.arange(10), 30)
    vals = rng.normal(size=10)[g_big] + rng.normal(scale=0.05, size=300)      # strong cluster effect
    p, lo, hi = M.cluster_bootstrap_diff(lambda i: vals[i].mean(), g_big, 400, 0)
    assert lo <= p <= hi
    _, lo_f, hi_f = M.cluster_bootstrap_diff(lambda i: vals[i].mean(), np.arange(300), 400, 0)   # every frame its own group
    assert (hi - lo) > 2 * (hi_f - lo_f)


# ------------------------------------------------------------- side matcher
def test_side_matcher_with_zero_options_equals_plain_matcher():
    Xe, ye, _, _ = make_letter_dataset(10, 2, seed=1)
    w = feature_weights(1, 1, 1, 2)
    plain = TemplateMatcher(feature_weights=w); plain.add(Xe, ye, np.zeros(len(ye)))
    aware = SideAwareMatcher(w); aware.add(augment_with_side(Xe), ye, np.zeros(len(ye)))
    Q = Xe[:20]
    assert np.allclose(plain.class_scores(Q), aware.class_scores(augment_with_side(Q)), atol=1e-4)


def test_side_penalty_pushes_away_templates_with_opposite_side():
    Xe, ye, _, _ = make_letter_dataset(10, 2, seed=1)
    aug = augment_with_side(Xe)
    w = feature_weights(1, 1, 1, 2)
    off = SideAwareMatcher(w); off.add(aug, ye, np.zeros(len(ye)))
    on = SideAwareMatcher(w, side_penalty=1.0); on.add(aug, ye, np.zeros(len(ye)))
    q = aug[:1].copy()
    q[0, FEATURE_LEN:] *= -1                           # query with both sides flipped
    assert np.all(on.class_scores(q) >= off.class_scores(q) - 1e-6)
    assert np.any(on.class_scores(q) > off.class_scores(q) + 0.5)


# ------------------------------------------------------------------ protocols
def _small_ds():
    Xe, ye, Xu, yu = make_letter_dataset(20, 20, seed=2, hard=2.0)
    return D.from_arrays(Xe, ye, Xu, yu, 5, synthetic=True)


def test_blocks_are_consecutive_and_balanced():
    blocks = D.assign_blocks(["A"] * 20 + ["B"] * 10, 5)
    assert list(blocks[:20]) == sorted(blocks[:20]) and set(blocks) == {0, 1, 2, 3, 4}
    assert np.bincount(blocks[:20]).tolist() == [4] * 5


def test_protocol_c_has_no_leak_between_train_and_test():
    ds = _small_ds()
    for f in D.protocol_c(ds):
        assert not set(f.train.tolist()) & set(f.test.tolist())
        assert (ds.src[f.test] == 1).all()
        test_blocks = set(ds.block[f.test].tolist())
        assert not test_blocks & set(ds.block[f.train][ds.src[f.train] == 1].tolist())


def test_protocol_a_split_is_disjoint_and_external_only():
    ds = _small_ds()
    f = D.protocol_a(ds)[0]
    assert not set(f.train.tolist()) & set(f.test.tolist())
    assert (ds.src[f.train] == SRC_EXTERNAL).all() and (ds.src[f.test] == SRC_EXTERNAL).all()


def test_protocol_b_only_uses_letters_present_in_imported_data():
    Xe, ye, Xu, yu = make_letter_dataset(10, 10, seed=3)
    keep = ~np.isin(ye, ["H", "J", "Y"])
    ds = D.from_arrays(Xe[keep], ye[keep], Xu, yu, 5)
    f = D.protocol_b(ds)[0]
    assert not set(ds.y[f.test]) & {"H", "J", "Y"}


def test_inner_folds_never_touch_outer_test_data():
    ds = _small_ds()
    outer = D.protocol_c(ds)[0]
    for inner in D.inner_folds(ds, outer.train, ext_cap=10):
        assert set(inner.train.tolist()) | set(inner.test.tolist()) <= set(outer.train.tolist())


def test_protocols_degrade_gracefully_without_user_or_external_data():
    Xe, ye, Xu, yu = make_letter_dataset(8, 8, seed=4)
    only_ext = D.from_arrays(Xe, ye, np.zeros((0, 132)), [], 5)
    assert list(D.all_protocols(only_ext)) == ["A"]
    only_user = D.from_arrays(np.zeros((0, 132)), [], Xu, yu, 5)
    assert list(D.all_protocols(only_user)) == ["C"]


# -------------------------------------------------------------- runner / e2e
def test_evaluate_returns_one_score_row_per_query_and_26_columns():
    ds = _small_ds()
    folds = D.protocol_c(ds)[:2]
    r = R.evaluate(lambda f: R.Config("t"), ds, folds)
    assert r.scores.shape == (sum(len(f.test) for f in folds), 26)
    s = r.summary()
    assert 0.0 <= s["accuracy"] <= 1.0 and s["n_queries"] == len(r.y_idx)


def test_tuning_prefers_the_profile_closest_to_ones_on_ties():
    ds = _small_ds()
    best, table = R.tune_weights(ds, D.protocol_c(ds)[0].train, grid=(1.0,), ext_cap=10)
    assert best == (1.0, 1.0, 1.0, 1.0) and len(table) == 1


def test_paired_comparison_of_a_result_with_itself_is_zero():
    ds = _small_ds()
    r = R.evaluate(lambda f: R.Config("t"), ds, D.protocol_c(ds)[:2])
    out = R.paired_comparison(r, r, n_boot=50)
    assert out["delta_accuracy"][0] == 0 and out["delta_mean_confidence"][0] == 0


@pytest.mark.slow
def test_full_runner_writes_report(tmp_path):
    from experiments.run_experiments import main
    main(["--synthetic", "--quick", "--n-boot", "40", "--out", str(tmp_path)])
    text = (tmp_path / "final_report.md").read_text()
    assert "SYNTHETIC DATA" in text and "## 4. Final performance comparison" in text
    assert (tmp_path / "results.json").exists() and (tmp_path / "fig_exp1_profiles.png").exists()
