"""Tests that need no webcam and no window.

Run:  python -m pytest tests -q
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

import calibrate
from logic import HoldToAdd, PracticeSession, Settings, Smoother
from matcher import SRC_EXTERNAL, SRC_USER, Prediction, TemplateMatcher, load_templates, save_templates
from utils import (EXTERNAL_TEMPLATES, FEATURE_LEN, FLAGS_START, HAND_LEN, INTER_START, LETTERS,
                   build_features, chart_cell, mirror_hands)

RNG = np.random.default_rng(0)


def random_hand() -> np.ndarray:
    """A made-up hand in MediaPipe-like coordinates."""
    base = RNG.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
    base[0] = 0
    return base + np.array([0.5, 0.5, 0.0], dtype=np.float32)


# --------------------------------------------------------------- features
def test_feature_length_for_zero_one_two_hands():
    assert FEATURE_LEN == 132
    zero = build_features([])
    assert zero.shape == (FEATURE_LEN,) and not zero.any()
    one = build_features([random_hand()])
    assert one.shape == (FEATURE_LEN,)
    assert one[FLAGS_START] == 1 and one[FLAGS_START + 1] == 0
    assert not one[HAND_LEN:2 * HAND_LEN].any()          # second hand slot is empty
    two = build_features([random_hand(), random_hand()])
    assert two[FLAGS_START] == 1 and two[FLAGS_START + 1] == 1


def test_wrist_relative_and_scaled():
    f = build_features([random_hand()])
    assert np.allclose(f[:3], 0)                           # wrist is the origin
    xy = f[:HAND_LEN].reshape(21, 3)[:, :2]
    assert np.isclose(np.max(np.linalg.norm(xy, axis=1)), 1.0, atol=1e-5)


def test_hand_order_does_not_matter():
    a, b = random_hand(), random_hand()
    b[:, 0] += 0.3                                         # b is clearly to the right of a
    assert np.allclose(build_features([a, b]), build_features([b, a]))


def test_position_and_size_do_not_matter():
    a, b = random_hand(), random_hand()
    b[:, 0] += 0.3
    base = build_features([a, b])
    shifted = build_features([a + np.array([0.1, -0.05, 0], np.float32), b + np.array([0.1, -0.05, 0], np.float32)])
    assert np.allclose(base, shifted, atol=1e-4)
    scaled = build_features([a * 0.7, b * 0.7])
    assert np.allclose(base, scaled, atol=1e-4)


def test_mirror_twice_is_identity():
    h = random_hand()
    assert np.allclose(mirror_hands(mirror_hands([h]))[0], h, atol=1e-6)


def test_chart_cells_cover_all_letters():
    rects = [chart_cell(c, 582, 357) for c in LETTERS]
    assert all(r is not None for r in rects)
    assert chart_cell("1", 582, 357) is None


# ---------------------------------------------------------------- matcher
def synthetic_templates(n_letters=8, per_letter=30, noise=0.03):
    protos = [build_features([random_hand(), random_hand()]) for _ in range(n_letters)]
    X, y = [], []
    for i, p in enumerate(protos):
        for _ in range(per_letter):
            X.append(p + RNG.normal(0, noise, p.shape).astype(np.float32))
            y.append(LETTERS[i])
    return protos, np.stack(X), y


def test_matcher_recognises_noisy_prototypes_and_rejects_far_input():
    protos, X, y = synthetic_templates()
    m = TemplateMatcher(reject_distance=2.0)
    m.add(X, y, [SRC_EXTERNAL] * len(y))
    hits = 0
    for i, p in enumerate(protos):
        pred = m.predict(p + RNG.normal(0, 0.03, p.shape).astype(np.float32))
        hits += pred.label == LETTERS[i]
        assert abs(sum(pc for _, pc in pred.top3)) <= 1.0 + 1e-6
    assert hits == len(protos)
    far = m.predict(RNG.normal(0, 3.0, FEATURE_LEN).astype(np.float32))
    assert far.label is None                                # too far from everything


def test_matcher_prefers_user_templates_when_tied():
    p = build_features([random_hand(), random_hand()])
    m = TemplateMatcher(user_weight=0.5)
    m.add(np.stack([p, p]), ["A", "B"], [SRC_EXTERNAL, SRC_USER])
    query = p + 0.1                                         # not identical, so distances are above zero
    assert m.predict(query).label == "B"
    m.configure(user_weight=1.0)                            # equal weight: the first template wins the tie
    assert m.predict(query).label == "A"


def test_matcher_without_templates_is_safe():
    m = TemplateMatcher()
    assert not m.available
    assert m.predict(np.zeros(FEATURE_LEN, np.float32)).label is None


def test_templates_roundtrip(tmp_path):
    _, X, y = synthetic_templates(3, 5)
    path = tmp_path / "t.npz"
    save_templates(path, X, y, [SRC_USER] * len(y), ["s1"] * len(y))
    X2, y2, src2, sess2 = load_templates(path)
    assert X2.shape == X.shape and list(y2) == y and set(src2) == {SRC_USER} and set(sess2) == {"s1"}


def test_wrong_feature_length_gives_clear_error(tmp_path):
    path = tmp_path / "bad.npz"
    save_templates(path, np.zeros((2, 10), np.float32), ["A", "B"], [0, 0])
    with pytest.raises(ValueError, match="expected 132"):
        TemplateMatcher.from_files([path])


# ------------------------------------------------------- smoothing / hold logic
def pred(label, conf=0.95):
    return Prediction(label, conf if label else 0.0)


def test_smoother_majority_vote():
    s = Smoother(5)
    out = [s.update(pred("A")) for _ in range(3)]
    assert out[-1][0] == "A"
    s.update(pred("B"))
    assert s.update(pred(None))[0] == "A"                   # A still holds 3 of 5
    s.update(pred(None))
    assert s.update(pred(None))[0] is None                  # no majority any more


def run_stream(hold, frames, dt=0.05, start=0.0):
    """Feed (label, conf, hands) frames at fixed time steps. Returns accepted events."""
    events, t = [], start
    for label, conf, hands in frames:
        ev = hold.update(label, conf, t, hands)
        if ev:
            events.append((round(t, 2), ev.kind, ev.letter))
        t += dt
    return events


def test_letter_added_after_hold_time():
    hold = HoldToAdd(hold_time=1.0, threshold=0.8)
    events = run_stream(hold, [("A", 0.95, True)] * 30)     # 1.5 s
    assert [(k, l) for _, k, l in events] == [("letter", "A")]
    assert events[0][0] >= 1.0


def test_same_letter_not_added_twice_without_a_break():
    hold = HoldToAdd(hold_time=0.5)
    events = run_stream(hold, [("A", 0.95, True)] * 80)     # 4 s of the same sign
    assert len(events) == 1


def test_same_letter_added_again_after_release():
    hold = HoldToAdd(hold_time=0.5, release_time=0.6)
    frames = [("L", 0.95, True)] * 15 + [(None, 0.0, False)] * 20 + [("L", 0.95, True)] * 15
    events = run_stream(hold, frames)
    assert [l for _, _, l in events] == ["L", "L"]


def test_different_stable_letter_unblocks():
    hold = HoldToAdd(hold_time=0.5)
    frames = [("A", 0.95, True)] * 15 + [("B", 0.95, True)] * 15 + [("A", 0.95, True)] * 15
    events = run_stream(hold, frames)
    assert [l for _, _, l in events] == ["A", "B", "A"]


def test_idle_low_confidence_and_no_hands_add_nothing():
    hold = HoldToAdd(hold_time=0.5, threshold=0.8)
    frames = ([("idle", 0.99, True)] * 40 + [("A", 0.5, True)] * 40 + [(None, 0.0, False)] * 40)
    assert run_stream(hold, frames) == []


def test_interrupted_hold_restarts():
    hold = HoldToAdd(hold_time=1.0)
    frames = [("A", 0.95, True)] * 15 + [("B", 0.95, True)] * 10 + [("A", 0.95, True)] * 15
    assert run_stream(hold, frames) == []                   # nothing was held for a full second


def test_auto_space_once_per_absence_and_never_at_start():
    hold = HoldToAdd(hold_time=0.3, auto_space=True, auto_space_delay=1.0)
    assert run_stream(hold, [(None, 0.0, False)] * 60) == []          # nothing typed yet, so no space
    hold.reset()
    frames = [("A", 0.95, True)] * 10 + [(None, 0.0, False)] * 100    # 5 s absent
    events = run_stream(hold, frames)
    assert [k for _, k, _ in events] == ["letter", "space"]           # exactly one space
    hold.set_text_state("A ")
    assert run_stream(hold, [(None, 0.0, False)] * 100, start=20) == []


def test_practice_only_letter_ignores_other_letters():
    hold = HoldToAdd(hold_time=0.5, only_letter="C")
    assert run_stream(hold, [("D", 0.95, True)] * 40) == []
    hold.reset()
    assert [l for _, _, l in run_stream(hold, [("C", 0.95, True)] * 20)] == ["C"]


def test_practice_session_never_repeats_target_in_a_row():
    session = PracticeSession(["A", "B", "C"], random.Random(1))
    targets = [session.next_target() for _ in range(50)]
    assert all(a != b for a, b in zip(targets, targets[1:]))
    session.succeed(); session.succeed(); session.skip()
    assert (session.score, session.streak, session.best_streak) == (2, 0, 2)


def test_settings_load_clamp_and_ignore_bad_values(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hold_time": 99, "threshold": "oops", "smoothing": 0, "mirror": False, "junk": 1}))
    s = Settings.load(path)
    assert s.hold_time == 2.5 and s.threshold == 0.80 and s.smoothing == 1 and s.mirror is False
    path.write_text("not json")
    assert Settings.load(path).hold_time == 1.0
    s.hold_time = 1.7
    s.save(path)
    assert Settings.load(path).hold_time == 1.7


# --------------------------------------------------- real imported data (optional)
needs_data = pytest.mark.skipif(not EXTERNAL_TEMPLATES.exists(), reason="run import_dataset.py first")


@needs_data
def test_imported_templates_have_expected_shape_and_letters():
    X, y, src, _ = load_templates(EXTERNAL_TEMPLATES)
    assert X.shape[1] == FEATURE_LEN and np.isfinite(X).all()
    letters = set(y.tolist())
    assert letters <= set(LETTERS)
    assert {"A", "B", "C", "W", "X"} <= letters             # C comes from the one-hand file
    assert not letters & {"H", "J", "Y"}                    # not in the dataset


@needs_data
def test_held_out_accuracy_on_imported_data():
    from evaluate import accuracy, make_matcher, split_indices
    X, y, _, _ = load_templates(EXTERNAL_TEMPLATES)
    tr, te = split_indices(y, 0.2, seed=3)
    assert accuracy(make_matcher(X[tr], y[tr]), X[te], y[te]) > 0.95


# --------------------------------------------- calibration flow with a fake camera
class FakeCap:
    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        pass


class FakeTracker:
    def __init__(self, *a, **k):
        self.hands = [random_hand(), random_hand()]

    def process(self, frame):
        return self.hands, None

    def draw(self, frame, result):
        pass

    def close(self):
        pass


def test_calibration_flow_saves_samples_for_chosen_letters(tmp_path, monkeypatch):
    import cv2
    out = tmp_path / "templates_user.npz"
    keys = iter([32] + [255] * 40 + [32] + [255] * 40 + [ord("q")])
    monkeypatch.setattr(calibrate, "USER_TEMPLATES", out)
    monkeypatch.setattr(calibrate, "COUNTDOWN_S", 0)
    monkeypatch.setattr(calibrate, "open_camera", lambda i: FakeCap())
    monkeypatch.setattr(calibrate, "HandTracker", FakeTracker)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys, ord("q")))
    monkeypatch.setattr("sys.argv", ["calibrate.py", "--letters", "H", "J", "--samples", "6", "--interval", "0"])
    calibrate.main()
    X, y, src, sess = load_templates(out)
    assert X.shape == (12, FEATURE_LEN)
    assert sorted(set(y.tolist())) == ["H", "J"] and set(src) == {SRC_USER}
    assert list(y).count("H") == 6 and list(y).count("J") == 6
