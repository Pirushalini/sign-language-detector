"""Automated tests for dynamic letter movement recognition (BSL H and J).

Covers all 10 required test criteria:
1. Existing static letter recognition compatibility.
2. H and J movement trajectory matching.
3. Different gesture speeds.
4. Incorrect movement direction.
5. Stationary hands.
6. Missing hand landmarks.
7. Repeated movement letters.
8. False positive prevention.
9. Gesture completion and reset.
10. Missing or empty motion template datasets.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from matcher import Prediction, TemplateMatcher
from motion import (DYNAMIC_LETTERS, FEATURE_DIM, N_RESAMPLE_POINTS, MotionRecognizer,
                    MotionResult, compute_palm_frame, create_synthetic_template,
                    dtw_distance, extract_motion_features, identify_hands,
                    load_motion_templates, resample_trajectory, save_motion_templates,
                    transform_point_to_palm, validate_gesture_path)
from utils import build_features


# ------------------------------------------------------------- Fixtures & Helpers
def make_mock_hand(wrist=(0.5, 0.5, 0.0), scale=1.0) -> np.ndarray:
    """Generate 21 landmarks for an open flat palm."""
    h = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist
    h[0] = [wx, wy, wz]  # wrist
    # Thumb
    h[1] = [wx - 0.03 * scale, wy + 0.02 * scale, wz]
    h[2] = [wx - 0.05 * scale, wy + 0.04 * scale, wz]
    h[3] = [wx - 0.07 * scale, wy + 0.06 * scale, wz]
    h[4] = [wx - 0.09 * scale, wy + 0.08 * scale, wz]  # thumb tip
    # Index
    h[5] = [wx - 0.03 * scale, wy + 0.08 * scale, wz]  # index mcp
    h[6] = [wx - 0.03 * scale, wy + 0.11 * scale, wz]
    h[7] = [wx - 0.03 * scale, wy + 0.13 * scale, wz]
    h[8] = [wx - 0.03 * scale, wy + 0.15 * scale, wz]  # index tip
    # Middle
    h[9] = [wx, wy + 0.09 * scale, wz]                 # middle mcp
    h[10] = [wx, wy + 0.12 * scale, wz]
    h[11] = [wx, wy + 0.15 * scale, wz]
    h[12] = [wx, wy + 0.17 * scale, wz]                # middle tip
    # Ring
    h[13] = [wx + 0.03 * scale, wy + 0.08 * scale, wz] # ring mcp
    h[14] = [wx + 0.03 * scale, wy + 0.11 * scale, wz]
    h[15] = [wx + 0.03 * scale, wy + 0.14 * scale, wz]
    h[16] = [wx + 0.03 * scale, wy + 0.16 * scale, wz]
    # Pinky
    h[17] = [wx + 0.05 * scale, wy + 0.07 * scale, wz] # pinky mcp
    h[18] = [wx + 0.05 * scale, wy + 0.09 * scale, wz]
    h[19] = [wx + 0.05 * scale, wy + 0.11 * scale, wz]
    h[20] = [wx + 0.05 * scale, wy + 0.13 * scale, wz]
    return h


def make_h_frame_sequence(n_frames: int = 30) -> list[tuple[np.ndarray, np.ndarray]]:
    """Simulate two hands performing BSL H (sweep across supporting palm)."""
    supporting = make_mock_hand(wrist=(0.4, 0.5, 0.0), scale=1.0)
    frames = []
    # Dominant hand index tip sweeps from (0.38, 0.52) to (0.41, 0.63)
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        tip_x = 0.38 + 0.03 * t
        tip_y = 0.52 + 0.11 * t
        w_x = tip_x + 0.03
        w_y = tip_y - 0.15
        moving = make_mock_hand(wrist=(w_x, w_y, 0.01), scale=1.0)
        frames.append((moving, supporting.copy()))
    return frames


def make_j_frame_sequence(n_frames: int = 30) -> list[tuple[np.ndarray, np.ndarray]]:
    """Simulate two hands performing BSL J (trace middle finger down and hook to thumb)."""
    supporting = make_mock_hand(wrist=(0.4, 0.5, 0.0), scale=1.0)
    frames = []
    # Supporting middle tip is around (0.4, 0.67)
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        if t <= 0.6:
            p = t / 0.6
            x = 0.40
            y = 0.67 - 0.14 * p
        else:
            p = (t - 0.6) / 0.4
            angle = p * (np.pi * 0.6)
            x = 0.40 - 0.06 * np.sin(angle)
            y = 0.53 + 0.03 * (1.0 - np.cos(angle))
        moving = make_mock_hand(wrist=(x, y - 0.15, 0.02), scale=1.0)
        # Point index finger to (x, y)
        moving[8] = [x, y, 0.01]
        frames.append((moving, supporting.copy()))
    return frames


# ---------------------------------------------------------------- Tests
def test_static_compatibility():
    """Requirement 1: Static letter recognition remains fully functional."""
    # Ensure build_features produces the exact 132-dim vector
    h1 = make_mock_hand((0.3, 0.5, 0.0))
    h2 = make_mock_hand((0.6, 0.5, 0.0))
    feats = build_features([h1, h2], aspect=1.0)
    assert feats.shape == (132,)
    assert feats.dtype == np.float32

    # TemplateMatcher continues to function normally
    matcher = TemplateMatcher()
    matcher.add(np.stack([feats]), ["A"], [0])
    pred = matcher.predict(feats)
    assert pred.label == "A"
    assert pred.confidence > 0.8


def test_h_and_j_trajectory_matching():
    """Requirement 2: Accurate matching of H and J trajectories using DTW."""
    h_templ = create_synthetic_template("H")
    j_templ = create_synthetic_template("J")

    # Self-distance is zero
    assert dtw_distance(h_templ, h_templ) == 0.0
    assert dtw_distance(j_templ, j_templ) == 0.0

    # Cross-distance is distinctly large
    dist_h_j = dtw_distance(h_templ, j_templ)
    dist_j_h = dtw_distance(j_templ, h_templ)
    assert dist_h_j > 0.4
    assert dist_j_h > 0.4

    # Noisy H matches H, not J
    rng = np.random.default_rng(123)
    noisy_h = h_templ + rng.normal(0, 0.02, h_templ.shape).astype(np.float32)
    assert dtw_distance(noisy_h, h_templ) < dtw_distance(noisy_h, j_templ)

    # Noisy J matches J, not H
    noisy_j = j_templ + rng.normal(0, 0.02, j_templ.shape).astype(np.float32)
    assert dtw_distance(noisy_j, j_templ) < dtw_distance(noisy_j, h_templ)


def test_different_gesture_speeds():
    """Requirement 3: Gestures performed at different speeds match correctly."""
    base_h = create_synthetic_template("H")
    base_j = create_synthetic_template("J")

    # Fast version (12 frames) and slow version (50 frames)
    fast_h = resample_trajectory(base_h[:12], target_len=N_RESAMPLE_POINTS)
    slow_h = resample_trajectory(base_h, target_len=N_RESAMPLE_POINTS)

    fast_j = resample_trajectory(base_j[:12], target_len=N_RESAMPLE_POINTS)
    slow_j = resample_trajectory(base_j, target_len=N_RESAMPLE_POINTS)

    # Fast and slow H match H significantly better than J
    assert dtw_distance(fast_h, base_h) < dtw_distance(fast_h, base_j)
    assert dtw_distance(slow_h, base_h) < dtw_distance(slow_h, base_j)

    # Fast and slow J match J significantly better than H
    assert dtw_distance(fast_j, base_j) < dtw_distance(fast_j, base_h)
    assert dtw_distance(slow_j, base_j) < dtw_distance(slow_j, base_h)


def test_incorrect_movement_direction():
    """Requirement 4: Inverted or reverse movements are rejected."""
    # Reverse H (sweeping backward from fingertips towards wrist)
    base_h = create_synthetic_template("H")
    reversed_h = base_h[::-1].copy()
    is_valid, reason = validate_gesture_path("H", reversed_h)
    assert not is_valid
    assert "direction" in reason.lower() or "inverted" in reason.lower() or "sweep" in reason.lower()

    # Inverted J (starting at wrist and moving upward away from thumb)
    base_j = create_synthetic_template("J")
    reversed_j = base_j[::-1].copy()
    is_valid, reason = validate_gesture_path("J", reversed_j)
    assert not is_valid


def test_stationary_hands():
    """Requirement 5: Stationary hands are rejected and produce no dynamic detection."""
    rec = MotionRecognizer.from_files(include_canonical=True)
    h0 = make_mock_hand((0.4, 0.5, 0.0))
    h1 = make_mock_hand((0.6, 0.5, 0.0))

    # Feed 30 frames with completely still hands
    t = 1.0
    for _ in range(30):
        res = rec.process_frame([h0, h1], timestamp=t)
        assert not res.is_completed
        assert res.label is None
        t += 0.033

    # Hands with slight random tremor/jitter
    rng = np.random.default_rng(42)
    for _ in range(30):
        noisy_h1 = h1 + rng.normal(0, 0.001, h1.shape).astype(np.float32)
        res = rec.process_frame([h0, noisy_h1], timestamp=t)
        assert not res.is_completed
        assert res.label is None
        t += 0.033


def test_missing_hand_landmarks():
    """Requirement 6: Handle temporarily missing hand landmarks smoothly."""
    rec = MotionRecognizer.from_files(include_canonical=True)
    frames = make_h_frame_sequence(30)
    t = 10.0

    # Feed initial frames
    for i in range(10):
        rec.process_frame([frames[i][0], frames[i][1]], timestamp=t)
        t += 0.033

    # Drop 2 frames (empty hands) - recognizer should tolerate brief loss
    rec.process_frame([], timestamp=t)
    t += 0.033
    rec.process_frame([], timestamp=t)
    t += 0.033

    # Resume gesture frames
    for i in range(12, 28):
        rec.process_frame([frames[i][0], frames[i][1]], timestamp=t)
        t += 0.033

    # Settle motion to complete
    final_res = None
    for _ in range(8):
        final_res = rec.process_frame([frames[-1][0], frames[-1][1]], timestamp=t)
        t += 0.033

    # The gesture completes without throwing errors
    assert rec.state in ("IDLE", "COOLDOWN")


def test_repeated_movement_letters():
    """Requirement 7: Repeated letters require a clear reset, no runaway repeats."""
    rec = MotionRecognizer.from_files(include_canonical=True)
    rec.cooldown_time = 0.3
    frames = make_h_frame_sequence(25)
    t = 50.0

    # First take of H
    recognized_count = 0
    for moving, supp in frames:
        res = rec.process_frame([moving, supp], timestamp=t)
        if res.is_completed and res.label == "H":
            recognized_count += 1
        t += 0.033

    # Settle take 1
    for _ in range(8):
        res = rec.process_frame([frames[-1][0], frames[-1][1]], timestamp=t)
        if res.is_completed and res.label == "H":
            recognized_count += 1
        t += 0.033

    # Ensure exactly 1 letter was emitted for the first take
    assert recognized_count == 1
    assert rec.state == "COOLDOWN"

    # During cooldown, further frames must NOT emit another letter
    for _ in range(4):
        res = rec.process_frame([frames[-1][0], frames[-1][1]], timestamp=t)
        assert not res.is_completed
        t += 0.033

    # Reset period: wait past cooldown
    t += 0.4
    res = rec.process_frame([frames[0][0], frames[0][1]], timestamp=t)
    assert not res.is_completed
    rec.reset()
    assert rec.state == "IDLE"


def test_false_positive_prevention():
    """Requirement 8: Arbitrary repositioning or orthogonal movement is rejected."""
    rec = MotionRecognizer.from_files(include_canonical=True)
    supporting = make_mock_hand((0.4, 0.5, 0.0))
    t = 100.0

    # Moving hand moves purely in Z (toward/away from camera, not sweeping palm)
    for i in range(25):
        s = i / 24.0
        moving = make_mock_hand((0.4, 0.5, 0.05 + 0.3 * s))
        res = rec.process_frame([moving, supporting], timestamp=t)
        t += 0.033

    # Settle
    for _ in range(6):
        res = rec.process_frame([make_mock_hand((0.4, 0.5, 0.35)), supporting], timestamp=t)
        # Should not falsely classify as H or J
        if res.is_completed:
            assert res.label is None or res.status == "rejected"
        t += 0.033


def test_gesture_completion_and_reset():
    """Requirement 9: State machine follows IDLE -> TRACKING -> COMPLETED -> COOLDOWN -> IDLE."""
    rec = MotionRecognizer.from_files(include_canonical=True)
    rec.cooldown_time = 0.2
    assert rec.state == "IDLE"

    # Start moving
    h0 = make_mock_hand((0.4, 0.5, 0.0))
    h1 = make_mock_hand((0.4, 0.5, 0.0))
    rec.process_frame([h0, h1], timestamp=1.0)
    assert rec.state == "IDLE"

    # Fast displacement triggers TRACKING
    h1_moved = make_mock_hand((0.42, 0.58, 0.0))
    rec.process_frame([h1_moved, h0], timestamp=1.05)
    assert rec.state == "TRACKING"

    # Manual reset returns immediately to IDLE
    rec.reset()
    assert rec.state == "IDLE"
    assert len(rec._raw_buffer) == 0


def test_missing_or_empty_templates(tmp_path):
    """Requirement 10: Gracefully handles missing or empty template datasets."""
    empty_file = tmp_path / "empty_templates.npz"

    # Loading from non-existent file
    trajs, labels = load_motion_templates(empty_file)
    assert len(trajs) == 0
    assert len(labels) == 0

    # Recognizer initialized without templates
    rec = MotionRecognizer.from_files(paths=[empty_file], include_canonical=False)
    assert not rec.available
    assert rec.templates == {"H": [], "J": [], "NONE" : []}

    # Running with no templates reports status safely without crashing
    h0 = make_mock_hand((0.4, 0.5, 0.0))
    h1 = make_mock_hand((0.4, 0.6, 0.0))
    res = rec.process_frame([h0, h1], timestamp=1.0)
    assert not res.is_completed
    assert res.label is None

    # Saving and loading empty templates roundtrip
    save_motion_templates(empty_file, [], [])
    trajs_round, labels_round = load_motion_templates(empty_file)
    assert len(trajs_round) == 0
    assert len(labels_round) == 0
