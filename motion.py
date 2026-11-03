"""Dynamic movement recognition module for British Sign Language (BSL).

Specialized for movement-based letters such as H and J.
Uses continuous hand landmark tracking, supporting-hand-relative coordinate
normalization, trajectory resampling, and Dynamic Time Warping (DTW).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import DATA_DIR

MOTION_TEMPLATES = DATA_DIR / "motion_templates.npz"
MOTION_USER_TEMPLATES = DATA_DIR / "motion_templates_user.npz"

# MediaPipe landmark indices
WRIST = 0
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

DYNAMIC_LETTERS = ["H", "J"]
N_RESAMPLE_POINTS = 30
FEATURE_DIM = 7  # [x8, y8, z8, x0, y0, z0, dist_middle_tip]


@dataclass
class MotionResult:
    """Result of motion recognition for a frame or completed sequence."""
    label: Optional[str] = None
    confidence: float = 0.0
    distance: float = float("inf")
    status: str = "idle"  # "idle", "tracking", "recognized", "rejected"
    is_completed: bool = False
    details: Dict[str, float] = field(default_factory=dict)


def compute_palm_frame(supporting_hand: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute local coordinate frame of the supporting hand.

    Returns:
        origin: 3D point (wrist)
        u_x: unit vector across palm (towards pinky)
        u_y: unit vector along palm (wrist to middle MCP)
        u_z: normal unit vector to the palm
        scale: palm length (wrist to middle MCP distance)
    """
    supp = np.asarray(supporting_hand, dtype=np.float32).reshape(21, 3)
    wrist = supp[WRIST]
    middle_mcp = supp[MIDDLE_MCP]
    pinky_mcp = supp[PINKY_MCP]
    index_mcp = supp[INDEX_MCP]

    # Longitudinal axis (wrist to middle MCP)
    v_y = middle_mcp - wrist
    scale = float(np.linalg.norm(v_y))
    if scale < 1e-5:
        scale = 0.1
    u_y = v_y / scale

    # Transverse axis (across palm)
    v_x = pinky_mcp - index_mcp
    v_x_proj = np.dot(v_x, u_y) * u_y
    w_x = v_x - v_x_proj
    w_x_norm = float(np.linalg.norm(w_x))
    if w_x_norm < 1e-5:
        u_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        u_x = w_x / w_x_norm

    # Normal axis
    u_z = np.cross(u_x, u_y)
    u_z_norm = float(np.linalg.norm(u_z))
    if u_z_norm > 1e-5:
        u_z /= u_z_norm
    else:
        u_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return wrist, u_x, u_y, u_z, scale


def transform_point_to_palm(point: np.ndarray, origin: np.ndarray,
                            u_x: np.ndarray, u_y: np.ndarray, u_z: np.ndarray,
                            scale: float) -> np.ndarray:
    """Transform a 3D point into the normalized supporting palm coordinate system."""
    diff = point - origin
    x_rel = float(np.dot(diff, u_x)) / scale
    y_rel = float(np.dot(diff, u_y)) / scale
    z_rel = float(np.dot(diff, u_z)) / scale
    return np.array([x_rel, y_rel, z_rel], dtype=np.float32)


def identify_hands(hands: Sequence[np.ndarray],
                   recent_speeds: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Identify moving hand (dominant) vs supporting hand (non-dominant).

    Returns:
        (moving_hand, supporting_hand, moving_idx, supporting_idx)
    """
    if len(hands) < 2:
        raise ValueError("At least two hands are required for dynamic BSL letter recognition.")

    h0 = np.asarray(hands[0], dtype=np.float32).reshape(21, 3)
    h1 = np.asarray(hands[1], dtype=np.float32).reshape(21, 3)

    # 1. If recent motion speeds are available, hand with higher speed is moving
    if recent_speeds is not None and len(recent_speeds) >= 2:
        s0, s1 = recent_speeds[0], recent_speeds[1]
        if s1 > s0 * 1.3:
            return h1, h0, 1, 0
        if s0 > s1 * 1.3:
            return h0, h1, 0, 1

    # 2. Postural heuristic for BSL:
    # Supporting hand is typically an open flat palm (all 4 fingers extended).
    # Moving hand: for J, index is extended while others are curled; for H, flat sweep.
    # Measure finger extension ratio for both hands.
    def openness(h: np.ndarray) -> float:
        wrist = h[WRIST]
        tips = [h[INDEX_TIP], h[MIDDLE_TIP], h[16], h[PINKY_TIP]]
        mcps = [h[INDEX_MCP], h[MIDDLE_MCP], h[13], h[PINKY_MCP]]
        ratios = [np.linalg.norm(t - wrist) / max(float(np.linalg.norm(m - wrist)), 1e-4)
                  for t, m in zip(tips, mcps)]
        return float(np.mean(ratios))

    open0 = openness(h0)
    open1 = openness(h1)

    # More open / flat hand tends to be the supporting base hand in BSL fingerspelling
    if open0 > open1 + 0.15:
        return h1, h0, 1, 0
    elif open1 > open0 + 0.15:
        return h0, h1, 0, 1

    # Default fallback: left-most hand is index 0
    if float(h0[WRIST, 0]) < float(h1[WRIST, 0]):
        return h1, h0, 1, 0
    return h0, h1, 0, 1


def extract_motion_features(moving_hand: np.ndarray, supporting_hand: np.ndarray) -> np.ndarray:
    """Extract palm-normalized 7D motion feature vector for one frame.

    Features:
        [x8, y8, z8]: normalized relative position of moving index fingertip
        [x0, y0, z0]: normalized relative position of moving wrist
        [dist_mid_tip]: Euclidean distance from moving index tip to supporting middle tip
    """
    m = np.asarray(moving_hand, dtype=np.float32).reshape(21, 3)
    s = np.asarray(supporting_hand, dtype=np.float32).reshape(21, 3)

    origin, u_x, u_y, u_z, scale = compute_palm_frame(s)

    p_idx = transform_point_to_palm(m[INDEX_TIP], origin, u_x, u_y, u_z, scale)
    p_wri = transform_point_to_palm(m[WRIST], origin, u_x, u_y, u_z, scale)

    # Distance to supporting middle tip (critical for distinguishing 'J' from 'H')
    dist_mid_tip = float(np.linalg.norm(m[INDEX_TIP] - s[MIDDLE_TIP])) / scale

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat[0:3] = p_idx
    feat[3:6] = p_wri
    feat[6] = dist_mid_tip
    return feat


def resample_trajectory(trajectory: np.ndarray, target_len: int = N_RESAMPLE_POINTS) -> np.ndarray:
    """Resample trajectory to a fixed number of temporal points along its path."""
    traj = np.asarray(trajectory, dtype=np.float32)
    if len(traj) == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    if len(traj) == 1:
        return np.repeat(traj, target_len, axis=0)
    if len(traj) == target_len:
        return traj.copy()

    # Cumulative arc length along moving index tip (features 0:3)
    diffs = traj[1:, :3] - traj[:-1, :3]
    step_lens = np.linalg.norm(diffs, axis=1)
    cum_dist = np.zeros(len(traj), dtype=np.float32)
    cum_dist[1:] = np.cumsum(step_lens)

    total_dist = cum_dist[-1]
    if total_dist < 1e-6:
        # Uniform temporal resampling if spatial displacement is negligible
        old_indices = np.linspace(0.0, 1.0, len(traj))
        new_indices = np.linspace(0.0, 1.0, target_len)
        resampled = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
        for d in range(traj.shape[1]):
            resampled[:, d] = np.interp(new_indices, old_indices, traj[:, d])
        return resampled

    # Resample along arc length
    new_dists = np.linspace(0.0, total_dist, target_len)
    resampled = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for d in range(traj.shape[1]):
        resampled[:, d] = np.interp(new_dists, cum_dist, traj[:, d])
    return resampled


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: Optional[int] = None) -> float:
    """Compute normalized Dynamic Time Warping (DTW) distance between two trajectories."""
    s1 = np.asarray(seq1, dtype=np.float32)
    s2 = np.asarray(seq2, dtype=np.float32)
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return float("inf")

    w = max(window if window is not None else int(max(n, m) * 0.4), abs(n - m))

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    # Weights: index tip coordinates (x, y, z) have weight 1.0,
    # wrist has weight 0.5, dist_mid_tip has weight 1.2
    weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.2], dtype=np.float32)
    if s1.shape[1] == len(weights):
        s1_w = s1 * weights
        s2_w = s2 * weights
    else:
        s1_w, s2_w = s1, s2

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)
        sub_s2 = s2_w[j_start - 1:j_end - 1]
        diff = s1_w[i - 1] - sub_s2
        dists = np.linalg.norm(diff, axis=-1)
        for idx, j in enumerate(range(j_start, j_end)):
            cost[i, j] = dists[idx] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n, m] / (n + m))


def validate_gesture_path(label: str, trajectory: np.ndarray) -> Tuple[bool, str]:
    """Verify biomechanical movement direction and trajectory constraints for H and J.

    Returns:
        (is_valid, reason)
    """
    traj = np.asarray(trajectory, dtype=np.float32)
    if len(traj) < 2:
        return False, "Trajectory too short"

    # Moving fingertip relative positions
    y_vals = traj[:, 1]
    x_vals = traj[:, 0]
    total_y_disp = y_vals[-1] - y_vals[0]
    max_y_disp = np.max(y_vals) - np.min(y_vals)

    if label == "H":
        # BSL 'H': Flat sweep forward along supporting palm (from wrist to fingertips, +Y direction)
        if total_y_disp < 0.15:
            return False, f"H requires forward sweep along palm (disp={total_y_disp:.2f} < 0.15)"
        if y_vals[-1] < y_vals[0]:
            return False, "H movement direction was inverted (moved toward wrist instead of fingers)"
        return True, "Valid H trajectory"

    elif label == "J":
        # BSL 'J': Starts at middle fingertip, sweeps down into palm (-Y) and hooks towards thumb
        # 1. Starts near middle fingertip
        start_dist_mid = traj[0, 6]
        if start_dist_mid > 0.60:
            return False, f"J must start near middle fingertip (dist={start_dist_mid:.2f} > 0.60)"

        # 2. Strokes down into palm (start must be higher than the end / bottom)
        if y_vals[-1] >= y_vals[0]:
            return False, "J movement was inverted (moved upwards instead of downwards)"

        downward_stroke = y_vals[0] - np.min(y_vals)
        if downward_stroke < 0.2:
            return False, f"J requires downward stroke along middle finger (stroke={downward_stroke:.2f} < 0.2)"

        # 3. Lateral curve/hook
        lateral_disp = np.max(x_vals) - np.min(x_vals)
        if lateral_disp < 0.08:
            return False, f"J requires a curved hook motion (lateral={lateral_disp:.2f} < 0.08)"

        return True, "Valid J trajectory"

    return True, "Unknown letter"


def create_synthetic_template(label: str, n_points: int = N_RESAMPLE_POINTS) -> np.ndarray:
    """Generate a canonical reference trajectory for H or J (used as initial reference)."""
    t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    traj = np.zeros((n_points, FEATURE_DIM), dtype=np.float32)

    if label == "H":
        # Sweeping across supporting palm from wrist (y=0.2) to fingertips (y=1.5)
        # x remains relatively centered with a slight diagonal sweep
        traj[:, 0] = -0.2 + 0.3 * t       # x: slightly across palm
        traj[:, 1] = 0.2 + 1.2 * t        # y: forward sweep from heel of palm off fingertips
        traj[:, 2] = 0.05 + 0.1 * np.sin(t * np.pi)  # z: close to palm surface
        traj[:, 3] = traj[:, 0] - 0.2     # moving wrist x
        traj[:, 4] = traj[:, 1] - 0.8     # moving wrist y
        traj[:, 5] = traj[:, 2] + 0.1
        traj[:, 6] = np.abs(traj[:, 1] - 1.4)  # distance to middle tip decreases then increases
    elif label == "J":
        # Starting at middle fingertip (x=0.0, y=1.35), tracing down to palm (y=0.4), curving towards thumb (x=-0.5, y=0.6)
        # Phase 1: downward stroke (first 60% of time)
        # Phase 2: curve/hook towards thumb (last 40% of time)
        for i, val in enumerate(t):
            if val <= 0.6:
                phase = val / 0.6
                x = 0.0 + 0.05 * np.sin(phase * np.pi)
                y = 1.35 - 0.95 * phase
            else:
                phase = (val - 0.6) / 0.4
                # Hook curve: curves towards thumb (negative x) and slightly upwards
                angle = phase * (np.pi * 0.6)
                x = -0.45 * np.sin(angle)
                y = 0.40 + 0.25 * (1.0 - np.cos(angle))
            traj[i, 0] = x
            traj[i, 1] = y
            traj[i, 2] = 0.05
            traj[i, 3] = x + 0.3
            traj[i, 4] = y - 0.6
            traj[i, 5] = 0.15
            # Distance to middle fingertip
            traj[i, 6] = float(np.sqrt(x ** 2 + (y - 1.35) ** 2))

    return traj


# -------------------------------------------------------- Template Storage
def save_motion_templates(path: Path, trajectories: Sequence[np.ndarray], labels: Sequence[str]) -> None:
    """Save normalized motion templates to .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trajs = [np.asarray(t, dtype=np.float32) for t in trajectories]
    np.savez_compressed(
        path,
        trajectories=np.stack(trajs) if trajs else np.zeros((0, N_RESAMPLE_POINTS, FEATURE_DIM), dtype=np.float32),
        labels=np.asarray(labels, dtype="<U8"),
    )


def load_motion_templates(path: Path) -> Tuple[List[np.ndarray], List[str]]:
    """Load motion templates from .npz file. Returns (trajectories, labels)."""
    if not path.exists():
        return [], []
    try:
        with np.load(path, allow_pickle=False) as d:
            if "trajectories" in d and "labels" in d:
                trajs = list(d["trajectories"])
                labels = list(d["labels"].astype(str))
                return trajs, labels
    except Exception:
        pass
    return [], []


# ------------------------------------------------------ Streaming Recognizer
class MotionRecognizer:
    """Real-time streaming dynamic letter recognizer using DTW."""

    def __init__(self,
                 reject_distance: float = 0.85,
                 min_path_length: float = 0.35,
                 start_speed_threshold: float = 0.35,
                 end_speed_threshold: float = 0.20,
                 cooldown_time: float = 0.5,
                 max_missing_frames: int = 4):
        self.reject_distance = reject_distance
        self.min_path_length = min_path_length
        self.start_speed_threshold = start_speed_threshold
        self.end_speed_threshold = end_speed_threshold
        self.cooldown_time = cooldown_time
        self.max_missing_frames = max_missing_frames

        self.templates: Dict[str, List[np.ndarray]] = {c: [] for c in DYNAMIC_LETTERS}
        self.state = "IDLE"  # "IDLE", "TRACKING", "COOLDOWN"
        self._raw_buffer: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._missing_count = 0
        self._cooldown_until = 0.0
        self._last_moving_hand: Optional[np.ndarray] = None
        self._last_supporting_hand: Optional[np.ndarray] = None
        self._moving_idx: Optional[int] = None
        self._last_recognized_letter: Optional[str] = None
        self._recent_speeds: List[Tuple[float, float]] = []
        self._last_feat: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None

    @property
    def available(self) -> bool:
        """True if there is at least one stored motion template."""
        return any(len(t) > 0 for t in self.templates.values())

    @property
    def is_tracking(self) -> bool:
        """True if a dynamic gesture is actively being traced."""
        return self.state == "TRACKING"

    @classmethod
    def from_files(cls, paths: Optional[Sequence[Path]] = None,
                   include_canonical: bool = True, **kwargs) -> "MotionRecognizer":
        """Load templates from default files or fallback to canonical templates."""
        rec = cls(**kwargs)
        file_paths = paths if paths is not None else [MOTION_USER_TEMPLATES, MOTION_TEMPLATES]
        loaded_count = 0
        for p in file_paths:
            p = Path(p)
            if p.exists():
                trajs, labels = load_motion_templates(p)
                for t, l in zip(trajs, labels):
                    if l in rec.templates:
                        rec.templates[l].append(resample_trajectory(t))
                        loaded_count += 1

        # If no recorded files exist, initialize with synthetic canonical prototypes
        if loaded_count == 0 and include_canonical:
            for letter in DYNAMIC_LETTERS:
                rec.templates[letter].append(create_synthetic_template(letter))
        return rec

    def add_template(self, label: str, trajectory: np.ndarray) -> None:
        """Add a reference template trajectory."""
        label = label.upper()
        if label not in self.templates:
            self.templates[label] = []
        self.templates[label].append(resample_trajectory(trajectory))

    def reset(self) -> None:
        """Reset internal streaming state."""
        self.state = "IDLE"
        self._raw_buffer.clear()
        self._timestamps.clear()
        self._missing_count = 0
        self._last_moving_hand = None
        self._last_supporting_hand = None
        self._moving_idx = None
        self._recent_speeds.clear()
        self._last_feat = None
        self._last_timestamp = None

    def process_frame(self, hands: Sequence[np.ndarray], timestamp: float) -> MotionResult:
        """Process one incoming video frame with hand landmarks.

        Returns:
            MotionResult with state, detected letter (when completed), and confidence.
        """
        now = timestamp

        # Cooldown check
        if self.state == "COOLDOWN":
            if now < self._cooldown_until:
                return MotionResult(status="cooldown")
            self.state = "IDLE"
            self.reset()

        # Check for hands presence
        if len(hands) < 2:
            if self.state == "TRACKING":
                self._missing_count += 1
                if self._missing_count <= self.max_missing_frames and len(self._raw_buffer) > 0:
                    # Tolerate brief frame loss by repeating last feature vector
                    self._raw_buffer.append(self._raw_buffer[-1].copy())
                    self._timestamps.append(now)
                    return MotionResult(status="tracking")
                else:
                    # Hands lost for too long -> finalize or abort
                    return self._finalize_or_abort(now)
            else:
                self.reset()
                return MotionResult(status="idle")

        self._missing_count = 0
        h0 = np.asarray(hands[0], dtype=np.float32).reshape(21, 3)
        h1 = np.asarray(hands[1], dtype=np.float32).reshape(21, 3)

        # Estimate instantaneous speeds
        s0, s1 = 0.0, 0.0
        if self._last_moving_hand is not None and self._last_supporting_hand is not None:
            # Match to previous hands by wrist distance
            d00 = np.linalg.norm(h0[WRIST] - self._last_moving_hand[WRIST])
            d11 = np.linalg.norm(h1[WRIST] - self._last_supporting_hand[WRIST])
            d01 = np.linalg.norm(h0[WRIST] - self._last_supporting_hand[WRIST])
            d10 = np.linalg.norm(h1[WRIST] - self._last_moving_hand[WRIST])
            if (d01 + d10) < (d00 + d11):
                h0, h1 = h1, h0

        # Hand identity assignment
        if self.state == "TRACKING" and self._moving_idx is not None:
            moving, supporting = (h0, h1) if self._moving_idx == 0 else (h1, h0)
        else:
            moving, supporting, m_idx, _ = identify_hands([h0, h1])
            self._moving_idx = m_idx

        self._last_moving_hand = moving.copy()
        self._last_supporting_hand = supporting.copy()

        # Extract normalized 7D motion feature vector
        feat = extract_motion_features(moving, supporting)

        # Compute speed of moving index tip relative to previous frame
        speed = 0.0
        if self._last_feat is not None and self._last_timestamp is not None:
            dt = max(now - self._last_timestamp, 1e-4)
            d_pos = np.linalg.norm(feat[:3] - self._last_feat[:3])
            speed = float(d_pos / dt)

        # State transition logic
        if self.state == "IDLE":
            if speed > self.start_speed_threshold:
                self.state = "TRACKING"
                if self._last_feat is not None and self._last_timestamp is not None:
                    self._raw_buffer.append(self._last_feat.copy())
                    self._timestamps.append(self._last_timestamp)
                self._raw_buffer.append(feat)
                self._timestamps.append(now)
                self._last_feat = feat.copy()
                self._last_timestamp = now
                return MotionResult(status="tracking")
            self._last_feat = feat.copy()
            self._last_timestamp = now
            return MotionResult(status="idle")

        elif self.state == "TRACKING":
            self._raw_buffer.append(feat)
            self._timestamps.append(now)
            self._last_feat = feat.copy()
            self._last_timestamp = now
            duration = now - self._timestamps[0]

            # Check if gesture reached max duration
            if duration > 1.8:
                return self._finalize_or_abort(now)

            # Check if gesture settled/decelerated after minimum duration
            if duration >= 0.25 and len(self._raw_buffer) >= 6:
                recent_speeds = []
                for i in range(len(self._raw_buffer) - 3, len(self._raw_buffer)):
                    dt = max(self._timestamps[i] - self._timestamps[i - 1], 1e-4)
                    dp = np.linalg.norm(self._raw_buffer[i][:3] - self._raw_buffer[i - 1][:3])
                    recent_speeds.append(dp / dt)

                mean_speed = float(np.mean(recent_speeds))
                if mean_speed < self.end_speed_threshold:
                    return self._finalize_or_abort(now)

            return MotionResult(status="tracking")

        return MotionResult(status="idle")

    def _finalize_or_abort(self, now: float) -> MotionResult:
        """Finalize accumulated motion sequence, classify via DTW, or reject."""
        if len(self._raw_buffer) < 5:
            self.reset()
            return MotionResult(status="rejected")

        raw_traj = np.array(self._raw_buffer, dtype=np.float32)
        # Check cumulative path length (reject stationary hands)
        diffs = raw_traj[1:, :3] - raw_traj[:-1, :3]
        total_path = float(np.sum(np.linalg.norm(diffs, axis=1)))

        if total_path < self.min_path_length:
            self.reset()
            return MotionResult(status="rejected", distance=999.0)

        # Resample to canonical 30 points
        candidate = resample_trajectory(raw_traj)

        # Match against stored templates
        best_letter: Optional[str] = None
        best_dist = float("inf")
        scores: Dict[str, float] = {}

        if not self.available:
            self.reset()
            return MotionResult(status="no_templates", distance=float("inf"))

        for letter, t_list in self.templates.items():
            if not t_list:
                continue
            dists = [dtw_distance(candidate, t) for t in t_list]
            min_d = float(np.min(dists))
            scores[letter] = min_d
            if min_d < best_dist:
                best_dist = min_d
                best_letter = letter

        # Check rejection threshold
        if best_letter is None or best_dist > self.reject_distance:
            self.reset()
            return MotionResult(status="rejected", distance=best_dist, details=scores)

        # Biomechanical direction validation
        valid, reason = validate_gesture_path(best_letter, candidate)
        if not valid:
            self.reset()
            return MotionResult(status="rejected", distance=best_dist, details={"validation_error": 1.0})

        # Recognized successfully
        confidence = float(np.clip(1.0 - (best_dist / max(self.reject_distance, 1e-6)), 0.5, 0.99))
        self._last_recognized_letter = best_letter
        self.state = "COOLDOWN"
        self._cooldown_until = now + self.cooldown_time
        self._raw_buffer.clear()
        self._timestamps.clear()

        return MotionResult(
            label=best_letter,
            confidence=confidence,
            distance=best_dist,
            status="recognized",
            is_completed=True,
            details=scores,
        )
