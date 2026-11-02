"""SignVisionAI - Task 4: Confidence Estimation.

Confidence-aware selective gesture recognition.

Decision signals
----------------
1. Absolute distance to the best class
2. Margin between the best and second-best classes
3. Frame-based stability (how many recent frames agree)
4. Time-based stability (how long the label has been continuously best)

IMPORTANT
---------
Scores passed to ``update`` must be per-class *distances* (smaller is
better), NOT probabilities (larger is better). For probabilistic ML
classifiers use ``probabilities_to_scores`` first, and calibrate a
*separate* ConfidenceEstimator configuration for them.

All thresholds are initial experimental values. They depend on how the
matcher normalises and measures distance and MUST be tuned on validation
data (held-out signers) before being trusted.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import NamedTuple

# Tolerance so threshold comparisons are not decided by float rounding
# (e.g. 0.42 - 0.30 may evaluate to 0.11999999999999994).
_EPS = 1e-9


@dataclass(frozen=True)
class Decision:
    status: str  # "strong" | "likely" | "uncertain" | "no_hand"
    label: str | None
    display: str
    best_distance: float
    second_distance: float
    margin: float
    stability: float
    accepted: bool


class _Frame(NamedTuple):
    label: str
    distance: float
    margin: float


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def probabilities_to_scores(probabilities, eps=1e-12):
    """Convert class probabilities into distance-like scores (-log p).

    Smaller is better, so the result can be passed to
    ``ConfidenceEstimator.update``. The estimator's margin then equals
    log(p_best / p_second), i.e. a log-odds margin. Thresholds on this
    scale are different from template-matching thresholds and must be
    calibrated separately (ideally after temperature scaling, because raw
    softmax outputs are often over-confident).

    Invalid probabilities (NaN, <0, >1) become NaN, which the estimator
    treats as an unusable frame.
    """
    scores = {}
    for label, p in probabilities.items():
        if _is_finite_number(p) and -1e-6 <= p <= 1 + 1e-6:
            scores[str(label)] = max(0.0, -math.log(max(float(p), eps)))
        else:
            scores[str(label)] = math.nan
    return scores


class ConfidenceEstimator:
    """Stateful per-stream estimator. Use ONE instance per video stream."""

    def __init__(
        self,
        strong_distance=0.55,
        reject_distance=1.50,
        strong_margin=0.30,
        minimum_margin=0.12,
        window_size=7,
        stable_ratio=0.80,
        min_stable_seconds=0.20,
        max_gap_seconds=0.50,
    ):
        numeric = (
            strong_distance,
            reject_distance,
            strong_margin,
            minimum_margin,
            stable_ratio,
            min_stable_seconds,
            max_gap_seconds,
        )
        if not all(_is_finite_number(v) for v in numeric):
            raise ValueError("Thresholds must be finite numbers")

        if not 0 <= strong_distance <= reject_distance:
            raise ValueError("Invalid distance thresholds")

        if not 0 <= minimum_margin <= strong_margin:
            raise ValueError("Invalid margin thresholds")

        if (
            not isinstance(window_size, int)
            or isinstance(window_size, bool)
            or window_size < 1
            or not 0 < stable_ratio <= 1
            or min_stable_seconds < 0
            or max_gap_seconds <= 0
        ):
            raise ValueError("Invalid stability parameters")

        self.strong_distance = strong_distance
        self.reject_distance = reject_distance
        self.strong_margin = strong_margin
        self.minimum_margin = minimum_margin
        self.window_size = window_size
        self.stable_ratio = stable_ratio
        self.min_stable_seconds = min_stable_seconds
        self.max_gap_seconds = max_gap_seconds

        # Integer frame count equivalent to stable_ratio (avoids float
        # comparison problems such as 3/5 >= 0.6).
        self._required_frames = max(
            1, math.ceil(stable_ratio * window_size - _EPS)
        )

        # Recent frames: a _Frame, or None for a rejected frame.
        self.history = deque(maxlen=window_size)
        self.last_label = None
        self.since = None
        self._last_time = None

    def reset(self):
        """Reset prediction history and stability timers."""
        self.history.clear()
        self.last_label = None
        self.since = None
        self._last_time = None

    def _result(
        self,
        status,
        label=None,
        best=math.inf,
        second=math.inf,
        margin=0.0,
        stability=0.0,
    ):
        """Create a consistent decision result."""
        if status == "strong":
            display = label
        elif status == "likely":
            display = f"{label}?"
        elif status == "no_hand":
            display = "No hand detected"
        else:
            display = "Uncertain gesture"

        return Decision(
            status=status,
            label=label,
            display=display,
            best_distance=best,
            second_distance=second,
            margin=margin,
            stability=stability,
            accepted=(status == "strong"),
        )

    def update(self, distances, hands_present=True, now=None):
        """Evaluate the current frame.

        distances: Mapping label -> distance (smaller is better).
        now: timestamp in seconds from a monotonic clock (optional).
        Raises TypeError if ``distances`` is not a Mapping and ValueError
        if ``now`` is not finite; both are programming errors that would
        otherwise silently produce "Uncertain gesture" forever.
        """
        if now is None:
            now = time.monotonic()
        if not _is_finite_number(now):
            raise ValueError("now must be a finite number of seconds")
        now = float(now)

        # 1. No hand detected
        if not hands_present:
            self.reset()
            return self._result("no_hand")

        if not isinstance(distances, Mapping):
            raise TypeError("distances must be a mapping of label -> distance")

        # Stale / non-monotonic timestamps: a pause in updates or a clock
        # going backwards must not count as "stable time".
        if self._last_time is not None and not (
            0 <= now - self._last_time <= self.max_gap_seconds
        ):
            self.reset()

        # 2. Insufficient candidate classes
        if len(distances) < 2:
            self.reset()
            return self._result("uncertain")

        # 3. Validate and rank class distances (smaller is better)
        try:
            ranked = sorted(
                (
                    (str(label), float(distance))
                    for label, distance in distances.items()
                ),
                key=lambda item: (item[1], item[0]),
            )
            if any(
                not math.isfinite(distance) or distance < 0
                for _, distance in ranked
            ):
                raise ValueError("Invalid distance")
        except (ValueError, TypeError, OverflowError):
            self.reset()
            return self._result("uncertain")

        # 4. Top two candidates
        best_label, best_distance = ranked[0]
        second_distance = ranked[1][1]
        margin = max(0.0, second_distance - best_distance)
        self._last_time = now

        # 5. Reject ambiguous or poor matches
        if (
            best_distance > self.reject_distance + _EPS
            or margin < self.minimum_margin - _EPS
        ):
            self.history.append(None)
            self.last_label = None
            self.since = None
            return self._result(
                "uncertain",
                best=best_distance,
                second=second_distance,
                margin=margin,
            )

        # 6. Stability tracking
        if best_label != self.last_label:
            self.last_label = best_label
            self.since = now

        self.history.append(_Frame(best_label, best_distance, margin))

        matching = [
            f for f in self.history if f is not None and f.label == best_label
        ]
        stability = len(matching) / len(self.history)

        stable = (
            len(self.history) == self.window_size
            and len(matching) >= self._required_frames
            and self.since is not None
            and (now - self.since) >= self.min_stable_seconds - _EPS
        )

        # 7. Strong: stable AND the evidence over the window (median, so
        # one lucky frame cannot lift a weak run) AND the current frame
        # all meet the strong thresholds.
        strong = (
            stable
            and best_distance <= self.strong_distance + _EPS
            and margin >= self.strong_margin - _EPS
            and statistics.median(f.distance for f in matching)
            <= self.strong_distance + _EPS
            and statistics.median(f.margin for f in matching)
            >= self.strong_margin - _EPS
        )

        status = "strong" if strong else "likely"
        return self._result(
            status,
            label=best_label,
            best=best_distance,
            second=second_distance,
            margin=margin,
            stability=stability,
        )
