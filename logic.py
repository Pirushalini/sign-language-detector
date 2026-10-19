"""Detection logic that does not depend on any window: settings, smoothing,
hold-to-add and practice scoring. Keeping it separate makes it easy to test.
"""
from __future__ import annotations

import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

from matcher import Prediction
from utils import IDLE, LETTERS


# ---------------------------------------------------------------- settings
@dataclass
class Settings:
    """User settings, saved to settings.json."""

    hold_time: float = 1.0          # seconds a sign must be held
    threshold: float = 0.80         # minimum confidence
    smoothing: int = 7              # frames used for majority voting
    max_distance: float = 1.5       # matches farther than this are ignored
    show_landmarks: bool = True
    mirror: bool = True             # mirror the camera picture (display only)
    auto_space: bool = False
    auto_space_delay: float = 2.0   # seconds without hands before a space is added
    camera_index: int = 0

    RANGES = {
        "hold_time": (0.3, 2.5), "threshold": (0.5, 0.99), "smoothing": (1, 15),
        "max_distance": (0.5, 3.0), "auto_space_delay": (1.0, 5.0), "camera_index": (0, 9),
    }

    def clamp(self) -> "Settings":
        for name, (lo, hi) in self.RANGES.items():
            value = getattr(self, name)
            value = min(max(value, lo), hi)
            setattr(self, name, int(round(value)) if isinstance(getattr(Settings, name), int) else float(value))
        return self

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings, ignoring anything missing or invalid."""
        s = cls()
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return s
        for f in fields(cls):
            if f.name in data and isinstance(data[f.name], (int, float, bool)):
                try:
                    setattr(s, f.name, type(getattr(s, f.name))(data[f.name]))
                except (TypeError, ValueError):
                    pass
        return s.clamp()

    def save(self, path: Path) -> None:
        try:
            Path(path).write_text(json.dumps(asdict(self), indent=2))
        except OSError:
            pass  # not being able to save settings should never crash the app


# ---------------------------------------------------------------- smoothing
class Smoother:
    """Majority vote over the last N frames.

    A letter is reported only if it wins more than half of the window. Its
    confidence is the average confidence of the frames that voted for it.
    Frames with no hands or no match count as empty votes.
    """

    def __init__(self, size: int = 7):
        self._buf: Deque[Tuple[Optional[str], float]] = deque(maxlen=max(1, size))

    def resize(self, size: int) -> None:
        self._buf = deque(self._buf, maxlen=max(1, size))

    def reset(self) -> None:
        self._buf.clear()

    def update(self, pred: Optional[Prediction]) -> Tuple[Optional[str], float]:
        label = pred.label if pred is not None else None
        conf = pred.confidence if (pred is not None and label) else 0.0
        self._buf.append((label, conf))
        votes = Counter(lab for lab, _ in self._buf if lab)
        if not votes:
            return None, 0.0
        winner, count = votes.most_common(1)[0]
        if count * 2 <= len(self._buf):
            return None, 0.0
        confs = [c for lab, c in self._buf if lab == winner]
        return winner, sum(confs) / len(confs)


# -------------------------------------------------------------- hold to add
@dataclass
class HoldEvent:
    kind: str            # "letter" or "space"
    letter: str = ""


class HoldToAdd:
    """Turns a stream of smoothed predictions into accepted letters.

    A letter is accepted after it has been the stable prediction, above the
    confidence threshold, for hold_time seconds. After that, the same letter is
    blocked until a different stable letter is shown or the sign is released
    (no usable prediction) for release_time seconds. This avoids "AAAAAA".
    Double letters therefore need a short break between the two letters.

    only_letter: if set, other letters are ignored (used by practice mode).
    """

    STABLE_TIME = 0.4  # how long a different letter must be shown to unblock

    def __init__(self, hold_time: float = 1.0, threshold: float = 0.8, release_time: float = 0.6,
                 auto_space: bool = False, auto_space_delay: float = 2.0,
                 only_letter: Optional[str] = None):
        self.hold_time = hold_time
        self.threshold = threshold
        self.release_time = release_time
        self.auto_space = auto_space
        self.auto_space_delay = auto_space_delay
        self.only_letter = only_letter
        self.reset()

    def reset(self) -> None:
        self._cand: Optional[str] = None
        self._cand_since = 0.0
        self._idle_since: Optional[float] = None
        self._last_added: Optional[str] = None
        self._absent_since: Optional[float] = None
        self._space_done = False
        self._can_space = False
        self.progress = 0.0
        self.blocked = False

    def set_text_state(self, text: str) -> None:
        """Tell the logic what the text looks like (for auto-space rules)."""
        self._can_space = bool(text) and not text.endswith(" ")

    def update(self, label: Optional[str], conf: float, now: float, hands_present: bool) -> Optional[HoldEvent]:
        usable = (label is not None and label != IDLE and conf >= self.threshold
                  and (self.only_letter is None or label == self.only_letter))
        cand = label if usable else None
        event: Optional[HoldEvent] = None
        self.blocked = False

        if cand is None:
            if self._idle_since is None:
                self._idle_since = now
            if now - self._idle_since >= self.release_time:
                self._last_added = None
            self._cand = None
            self.progress = 0.0
        else:
            self._idle_since = None
            if cand != self._cand:
                self._cand, self._cand_since = cand, now
            age = now - self._cand_since
            if self._last_added is not None and cand != self._last_added \
                    and age >= min(self.STABLE_TIME, self.hold_time / 2):
                self._last_added = None
            if cand == self._last_added:
                self.blocked = True
                self.progress = 0.0
            else:
                self.progress = min(age / max(self.hold_time, 1e-6), 1.0)
                if self.progress >= 1.0:
                    event = HoldEvent("letter", cand)
                    self._last_added = cand
                    self._can_space = True
                    self._space_done = False
                    self.progress = 0.0

        # Auto-space: only after hands have been absent for a while, once per absence.
        if hands_present:
            self._absent_since = None
            self._space_done = False
        else:
            if self._absent_since is None:
                self._absent_since = now
            if (event is None and self.auto_space and self.only_letter is None and self._can_space
                    and not self._space_done and now - self._absent_since >= self.auto_space_delay):
                event = HoldEvent("space")
                self._space_done = True
                self._can_space = False
        return event


# ------------------------------------------------------------------ practice
class PracticeSession:
    """Random target letters with score and streak."""

    def __init__(self, available: Sequence[str], rng: Optional[random.Random] = None):
        self.available: List[str] = [c for c in available if c in LETTERS]
        self.rng = rng or random.Random()
        self.target: Optional[str] = None
        self.score = 0
        self.streak = 0
        self.best_streak = 0

    def next_target(self) -> Optional[str]:
        """Pick a random letter, never the same one twice in a row (if possible)."""
        if not self.available:
            self.target = None
            return None
        options = [c for c in self.available if c != self.target] or self.available
        self.target = self.rng.choice(options)
        return self.target

    def succeed(self) -> None:
        self.score += 1
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)

    def skip(self) -> None:
        self.streak = 0
