"""GUI tests. They need a display (on a server use: xvfb-run -a python -m pytest tests -q).

The window is built for real. The camera and hand tracker are replaced by fakes,
so no webcam is needed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")

import app as appmod
from logic import Settings
from matcher import SRC_EXTERNAL, Prediction, TemplateMatcher
from utils import LETTERS, build_features

RNG = np.random.default_rng(1)
ASPECT = 640 / 480


def hand() -> np.ndarray:
    h = RNG.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
    h[0] = 0
    return h + np.array([0.5, 0.5, 0.0], dtype=np.float32)


PROTOS = {c: [hand(), hand()] for c in "ABCDE"}


def make_matcher() -> TemplateMatcher:
    m = TemplateMatcher(reject_distance=2.0)
    X, y = [], []
    for c, hands in PROTOS.items():
        for _ in range(10):
            X.append(build_features(hands, ASPECT) + RNG.normal(0, 0.01, 132).astype(np.float32))
            y.append(c)
    m.add(np.stack(X), y, [SRC_EXTERNAL] * len(y))
    return m


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    created = []

    def factory(with_templates=True):
        monkeypatch.setattr(appmod, "SETTINGS_PATH", tmp_path / "settings.json")
        if with_templates:
            monkeypatch.setattr(appmod.TemplateMatcher, "from_files", classmethod(lambda cls, *a, **k: make_matcher()))
        else:
            monkeypatch.setattr(appmod.TemplateMatcher, "from_files", classmethod(lambda cls, *a, **k: TemplateMatcher()))
        try:
            a = appmod.App(autostart=False, scan_cameras=False)
        except tk.TclError as exc:
            pytest.skip(f"no display available: {exc}")
        a.settings.hold_time = 0.3
        a.apply_settings()
        created.append(a)
        return a

    yield factory
    for a in created:
        try:
            a.stop_camera()
            a.destroy()
        except tk.TclError:
            pass


def pump(a, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        a._drain()
        a.update()
        time.sleep(0.02)


FRAME = np.full((360, 640, 3), 120, np.uint8)


def feed(a, label, seconds, hands=2, conf=0.95):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pred = Prediction(label, conf, [(label, conf)]) if label else None
        a.queue.put({"type": "frame", "image": FRAME, "hands": hands if label else 0, "pred": pred, "fps": 30.0})
        a._drain()
        a.update()
        time.sleep(0.02)


def test_letters_are_added_erased_and_reset(make_app, monkeypatch):
    a = make_app()
    feed(a, "A", 0.6)
    feed(a, None, 0.8)
    feed(a, "B", 0.6)
    assert a.text == "AB"
    assert a.textbox.get("1.0", "end").strip() == "AB"
    a.erase()
    assert a.text == "A"
    a.add_space()
    assert a.text == "A "
    a.reset_text()
    assert a.text == ""


def test_reset_asks_before_clearing_long_text(make_app, monkeypatch):
    a = make_app()
    a.text = "X" * 20
    monkeypatch.setattr(appmod.messagebox, "askyesno", lambda *args, **kw: False)
    a.reset_text()
    assert a.text == "X" * 20
    monkeypatch.setattr(appmod.messagebox, "askyesno", lambda *args, **kw: True)
    a.reset_text()
    assert a.text == ""


def test_keyboard_shortcuts(make_app):
    a = make_app()
    a.text = "AB"
    a.focus_force()
    a.update()
    a.event_generate("<BackSpace>")
    a.update()
    assert a.text == "A"
    a.event_generate("<space>")
    a.update()
    assert a.text == "A "
    a.event_generate("<Escape>")
    a.update()
    assert a.text == ""


def test_copy_puts_text_on_the_clipboard(make_app):
    a = make_app()
    a.text = "HELLO"
    a.copy_text()
    a.update()
    assert a.clipboard_get() == "HELLO"
    assert "Copied" in a.text_note.cget("text")


def test_idle_and_one_second_flicker_add_nothing(make_app):
    a = make_app()
    a.settings.hold_time = 1.0
    a.apply_settings()
    a.hold.hold_time = 1.0
    feed(a, "A", 0.4)
    feed(a, "B", 0.4)
    feed(a, None, 0.3)
    assert a.text == ""


def test_status_messages(make_app):
    a = make_app()
    feed(a, None, 0.1)
    assert a.status_label.cget("text") == "No hands detected"
    feed(a, "A", 0.1, hands=1, conf=0.3)
    assert "1 hand detected" in a.status_label.cget("text")
    assert "both hands" in a.status_label.cget("text")
    feed(a, "A", 0.1, hands=2)
    assert a.status_label.cget("text") == "2 hands detected"


def test_practice_mode_scores_and_does_not_touch_text(make_app):
    a = make_app()
    a.toggle_practice()
    assert a.practice is not None and a.target_label.cget("text") in PROTOS
    target = a.target_label.cget("text")
    feed(a, target, 0.7)
    assert a.practice.score == 1 and a.practice.streak == 1
    assert a.text == ""                                    # letters are not added to the text in practice
    assert "Correct" in a.practice_message.cget("text")
    pump(a, 1.4)                                           # next target appears after a short pause
    assert a.practice.target is not None
    a.skip_practice()
    assert a.practice.streak == 0
    a.toggle_practice()
    assert a.practice is None


def test_wrong_letter_in_practice_does_not_score(make_app):
    a = make_app()
    a.toggle_practice()
    target = a.target_label.cget("text")
    wrong = next(c for c in PROTOS if c != target)
    feed(a, wrong, 0.8)
    assert a.practice.score == 0


def test_app_opens_without_templates_and_explains(make_app):
    a = make_app(with_templates=False)
    assert "No templates found" in a.banner.cget("text")
    feed(a, None, 0.1)
    a.queue.put({"type": "frame", "image": FRAME, "hands": 2, "pred": None, "fps": 20.0})
    pump(a, 0.1)                                           # frames still display, nothing crashes
    a.toggle_practice()
    assert a.practice is None
    assert "No letters to practise" in a.practice_message.cget("text")


def test_error_message_is_shown_in_the_banner(make_app):
    a = make_app()
    a.queue.put({"type": "error", "message": "Cannot open camera 3."})
    pump(a, 0.1)
    assert a.banner.cget("text") == "Cannot open camera 3."


def test_settings_window_changes_apply_and_save(make_app, tmp_path):
    a = make_app()
    a.open_settings()
    a.update()
    win = a.settings_window
    slider, label, fmt = win._sliders["hold_time"]
    slider.set(2.0)
    win._on_slider("hold_time", 2.0)
    assert a.settings.hold_time == 2.0 and a.hold.hold_time == 2.0 and "2.0" in label.cget("text")
    win._on_slider("smoothing", 11.4)
    assert a.settings.smoothing == 11
    win._switches["auto_space"].select()
    win._on_switch("auto_space")
    assert a.settings.auto_space is True and a.hold.auto_space is True
    assert Settings.load(tmp_path / "settings.json").hold_time == 2.0
    win._reset()
    assert a.settings.hold_time == 1.0 and a.settings.auto_space is False
    win.destroy()


def test_reference_chart_tab_shows_the_image(make_app):
    a = make_app()
    a.update()
    a.update_idletasks()
    a._fit_chart()
    assert a._chart_pil is not None
    assert a._chart_image is not None


# ------------------------------------------- the real worker thread with a fake camera
class FakeCap:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, *_):
        return True

    def read(self):
        time.sleep(0.01)
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


class FakeTracker:
    letter = "A"

    def __init__(self, *a, **k):
        pass

    def process(self, frame):
        return PROTOS[FakeTracker.letter], None

    def draw(self, frame, result):
        pass

    def close(self):
        pass


def test_worker_reports_a_camera_that_cannot_be_opened(make_app, monkeypatch):
    a = make_app()
    monkeypatch.setattr(appmod, "open_camera", lambda i: FakeCap(opened=False))
    a.start_camera()
    pump(a, 0.6)
    assert "Cannot open camera" in a.banner.cget("text")
    assert a.start_button.cget("text") == "Start" and a.worker is None


def test_worker_pipeline_adds_a_letter_and_supports_pause_and_stop(make_app, monkeypatch):
    a = make_app()
    cap = FakeCap()
    FakeTracker.letter = "C"
    monkeypatch.setattr(appmod, "open_camera", lambda i: cap)
    monkeypatch.setattr(appmod, "HandTracker", FakeTracker)
    a.start_camera()
    pump(a, 1.6)
    assert a.text == "C"
    assert a.status_label.cget("text") == "2 hands detected"
    assert a.fps_label.cget("text").startswith("FPS: ") and a.fps_label.cget("text") != "FPS: -"
    a.toggle_pause()
    assert a.worker.paused.is_set() and a.pause_button.cget("text") == "Resume"
    pump(a, 0.4)
    assert a.status_label.cget("text") == "Paused"
    a.toggle_pause()
    assert a.pause_button.cget("text") == "Pause"
    t0 = time.monotonic()
    a.stop_camera()
    assert time.monotonic() - t0 < 2.5 and cap.released and a.worker is None
    assert a.start_button.cget("text") == "Start"


def test_camera_that_disappears_stops_cleanly(make_app, monkeypatch):
    a = make_app()

    class DyingCap(FakeCap):
        def read(self):
            return False, None

    monkeypatch.setattr(appmod, "open_camera", lambda i: DyingCap())
    monkeypatch.setattr(appmod, "HandTracker", FakeTracker)
    a.start_camera()
    pump(a, 1.8)
    assert "stopped sending pictures" in a.banner.cget("text")
    assert a.start_button.cget("text") == "Start"


def test_letters_constant_is_complete():
    assert LETTERS[0] == "A" and LETTERS[-1] == "Z" and len(LETTERS) == 26
