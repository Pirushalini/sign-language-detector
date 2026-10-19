"""Sign Language Detector: desktop app (CustomTkinter, light theme).

Camera feed on the left, detected letter and text on the right, reference chart
and practice mode below. Detection runs in a background thread; the window only
draws results, so it stays responsive.

Run:  python app.py
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import fields
from tkinter import messagebox
from typing import Dict, List, Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from logic import HoldToAdd, PracticeSession, Settings, Smoother
from matcher import Prediction, TemplateMatcher
from utils import CHART_PATH, SETTINGS_PATH, HandTracker, build_features, list_cameras, open_camera

# ------------------------------------------------------------------ style
BG = "#FFFFFF"
PANEL = "#F3F4F6"
BORDER = "#E5E7EB"
ACCENT = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
TEXT = "#111827"
MUTED = "#4B5563"
GREEN = "#16A34A"
AMBER = "#D97706"
RED = "#DC2626"
GRAY = "#9CA3AF"
SHORTCUT_NOTE = "Shortcuts: Backspace erase, Space space, Esc reset"


def font_family() -> Optional[str]:
    if sys.platform.startswith("win"):
        return "Segoe UI"
    if sys.platform == "darwin":
        return "Helvetica Neue"
    return None


def make_font(size: int, weight: str = "normal") -> ctk.CTkFont:
    return ctk.CTkFont(family=font_family(), size=size, weight=weight)


def primary_button(master, text, command, **kw):
    kw.setdefault("height", 36)
    return ctk.CTkButton(master, text=text, command=command, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                         text_color="#FFFFFF", corner_radius=8, font=make_font(14, "bold"), **kw)


def secondary_button(master, text, command, **kw):
    kw.setdefault("height", 36)
    return ctk.CTkButton(master, text=text, command=command, fg_color=BORDER, hover_color="#D1D5DB",
                         text_color=TEXT, corner_radius=8, font=make_font(14), **kw)


# ------------------------------------------------------------ camera worker
class DetectorWorker(threading.Thread):
    """Reads the camera, tracks hands and matches templates in the background.

    Results go to a queue as dictionaries:
        {"type": "frame", "image": RGB array, "hands": int, "pred": Prediction or None, "fps": float}
        {"type": "status", "state": "starting" | "running" | "stopped"}
        {"type": "error", "message": str}
    """

    def __init__(self, camera_index: int, matcher: TemplateMatcher, settings: Settings, out: "queue.Queue"):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.matcher = matcher
        self.settings = settings
        self.out = out
        self._stop_flag = threading.Event()
        self.paused = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def _put(self, msg: dict) -> None:
        try:
            self.out.put_nowait(msg)
        except queue.Full:
            try:
                self.out.get_nowait()       # drop the oldest message
            except queue.Empty:
                pass
            try:
                self.out.put_nowait(msg)
            except queue.Full:
                pass

    def run(self) -> None:
        self._put({"type": "status", "state": "starting"})
        cap = open_camera(self.camera_index)
        if cap is None or not cap.isOpened():
            self._put({"type": "error", "message": f"Cannot open camera {self.camera_index}. Close other apps "
                       "that use the webcam, or pick another camera."})
            self._put({"type": "status", "state": "stopped"})
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        tracker = HandTracker()
        fps, last_t, failures = 0.0, time.monotonic(), 0
        paused_image: Optional[np.ndarray] = None
        last_rgb: Optional[np.ndarray] = None
        try:
            self._put({"type": "status", "state": "running"})
            while not self._stop_flag.is_set():
                if self.paused.is_set():
                    if paused_image is None and last_rgb is not None:
                        paused_image = last_rgb.copy()
                        cv2.putText(paused_image, "Paused", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                    (255, 255, 255), 5, cv2.LINE_AA)
                        cv2.putText(paused_image, "Paused", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                    (37, 99, 235), 2, cv2.LINE_AA)
                    if paused_image is not None:
                        self._put({"type": "frame", "image": paused_image, "hands": -1, "pred": None,
                                   "fps": 0.0, "paused": True})
                    time.sleep(0.1)
                    continue
                paused_image = None

                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures > 30:
                        self._put({"type": "error", "message": "The camera stopped sending pictures. It may "
                                   "have been disconnected. Reconnect it and press Start."})
                        break
                    time.sleep(0.03)
                    continue
                failures = 0

                h, w = frame.shape[:2]
                hands, result = tracker.process(frame)       # always on the raw frame
                pred: Optional[Prediction] = None
                if hands and self.matcher.available:
                    pred = self.matcher.predict(build_features(hands, aspect=w / h))
                if self.settings.show_landmarks:
                    tracker.draw(frame, result)
                if self.settings.mirror:
                    frame = cv2.flip(frame, 1)               # display only
                if w > 960:
                    frame = cv2.resize(frame, (960, int(h * 960 / w)))
                last_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                now = time.monotonic()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
                self._put({"type": "frame", "image": last_rgb, "hands": len(hands), "pred": pred, "fps": fps})
        except Exception as exc:  # keep any tracking problem away from the user as a traceback
            self._put({"type": "error", "message": f"Camera or tracking problem: {exc}"})
        finally:
            cap.release()
            tracker.close()
            self._put({"type": "status", "state": "stopped"})


# ---------------------------------------------------------- settings window
class SettingsWindow(ctk.CTkToplevel):
    """Sliders and switches. Changes apply immediately and are saved."""

    def __init__(self, app: "App"):
        super().__init__(app)
        self.app = app
        self.title("Settings")
        self.geometry("460x640")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.transient(app)
        self.columnconfigure(0, weight=1)
        self._sliders: Dict[str, tuple] = {}
        self._switches: Dict[str, ctk.CTkSwitch] = {}

        ctk.CTkLabel(self, text="Settings", font=make_font(20, "bold"), text_color=TEXT).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(16, 0))
        row = 1
        for key, label, lo, hi, steps, fmt in [
            ("hold_time", "Hold time to add a letter", 0.3, 2.5, 22, lambda v: f"{v:.1f} s"),
            ("threshold", "Confidence threshold", 0.5, 0.99, 49, lambda v: f"{v * 100:.0f}%"),
            ("smoothing", "Smoothing frames", 1, 15, 14, lambda v: f"{int(round(v))}"),
            ("max_distance", "Max match distance (higher accepts looser matches)", 0.5, 3.0, 50, lambda v: f"{v:.2f}"),
        ]:
            self._add_slider(row, key, label, lo, hi, steps, fmt)
            row += 2
        for key, label in [("show_landmarks", "Show hand landmarks"), ("mirror", "Mirror camera picture"),
                           ("auto_space", "Add a space automatically when hands are gone")]:
            sw = ctk.CTkSwitch(self, text=label, command=lambda k=key: self._on_switch(k), progress_color=ACCENT,
                               text_color=TEXT, font=make_font(13))
            sw.grid(row=row, column=0, columnspan=2, sticky="w", padx=20, pady=(14, 0))
            self._switches[key] = sw
            row += 1
        self._add_slider(row, "auto_space_delay", "Auto-space delay", 1.0, 5.0, 8, lambda v: f"{v:.1f} s")
        row += 2
        bar = ctk.CTkFrame(self, fg_color=BG)
        bar.grid(row=row, column=0, columnspan=2, sticky="ew", padx=20, pady=24)
        secondary_button(bar, "Reset to defaults", self._reset, width=160).pack(side="left")
        primary_button(bar, "Close", self.destroy, width=100).pack(side="right")
        self._refresh()

    def _add_slider(self, row, key, label, lo, hi, steps, fmt) -> None:
        ctk.CTkLabel(self, text=label, text_color=TEXT, font=make_font(13), anchor="w", wraplength=330,
                     justify="left").grid(row=row, column=0, sticky="w", padx=(20, 0), pady=(14, 0))
        value_label = ctk.CTkLabel(self, text="", text_color=MUTED, font=make_font(13))
        value_label.grid(row=row, column=1, sticky="e", padx=20, pady=(14, 0))
        slider = ctk.CTkSlider(self, from_=lo, to=hi, number_of_steps=steps, progress_color=ACCENT,
                               button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                               command=lambda v, k=key: self._on_slider(k, v))
        slider.grid(row=row + 1, column=0, columnspan=2, sticky="ew", padx=20, pady=(4, 0))
        self._sliders[key] = (slider, value_label, fmt)

    def _on_slider(self, key: str, value: float) -> None:
        if key == "smoothing":
            value = int(round(value))
        setattr(self.app.settings, key, value)
        self.app.apply_settings()
        _, value_label, fmt = self._sliders[key]
        value_label.configure(text=fmt(value))

    def _on_switch(self, key: str) -> None:
        setattr(self.app.settings, key, bool(self._switches[key].get()))
        self.app.apply_settings()

    def _refresh(self) -> None:
        for key, (slider, value_label, fmt) in self._sliders.items():
            value = getattr(self.app.settings, key)
            slider.set(value)
            value_label.configure(text=fmt(value))
        for key, sw in self._switches.items():
            sw.select() if getattr(self.app.settings, key) else sw.deselect()

    def _reset(self) -> None:
        defaults = Settings()
        for f in fields(Settings):
            if f.name != "camera_index":
                setattr(self.app.settings, f.name, getattr(defaults, f.name))
        self.app.apply_settings()
        self._refresh()


# ------------------------------------------------------------------- main app
class App(ctk.CTk):
    def __init__(self, autostart: bool = True, scan_cameras: bool = True):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("Sign Language Detector")
        self.geometry("1280x820")
        self.minsize(1100, 720)
        self.configure(fg_color=BG)

        self.settings = Settings.load(SETTINGS_PATH)
        self.matcher, self.model_message = self._load_matcher()
        self.smoother = Smoother(self.settings.smoothing)
        self.hold = HoldToAdd()
        self.practice: Optional[PracticeSession] = None
        self.practice_hold: Optional[HoldToAdd] = None
        self._practice_pause_until = 0.0
        self.text = ""
        self.queue: "queue.Queue" = queue.Queue(maxsize=4)
        self.worker: Optional[DetectorWorker] = None
        self.cam_state = "stopped"
        self.error_message = ""
        self.settings_window: Optional[SettingsWindow] = None
        self._cam_image = None
        self._chart_image = None
        self._chart_pil: Optional[Image.Image] = None
        self._autostart = autostart
        self._cache: Dict[str, object] = {}

        self._build_ui()
        self.apply_settings()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_banner()
        self._set_status("Looking for cameras..." if scan_cameras else "Camera stopped", GRAY)
        if scan_cameras:
            threading.Thread(target=self._scan_cameras, daemon=True).start()
        self.after(15, self._poll)

    # ------------------------------------------------------------- setup
    def _load_matcher(self):
        try:
            matcher = TemplateMatcher.from_files(reject_distance=self.settings.max_distance)
        except ValueError as exc:
            return TemplateMatcher(), str(exc)
        if not matcher.available:
            return matcher, ("No templates found. Run import_dataset.py and/or calibrate.py first. "
                             "Predictions are disabled until then.")
        return matcher, ""

    def _card(self, master, **grid):
        card = ctk.CTkFrame(master, fg_color=PANEL, corner_radius=12)
        card.grid(**grid)
        return card

    def _build_ui(self) -> None:
        # Top bar
        top = ctk.CTkFrame(self, fg_color=BG, height=56)
        top.pack(fill="x", padx=16, pady=(10, 0))
        ctk.CTkLabel(top, text="Sign Language Detector", font=make_font(22, "bold"), text_color=TEXT).pack(side="left")
        secondary_button(top, "Settings", self.open_settings, width=100).pack(side="right")

        main = ctk.CTkFrame(self, fg_color=BG)
        main.pack(fill="both", expand=True, padx=16, pady=10)
        main.columnconfigure(0, weight=3, uniform="cols")
        main.columnconfigure(1, weight=2, uniform="cols")
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        self._build_left(main)
        self._build_right(main)
        self._build_tabs(main)

    def _build_left(self, main) -> None:
        left = self._card(main, row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.cam_box = ctk.CTkFrame(left, fg_color=BORDER, corner_radius=8)
        self.cam_box.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        self.cam_box.grid_propagate(False)
        self.cam_label = ctk.CTkLabel(self.cam_box, text="Camera stopped", text_color=MUTED, font=make_font(16),
                                      compound="center", fg_color="transparent")
        self.cam_label.place(relx=0.5, rely=0.5, anchor="center")

        self.banner = ctk.CTkLabel(left, text="", font=make_font(13), anchor="w", justify="left", wraplength=620)
        self.banner.grid(row=1, column=0, sticky="ew", padx=14)

        status = ctk.CTkFrame(left, fg_color="transparent")
        status.grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 0))
        self.dot = ctk.CTkFrame(status, width=12, height=12, corner_radius=6, fg_color=GRAY)
        self.dot.pack(side="left", padx=(0, 8))
        self.status_label = ctk.CTkLabel(status, text="", font=make_font(14), text_color=TEXT)
        self.status_label.pack(side="left")
        self.fps_label = ctk.CTkLabel(status, text="FPS: -", font=make_font(14), text_color=MUTED)
        self.fps_label.pack(side="right")

        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 12))
        self.start_button = primary_button(controls, "Start", self.toggle_camera, width=90)
        self.start_button.pack(side="left")
        self.pause_button = secondary_button(controls, "Pause", self.toggle_pause, width=90, state="disabled")
        self.pause_button.pack(side="left", padx=8)
        self.camera_menu = ctk.CTkOptionMenu(controls, values=["Camera 0"], command=self._on_camera_choice,
                                             fg_color=BG, button_color=BORDER, button_hover_color="#D1D5DB",
                                             text_color=TEXT, dropdown_fg_color=BG, dropdown_text_color=TEXT,
                                             dropdown_hover_color=PANEL, font=make_font(13), width=130, height=36)
        self.camera_menu.pack(side="left", padx=(8, 0))

    def _build_right(self, main) -> None:
        right = ctk.CTkFrame(main, fg_color=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # Detected letter card
        letter = self._card(right, row=0, column=0, sticky="ew", pady=(0, 8))
        letter.columnconfigure(1, weight=1)
        ctk.CTkLabel(letter, text="DETECTED LETTER", font=make_font(12, "bold"), text_color=MUTED).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(10, 0))
        self.letter_label = ctk.CTkLabel(letter, text="-", font=make_font(56, "bold"), text_color=TEXT)
        self.letter_label.grid(row=1, column=0, columnspan=3, pady=(0, 2))
        ctk.CTkLabel(letter, text="Confidence", font=make_font(13), text_color=MUTED).grid(
            row=2, column=0, sticky="w", padx=(16, 8))
        self.conf_bar = ctk.CTkProgressBar(letter, height=10, progress_color=ACCENT, fg_color=BORDER)
        self.conf_bar.set(0)
        self.conf_bar.grid(row=2, column=1, sticky="ew")
        self.conf_value = ctk.CTkLabel(letter, text="0%", font=make_font(13), text_color=TEXT, width=48)
        self.conf_value.grid(row=2, column=2, padx=(8, 16))
        self.hold_caption = ctk.CTkLabel(letter, text="Hold", font=make_font(13), text_color=MUTED)
        self.hold_caption.grid(row=3, column=0, sticky="w", padx=(16, 8), pady=(8, 0))
        self.hold_bar = ctk.CTkProgressBar(letter, height=10, progress_color=GREEN, fg_color=BORDER)
        self.hold_bar.set(0)
        self.hold_bar.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(0, 16), pady=(8, 0))
        self.top3_label = ctk.CTkLabel(letter, text="Top 3:  -", font=make_font(13), text_color=TEXT, anchor="w")
        self.top3_label.grid(row=4, column=0, columnspan=3, sticky="w", padx=16, pady=(6, 10))

        # Detected text card
        text_card = self._card(right, row=1, column=0, sticky="nsew")
        text_card.columnconfigure((0, 1, 2, 3), weight=1)
        text_card.rowconfigure(1, weight=1)
        ctk.CTkLabel(text_card, text="DETECTED TEXT", font=make_font(12, "bold"), text_color=MUTED).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=16, pady=(10, 4))
        self.textbox = ctk.CTkTextbox(text_card, height=60, font=make_font(28, "bold"), fg_color=BG,
                                      text_color=TEXT, border_width=1, border_color=BORDER, corner_radius=8,
                                      wrap="word")
        self.textbox.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=16)
        self.textbox.configure(state="disabled")
        for col, (name, command) in enumerate([("Erase", self.erase), ("Space", self.add_space),
                                               ("Reset", self.reset_text), ("Copy", self.copy_text)]):
            secondary_button(text_card, name, command).grid(
                row=2, column=col, sticky="ew", padx=(16 if col == 0 else 3, 16 if col == 3 else 3), pady=(10, 0))
        self.text_note = ctk.CTkLabel(text_card, text=SHORTCUT_NOTE, font=make_font(12), text_color=MUTED, anchor="w")
        self.text_note.grid(row=3, column=0, columnspan=4, sticky="w", padx=16, pady=(8, 10))

    def _build_tabs(self, main) -> None:
        tabs = ctk.CTkTabview(main, height=230, fg_color=PANEL, corner_radius=12,
                              segmented_button_fg_color=BORDER, segmented_button_selected_color=BG,
                              segmented_button_selected_hover_color=BG, segmented_button_unselected_color=BORDER,
                              segmented_button_unselected_hover_color="#D1D5DB", text_color=TEXT)
        tabs.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.tabs = tabs
        chart_tab = tabs.add("Reference Chart")
        practice_tab = tabs.add("Practice Mode")

        # Reference chart
        self.chart_box = ctk.CTkFrame(chart_tab, fg_color=PANEL)
        self.chart_box.pack(fill="both", expand=True)
        self.chart_box.pack_propagate(False)
        self.chart_label = ctk.CTkLabel(self.chart_box, text="", text_color=MUTED, font=make_font(14))
        self.chart_label.place(relx=0.5, rely=0.5, anchor="center")
        if CHART_PATH.exists():
            try:
                self._chart_pil = Image.open(CHART_PATH).convert("RGB")
            except OSError:
                self._chart_pil = None
        if self._chart_pil is None:
            self.chart_label.configure(text="Reference chart not found. Put your chart image at "
                                            "assets/alphabet_chart.png")
        else:
            secondary_button(self.chart_box, "Open larger", self.open_chart_large, width=110, height=30).place(
                relx=1.0, rely=0.0, anchor="ne", x=-4, y=2)
            self.chart_box.bind("<Configure>", lambda e: self.after(60, self._fit_chart))

        # Practice mode
        practice_tab.columnconfigure((0, 1, 2), weight=1)
        left = ctk.CTkFrame(practice_tab, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=12)
        self.practice_button = primary_button(left, "Start Practice", self.toggle_practice, width=150)
        self.practice_button.pack(pady=(24, 8))
        self.skip_button = secondary_button(left, "Skip", self.skip_practice, width=150, state="disabled")
        self.skip_button.pack()
        middle = ctk.CTkFrame(practice_tab, fg_color="transparent")
        middle.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(middle, text="Target letter", font=make_font(13), text_color=MUTED).pack(pady=(14, 0))
        self.target_label = ctk.CTkLabel(middle, text="-", font=make_font(72, "bold"), text_color=ACCENT)
        self.target_label.pack()
        self.practice_sign = ctk.CTkLabel(middle, text="Your sign: -", font=make_font(14), text_color=TEXT)
        self.practice_sign.pack()
        stats = ctk.CTkFrame(practice_tab, fg_color="transparent")
        stats.grid(row=0, column=2, sticky="nsew", padx=12)
        self.score_label = ctk.CTkLabel(stats, text="Score: 0", font=make_font(18, "bold"), text_color=TEXT)
        self.score_label.pack(pady=(28, 2), anchor="w")
        self.streak_label = ctk.CTkLabel(stats, text="Streak: 0", font=make_font(15), text_color=TEXT)
        self.streak_label.pack(anchor="w")
        self.best_label = ctk.CTkLabel(stats, text="Best streak: 0", font=make_font(15), text_color=MUTED)
        self.best_label.pack(anchor="w")
        self.practice_message = ctk.CTkLabel(stats, text="Press Start Practice, then hold the target sign.",
                                             font=make_font(13), text_color=MUTED, wraplength=260, justify="left")
        self.practice_message.pack(pady=(10, 0), anchor="w")

    def _bind_keys(self) -> None:
        self.bind("<BackSpace>", lambda e: self.erase())
        self.bind("<space>", lambda e: self.add_space())
        self.bind("<Escape>", lambda e: self.reset_text())

    # --------------------------------------------------------- small helpers
    def _set_label(self, widget, key: str, **kwargs) -> None:
        """Configure a widget only when the value changed (avoids needless redraws)."""
        if self._cache.get(key) != kwargs:
            self._cache[key] = kwargs
            widget.configure(**kwargs)

    def _set_status(self, text: str, color: str) -> None:
        self._set_label(self.status_label, "status_text", text=text)
        self._set_label(self.dot, "status_dot", fg_color=color)

    def _refresh_banner(self) -> None:
        if self.error_message:
            self.banner.configure(text=self.error_message, text_color=RED)
        elif self.model_message:
            self.banner.configure(text=self.model_message, text_color=AMBER)
        else:
            self.banner.configure(text="")

    # ----------------------------------------------------------- camera control
    def _scan_cameras(self) -> None:
        found = list_cameras()
        try:
            self.queue.put({"type": "cameras", "list": found})
        except queue.Full:
            pass

    def _on_cameras(self, found: List[int]) -> None:
        if not found:
            self.camera_menu.configure(values=["No camera"])
            self.camera_menu.set("No camera")
            self.error_message = ("No camera found. Check that a webcam is connected and not used by another app, "
                                  "then restart the program.")
            self._refresh_banner()
            self._set_status("Camera stopped", GRAY)
            return
        values = [f"Camera {i}" for i in found]
        self.camera_menu.configure(values=values)
        if self.settings.camera_index not in found:
            self.settings.camera_index = found[0]
        self.camera_menu.set(f"Camera {self.settings.camera_index}")
        self._set_status("Camera stopped", GRAY)
        if self._autostart and self.worker is None:
            self.start_camera()

    def _on_camera_choice(self, choice: str) -> None:
        if not choice.startswith("Camera "):
            return
        self.settings.camera_index = int(choice.split()[1])
        self.settings.save(SETTINGS_PATH)
        if self.worker is not None:
            self.start_camera()

    def toggle_camera(self) -> None:
        if self.worker is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self) -> None:
        self.stop_camera()
        self.error_message = ""
        self._refresh_banner()
        self.smoother.reset()
        self.hold.reset()
        self.hold.set_text_state(self.text)
        self.worker = DetectorWorker(self.settings.camera_index, self.matcher, self.settings, self.queue)
        self.worker.start()
        self.start_button.configure(text="Stop")
        self.pause_button.configure(state="normal", text="Pause")
        self._set_status("Starting camera...", AMBER)

    def stop_camera(self) -> None:
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.stop()
            worker.join(timeout=2.0)
        self.start_button.configure(text="Start")
        self.pause_button.configure(state="disabled", text="Pause")
        self.cam_state = "stopped"
        self._set_status("Camera stopped", GRAY)
        self._set_label(self.fps_label, "fps", text="FPS: -")
        self._show_placeholder("Camera stopped")
        self._clear_detection()

    def toggle_pause(self) -> None:
        if self.worker is None:
            return
        if self.worker.paused.is_set():
            self.worker.paused.clear()
            self.pause_button.configure(text="Pause")
            self.smoother.reset()
        else:
            self.worker.paused.set()
            self.pause_button.configure(text="Resume")
            self._set_status("Paused", AMBER)
            self._clear_detection()

    def _show_placeholder(self, text: str) -> None:
        self._cam_image = ctk.CTkImage(light_image=Image.new("RGB", (4, 4), BORDER), size=(4, 4))
        self.cam_label.configure(image=self._cam_image, text=text)

    # --------------------------------------------------------------- polling
    def _poll(self) -> None:
        self._drain()
        self.after(15, self._poll)

    def _drain(self) -> None:
        latest = None
        while True:
            try:
                msg = self.queue.get_nowait()
            except queue.Empty:
                break
            kind = msg.get("type")
            if kind == "frame":
                paused_now = self.worker is not None and self.worker.paused.is_set()
                if paused_now and not msg.get("paused"):
                    continue          # a live frame queued just before Pause was pressed: ignore it
                latest = msg["image"]
                if not msg.get("paused"):
                    self._on_result(msg)
            elif kind == "status":
                self._on_state(msg["state"])
            elif kind == "error":
                self.error_message = msg["message"]
                self._refresh_banner()
            elif kind == "cameras":
                self._on_cameras(msg["list"])
        if latest is not None:
            self._show_frame(latest)

    def _on_state(self, state: str) -> None:
        self.cam_state = state
        if state == "starting":
            self._set_status("Starting camera...", AMBER)
        elif state == "stopped" and self.worker is not None and not self.worker.is_alive():
            # The worker ended by itself (for example the camera was unplugged).
            self.worker = None
            self.start_button.configure(text="Start")
            self.pause_button.configure(state="disabled", text="Pause")
            self._set_status("Camera stopped", GRAY)
            self._show_placeholder("Camera stopped")
            self._clear_detection()

    def _show_frame(self, rgb: np.ndarray) -> None:
        box_w, box_h = self.cam_box.winfo_width(), self.cam_box.winfo_height()
        if box_w < 50 or box_h < 50:
            return
        h, w = rgb.shape[:2]
        scale = min(box_w / w, box_h / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        self._cam_image = ctk.CTkImage(light_image=Image.fromarray(rgb), size=size)
        self.cam_label.configure(image=self._cam_image, text="")

    # ------------------------------------------------------- detection result
    def _on_result(self, msg: dict) -> None:
        n_hands: int = msg["hands"]
        pred: Optional[Prediction] = msg["pred"]
        self._set_label(self.fps_label, "fps", text=f"FPS: {msg['fps']:.0f}")
        label, conf = self.smoother.update(pred if n_hands > 0 else None)
        now = time.monotonic()
        practicing = self.practice is not None and self.practice_hold is not None
        hold = self.practice_hold if practicing else self.hold
        event = None
        if not (practicing and now < self._practice_pause_until):
            event = hold.update(label, conf, now, n_hands > 0)

        self._update_detected(label, conf, pred, hold)
        self._update_status(n_hands, label, conf)
        if practicing:
            self._set_label(self.practice_sign, "pr_sign", text=f"Your sign: {label or '-'}")
        if event is None:
            return
        if practicing:
            if event.kind == "letter":
                self._practice_success()
        elif event.kind == "letter":
            self.text += event.letter
            self._refresh_text()
        elif event.kind == "space":
            self.text += " "
            self._refresh_text()

    def _update_detected(self, label, conf, pred, hold: HoldToAdd) -> None:
        self._set_label(self.letter_label, "letter", text=label or "-")
        self._set_label(self.conf_value, "conf_text", text=f"{conf * 100:.0f}%")
        self.conf_bar.set(conf)
        self.hold_bar.set(hold.progress)
        caption = "Hold (release to repeat)" if hold.blocked else "Hold"
        self._set_label(self.hold_caption, "hold_caption", text=caption)
        if pred is not None and pred.top3:
            top = "   ".join(f"{c} {p * 100:.0f}%" for c, p in pred.top3)
        else:
            top = "-"
        self._set_label(self.top3_label, "top3", text=f"Top 3:  {top}")

    def _update_status(self, n_hands: int, label, conf) -> None:
        if n_hands == 0:
            self._set_status("No hands detected", AMBER)
        elif n_hands == 1:
            text = "1 hand detected"
            if self.matcher.available and (label is None or conf < self.settings.threshold):
                text += " - show both hands for best results"
            self._set_status(text, AMBER)
        else:
            self._set_status(f"{n_hands} hands detected", GREEN)

    def _clear_detection(self) -> None:
        self.smoother.reset()
        self.hold.reset()
        self.hold.set_text_state(self.text)
        self._set_label(self.letter_label, "letter", text="-")
        self._set_label(self.conf_value, "conf_text", text="0%")
        self._set_label(self.top3_label, "top3", text="Top 3:  -")
        self.conf_bar.set(0)
        self.hold_bar.set(0)

    # ------------------------------------------------------------- text panel
    def _refresh_text(self) -> None:
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", self.text)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")
        self.hold.set_text_state(self.text)

    def erase(self) -> None:
        if self.text:
            self.text = self.text[:-1]
            self._refresh_text()

    def add_space(self) -> None:
        self.text += " "
        self._refresh_text()

    def reset_text(self) -> None:
        if not self.text:
            return
        if len(self.text) > 15 and not messagebox.askyesno("Reset", "Clear all detected text?", parent=self):
            return
        self.text = ""
        self._refresh_text()

    def copy_text(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.text)
        self._set_label(self.text_note, "note", text="Copied", text_color=GREEN)
        self.after(1500, lambda: self._set_label(self.text_note, "note", text=SHORTCUT_NOTE, text_color=MUTED))

    # ---------------------------------------------------------- practice mode
    def toggle_practice(self) -> None:
        if self.practice is not None:
            self.practice = None
            self.practice_hold = None
            self.practice_button.configure(text="Start Practice")
            self.skip_button.configure(state="disabled")
            self.target_label.configure(text="-")
            self.practice_message.configure(text="Practice stopped. Detected letters are added to the text again.",
                                            text_color=MUTED)
            self._set_label(self.text_note, "note", text=SHORTCUT_NOTE, text_color=MUTED)
            self._clear_detection()
            return
        letters = self.matcher.classes if self.matcher.available else []
        if not letters:
            self.practice_message.configure(text="No letters to practise yet. Run import_dataset.py or "
                                                 "calibrate.py first.", text_color=AMBER)
            return
        self.practice = PracticeSession(letters)
        self.practice_button.configure(text="Stop Practice")
        self.skip_button.configure(state="normal")
        self.score_label.configure(text="Score: 0")
        self.streak_label.configure(text="Streak: 0")
        self.best_label.configure(text="Best streak: 0")
        self._new_practice_target()
        self._set_label(self.text_note, "note", text="Practice mode is on: letters are not added to the text.",
                        text_color=MUTED)
        self.after(0, self._clear_detection)

    def _new_practice_target(self) -> None:
        if self.practice is None:
            return
        target = self.practice.next_target()
        self.practice_hold = HoldToAdd(self.settings.hold_time, self.settings.threshold, only_letter=target)
        self.target_label.configure(text=target or "-")
        self.practice_message.configure(text="Hold the target sign until the bar is full.", text_color=MUTED)
        self.smoother.reset()

    def _practice_success(self) -> None:
        assert self.practice is not None
        self.practice.succeed()
        self.score_label.configure(text=f"Score: {self.practice.score}")
        self.streak_label.configure(text=f"Streak: {self.practice.streak}")
        self.best_label.configure(text=f"Best streak: {self.practice.best_streak}")
        self.practice_message.configure(text="Correct", text_color=GREEN)
        self._practice_pause_until = time.monotonic() + 1.2
        self.after(1200, self._new_practice_target)

    def skip_practice(self) -> None:
        if self.practice is None:
            return
        self.practice.skip()
        self.streak_label.configure(text="Streak: 0")
        self._new_practice_target()

    # --------------------------------------------------------------- settings
    def open_settings(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        self.settings_window = SettingsWindow(self)

    def apply_settings(self) -> None:
        s = self.settings.clamp()
        self.smoother.resize(s.smoothing)
        for h in (self.hold, self.practice_hold):
            if h is not None:
                h.hold_time, h.threshold = s.hold_time, s.threshold
        self.hold.auto_space, self.hold.auto_space_delay = s.auto_space, s.auto_space_delay
        self.matcher.reject_distance = s.max_distance
        s.save(SETTINGS_PATH)

    # ------------------------------------------------------------------ chart
    def _fit_chart(self) -> None:
        if self._chart_pil is None:
            return
        box_w, box_h = self.chart_box.winfo_width(), self.chart_box.winfo_height()
        if box_w < 50 or box_h < 50:
            return
        iw, ih = self._chart_pil.size
        scale = min(box_w / iw, box_h / ih)
        size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        if self._cache.get("chart_size") == size:
            return
        self._cache["chart_size"] = size
        self._chart_image = ctk.CTkImage(light_image=self._chart_pil, size=size)
        self.chart_label.configure(image=self._chart_image)

    def open_chart_large(self) -> None:
        if self._chart_pil is None:
            return
        win = ctk.CTkToplevel(self)
        win.title("Reference Chart")
        win.configure(fg_color=BG)
        iw, ih = self._chart_pil.size
        scale = min(1000 / iw, (self.winfo_screenheight() - 160) / ih)
        size = (int(iw * scale), int(ih * scale))
        win._image = ctk.CTkImage(light_image=self._chart_pil, size=size)   # keep a reference
        ctk.CTkLabel(win, text="", image=win._image).pack(padx=10, pady=10)
        win.after(100, win.lift)

    # ------------------------------------------------------------------ close
    def _on_close(self) -> None:
        self.stop_camera()
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
