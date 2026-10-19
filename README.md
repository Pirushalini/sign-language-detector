# Sign Language Detector (two-handed fingerspelling, A to Z)

A desktop app that reads the two-handed manual alphabet from your webcam and writes the letters into a
text box. Everything runs on your own computer. There is **no model training**: the app compares what
the camera sees with stored example hand shapes (template matching).

The signs follow the chart in `assets/alphabet_chart.png` (British-style manual alphabet; C is one-handed,
the other letters use both hands).

## How it works

```
webcam -> MediaPipe Hands (21 points per hand) -> 132 numbers -> compare with stored templates
       -> smoothing -> hold the sign for about 1 second -> letter is added to the text
```

1. **MediaPipe** finds up to two hands and gives 21 landmarks for each.
2. The landmarks are turned into a 132-value feature vector. Each hand is moved to its own wrist and
   scaled by its own size, so the position and the distance from the camera do not matter. The relative
   position of the two hands is added as extra values, because many letters depend on it.
3. The **template matcher** measures the distance to every stored example. For each letter it averages
   the 3 closest examples, then turns the distances into probabilities. The best letter and its
   confidence are shown, together with the Top 3.
4. A **vote over the last 7 frames** removes flicker. A letter is only added when it is held above the
   confidence threshold for the hold time (default 1.0 s).

Where the templates come from:

- **Imported templates** from a public landmark dataset (see `NOTICE.md`). It has 23 letters; **H, J and Y are missing**.
- **Your own templates**, recorded with the guided calibration tool. These are preferred when both match.

Nothing is fitted to the data. If you add or remove templates, the next prediction changes immediately.

## Setup

You need Python 3.9 to 3.12 and a webcam.

**Windows (PowerShell)**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS or Linux**

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Linux you may also need Tk: `sudo apt install python3-tk`.

## Run order

```
python import_dataset.py            # 1. convert the dataset CSV files into templates (once)
python calibrate.py --letters H J Y # 2. record the three missing letters from the chart
python evaluate.py                  # 3. optional: accuracy report and charts in reports/
python app.py                       # 4. start the detector
```

The zip already contains the converted templates (`data/templates_external.npz`), so you can skip step 1
and go straight to step 2. Step 1 is only needed to rebuild them: download `two_hand_dataset.csv` and
`one_hand_dataset.csv` from the Kaggle page named in `NOTICE.md` and put them in `data/external/`.

### Calibration tool

A window shows your camera and the reference chart with the current letter highlighted.

| Key | Action |
|-----|--------|
| SPACE | start recording the current letter (3 second countdown, about 40 samples) |
| N / P | next / previous letter |
| Q or ESC | quit (samples are saved after every letter) |

Useful options: `--letters A B C`, `--samples 60`, `--camera 1`, `--append`. With no options it records all
26 letters. Recording all letters yourself gives the best accuracy for you, because the templates then
match your hands and your camera.

### Using the app

- **Start / Stop**, **Pause / Resume**, and the camera menu are under the camera picture.
- The right side shows the detected letter, confidence, the hold bar and the Top 3, then the detected text.
- Text buttons: **Erase** (last character), **Space**, **Reset** (asks first when the text is long), **Copy**.
- Keyboard: `BackSpace` erases, `Space` adds a space, `Esc` resets.
- Tabs at the bottom: **Reference Chart** (with an Open larger button) and **Practice Mode** (the app
  shows a letter, you sign it, and it keeps a score and a streak).
- **Settings** has the hold time, confidence threshold, smoothing, maximum distance, show landmarks,
  mirror view, auto-space and camera number. Changes are saved in `settings.json`.

Double letters (for example the two L in HELLO): sign the first L, relax or lower your hands for a
moment (about 0.6 s), then sign it again. The same letter is not added twice while you keep holding it.

## Tests

```
python -m pytest tests -q
```

On a server without a screen, run the window tests under a virtual display:
`xvfb-run -a python -m pytest tests -q`. The tests use a fake camera, so no webcam is needed.

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| "Cannot open camera" | Close Zoom, Teams or the browser, or choose another camera in the menu. |
| "No templates found" banner | Run `python import_dataset.py`, or record letters with `calibrate.py`. |
| Wrong letters | Record your own samples (`calibrate.py`), raise the threshold, or move so both hands are fully visible. |
| Slow video | Close other apps, or lower the camera resolution in good light. |
| Left-handed signing | Run `python import_dataset.py --mirror` to add mirrored copies. |
| `pip install` fails on mediapipe | Check `python --version`. Use 3.9 to 3.12. |

## Limitations (please read)

- The dataset's 99.8% accuracy in `evaluate.py` comes from a random split of near-identical frames, so it
  is **optimistic**. A fair test is imported templates against your own recordings. After you run
  `calibrate.py`, `evaluate.py` prints that second result.
- The dataset probably comes from one signer and one camera. Other hands, lighting and cameras give lower
  accuracy until you add your own templates.
- The image shape of the dataset is not known. The importer assumes 16:9 (`--aspect`). If imported letters
  match badly but your own samples work well, try `--aspect 1.333`.
- Letters with very similar shapes (for example T and N, or I and E) can be confused.
- MediaPipe sometimes loses a hand when the hands overlap, which blocks some letters for a moment.
- Only the 26 letters of the chart are supported (no numbers, no words or sentences).

## Project files

| File | Purpose |
|------|---------|
| `app.py` | the CustomTkinter window and the camera thread |
| `logic.py` | settings, smoothing, hold-to-add and practice logic (no GUI code) |
| `matcher.py` | template matcher (distance, voting, probabilities) |
| `utils.py` | hand tracking, feature vector, camera helpers, chart cells |
| `import_dataset.py` | converts the public CSV files into templates |
| `calibrate.py` | guided recording of your own templates |
| `evaluate.py` | accuracy report, confusion matrix and ablations |
| `tests/` | automated tests |

## Credits

Landmark data: erentatepe, "BSL Numbers and Alphabet Hand Position for MediaPipe" (Kaggle), from the
Brunel University final-year project by datMaul. See `NOTICE.md` for the license situation.
Hand tracking: Google MediaPipe. Interface: CustomTkinter.
