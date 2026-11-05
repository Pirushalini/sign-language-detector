# Experiments: weighting, palm side and confidence

This adds four experiments to the template matcher. Nothing is trained: every method is a distance
rule over stored templates, as in the rest of the project.

## What you need to run it

| Experiment | Data needed | How to get it |
|---|---|---|
| 1 Landmark weighting | `data/templates_external.npz` and/or `data/templates_user.npz` | `import_dataset.py`, `calibrate.py` |
| 2 Palm-side identification | `data/palm_side_labels.npz` (ground-truth palm/back labels) | `python record_palm_side.py` (about 6 min) |
| 3 Confidence analysis | same templates as Experiment 1 | - |
| 4 Final report | results of 1 to 3 | written automatically |

**The zip you received contains no template data** (`data/` is empty; the `.npz` files are in `.gitignore`).
So no accuracy numbers are included here. Run the commands below on your machine to get them.

```
python record_palm_side.py                       # once, for Experiment 2
python -m experiments.run_experiments            # all experiments -> reports/experiments/
python -m experiments.run_experiments --exp 1 3  # only some
python -m experiments.run_experiments --synthetic --quick   # check that the code runs (numbers are meaningless)
python -m pytest tests/test_experiments.py -q    # tests
```

Output in `reports/experiments/`: `final_report.md` (tables, figures, verdicts), `results.json` (every number),
`fig_*.png`. The report is generated from the numbers; nothing in its tables is hand-written.

## Experiment 1: landmark weighting

The 132-value feature vector is compared with a weighted Euclidean distance. A weight scales the squared difference
of a feature (0 switches it off). Regions (MediaPipe landmark numbers):

| Region | Landmarks |
|---|---|
| Fingertips | 4, 8, 12, 16, 20 |
| Finger joints | 2, 3, 6, 7, 10, 11, 14, 15, 18, 19 |
| Palm and wrist | 0, 1, 5, 9, 13, 17 (the wrist is always 0 in the wrist-relative features, so this is the five palm-base points) |
| Relative position of the two hands | the 4 wrist-to-wrist values |

Strategies, written (fingertips, joints, palm, inter-hand). All are fixed before looking at results:

| Name | Weights | Idea |
|---|---|---|
| `uniform` | (1, 1, 1, 1) | the requested baseline: every feature equal |
| `repo_default` | (1, 1, 1, 2) | what `matcher.py` does today (inter-hand weight 2) |
| `region_balanced` | (1.33, 0.67, 1.33, 1) | each region gets the same total influence (uniform lets the 10 joint points outweigh the 5 fingertips) |
| `tips_emphasis` | (2, 1, 0.5, 2) | fingertips matter most |
| `palm_emphasis` | (0.5, 1, 2, 2) | palm orientation matters most |
| `inter_emphasis` | (1, 1, 1, 4) | hand arrangement matters most |
| `region_tuned` | found by search | 3x3x3x3 grid over {0.5, 1, 2}, chosen on validation folds **inside** each training set |

Also reported: a one-region-at-a-time sweep (weights 0 to 4) and a depth (z) weight sweep. The two presence flags
keep weight 3 in every strategy, because they only guard against matching a one-hand frame to a two-hand template.

## Experiment 2: palm side

For each hand, is the palm or the back turned to the camera? The key quantity is the signed area `s` of the triangle
wrist, index knuckle (5), pinky knuckle (17) in the picture. Its sign flips when the hand is turned over **or** when a
right hand is swapped for a left hand, so the side is `palm <=> s * chirality < 0` once the chirality is known.
The methods differ in where the chirality comes from:

| Method | Chirality from | Needs |
|---|---|---|
| G1 | MediaPipe's Left/Right label | the label (`G1b` tests the swapped mapping, because MediaPipe assumes a mirrored picture) |
| G2 | slot: hand on the left of the picture = right hand | two hands in a natural arrangement |
| G3 | 3D landmarks: thumb relative to the palm plane | MediaPipe's z values |
| G4 | nearest labelled recordings vote (kNN) | labelled recordings |

Reported: accuracy (strict, "unsure" counts as wrong), accuracy on decided hands, coverage, recall and precision per side,
confusion matrices, a breakdown by situation (one or two hands, physical hand, slot, pose), an accuracy-vs-coverage curve,
and how often each chirality source is right on single-hand clips.

**Ground truth comes from the recording protocol**, not from a model: `record_palm_side.py` tells you which hand and which
side to show. Single-hand clips also reveal the physical hand, which checks MediaPipe's label mapping directly.

## Experiment 3: confidence with and without palm side

Palm-side values are added to the matcher in two ways (`experiments/side_matcher.py`):
- **feature**: two extra columns (one per hand slot, value in [-1, 1]), weight 2;
- **penalty**: +0.5 distance for each hand whose *decided* side disagrees with the template's.

Template files carry no MediaPipe label, so templates and queries both use the slot rule (G2) here.
Settings are fixed in advance and a sensitivity sweep is reported.

Raw confidence depends on the distance scale: add a penalty and the same temperature gives higher confidence even if
nothing improved. So the report shows the raw numbers the app would show **and** scale-free measures:
AUROC (does confidence rank right answers above wrong ones), risk-coverage, and ECE after fitting one temperature.
Each palm-side configuration is compared with the same configuration **without** palm side on identical queries,
and classified as improving accuracy and/or confidence reliability, mixed, or no reliable change.

A plausible outcome, stated before running: little or no gain. The sign of `s` is already contained in the wrist-relative
x and y values of the feature vector, so the palm-side value is partly redundant. The experiment tests that.

## Evaluation protocols

| | Templates | Queries | Meaning |
|---|---|---|---|
| A | imported, 80 % | imported, 20 % random | same as `evaluate.py`; optimistic (near-identical neighbouring frames) |
| B | all imported | your recordings | new signer and camera; letters missing from the import (H, J, Y) are excluded |
| **C** (primary) | imported + your recordings minus one block | the held-out block of your recordings | how the app really runs; blocks are consecutive chunks of a letter's recording so look-alike frames never straddle the split |

Statistics: 95 % cluster-bootstrap intervals (letter x block groups are resampled, not frames). "Better" or "worse" is only
claimed when the paired-difference interval excludes 0. The best configuration is the highest-accuracy one, or the
simplest one that is not reliably worse than it. McNemar p-values are printed but treat frames as independent, so read them as optimistic.

## Limits you should know

- Several strategies are compared at once, so one significant result in many can be chance. Trust effects that appear in more than one protocol.
- Protocols B and C use one signer and one camera. Intervals describe sampling noise, not the variation between people.
- Strengths and limitations in the report's section 4 are reasoned from how the methods work; only the tables are measured.
- Palm-side code was checked against a synthetic hand model (sign conventions, edge-on behaviour, failure cases). That proves the
  geometry is implemented as intended, not that MediaPipe's landmarks on real video behave the same way. Experiment 2 on your recordings answers that.

## Files

| File | Purpose |
|---|---|
| `experiments/regions.py` | landmark regions and weight profiles |
| `experiments/palm_side.py` | palm-side methods G1 to G4 and their evaluation |
| `experiments/palm_labels.py` | format of the labelled recordings |
| `experiments/side_matcher.py` | matcher with palm-side feature / penalty |
| `experiments/data.py` | datasets, protocols, inner validation folds |
| `experiments/runner.py` | evaluation, tuning, paired comparisons |
| `experiments/metrics.py` | accuracy, calibration, AUROC, bootstrap |
| `experiments/run_experiments.py` | runs everything, writes the report |
| `experiments/synthetic.py` | synthetic hand model (tests and `--synthetic` only) |
| `record_palm_side.py` | guided recording of palm/back ground truth |
| `matcher.py` | now accepts `feature_weights` (default behaviour unchanged) |
