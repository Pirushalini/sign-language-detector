"""Run all four experiments and write the comparative report.

    python -m experiments.run_experiments                 # real data in data/
    python -m experiments.run_experiments --synthetic     # pipeline check only (numbers are meaningless)
    python -m experiments.run_experiments --exp 1 3       # only some experiments
    python -m experiments.run_experiments --quick         # smaller search grid, fewer bootstrap rounds

Outputs (reports/experiments/):  final_report.md, results.json, fig_*.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import LETTERS, REPORTS_DIR, DATA_DIR

from . import data as D
from . import metrics as M
from . import palm_side as PS
from . import runner as R
from .regions import GRID_VALUES, PROFILES, SWEEP_VALUES

PALM_LABELS_DEFAULT = DATA_DIR / "palm_side_labels.npz"
OUT_DEFAULT = REPORTS_DIR / "experiments"

# Settings of Experiment 3, fixed before looking at results.
SIDE_FEATURE_WEIGHT = 2.0
SIDE_PENALTY = 0.5
SIDE_FEATURE_SWEEP = (0.5, 1.0, 2.0, 4.0)
SIDE_PENALTY_SWEEP = (0.25, 0.5, 1.0, 2.0)
SEP = " — "      # separates the base weighting from the palm-side variant in a configuration name


# ================================================================== helpers
def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_ci(t, nd=3):
    p, lo, hi = t
    return f"{p:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def verdict(t) -> str:
    _, lo, hi = t
    if lo > 0:
        return "better (interval excludes 0)"
    if hi < 0:
        return "worse (interval excludes 0)"
    return "no reliable difference (interval includes 0)"


def classify_effect(eff) -> str:
    """One label for the paired effect of palm-side identification (accuracy and confidence reliability)."""
    a, r = eff["delta_accuracy"], eff["delta_auroc"]
    acc = "up" if a[1] > 0 else "down" if a[2] < 0 else "same"
    rel = "up" if r[1] > 0 else "down" if r[2] < 0 else "same"
    return {
        ("up", "up"): "improves accuracy and confidence reliability",
        ("up", "same"): "improves accuracy, reliability unchanged",
        ("up", "down"): "MIXED: accuracy up, confidence reliability down",
        ("same", "up"): "improves confidence reliability only",
        ("same", "same"): "no reliable change",
        ("same", "down"): "worse confidence reliability, accuracy unchanged",
        ("down", "up"): "MIXED: accuracy down, confidence reliability up",
        ("down", "same"): "worse accuracy",
        ("down", "down"): "worse accuracy and confidence reliability",
    }[(acc, rel)]


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def abs_acc_ci(res: R.Result, n_boot: int):
    r = M.summarise_predictions(res.scores, res.y_idx, R.DEFAULT_TEMPERATURE)
    return M.cluster_bootstrap_diff(lambda i: r["correct"][i].mean(), res.groups, n_boot, 0)


def choose_best(results: Dict[str, R.Result], complexity: Dict[str, int], n_boot: int) -> Tuple[str, Dict[str, tuple]]:
    """Best accuracy; among configurations NOT reliably worse than it, the simplest one."""
    acc = {k: float(np.mean(r.scores.argmin(axis=1) == r.y_idx)) for k, r in results.items()}
    top = max(acc, key=acc.get)
    cmp = {}
    for k in results:
        if k != top:
            cmp[k] = R.paired_comparison(results[top], results[k], n_boot)["delta_accuracy"]
    candidates = [top] + [k for k, t in cmp.items() if t[1] <= 0 <= t[2]]
    best = min(candidates, key=lambda k: (complexity.get(k, 0), -acc[k]))
    return best, cmp


# ================================================================== Experiment 1
def run_exp1(ds, protocols, args, out: Path):
    print("\n=== Experiment 1: landmark weighting ===")
    primary = D.primary_protocol(protocols)
    res: Dict[str, Dict[str, R.Result]] = {}
    for pname, folds in protocols.items():
        res[pname] = {}
        for name, prof in PROFILES.items():
            res[pname][name] = R.evaluate(lambda f, n=name, p=prof: R.Config(n, profile=p,
                                          complexity=0 if n in ("uniform", "repo_default") else 1), ds, folds, name)
        t0 = time.time()
        grid = (0.5, 1.0, 2.0) if not args.quick else (0.5, 1.0, 2.0)
        res[pname]["region_tuned"] = R.evaluate_tuned(ds, folds, "region_tuned", grid=grid,
                                                      ext_cap=40 if args.quick else 100)
        print(f"  protocol {pname}: done (tuning {time.time() - t0:.0f}s)")
    complexity = {n: (0 if n in ("uniform", "repo_default") else 2 if n == "region_tuned" else 1) for n in res[primary]}
    best, cmp_best = choose_best(res[primary], complexity, args.n_boot)

    summaries = {p: {n: r.summary() for n, r in rs.items()} for p, rs in res.items()}
    acc_ci = {p: {n: abs_acc_ci(r, args.n_boot) for n, r in rs.items()} for p, rs in res.items()}
    vs_uniform = {p: {n: R.paired_comparison(rs["uniform"], r, args.n_boot) for n, r in rs.items() if n != "uniform"}
                  for p, rs in res.items()}
    sweep = R.sweep_regions(ds, protocols[primary], SWEEP_VALUES)
    tuned_profiles = {p: [list(map(float, c)) for c in rs["region_tuned"].chosen_profiles] for p, rs in res.items()}

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(PROFILES) + ["region_tuned"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(protocols)
    for k, p in enumerate(protocols):
        pts = [acc_ci[p][n] for n in names]
        ax.bar(np.arange(len(names)) + k * width, [t[0] for t in pts], width, label={"A": "A random split", "B": "B imported \u2192 own", "C": "C combined, block CV"}[p],
               yerr=[[t[0] - t[1] for t in pts], [t[2] - t[0] for t in pts]], capsize=2)
    ax.set_xticks(np.arange(len(names)) + width * (len(protocols) - 1) / 2, names, rotation=20)
    lo = min(t[1] for p in acc_ci for t in acc_ci[p].values())
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    ax.set_ylabel("accuracy (95 % interval)")
    ax.set_title("Experiment 1: weighting strategies")
    ax.legend(title="protocol")
    fig.tight_layout(); fig.savefig(out / "fig_exp1_profiles.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for region, vals in sweep.items():
        ax.plot(list(vals), list(vals.values()), marker="o", label=region)
    ax.set_xlabel("weight of that region (others = 1)")
    ax.set_ylabel(f"accuracy, protocol {primary}")
    ax.set_title("Experiment 1: one region at a time")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_exp1_sweep.png", dpi=130); plt.close(fig)

    return dict(res=res, summaries=summaries, acc_ci=acc_ci, vs_uniform=vs_uniform, sweep=sweep, best=best,
                cmp_best=cmp_best, primary=primary, tuned_profiles=tuned_profiles)


# ================================================================== Experiment 2
def load_palm_labels(path: Path, synthetic: bool):
    if synthetic:
        from .synthetic import make_palm_side_dataset
        return make_palm_side_dataset(30, seed=0)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def run_exp2(args, out: Path):
    print("\n=== Experiment 2: palm-side identification ===")
    d = load_palm_labels(Path(args.palm_labels), args.synthetic)
    if d is None:
        print(f"  no labelled recordings at {args.palm_labels}: skipped (run record_palm_side.py)")
        return None
    aspect = float(d["aspect"])
    hands, present = PS.hands_from_landmarks(d["landmarks"], aspect)
    side, mp, true_hand, block = d["side"], d["mp_label"], d["true_hand"], d["block"]
    pose = d["pose"]
    n_hands = present.sum(axis=1)

    values = {
        "G1 label (MediaPipe Left/Right)": PS.g1_label(hands, present, mp),
        "G1b label, mapping swapped": PS.g1_label(hands, present, -mp),
        "G2 slot (left of picture = right hand)": PS.g2_slot(hands, present),
        "G3 3D thumb/palm-plane chirality": PS.g3_intrinsic(hands, present),
    }
    # G4: grouped cross-validation over recording blocks
    from sklearn.model_selection import GroupKFold
    v4 = np.zeros(present.shape, np.float32)
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(block)))).split(hands, groups=block):
        knn = PS.KNNSide(k=7).fit(hands[tr], present[tr], side[tr])
        v4[te] = knn.score(hands[te], present[te])
    values["G4 kNN on labelled recordings"] = v4

    delta = args.side_delta
    overall = {n: PS.evaluate_side(v, side, present, delta) for n, v in values.items()}
    subgroups: Dict[str, Dict[str, Dict[str, float]]] = {}
    masks = {
        "one hand in picture": (n_hands == 1)[:, None] & np.ones((1, 2), bool),
        "two hands in picture": (n_hands == 2)[:, None] & np.ones((1, 2), bool),
        "physical right hand (single-hand clips)": true_hand == 1,
        "physical left hand (single-hand clips)": true_hand == -1,
        "hand slot 0 (left of picture)": np.arange(2)[None, :] == 0,
        "hand slot 1 (right of picture)": np.arange(2)[None, :] == 1,
    }
    for style in np.unique(pose):
        masks[f"pose: {style}"] = (pose == style)[:, None] & np.ones((1, 2), bool)
    for gname, mk in masks.items():
        subgroups[gname] = {n: PS.evaluate_side(v, side, present, delta, mask=mk) for n, v in values.items()}
    chir = PS.chirality_accuracy(hands, present, mp, true_hand)

    deltas = np.linspace(0.0, 0.9, 19)
    curves = {}
    for n, v in values.items():
        cov, acc = [], []
        for dl in deltas:
            r = PS.evaluate_side(v, side, present, float(dl))
            cov.append(r["coverage"]); acc.append(r["accuracy_decided"])
        curves[n] = (cov, acc)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n, (cov, acc) in curves.items():
        ax.plot(cov, acc, marker=".", label=n)
    ax.set_xlabel("coverage (share of hands decided)")
    ax.set_ylabel("accuracy on decided hands")
    ax.set_title("Experiment 2: accuracy vs coverage")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_exp2_coverage.png", dpi=130); plt.close(fig)

    return dict(overall=overall, subgroups=subgroups, chirality=chir, delta=delta,
                n_frames=int(len(side)), n_hands=int(present.sum()), curves={k: v for k, v in curves.items()})


# ================================================================== Experiment 3
def run_exp3(ds, protocols, exp1, args, out: Path):
    print("\n=== Experiment 3: confidence with / without palm side ===")
    primary = D.primary_protocol(protocols)
    bases = {"current system (repo_default)": PROFILES["repo_default"]}
    best_name = exp1["best"] if exp1 else None
    if best_name and best_name in PROFILES and PROFILES[best_name] != PROFILES["repo_default"]:
        bases[f"best of Exp. 1 ({best_name})"] = PROFILES[best_name]
    elif best_name == "region_tuned":
        prof = tuple(np.mean(exp1["tuned_profiles"][primary], axis=0))
        bases["best of Exp. 1 (region_tuned, mean weights)"] = prof

    out_res: Dict[str, Dict[str, R.Result]] = {}
    for pname, folds in protocols.items():
        out_res[pname] = {}
        for bname, prof in bases.items():
            variants = {
                "without palm side": R.Config("none", prof),
                f"palm-side feature (w={SIDE_FEATURE_WEIGHT:g})": R.Config("feat", prof, side_weight=SIDE_FEATURE_WEIGHT, complexity=3),
                f"palm-side penalty (+{SIDE_PENALTY:g})": R.Config("pen", prof, side_penalty=SIDE_PENALTY, complexity=3),
            }
            for vname, cfg in variants.items():
                key = f"{bname}{SEP}{vname}"
                out_res[pname][key] = R.evaluate(lambda f, c=cfg: c, ds, folds, key)
    summaries = {p: {n: r.summary() for n, r in rs.items()} for p, rs in out_res.items()}

    # paired effect of palm side, per base and protocol
    effects = {}
    for pname, rs in out_res.items():
        effects[pname] = {}
        for bname in bases:
            ref = rs[f"{bname}{SEP}without palm side"]
            for vname in [k.split(SEP)[1] for k in rs if k.startswith(bname) and "without" not in k]:
                key = f"{bname}{SEP}{vname}"
                eff = R.paired_comparison(ref, rs[key], args.n_boot)
                s0, s1 = summaries[pname][ref.name], summaries[pname][key]
                eff["delta_ece_fitted_T"] = (s1["ece_fitted_T"] - s0["ece_fitted_T"],) * 3
                eff["delta_mean_conf_wrong"] = (s1["mean_conf_wrong"] - s0["mean_conf_wrong"],) * 3
                effects[pname][key] = eff

    # sensitivity on the primary protocol, first base only
    first_base = next(iter(bases.items()))
    sens = {"feature": {}, "penalty": {}}
    for w in SIDE_FEATURE_SWEEP:
        r = R.evaluate(lambda f, w=w: R.Config("s", first_base[1], side_weight=w), ds, protocols[primary])
        sens["feature"][w] = r.summary()
    for lam in SIDE_PENALTY_SWEEP:
        r = R.evaluate(lambda f, lam=lam: R.Config("s", first_base[1], side_penalty=lam), ds, protocols[primary])
        sens["penalty"][lam] = r.summary()
    sens["none"] = summaries[primary][f"{first_base[0]}{SEP}without palm side"]

    # how often is a side actually decided on the queries?
    test_idx = np.concatenate([f.test for f in protocols[primary]])
    sv = PS.side_values_for_templates(ds.X[test_idx])
    pres = PS.hands_from_features(ds.X[test_idx])[1]
    decided_rate = float(np.mean(np.abs(sv[pres]) >= 0.5)) if pres.any() else float("nan")

    # figures: reliability + risk-coverage on primary protocol, base 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = out_res[primary]
    keys = [k for k in rs if k.startswith(first_base[0])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k in keys:
        r = M.summarise_predictions(rs[k].scores, rs[k].y_idx, R.DEFAULT_TEMPERATURE)
        bins = M.reliability_bins(r["conf"], r["correct"])
        axes[0].plot([b[0] for b in bins], [b[1] for b in bins], marker="o", label=k.split(SEP)[1])
        rc = M.risk_coverage(r["conf"], r["correct"])
        axes[1].plot([c for c, _ in rc], [a for _, a in rc], marker=".", label=k.split(SEP)[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("mean confidence"); axes[0].set_ylabel("accuracy"); axes[0].set_title("Reliability (default temperature)")
    axes[1].set_xlabel("coverage (most confident answers first)"); axes[1].set_ylabel("accuracy"); axes[1].set_title("Risk-coverage")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Experiment 3, protocol {primary}, base: {first_base[0]}")
    fig.tight_layout(); fig.savefig(out / "fig_exp3_confidence.png", dpi=130); plt.close(fig)

    return dict(res=out_res, summaries=summaries, effects=effects, sens=sens, bases=list(bases), primary=primary,
                decided_rate=decided_rate)


# ================================================================== report
STRENGTHS = """\
**Design analysis (reasoned from how each method works; the tables above are what was measured).**

| Approach | Strengths | Limitations |
|---|---|---|
| Uniform weighting | No assumptions; nothing to tune. Easy to explain. | Joints count twice as much as fingertips (10 vs 5 landmarks); noisy z values count as much as x and y. |
| Region-based weighting (fixed profiles) | Encodes domain knowledge (fingertips and the hand arrangement decide most letters). Free at run time. | The right weights depend on the data. A profile that helps one signer or camera can hurt another. |
| Region-based weighting (tuned) | Adapts to your own recordings. | Needs your own recordings; with few samples it can over-fit the grid. Uses only validation folds inside the training data here. |
| Palm side, G1 (MediaPipe label) | Simple and fast. Needs no extra data. | The label is wrong on some frames, especially for overlapping hands or odd angles, and its Left/Right meaning depends on whether the picture is mirrored. Not available for the imported templates. |
| Palm side, G2 (slot rule) | Needs no label. Works for templates too. | Assumes the hand on the left of the picture is the right hand. Fails for crossed hands, one-hand frames of the left hand and swapped signing roles. |
| Palm side, G3 (3D chirality) | Needs no label and no assumption about where the hands are. | Depends on MediaPipe's z values, which are the least accurate coordinate. A thumb lying in the palm plane gives no clear answer. |
| Palm side, G4 (kNN) | Learns what your camera really produces. | Needs labelled recordings of both sides and every hand role. Cannot beat its labels. |
| Palm-side feature / penalty in the matcher | One extra signal that separates mirror-like signs. | The side sign is already implicit in the wrist-relative x and y values of the 132-value feature, so the information is partly redundant. A wrong side estimate adds a wrong penalty. Edge-on hands give no usable side. |
"""


def build_report(ds, protocols, e1, e2, e3, args, out: Path, runtime: float) -> str:
    L: List[str] = []
    L.append("# Template-matching improvements: comparative report\n")
    if ds.synthetic:
        L.append("> **SYNTHETIC DATA. PIPELINE CHECK ONLY.** The numbers below come from a computer-generated hand model, "
                 "not from real signers. They prove that the code runs, not that any method is better. "
                 "Re-run without `--synthetic` on your own data.\n")
    L.append(f"*Generated by `python -m experiments.run_experiments`. Data: {ds.description}. "
             f"Bootstrap rounds: {args.n_boot}. Run time: {runtime:.0f} s.*\n")
    L.append("## 0. How to read this report\n")
    L.append("Protocols (see `experiments/data.py`):\n")
    for p in protocols:
        n_q = sum(len(f.test) for f in protocols[p])
        L.append(f"- **{D.PROTOCOL_TITLES[p]}**: {n_q} test queries")
    primary = D.primary_protocol(protocols)
    L.append(f"\nThe **primary protocol is {primary}**; configurations are ranked on it. Intervals are 95 % cluster-bootstrap "
             "intervals (whole letter-blocks are resampled, because neighbouring frames are near-identical). "
             "\"Better / worse\" is only claimed when the interval for the paired difference excludes 0.\n")
    if "A" in protocols and primary != "A":
        L.append("Protocol A is shown for comparison with `evaluate.py` but is optimistic: its test frames have near-identical twins in the templates.\n")
    missing = {"A": "imported templates (data/templates_external.npz)", "B": "imported templates AND your own recordings",
               "C": "your own recordings (data/templates_user.npz)"}
    not_run = [p for p in "ABC" if p not in protocols]
    if not_run:
        L.append("**Protocols that could not run (missing data):** " + "; ".join(f"{p} needs {missing[p]}" for p in not_run) +
                 ". Import the dataset (`import_dataset.py`) and/or record letters (`calibrate.py`) for the full comparison.\n")

    # ---------------- Exp 1
    if e1:
        L.append("## 1. Landmark weighting strategies\n")
        L.append("Weights multiply the squared coordinate differences. Regions: fingertips (4, 8, 12, 16, 20), finger joints "
                 "(2, 3, 6, 7, ...), palm and wrist (0, 1, 5, 9, 13, 17), and the inter-hand features (wrist-to-wrist vector). "
                 "Profiles are written (fingertips, joints, palm, inter-hand):\n")
        for n, pr in PROFILES.items():
            L.append(f"- `{n}`: ({', '.join(f'{v:.2f}' for v in pr)})")
        L.append("- `region_tuned`: best of a 3x3x3x3 grid over {0.5, 1, 2}, chosen on validation folds inside each training set\n")
        L.append("Note: scaling all weights up or down does not change which letter wins, but it does change the distances and therefore "
                 "the confidence (the temperature is fixed at 0.12). Compare strategies by accuracy and AUROC; \"mean conf.\" alone says little.\n")
        for p in protocols:
            L.append(f"\n**Protocol {p}**\n")
            rows = []
            for n in list(PROFILES) + ["region_tuned"]:
                s = e1["summaries"][p][n]
                ci = e1["acc_ci"][p][n]
                vs = "" if n == "uniform" else verdict(e1["vs_uniform"][p][n]["delta_accuracy"])
                dd = "" if n == "uniform" else fmt_ci(e1["vs_uniform"][p][n]["delta_accuracy"])
                rows.append([n, f"{fmt(ci[0])} [{fmt(ci[1])}, {fmt(ci[2])}]", fmt(s["macro_f1"]), fmt(s["top3_accuracy"]),
                             fmt(s["mean_confidence"]), fmt(s["auroc"]), dd, vs])
            L.append(md_table(["strategy", "accuracy [95 %]", "macro-F1", "top-3", "mean conf.", "AUROC", "Δ accuracy vs uniform", "verdict"], rows))
        L.append(f"\nTuned weights per fold (protocol {primary}): " +
                 "; ".join("(" + ", ".join(f"{v:g}" for v in w) + ")" for w in e1["tuned_profiles"][primary]) + "\n")
        L.append(f"\n**One region at a time** (protocol {primary}; the other regions stay at 1; weight 0 switches the region off):\n")
        vals = list(next(iter(e1["sweep"].values())))
        L.append(md_table(["region"] + [f"w={v:g}" for v in vals],
                          [[r] + [fmt(a) for a in e1["sweep"][r].values()] for r in e1["sweep"]]))
        L.append("\n![weighting](fig_exp1_profiles.png)\n\n![sweep](fig_exp1_sweep.png)\n")
        top_region = max(e1["sweep"], key=lambda r: max(e1["sweep"][r].values()) - min(e1["sweep"][r].values()))
        L.append(f"Most sensitive region in the sweep: **{top_region}** (accuracy range "
                 f"{fmt(min(e1['sweep'][top_region].values()))} to {fmt(max(e1['sweep'][top_region].values()))}).\n")
        L.append(f"**Result:** the best strategy on protocol {primary} under the rule \"highest accuracy, then the simplest one that is not reliably worse\" is **`{e1['best']}`**.\n")

    # ---------------- Exp 2
    L.append("## 2. Palm-side identification\n")
    if not e2:
        L.append("Skipped: no labelled recordings found. Run `python record_palm_side.py` (about 5 minutes) and re-run this script.\n")
    else:
        L.append(f"{e2['n_frames']} labelled frames, {e2['n_hands']} hands. A hand counts as *decided* when |value| ≥ {e2['delta']:g}; "
                 "an undecided hand counts as wrong in `accuracy (all)` and is excluded from `accuracy (decided)`.\n")
        rows = []
        for n, r in e2["overall"].items():
            rows.append([n, fmt(r["accuracy_all"]), fmt(r["accuracy_decided"]), fmt(r["coverage"]),
                         f"{fmt(r['recall_palm'])} / {fmt(r['recall_back'])}", f"{fmt(r['precision_palm'])} / {fmt(r['precision_back'])}"])
        L.append(md_table(["method", "accuracy (all)", "accuracy (decided)", "coverage", "recall palm / back", "precision palm / back"], rows))
        if e2["chirality"]:
            c = e2["chirality"]
            L.append(f"\n**Where does the chirality (left- vs right-shaped hand) come from?** On {c['n']} single-hand frames with a known physical hand: "
                     f"MediaPipe label (documented mapping) right {fmt(c['label_correct'])}, label with swapped mapping right {fmt(c['label_correct_if_swapped'])}, "
                     f"slot rule right {fmt(c['slot_correct'])}, 3D estimate right {fmt(c['intrinsic_correct'])}.")
            if c["label_correct_if_swapped"] > c["label_correct"] + 0.2:
                L.append("**The label mapping looks swapped for your MediaPipe version/pipeline.** Use row G1b, or flip the sign in `palm_side.g1_label`.")
        L.append("\n**Breakdown by situation (accuracy (all)):**\n")
        names = list(e2["overall"])
        L.append(md_table(["situation"] + [n.split(" ")[0] for n in names],
                          [[g] + [fmt(e2["subgroups"][g][n].get("accuracy_all", float("nan"))) for n in names] for g in e2["subgroups"]]))
        L.append("\n![coverage](fig_exp2_coverage.png)\n")
        L.append("Confusion matrices (rows true palm / back, columns predicted palm / back / unsure):\n")
        for n, r in e2["overall"].items():
            L.append(f"- {n}: {r['confusion']}")
        best2 = max((n for n in e2["overall"] if not n.startswith("G1b")), key=lambda n: e2["overall"][n]["accuracy_all"])
        L.append(f"\n**Result:** highest overall accuracy: **{best2}** ({fmt(e2['overall'][best2]['accuracy_all'])}). "
                 "Check the breakdown before choosing: a method that is good on average can still fail on one hand role.\n")

    # ---------------- Exp 3
    if e3:
        L.append("## 3. Does knowing the palm side change the confidence?\n")
        L.append(f"Palm-side values for templates and queries come from the slot rule (G2), because the template files carry no MediaPipe label. "
                 f"Two ways to use them: a **feature** (two extra columns, weight {SIDE_FEATURE_WEIGHT:g}) and a **penalty** (+{SIDE_PENALTY:g} distance per hand whose decided side disagrees). "
                 f"Settings were fixed in advance; a sensitivity sweep follows. On the test queries a side was decided for {fmt(e3['decided_rate'])} of the hands.\n")
        L.append("The raw confidence depends on the distance scale, so a higher number alone does not mean better. "
                 "**AUROC** (does confidence rank right answers above wrong ones) and **ECE after fitting one temperature** are scale-free.\n")
        for p in protocols:
            L.append(f"\n**Protocol {p}**\n")
            rows = []
            for k, s in e3["summaries"][p].items():
                eff = e3["effects"][p].get(k)
                rows.append([k, fmt(s["accuracy"]), fmt(s["mean_confidence"]), fmt(s["mean_conf_correct"]), fmt(s["mean_conf_wrong"]),
                             fmt(s["auroc"]), fmt(s["ece_default_T"]), fmt(s["ece_fitted_T"]),
                             f"{fmt(s['coverage_at_app_threshold'])} / {fmt(s['precision_at_app_threshold'])}",
                             "" if eff is None else fmt_ci(eff["delta_accuracy"]),
                             "" if eff is None else fmt_ci(eff["delta_auroc"], 3)])
            L.append(md_table(["configuration", "acc.", "mean conf.", "conf. (right)", "conf. (wrong)", "AUROC", "ECE", "ECE fitted T",
                               "accepted at 0.80 / precision", "Δ acc. vs without", "Δ AUROC vs without"], rows))
        L.append("\n![confidence](fig_exp3_confidence.png)\n")
        L.append(f"\n**Paired effect of palm-side identification (protocol {e3['primary']}):**\n")
        rows = []
        for k, eff in e3["effects"][e3["primary"]].items():
            rows.append([k, fmt_ci(eff["delta_accuracy"]), verdict(eff["delta_accuracy"]), fmt_ci(eff["delta_mean_confidence"]),
                         fmt_ci(eff["delta_auroc"]), verdict(eff["delta_auroc"]), f"{eff['mcnemar_p'][0]:.3f}"])
        L.append(md_table(["configuration", "Δ accuracy", "accuracy verdict", "Δ mean confidence", "Δ AUROC", "reliability verdict", "McNemar p (optimistic)"], rows))
        L.append("\n**Sensitivity of the settings (protocol " + e3["primary"] + ", first base):**\n")
        rows = [["without", "-", fmt(e3["sens"]["none"]["accuracy"]), fmt(e3["sens"]["none"]["mean_confidence"]), fmt(e3["sens"]["none"]["auroc"]), fmt(e3["sens"]["none"]["ece_fitted_T"])]]
        for kind in ("feature", "penalty"):
            for v, s in e3["sens"][kind].items():
                rows.append([f"side {kind}", f"{v:g}", fmt(s["accuracy"]), fmt(s["mean_confidence"]), fmt(s["auroc"]), fmt(s["ece_fitted_T"])])
        L.append(md_table(["variant", "setting", "accuracy", "mean conf.", "AUROC", "ECE fitted T"], rows))
        L.append("")
        L.append("**Classification of each configuration (protocol " + e3["primary"] + "):**\n")
        for k, e in e3["effects"][e3["primary"]].items():
            L.append(f"- {k}: {classify_effect(e)}")
        labels = [classify_effect(e) for e in e3["effects"][e3["primary"]].values()]
        n_good = sum(1 for l in labels if l.startswith("improves") and "reliability down" not in l)
        L.append(f"\n**Result:** {n_good} of {len(labels)} palm-side configurations improved on the system without palm side "
                 "without hurting the confidence ranking. Mixed results (higher accuracy but a worse confidence ranking) "
                 "mean the extra signal fixes some errors but makes the confidence less trustworthy; do not adopt such a configuration before re-fitting the temperature.")
        L.append("")

    # ---------------- Exp 4
    L.append("## 4. Final performance comparison\n")
    rows = []
    allres: Dict[str, Tuple[str, dict, tuple]] = {}
    if e1:
        for n in list(PROFILES) + ["region_tuned"]:
            allres[f"weighting: {n}"] = (n, e1["summaries"][primary][n], e1["acc_ci"][primary][n])
    if e3:
        for k, s in e3["summaries"][primary].items():
            allres[f"palm side: {k}"] = (k, s, None)
    for k, (_, s, ci) in allres.items():
        rows.append([k, fmt(s["accuracy"]), fmt(s["macro_f1"]), fmt(s["top3_accuracy"]), fmt(s["mean_confidence"]), fmt(s["auroc"]),
                     fmt(s["ece_fitted_T"]), f"{fmt(s['coverage_at_app_threshold'])} / {fmt(s['precision_at_app_threshold'])}",
                     fmt(s["median_best_distance"], 2), fmt(s["ms_per_query"], 3)])
    L.append(f"All configurations on protocol **{primary}**:\n")
    L.append(md_table(["configuration", "accuracy", "macro-F1", "top-3", "mean conf.", "AUROC", "ECE fitted T",
                       "accepted at 0.80 / precision", "median best dist.", "ms / query (batch)"], rows))
    L.append("")
    if e1:
        L.append(f"**Best weighting configuration:** `{e1['best']}` (rule: highest accuracy; among configurations not reliably worse, the simplest).")
        close = [k for k, t in e1["cmp_best"].items() if t[1] <= 0 <= t[2]]
        if close:
            L.append(f"Not reliably different from the top accuracy: {', '.join(close)}.")
    if e3:
        eff = e3["effects"][e3["primary"]]
        good = [k for k, e in eff.items() if classify_effect(e) in ("improves accuracy and confidence reliability",
                                                                    "improves accuracy, reliability unchanged",
                                                                    "improves confidence reliability only")]
        mixed = [k for k, e in eff.items() if classify_effect(e).startswith("MIXED")]
        if good:
            L.append("**Palm-side identification:** reliably better, without hurting confidence reliability: " + "; ".join(good) + ".")
        elif mixed:
            L.append("**Palm-side identification:** mixed. It changes accuracy and confidence in opposite directions for: " + "; ".join(mixed) +
                     ". Not recommended for the matcher until the temperature is re-fitted and the effect is confirmed on protocol B.")
        else:
            L.append("**Palm-side identification:** no reliable gain on this data, so not recommended for the matcher. "
                     "It can still be useful as a separate on-screen hint.")
    L.append("\n### Strengths and limitations\n")
    L.append(STRENGTHS)
    L.append("### Recommended improvements\n")
    recs = [
        "Use the best configuration above in `matcher.py` only if its advantage holds on protocol B or C (data from a person who is not in the templates). A gain that appears only on protocol A is not a gain.",
        "Record more of your own letters (`calibrate.py --append`) in more than one session. Domain shift between the imported data and a new signer is usually the largest error source; weighting and palm side are second-order effects.",
        "Fit one temperature per final configuration (`fitted_T` in `results.json`) so the 0.80 threshold means what it says. Changing weights changes the distance scale and therefore the confidence.",
        "If palm side is used, take the chirality from a source that Experiment 2 shows to be reliable for your camera, and leave edge-on hands undecided instead of forcing a side.",
        "Collect labelled palm/back recordings with `record_palm_side.py` in several lighting conditions before trusting a palm-side number.",
    ]
    L += [f"{i + 1}. {r}" for i, r in enumerate(recs)]
    L.append("\n### Threats to validity\n")
    L.append("- Test frames in protocols B and C come from one person and one camera. Intervals reflect frame sampling, not new signers.\n"
             "- Weights tuned on one protocol are only validated on folds that were not used for tuning, but the profile list was chosen by the author. Several profiles are compared, so one \"significant\" result in many is expected by chance; prefer effects that appear in more than one protocol.\n"
             "- Template files carry no raw landmarks, so Experiment 3 uses the slot rule for the palm side. Experiment 2 compares the other sources on separate labelled recordings.\n")
    return "\n".join(L)


# ================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--synthetic", action="store_true", help="use generated hands (pipeline check only)")
    ap.add_argument("--palm-labels", default=str(PALM_LABELS_DEFAULT))
    ap.add_argument("--blocks", type=int, default=5, help="blocks per letter for protocol C")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--side-delta", type=float, default=0.15, help="|value| needed to call a side (Experiment 2)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.quick:
        args.n_boot = min(args.n_boot, 200)

    out = Path(args.out) if args.out else (OUT_DEFAULT.parent / "experiments_synthetic" if args.synthetic else OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.synthetic:
        from .synthetic import make_letter_dataset
        Xe, ye, Xu, yu = make_letter_dataset(per_letter_ext=60, per_letter_user=40, seed=0, hard=2.5)
        ds = D.from_arrays(Xe, ye, Xu, yu, args.blocks, description="SYNTHETIC hands (not real data)", synthetic=True)
    else:
        ds = D.load_real(args.blocks)
    protocols = D.all_protocols(ds)
    print("Data:", ds.description, "| protocols:", list(protocols))

    e1 = run_exp1(ds, protocols, args, out) if 1 in args.exp else None
    e2 = run_exp2(args, out) if 2 in args.exp else None
    e3 = run_exp3(ds, protocols, e1, args, out) if 3 in args.exp else None

    report = build_report(ds, protocols, e1, e2, e3, args, out, time.time() - t0)
    (out / "final_report.md").write_text(report)

    def jsonable(o):
        if isinstance(o, dict):
            return {str(k): jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, R.Result):
            return None
        return o

    dump = {"data": ds.description, "synthetic": ds.synthetic,
            "exp1": jsonable({k: v for k, v in (e1 or {}).items() if k != "res"}),
            "exp2": jsonable({k: v for k, v in (e2 or {}).items() if k != "curves"}),
            "exp3": jsonable({k: v for k, v in (e3 or {}).items() if k != "res"})}
    (out / "results.json").write_text(json.dumps(dump, indent=2))
    print(f"\nReport: {out / 'final_report.md'}\nNumbers: {out / 'results.json'}")


if __name__ == "__main__":
    main()
