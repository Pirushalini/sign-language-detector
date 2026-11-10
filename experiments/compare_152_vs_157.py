"""READ-ONLY exploratory comparison of 132D, 152D and 157D BSL static features.

Run from the sign-language-detector repo root, alongside geometric_features.py:

python compare_152_vs_157.py \
    --user data/templates_user_eio_validation.npz \
    --test-session eio_validation_20261110_130806

IMPORTANT:
- No changes to .npz files, app.py or motion.py.
- Train-only scaling; no samples from the held-out session used in training.
- Contact features reconstructed from 132-D stored vectors are APPROXIMATE.
- The provided test session has already guided development: results are exploratory,
  not unbiased final performance. Record a separate untouched session for final test.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geometric_features import features_152, augmented_features_157, contact_features

EXCLUDED = {"H", "J", "NONE"}


def read_npz(path: Path, require_session: bool = False):
    if not path.is_file():
        raise SystemExit(f"File does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "X" not in data.files or "y" not in data.files:
            raise SystemExit(f"{path}: expected X and y arrays")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"]).astype(str)
        if require_session and "session" not in data.files:
            raise SystemExit(f"{path}: no session field, cannot hold out a session")
        sessions = (np.asarray(data["session"]).astype(str)
                    if "session" in data.files else None)
    if X.ndim != 2 or X.shape[1] != 132 or len(X) != len(y):
        raise SystemExit(f"{path}: expected X=(N,132) with matching y, got {X.shape}")
    if sessions is not None and len(sessions) != len(y):
        raise SystemExit(f"{path}: session field does not match sample count")
    if not np.all(np.isfinite(X)):
        raise SystemExit(f"{path}: nonfinite features")
    keep = ~np.isin(y, list(EXCLUDED))
    return X[keep], y[keep], sessions[keep] if sessions is not None else None


def predict(Xtrain, ytrain, Xtest, weights=None):
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, class_weight="balanced"))
    fit_args = ({"svc__sample_weight": weights} if weights is not None else {})
    model.fit(Xtrain, ytrain, **fit_args)
    return model.predict(Xtest)


def show_result(title, truth, pred):
    letters = sorted(set(truth))
    accuracy = accuracy_score(truth, pred)
    f1 = f1_score(truth, pred, labels=letters, average="macro", zero_division=0)
    print(f"{title:<32} {accuracy:8.2%} {f1:9.4f}")
    print("  Letter  correct/total  most common mistakes")
    for l in letters:
        rows = pred[truth == l]
        n_correct = int((rows == l).sum())
        wrong = Counter(rows[rows != l]).most_common(3)
        errors = ", ".join(f"{label}:{cnt}" for label, cnt in wrong) if wrong else "none"
        print(f"  {l:>6}  {n_correct:>3}/{len(rows):<3}       {errors}")
    return accuracy, f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--external", type=Path, default=Path("data/templates_external.npz"))
    ap.add_argument("--user", type=Path, default=Path("data/templates_user_eio_validation.npz"))
    ap.add_argument("--test-session", required=True,
                    help="Exactly one user recording session to hold out")
    ap.add_argument("--personal-weight", type=float, default=1.0,
                    help="Optional personal training sample weight; 1 means unweighted")
    args = ap.parse_args()
    if args.personal_weight <= 0:
        raise SystemExit("--personal-weight must be positive")

    Xe, ye, _ = read_npz(args.external)
    Xu, yu, su = read_npz(args.user, require_session=True)
    valid_sessions = sorted(set(su))
    if args.test_session not in valid_sessions:
        raise SystemExit(f"Unknown test session: {args.test_session}\nAvailable: {valid_sessions}")
    # Evaluate classes present in external training, leaving unmatched classes out of the comparison.
    shared_letters = sorted(set(ye) & set(yu))
    Xe, ye = Xe[np.isin(ye, shared_letters)], ye[np.isin(ye, shared_letters)]
    keep_u = np.isin(yu, shared_letters)
    Xu, yu, su = Xu[keep_u], yu[keep_u], su[keep_u]

    test_mask = su == args.test_session
    Xtest, ytest = Xu[test_mask], yu[test_mask]
    Xpersonal, ypersonal = Xu[~test_mask], yu[~test_mask]
    if len(Xtest) == 0 or len(Xpersonal) == 0:
        raise SystemExit("Need separate personal training and test-session samples")
    if len(set(ye) | set(ypersonal)) < 2:
        raise SystemExit("At least two training letters are needed")
    if not set(ytest).issubset(set(ye) | set(ypersonal)):
        raise SystemExit("Test contains letters missing from training")

    print("READ-ONLY, EXPLORATORY experiment (not a final independent test)")
    print("External:", args.external, "User:", args.user)
    print("Test session:", args.test_session)
    print(f"External training: {len(Xe)}, earlier personal training: {len(Xpersonal)}, "
          f"held-out test: {len(Xtest)}")
    print("Test letters:", ", ".join(sorted(set(ytest))))
    print("No held-out session samples in model training: True")
    print("NOTE: contact features are approximate due to missing per-hand scale ratios")

    Xtrain = np.concatenate((Xe, Xpersonal), axis=0)
    ytrain = np.concatenate((ye, ypersonal), axis=0)
    w = (np.concatenate((np.ones(len(ye)),
                         np.full(len(ypersonal), args.personal_weight)))
         if args.personal_weight != 1.0 else None)
    # Identical train/test rows for every model, train-only StandardScaler fit.
    experiments = [
        ("Original 132-D", Xtrain, Xtest),
        ("Distance + angles 152-D", features_152(Xtrain), features_152(Xtest)),
        ("152-D + contacts 157-D", augmented_features_157(Xtrain), augmented_features_157(Xtest)),
    ]
    print(f"\n{'MODEL':<32} {'Accuracy':>8} {'Macro F1':>9}")
    results = {}
    for label, Xtr, Xte in experiments:
        pred = predict(Xtr, ytrain, Xte, w)
        results[label] = show_result(label, ytest, pred)

    contacts = contact_features(Xtest)
    present = (Xtest[:, 130] > 0.5) & (Xtest[:, 131] > 0.5)
    print("\nCONTACT FEATURE CHECK")
    print(f"Two-hand test examples: {int(present.sum())}/{len(Xtest)}")
    if present.any():
        print("Mean approximate tip distances (thumb,index,middle,ring,pinky):",
              np.round(contacts[present].mean(axis=0), 3).tolist())
        print("Std across test examples:",
              np.round(contacts[present].std(axis=0), 3).tolist())

    diff = 100 * (results["152-D + contacts 157-D"][0]
                  - results["Distance + angles 152-D"][0])
    print(f"\n157D minus 152D accuracy: {diff:+.2f} percentage points")
    print("Use a NEW untouched session/person later for final accuracy assessment.")
    print("No files modified. H/J motion recognition is not affected.")


if __name__ == "__main__":
    main()
