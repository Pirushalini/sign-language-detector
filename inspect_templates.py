from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("template_analysis")

EXTERNAL_FILE = DATA_DIR / "templates_external.npz"
USER_FILE = DATA_DIR / "templates_user.npz"

FEATURE_LEN = 132
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3

# Standard MediaPipe hand skeleton connections.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --------------------------------------------------
# Load templates
# --------------------------------------------------

def load_template_file(path):
    if not path.exists():
        print(f"File not found: {path}")
        return None

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)

    if X.ndim != 2 or X.shape[1] != FEATURE_LEN:
        raise ValueError(
            f"{path} has shape {X.shape}. "
            f"Expected (number_of_samples, {FEATURE_LEN})."
        )

    return X, y


external_data = load_template_file(EXTERNAL_FILE)
user_data = load_template_file(USER_FILE)

if external_data is None:
    raise SystemExit(
        "External templates are missing. Import the dataset first."
    )

if user_data is None:
    raise SystemExit(
        "User templates are missing. Run calibration first."
    )

X_external, y_external = external_data
X_user, y_user = user_data

letters = sorted(set(y_external) | set(y_user))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "landmark_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print(f"External templates: {len(X_external)}")
print(f"User templates:     {len(X_user)}")
print(f"Letters found:      {letters}")


# --------------------------------------------------
# Feature helpers
# --------------------------------------------------

def hand_landmarks(X, hand_index):
    """
    Extract the 21 x 3 landmark coordinates for one hand.

    hand_index = 0: first hand, features 0:63
    hand_index = 1: second hand, features 63:126
    """
    start = hand_index * 63
    end = start + 63

    return X[:, start:end].reshape(-1, 21, 3)


def hand_present(X, hand_index):
    """
    Hand-presence flags are features 130 and 131.
    """
    flag_index = 130 + hand_index
    return X[:, flag_index] > 0.5


def weighted_distance_features(X):
    """
    Apply the same feature-group weights used by the matcher.

    This helper does NOT apply the matcher's personal-template
    distance discount. That discount depends on template source.
    """
    X = X.copy()

    # Wrist relationship features: 126:130
    X[:, 126:130] *= np.sqrt(2.0)

    # Hand-presence flags: 130:132
    X[:, 130:132] *= np.sqrt(3.0)

    return X


# --------------------------------------------------
# Analyse every letter
# --------------------------------------------------

rows = []

for letter in letters:

    ext = X_external[y_external == letter]
    usr = X_user[y_user == letter]

    row = {
        "letter": letter,
        "external_count": len(ext),
        "user_count": len(usr),
        "mean_feature_difference": np.nan,
        "mean_absolute_feature_difference": np.nan,
        "user_to_nearest_external_mean": np.nan,
        "user_to_nearest_external_median": np.nan,
    }

    if len(ext) > 0 and len(usr) > 0:

        # Difference between the average feature vectors.
        ext_mean = ext.mean(axis=0)
        usr_mean = usr.mean(axis=0)

        row["mean_feature_difference"] = float(
            np.linalg.norm(ext_mean - usr_mean)
        )

        row["mean_absolute_feature_difference"] = float(
            np.mean(np.abs(ext_mean - usr_mean))
        )

        # Compare every user template with the closest external
        # template belonging to the SAME letter.
        ext_weighted = weighted_distance_features(ext)
        usr_weighted = weighted_distance_features(usr)

        # Pairwise weighted Euclidean distances.
        differences = (
            usr_weighted[:, None, :]
            - ext_weighted[None, :, :]
        )

        distances = np.sqrt(
            np.sum(differences ** 2, axis=2)
        )

        nearest_distances = distances.min(axis=1)

        row["user_to_nearest_external_mean"] = float(
            nearest_distances.mean()
        )

        row["user_to_nearest_external_median"] = float(
            np.median(nearest_distances)
        )

    rows.append(row)

# ----------------------------------------------
    # Plot average landmark positions
    # ----------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Define distinct colors for the two template sources
    source_colors = {
        "External": "dodgerblue",  # Color for external templates
        "User": "crimson"          # Color for user templates
    }

    for hand_index, ax in enumerate(axes):

        plotted_any = False

        for X, y, source_name in [
            (X_external, y_external, "External"),
            (X_user, y_user, "User"),
        ]:

            class_samples = X[y == letter]

            if len(class_samples) == 0:
                continue

            # Only average samples where this hand is present.
            present = hand_present(class_samples, hand_index)

            if not np.any(present):
                continue

            landmarks = hand_landmarks(
                class_samples[present],
                hand_index
            )

            mean_landmarks = landmarks.mean(axis=0)
            color = source_colors[source_name]

            ax.scatter(
                mean_landmarks[:, 0],
                mean_landmarks[:, 1],
                label=source_name,
                color=color,
                s=35
            )

            for start, end in CONNECTIONS:
                ax.plot(
                    [
                        mean_landmarks[start, 0],
                        mean_landmarks[end, 0]
                    ],
                    [
                        mean_landmarks[start, 1],
                        mean_landmarks[end, 1]
                    ],
                    color=color,
                    linewidth=1.5
                )

            # Label the 21 landmark indices.
            for i, (x, y_coord, z) in enumerate(mean_landmarks):
                ax.annotate(
                    str(i),
                    (x, y_coord),
                    fontsize=7
                )

            plotted_any = True

        ax.set_title(
            f"Letter {letter}: "
            f"{'First' if hand_index == 0 else 'Second'} hand"
        )

        ax.set_xlabel("Normalised relative X")
        ax.set_ylabel("Normalised relative Y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)

        if plotted_any:
            # Avoid duplicate legend entries if both sources are plotted
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
        else:
            ax.text(
                0.5, 0.5,
                "No samples with this hand",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

        # Image coordinates usually increase downward.
        ax.invert_yaxis()

    fig.suptitle(
        f"Letter {letter}: external vs personal templates"
    )

    fig.tight_layout()

    safe_letter = "".join(
        c for c in letter if c.isalnum()
    )

    fig.savefig(
        PLOT_DIR / f"letter_{safe_letter}.png",
        dpi=160
    )

    plt.close(fig)


# --------------------------------------------------
# Save report
# --------------------------------------------------

report = pd.DataFrame(rows)

report.to_csv(
    OUTPUT_DIR / "template_comparison.csv",
    index=False
)

print("\nTemplate comparison report:")
print(
    report.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print(f"\nCSV saved to: {OUTPUT_DIR / 'template_comparison.csv'}")
print(f"Plots saved to: {PLOT_DIR.resolve()}")