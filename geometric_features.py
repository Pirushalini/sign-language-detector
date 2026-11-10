
"""Geometric feature extraction for BSL static recognition.

Original: 132D landmarks
152D: 132D + 10 tip-palm distances + 10 joint angles
157D: 152D + 5 approximate cross-hand contact distances

IMPORTANT:
The original 132D format normalizes each hand separately.
Exact cross-hand distances cannot be recovered from it because
the original individual hand scales are not stored.

The contact features below use the inter-wrist vector and assume
that both hand scales are similar. They are approximate features
for offline experiments, not exact physical contact measurements.
"""

import numpy as np

TIPS = (4, 8, 12, 16, 20)
PIP = (3, 6, 10, 14, 18)
MCP = (2, 5, 9, 13, 17)

DISTANCE_INDICES = list(range(0, 5)) + list(range(13, 18))
ANGLE_INDICES = list(range(5, 10)) + list(range(18, 23))


def _as_feature_matrix(X):
    A = np.asarray(X, dtype=np.float32)
    single = A.ndim == 1

    if single:
        A = A.reshape(1, -1)

    if A.ndim != 2 or A.shape[1] != 132:
        raise ValueError(
            f"Expected (132,) or (N,132), got {A.shape}"
        )

    if not np.all(np.isfinite(A)):
        raise ValueError("Features contain NaN or Inf")

    return A, single


# --------------------------------------------------
# 1. Finger joint angle
# --------------------------------------------------

def _angle(a, b, c):
    ab = a - b
    cb = c - b

    denominator = (
        np.linalg.norm(ab) * np.linalg.norm(cb)
    )

    value = np.dot(ab, cb) / max(
        float(denominator), 1e-8
    )

    return float(
        np.arccos(np.clip(value, -1.0, 1.0)) / np.pi
    )


# --------------------------------------------------
# 2. Original geometric features for one hand
# --------------------------------------------------

def _one_hand(h):
    palm = np.mean(
        h[[0, 5, 9, 13, 17]], axis=0
    )

    # Five fingertip-to-palm distances
    tip_palm = np.linalg.norm(
        h[list(TIPS)] - palm, axis=1
    )

    # Five finger bend angles
    bends = np.asarray([
        _angle(h[m], h[p], h[t])
        for m, p, t in zip(MCP, PIP, TIPS)
    ], dtype=np.float32)

    # Palm orientation
    u = h[17] - h[5]
    v = h[9] - h[0]

    normal = np.cross(u, v)
    normal = normal / max(
        float(np.linalg.norm(normal)), 1e-8
    )

    return np.concatenate(
        (tip_palm, bends, normal)
    ).astype(np.float32)


# --------------------------------------------------
# 3. Existing 26D geometric features
# --------------------------------------------------

def geometric_features(X):
    A, single = _as_feature_matrix(X)

    output = np.zeros(
        (len(A), 26), dtype=np.float32
    )

    for i, row in enumerate(A):
        for hand_idx in range(2):

            if row[130 + hand_idx] > 0.5:

                start = hand_idx * 63
                end = start + 63

                h = row[start:end].reshape(21, 3)

                output[
                    i,
                    hand_idx * 13:(hand_idx + 1) * 13
                ] = _one_hand(h)

    return output[0] if single else output


# --------------------------------------------------
# 4. Existing 158D features (backward compatible)
# --------------------------------------------------

def augmented_features(X):
    A, single = _as_feature_matrix(X)

    result = np.concatenate(
        [A, geometric_features(A)], axis=1
    ).astype(np.float32)

    return result[0] if single else result


# --------------------------------------------------
# 5. Selected 152D features (without orientation)
# --------------------------------------------------

def features_152(X):
    A, single = _as_feature_matrix(X)

    geo = geometric_features(A)

    selected = geo[
        :, DISTANCE_INDICES + ANGLE_INDICES
    ]

    result = np.concatenate(
        [A, selected], axis=1
    ).astype(np.float32)

    return result[0] if single else result


# --------------------------------------------------
# 6. Identify likely pointing hand
# --------------------------------------------------

def _pointing_score(h):
    # Index finger straightness
    index = _angle(
        h[5], h[6], h[8]
    )

    # Straightness of other fingers
    others = np.mean([
        _angle(h[9], h[10], h[12]),
        _angle(h[13], h[14], h[16]),
        _angle(h[17], h[18], h[20])
    ])

    return float(index - others)


# --------------------------------------------------
# 7. NEW: Inter-hand fingertip contact features
# --------------------------------------------------

def contact_features(X):
    """
    Calculate five approximate distances:

    Pointing index tip -> Supporting thumb tip (4)
    Pointing index tip -> Supporting index tip (8)
    Pointing index tip -> Supporting middle tip (12)
    Pointing index tip -> Supporting ring tip (16)
    Pointing index tip -> Supporting pinky tip (20)

    Uses x/y coordinates and stored inter-wrist offset.
    Results are approximate because each hand was
    originally normalized using its own scale.
    """

    A, single = _as_feature_matrix(X)

    output = np.zeros(
        (len(A), 5), dtype=np.float32
    )

    for i, row in enumerate(A):

        # Dynamic contact features need two hands
        if row[130] <= 0.5 or row[131] <= 0.5:
            continue

        hand0 = row[0:63].reshape(21, 3)
        hand1 = row[63:126].reshape(21, 3)

        # Wrist 1 - wrist 0, in average-hand-scale units
        wrist_delta = row[126:128]

        # Case A: Hand 0 is pointing
        # Hand 1 is supporting
        distances0 = np.linalg.norm(
            hand0[8, :2] -
            (hand1[list(TIPS), :2] + wrist_delta),
            axis=1
        )

        # Case B: Hand 1 is pointing
        # Hand 0 is supporting
        distances1 = np.linalg.norm(
            (hand1[8, :2] + wrist_delta) -
            hand0[list(TIPS), :2],
            axis=1
        )

        score0 = _pointing_score(hand0)
        score1 = _pointing_score(hand1)

        # Choose likely pointing hand
        if score0 > score1 + 0.05:
            output[i] = distances0

        elif score1 > score0 + 0.05:
            output[i] = distances1

        else:
            # Fallback when hand roles are uncertain
            if np.min(distances0) <= np.min(distances1):
                output[i] = distances0
            else:
                output[i] = distances1

    return output[0] if single else output


# --------------------------------------------------
# 8. NEW: Complete 157D feature vector
# --------------------------------------------------

def augmented_features_157(X):
    """
    Original 132 features
    + 10 tip-to-palm distances
    + 10 joint angles
    + 5 cross-hand contact distances
    = 157 features
    """

    A, single = _as_feature_matrix(X)

    base_152 = features_152(A)
    contacts = contact_features(A)

    result = np.concatenate(
        [base_152, contacts], axis=1
    ).astype(np.float32)

    return result[0] if single else result


# --------------------------------------------------
# 9. Quick test
# --------------------------------------------------

if __name__ == "__main__":

    sample = np.zeros(
        (3, 132), dtype=np.float32
    )

    sample[:, 130:132] = 1.0
    sample[:, 126:128] = [1.0, 0.0]

    print(
        "Original geometry:",
        geometric_features(sample).shape
    )

    print(
        "Original augmented:",
        augmented_features(sample).shape
    )

    print(
        "152D features:",
        features_152(sample).shape
    )

    print(
        "Contact features:",
        contact_features(sample).shape
    )

    print(
        "157D features:",
        augmented_features_157(sample).shape
    )
