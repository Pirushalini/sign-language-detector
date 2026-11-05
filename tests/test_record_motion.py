import numpy as np

from motion import FEATURE_DIM, N_RESAMPLE_POINTS, load_motion_templates
from record_motion import _save_accepted_take


def test_accepted_takes_are_saved_and_accumulated(tmp_path):
    old_h = np.full((N_RESAMPLE_POINTS, FEATURE_DIM), -1, dtype=np.float32)
    old_j = np.full((N_RESAMPLE_POINTS, FEATURE_DIM), -2, dtype=np.float32)
    first_h = np.full((N_RESAMPLE_POINTS, FEATURE_DIM), 1, dtype=np.float32)
    second_h = np.full((N_RESAMPLE_POINTS, FEATURE_DIM), 2, dtype=np.float32)
    saved_data = {"H": [old_h], "J": [old_j]}
    replaced_letters = set()
    path = tmp_path / "motion_templates.npz"

    assert _save_accepted_take(
        saved_data, "H", first_h, False, replaced_letters, path
    ) == 1
    assert _save_accepted_take(
        saved_data, "H", second_h, False, replaced_letters, path
    ) == 2

    trajectories, labels = load_motion_templates(path)
    assert labels.count("H") == 2
    assert labels.count("J") == 1
    assert np.array_equal(trajectories[0], first_h)
    assert np.array_equal(trajectories[1], second_h)
    assert np.array_equal(trajectories[2], old_j)
