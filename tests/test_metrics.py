import numpy as np
import pytest

from airoa.metrics.smoothness import trajectory_metrics


def test_constant_chunk_is_smooth_and_has_no_gripper_flips():
    actions = np.ones((10, 7), dtype=np.float32)
    metrics = trajectory_metrics(actions)
    assert metrics["xyz_action_jerk"] == 0
    assert metrics["rotation_jerk"] == 0
    assert metrics["consecutive_action_cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["gripper_sign_flips"] == 0
    assert metrics["episode_steps"] == 10
