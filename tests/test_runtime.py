import numpy as np
import pytest

from airoa.parc.runtime import Pi05ParcPolicy


def make_obs():
    return {
        "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def test_action_queue_horizon_and_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("AIROA_POLICY_BACKEND", "stub")
    policy = Pi05ParcPolicy(tmp_path, execution_horizon=5)
    obs = make_obs()
    first = policy.get_action(obs)
    second = policy.get_action(obs)
    assert first.shape == (7,) and first.dtype == np.float32
    np.testing.assert_array_equal(first, np.zeros(7, dtype=np.float32))
    np.testing.assert_allclose(second, np.full(7, 0.01, dtype=np.float32))
    assert len(policy.action_queue) == 3
    policy.reset("new instruction")
    assert len(policy.action_queue) == 0
    assert policy.previous_chunk is None
    assert policy.instruction == "new instruction"
    np.testing.assert_array_equal(policy.get_action(obs), first)


@pytest.mark.parametrize("horizon", [5, 8, 10])
def test_supported_execution_horizons(monkeypatch, tmp_path, horizon):
    monkeypatch.setenv("AIROA_POLICY_BACKEND", "stub")
    policy = Pi05ParcPolicy(tmp_path, execution_horizon=horizon)
    policy.get_action(make_obs())
    assert len(policy.action_queue) == horizon - 1
