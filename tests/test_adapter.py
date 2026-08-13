import numpy as np

from airoa.parc.adapter import parc_observation_to_pi05, quaternion_to_axis_angle


def make_obs():
    image = np.arange(128 * 128 * 3, dtype=np.uint32).reshape(128, 128, 3).astype(np.uint8)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image.copy(),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.array([1, 2, 3], dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
    }


def test_axis_angle_matches_identity_and_quarter_turn():
    np.testing.assert_array_equal(quaternion_to_axis_angle(np.array([0, 0, 0, 1])), np.zeros(3))
    angle = np.pi / 2
    quat = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)])
    np.testing.assert_allclose(quaternion_to_axis_angle(quat), [0, 0, angle], atol=1e-6)


def test_parc_adapter_rotates_both_cameras_and_builds_8d_state():
    obs = make_obs()
    result = parc_observation_to_pi05(obs, "do the task")
    expected = obs["agentview_image"][::-1, ::-1].transpose(2, 0, 1) / 255.0
    np.testing.assert_allclose(result["observation.images.image"].numpy(), expected)
    np.testing.assert_allclose(result["observation.images.image2"].numpy(), expected)
    np.testing.assert_allclose(
        result["observation.state"].numpy(), [1, 2, 3, 0, 0, 0, 0.04, -0.04]
    )
    assert result["observation.state"].shape == (8,)
    assert result["task"] == "do the task"
