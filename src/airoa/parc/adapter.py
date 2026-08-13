"""Patches PARC observations into the exact LeRobot π0.5 LIBERO feature contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from airoa.constants import FRONT_POLICY_KEY, STATE_KEY, WRIST_POLICY_KEY

REQUIRED_OBSERVATION_SHAPES = {
    "agentview_image": (128, 128, 3),
    "robot0_eye_in_hand_image": (128, 128, 3),
    "robot0_joint_pos": (7,),
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_qpos": (2,),
}


def quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """robosuite/OpenPI LIBERO x,y,z,w quaternion conversion."""
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {quat.shape}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator).astype(np.float32)


def rotate_libero_camera(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError(f"PARC camera must be uint8 (128,128,3), got {image.dtype} {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def _image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)


def parc_observation_to_pi05(obs: dict[str, np.ndarray], instruction: str) -> dict[str, Any]:
    for key, shape in REQUIRED_OBSERVATION_SHAPES.items():
        if key not in obs:
            raise KeyError(f"Missing PARC observation key: {key}")
        if tuple(np.asarray(obs[key]).shape) != shape:
            raise ValueError(f"{key} expected {shape}, got {np.asarray(obs[key]).shape}")
    front = rotate_libero_camera(obs["agentview_image"])
    wrist = rotate_libero_camera(obs["robot0_eye_in_hand_image"])
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quaternion_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid 8D LIBERO state: {state}")
    return {
        FRONT_POLICY_KEY: _image_tensor(front),
        WRIST_POLICY_KEY: _image_tensor(wrist),
        STATE_KEY: torch.from_numpy(state),
        "task": instruction,
    }
