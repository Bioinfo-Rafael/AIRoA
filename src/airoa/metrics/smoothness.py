"""Comparable local trajectory proxies; these are not the private PARC score."""

from __future__ import annotations

import numpy as np


def _mean_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values, axis=-1).mean()) if len(values) else 0.0


def _cosine_similarity(actions: np.ndarray) -> float:
    if len(actions) < 2:
        return 1.0
    left, right = actions[:-1], actions[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 1e-8
    if not valid.any():
        return 1.0
    return float(np.mean(np.sum(left[valid] * right[valid], axis=1) / denominator[valid]))


def _sparc_proxy(signal: np.ndarray) -> float:
    """Spectral arc length proxy where values closer to zero are smoother."""
    if len(signal) < 4:
        return 0.0
    speed = np.linalg.norm(signal, axis=1)
    spectrum = np.abs(np.fft.rfft(speed - speed.mean()))
    if spectrum.max() <= 1e-12:
        return 0.0
    spectrum /= spectrum.max()
    frequency = np.linspace(0.0, 1.0, len(spectrum))
    return float(-np.sum(np.sqrt(np.diff(frequency) ** 2 + np.diff(spectrum) ** 2)))


def trajectory_metrics(actions: np.ndarray) -> dict[str, float | int]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected [T,7] actions, got {actions.shape}")
    xyz_jerk = np.diff(actions[:, :3], n=2, axis=0)
    rotation_jerk = np.diff(actions[:, 3:6], n=2, axis=0)
    gripper = np.sign(actions[:, 6])
    nonzero = gripper != 0
    gripper_nonzero = gripper[nonzero]
    flips = int(np.sum(gripper_nonzero[1:] != gripper_nonzero[:-1])) if len(gripper_nonzero) > 1 else 0
    return {
        "episode_steps": int(len(actions)),
        "mean_action_norm": _mean_norm(actions),
        "xyz_action_jerk": _mean_norm(xyz_jerk),
        "rotation_jerk": _mean_norm(rotation_jerk),
        "consecutive_action_cosine_similarity": _cosine_similarity(actions[:, :6]),
        "gripper_sign_flips": flips,
        "trajectory_length": float(np.linalg.norm(actions[:, :3], axis=1).sum()),
        "rotation_length": float(np.linalg.norm(actions[:, 3:6], axis=1).sum()),
        "sparc_proxy": _sparc_proxy(actions[:, :6]),
    }
