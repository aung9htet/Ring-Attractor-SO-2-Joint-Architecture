"""Ordered rotation and lift/pitch/yaw target mathematics."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError("Unknown axis {!r}".format(axis))


def forward_kinematics(joint_angles: np.ndarray, joint_axes: Sequence[str]) -> np.ndarray:
    rotation = np.eye(3)
    for axis, angle in zip(joint_axes, joint_angles):
        rotation = rotation @ rotation_matrix(axis, angle)
    return rotation


def lift_pitch_yaw_from_rotation(rotation: np.ndarray) -> Tuple[float, float, float]:
    pitch = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
    lift = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return lift, pitch, yaw


def lift_pitch_yaw_from_angles(
    joint_angles: np.ndarray, joint_axes: Sequence[str]
) -> Tuple[float, float, float]:
    return lift_pitch_yaw_from_rotation(forward_kinematics(joint_angles, joint_axes))


def resample_profile(profile: np.ndarray, output_size: int) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    if len(values) == output_size:
        return values
    x_old = np.linspace(0, output_size - 1, len(values))
    return np.interp(np.arange(output_size), x_old, values)
