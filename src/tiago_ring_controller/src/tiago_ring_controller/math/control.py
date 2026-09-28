"""Pure counterparts of duplicated robot-control preprocessing helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np


class LegacyControlProfileLike(Protocol):
    ring_edge_margin_fraction: Optional[float]

    def effective_half_width(self, requested_half_width: int = 5) -> int:
        ...


def exponential_filter(spike_history: np.ndarray, dt_s: float, tau: float) -> np.ndarray:
    alpha = np.exp(-dt_s / tau)
    filtered = []
    drive = 0.0
    for spike in spike_history:
        drive = alpha * drive + (1.0 - alpha) * spike
        filtered.append(drive)
    return np.array(filtered, dtype=float)


def apply_delay(drive_history: np.ndarray, delay_steps: int) -> np.ndarray:
    drive = np.array(drive_history, dtype=float)
    if delay_steps <= 0:
        return drive.copy()
    delayed = np.zeros_like(drive)
    if delay_steps < len(drive):
        delayed[delay_steps:] = drive[:-delay_steps]
    return delayed


def decode_velocity(delayed_drive: float, gain_positive: float, gain_negative: float) -> float:
    if delayed_drive >= 0.0:
        return gain_positive * delayed_drive
    return gain_negative * (-delayed_drive)


def decoder_features(
    spike_history: np.ndarray, dt_s: float, tau: float, delay_steps: int
) -> Dict[str, Any]:
    filtered = exponential_filter(spike_history, dt_s, tau)
    delayed = apply_delay(filtered, delay_steps)
    drive_pos = np.maximum(delayed, 0.0)
    drive_neg = np.maximum(-delayed, 0.0)
    return {
        "filtered_drive": filtered,
        "delayed_drive": delayed,
        "drive_pos": drive_pos,
        "drive_neg": drive_neg,
        "S_pos": float(np.sum(drive_pos) * dt_s),
        "S_neg": float(np.sum(drive_neg) * dt_s),
    }


def signed_gain_signal(left_count: float, right_count: float, spike_scale: float = 1.0) -> float:
    return (right_count - left_count) * spike_scale


def legacy_joint_to_ring_index(
    position: float,
    joint_min: float,
    joint_max: float,
    population_size: int,
    half_width: int,
) -> int:
    index = int(
        (position - joint_min)
        / (joint_max - joint_min)
        * (population_size - half_width * 2)
    ) + half_width
    return max(half_width, min(index, population_size - 1 - half_width))


def calibration_joint_to_ring_index(
    position: float,
    joint_min: float,
    joint_max: float,
    population_size: int,
    half_width: int,
    edge_margin_fraction: float = 0.10,
) -> int:
    denom = joint_max - joint_min
    if denom <= 0:
        return max(half_width, min(population_size // 2, population_size - 1 - half_width))
    edge_margin = int(round(edge_margin_fraction * population_size))
    lower = max(half_width, edge_margin)
    upper = min(population_size - 1 - half_width, population_size - edge_margin)
    if upper < lower:
        lower = half_width
        upper = population_size - 1 - half_width
    phase = (position - joint_min) / denom
    phase = float(np.clip(phase, 0.0, 1.0))
    inject = int(round(lower + phase * (upper - lower)))
    return max(lower, min(inject, upper))


def profile_joint_to_ring_index(
    profile: LegacyControlProfileLike,
    position: float,
    joint_min: float,
    joint_max: float,
    population_size: int,
    requested_half_width: int = 5,
) -> int:
    half_width = profile.effective_half_width(requested_half_width)
    if profile.ring_edge_margin_fraction is not None:
        return calibration_joint_to_ring_index(
            position,
            joint_min,
            joint_max,
            population_size,
            half_width,
            profile.ring_edge_margin_fraction,
        )
    return legacy_joint_to_ring_index(
        position, joint_min, joint_max, population_size, half_width
    )
