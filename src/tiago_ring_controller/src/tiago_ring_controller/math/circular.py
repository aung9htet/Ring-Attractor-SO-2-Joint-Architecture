"""Circular coordinate and population-vector operations."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def preferred_angles(n: int) -> np.ndarray:
    return 2.0 * np.pi * np.arange(n) / n


def index_to_phase(index: int, population_size: int) -> float:
    return 2.0 * np.pi * index / population_size


def angle_to_neuron_index(angle: float, population_size: int) -> int:
    normalized = np.mod(angle, 2.0 * np.pi) / (2.0 * np.pi)
    # Built-in round (including its tie behavior) is used by the source.
    return int(round(normalized * population_size)) % population_size


def angle_to_ring_index(
    angle: Union[float, np.ndarray], ring_size: int
) -> Union[float, np.ndarray]:
    angle_array = np.asarray(angle, dtype=float)
    result = np.mod(angle_array, 2.0 * np.pi) / (2.0 * np.pi) * ring_size
    return float(result) if np.ndim(result) == 0 else result


def ring_index_to_angle(
    index: Union[float, np.ndarray], ring_size: int
) -> Union[float, np.ndarray]:
    index_array = np.asarray(index, dtype=float)
    result = np.mod(index_array, ring_size) / ring_size * 2.0 * np.pi
    return float(result) if np.ndim(result) == 0 else result


def decode_population_angle(spike_counts: np.ndarray) -> float:
    """Builder/multi-ring profile decoder, including minimum subtraction."""

    counts = np.asarray(spike_counts, dtype=float)
    normalized = np.maximum(counts - np.min(counts), 0.0)
    angles = preferred_angles(len(counts))
    sin_sum = np.dot(normalized, np.sin(angles))
    cos_sum = np.dot(normalized, np.cos(angles))
    return float(np.arctan2(sin_sum, cos_sum))


def decode_sawtooth_profile(profile: np.ndarray) -> float:
    return decode_population_angle(profile)


def circular_signed_difference(
    target: Union[float, np.ndarray],
    current: Union[float, np.ndarray],
    period: float,
) -> Union[float, np.ndarray]:
    delta = target - current
    delta = ((delta + period / 2.0) % period) - period / 2.0
    return delta


def circular_angle_error(
    estimate: Union[float, np.ndarray], true: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    result = np.angle(np.exp(1j * (np.asarray(estimate) - np.asarray(true))))
    return float(result) if np.ndim(result) == 0 else result


def decode_circular_centroid(rate_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate ``SingleRingModel._decode_circular_centroid``."""

    rates = np.asarray(rate_matrix)
    n_neurons = rates.shape[0]
    angles = 2.0 * np.pi * np.arange(n_neurons) / n_neurons
    centroid_idx = np.full(rates.shape[1], np.nan)
    strength = np.zeros(rates.shape[1])
    for time_idx in range(rates.shape[1]):
        activity = rates[:, time_idx]
        if np.sum(activity) <= 0:
            continue
        z = np.sum(activity * np.exp(1j * angles))
        theta = np.angle(z) % (2.0 * np.pi)
        centroid_idx[time_idx] = theta / (2.0 * np.pi) * n_neurons
        strength[time_idx] = np.abs(z) / np.sum(activity)
    return centroid_idx, strength


def circular_index_nearest(
    decoded: np.ndarray, true: np.ndarray, ring_size: int
) -> np.ndarray:
    decoded_array = np.asarray(decoded, dtype=float)
    true_array = np.asarray(true, dtype=float)
    size = float(ring_size)
    return true_array + ((decoded_array - true_array + size / 2.0) % size - size / 2.0)
