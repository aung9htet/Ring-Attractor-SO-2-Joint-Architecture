"""Exact pure counterparts of the two historical ring implementations."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def legacy_ring_distances(population_size: int, max_distance: float = 50) -> np.ndarray:
    """Replicate ``Ring_Attractor.neuron_distance`` operation-for-operation."""

    if population_size % 2 == 0:
        distance = np.append(
            0,
            np.linspace(0, max_distance, int(population_size / 2) + 1)[
                1 : int(population_size / 2)
            ],
        )
        distance = np.append(distance, max_distance)
        distance = np.append(
            distance,
            np.linspace(max_distance, 0, int(population_size / 2) + 1)[
                1 : int(population_size / 2)
            ],
        )
    else:
        distance = np.append(
            0,
            np.linspace(0, max_distance, int(population_size / 2) + 1)[1:],
        )
        distance = np.append(
            distance,
            np.linspace(max_distance, 0, int(population_size / 2) + 1)[
                : int(population_size / 2)
            ],
        )
    return distance


def builder_ring_distances(population_size: int, max_distance: float = 50) -> np.ndarray:
    """Replicate ``builders.RingAttractor._compute_ring_distances``."""

    ring_half_size = population_size // 2
    if population_size % 2 == 0:
        forward = np.linspace(0, max_distance, ring_half_size + 1)[:-1]
        backward = np.linspace(max_distance, 0, ring_half_size + 1)[:-1]
    else:
        forward = np.linspace(0, max_distance, ring_half_size + 1)
        backward = np.linspace(max_distance, 0, ring_half_size)
    return np.concatenate([forward, backward])


def legacy_ring_weight(
    distance: float, sd_1: float = 10, sd_2: float = 5
) -> np.floating:
    """Replicate the fixed-width legacy Mexican-hat calculation."""

    weight = 3 * (
        (
            sd_2 * np.exp((-(distance ** 2)) / (2 * (sd_1 ** 2)))
            - sd_1 * np.exp((-(distance ** 2)) / (2 * (sd_2 ** 2)))
        )
        / (sd_2 - sd_1)
    )
    return np.around(weight, 3)


def builder_ring_weight(
    distance: float,
    population_size: int,
    excitation_std_dev: float = 10,
    inhibition_std_dev: float = 5,
) -> float:
    """Replicate the population-scaled builder Mexican-hat calculation."""

    excitation_scale = excitation_std_dev / (population_size / 100)
    inhibition_scale = inhibition_std_dev / (population_size / 100)
    dist_sq = distance ** 2
    excitation_component = inhibition_scale * np.exp(
        -dist_sq / (2 * excitation_scale ** 2)
    )
    inhibition_component = excitation_scale * np.exp(
        -dist_sq / (2 * inhibition_scale ** 2)
    )
    weight = 3 * (excitation_component - inhibition_component) / (
        inhibition_scale - excitation_scale
    )
    return float(np.around(weight, 3))


def recurrent_connection_indices(population_size: int) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(pre, post, distance_index)`` in legacy connection order."""

    for pre_idx in range(population_size):
        for distance_idx in range(population_size):
            yield pre_idx, (pre_idx + distance_idx) % population_size, distance_idx


def stimulus_target_indices(
    population_size: int, center_index: int = 0, half_width: int = 5
) -> np.ndarray:
    """Return target indices in the exact inclusive connection order."""

    return np.array(
        [
            (center_index + offset) % population_size
            for offset in range(-half_width, half_width + 1)
        ],
        dtype=int,
    )


def generate_center_indices(population_size: int, num_positions: int) -> np.ndarray:
    """Evenly spaced non-zero positions used by homeostasis training."""

    return np.array(
        [
            int(k * population_size / (num_positions + 1)) % population_size
            for k in range(1, num_positions + 1)
        ]
    )


def ring_weight_matrix(
    population_size: int,
    variant: str = "legacy",
    max_distance: float = 50,
    excitation_std_dev: float = 10,
    inhibition_std_dev: float = 5,
) -> np.ndarray:
    """Recurrent weights as a ``(pre, post)`` matrix, one entry per synapse.

    Row ``pre`` holds the weights the legacy loop assigns in order
    ``post = (pre + distance_index) % N``; the diagonal is the self-connection
    (distance 0).  Both ring variants are supported; the values are the same
    rounded numbers the per-synapse builders pass to ``Connect``.
    """

    size = int(population_size)
    if size < 1:
        raise ValueError("population_size must be positive")
    if variant == "legacy":
        distances = legacy_ring_distances(size, max_distance)
        profile = np.array(
            [legacy_ring_weight(d, sd_1=excitation_std_dev, sd_2=inhibition_std_dev) for d in distances],
            dtype=float,
        )
    elif variant == "builder":
        distances = builder_ring_distances(size, max_distance)
        profile = np.array(
            [
                builder_ring_weight(
                    d,
                    population_size=size,
                    excitation_std_dev=excitation_std_dev,
                    inhibition_std_dev=inhibition_std_dev,
                )
                for d in distances
            ],
            dtype=float,
        )
    else:
        raise ValueError("Unknown ring variant: {}".format(variant))
    pre = np.arange(size)[:, None]
    shift = np.arange(size)[None, :]
    matrix = np.zeros((size, size), dtype=float)
    matrix[pre, (pre + shift) % size] = profile[None, :]
    return matrix
