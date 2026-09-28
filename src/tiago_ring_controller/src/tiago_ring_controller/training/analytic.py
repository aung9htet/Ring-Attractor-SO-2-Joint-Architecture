"""Analytically derived feature/readout artifacts (no optimization)."""

from __future__ import annotations

import numpy as np

from ..features import embed_homeostasis_weights
from ..math.fourier import build_push_pull_sine_weights, scalar_ramp_weights


def build_fourier_weights(population_size: int, num_fourier_k: int) -> np.ndarray:
    """Compatibility name for the matrix emitted by both ring trainers."""

    return build_push_pull_sine_weights(population_size, num_fourier_k)


def build_homeostasis_weight_matrix(signed_weights: np.ndarray) -> np.ndarray:
    return embed_homeostasis_weights(signed_weights)


def build_scalar_ramp_weights(
    output_ring_size: int, weight_scale: float = 100.0
) -> np.ndarray:
    return scalar_ramp_weights(output_ring_size, weight_scale)
