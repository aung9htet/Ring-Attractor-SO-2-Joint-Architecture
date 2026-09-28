"""Analytic push-pull Fourier feature construction."""

from __future__ import annotations

from typing import List

import numpy as np

from .circular import preferred_angles


def build_push_pull_sine_weights(population_size: int, num_fourier_k: int) -> np.ndarray:
    """Return the exact ``N x 2K`` matrix written by both legacy trainers."""

    theta = 2 * np.pi * np.arange(population_size) / population_size
    weights = np.zeros((population_size, 2 * num_fourier_k))
    for k in range(1, num_fourier_k + 1):
        w_sin = np.sin(k * theta)
        i0 = 2 * (k - 1)
        weights[:, i0] = np.maximum(w_sin, 0.0)
        weights[:, i0 + 1] = np.maximum(-w_sin, 0.0)
    return weights


def fourier_feature_names(num_fourier_k: int) -> List[str]:
    return [
        name
        for k in range(1, num_fourier_k + 1)
        for name in ("sin_pos_k{}".format(k), "sin_neg_k{}".format(k))
    ]


def signed_harmonic_features(push_pull_features: np.ndarray) -> np.ndarray:
    features = np.asarray(push_pull_features, dtype=float)
    return features[..., 0::2] - features[..., 1::2]


def preprocess_activity(activity: np.ndarray) -> np.ndarray:
    values = np.asarray(activity)
    centered = values - np.mean(values)
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 0 else centered


def decode_first_sine_arcsin(signed_counts: np.ndarray) -> float:
    counts = np.asarray(signed_counts)
    return float(np.arcsin(np.clip(counts[0], -1.0, 1.0))) if counts.size else np.nan


def target_cosine_modulation(angle: float, output_ring_size: int, amplitude: float) -> np.ndarray:
    theta = preferred_angles(output_ring_size)
    return amplitude * np.cos(theta - angle)


def scalar_ramp_weights(output_ring_size: int, weight_scale: float = 100.0) -> np.ndarray:
    return weight_scale * np.linspace(0.0, 1.0, output_ring_size)
