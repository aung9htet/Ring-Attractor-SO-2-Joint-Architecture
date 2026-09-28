"""Pure specifications for Fourier and signed product feature layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .math.circular import preferred_angles


SIGNED_PRODUCT_TERM_NAMES = (
    "cos1",
    "sin1",
    "cos2",
    "sin2",
    "cos1cos2",
    "cos1sin2",
    "sin1cos2",
    "sin1sin2",
)


def signed_product_term_names() -> List[str]:
    return list(SIGNED_PRODUCT_TERM_NAMES)


def signed_product_feature_order() -> List[str]:
    order = []
    for term_name in SIGNED_PRODUCT_TERM_NAMES:
        order.append("{}_pos".format(term_name))
        order.append("{}_neg".format(term_name))
    return order


def signed_product_feature_dimension(feature_grid_size: int) -> int:
    return 2 * len(SIGNED_PRODUCT_TERM_NAMES) * feature_grid_size * feature_grid_size


def mapped_ring_indices(population_size: int, feature_grid_size: int) -> np.ndarray:
    return np.array(
        [
            int(round(i * population_size / feature_grid_size)) % population_size
            for i in range(feature_grid_size)
        ],
        dtype=int,
    )


def signed_product_term_values(
    population_size: int, feature_grid_size: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    indices = mapped_ring_indices(population_size, feature_grid_size)
    theta_side = preferred_angles(population_size)[indices]
    ones = np.ones_like(theta_side)
    return {
        "cos1": (np.cos(theta_side), ones),
        "sin1": (np.sin(theta_side), ones),
        "cos2": (ones, np.cos(theta_side)),
        "sin2": (ones, np.sin(theta_side)),
        "cos1cos2": (np.cos(theta_side), np.cos(theta_side)),
        "cos1sin2": (np.cos(theta_side), np.sin(theta_side)),
        "sin1cos2": (np.sin(theta_side), np.cos(theta_side)),
        "sin1sin2": (np.sin(theta_side), np.sin(theta_side)),
    }


def flatten_signed_counts(
    counts: Mapping[str, np.ndarray], feature_order: Sequence[str] = ()
) -> np.ndarray:
    order = (
        list(feature_order)
        if len(feature_order) > 0
        else signed_product_feature_order()
    )
    parts = [np.asarray(counts[name], dtype=float).ravel() for name in order]
    return np.concatenate(parts)


def unflatten_signed_vector(
    vector: np.ndarray,
    feature_grid_size: int,
    feature_order: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    order = (
        list(feature_order)
        if len(feature_order) > 0
        else signed_product_feature_order()
    )
    cell_count = feature_grid_size * feature_grid_size
    maps = {}
    start = 0
    for name in order:
        maps[name] = vector[start : start + cell_count].reshape(
            feature_grid_size, feature_grid_size
        )
        start += cell_count
    return maps


def signed_vector_term_sums(
    vector: np.ndarray,
    feature_grid_size: int,
    feature_order: Sequence[str] = (),
) -> Dict[str, float]:
    maps = unflatten_signed_vector(vector, feature_grid_size, feature_order)
    return {
        term: float(np.sum(maps[term + "_pos"]) - np.sum(maps[term + "_neg"]))
        for term in SIGNED_PRODUCT_TERM_NAMES
    }


def signed_vector_term_activity(
    vector: np.ndarray,
    feature_grid_size: int,
    feature_order: Sequence[str] = (),
) -> Dict[str, float]:
    maps = unflatten_signed_vector(vector, feature_grid_size, feature_order)
    return {
        term: float(
            np.sum(np.abs(np.asarray(maps[term + "_pos"], dtype=float)))
            + np.sum(np.abs(np.asarray(maps[term + "_neg"], dtype=float)))
        )
        for term in SIGNED_PRODUCT_TERM_NAMES
    }


def homeostasis_dataset(states: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate ``HomeostasisTrainer._build_dataset``."""

    x_rows = []
    targets = []
    phases = []
    # The legacy method derives K from the first state and therefore raises
    # IndexError for an empty library; keep that contract.
    num_harmonics = len(states[0]["features"]) // 2
    for state in states:
        feature = np.asarray(state["features"])
        signed = []
        for harmonic_idx in range(num_harmonics):
            i0 = 2 * harmonic_idx
            signed.append(float(feature[i0] - feature[i0 + 1]))
        x_rows.append(signed)
        phase = float(state["phase"])
        targets.append(phase / (2.0 * np.pi))
        phases.append(phase)
    return (
        np.array(x_rows, dtype=float),
        np.array(targets, dtype=float),
        np.array(phases, dtype=float),
    )


def embed_homeostasis_weights(signed_weights: np.ndarray) -> np.ndarray:
    """Embed K signed weights into the current ``(4K, 2)`` artifact layout."""

    weights = np.asarray(signed_weights)
    push_pull = np.array([value for value in weights for value in (value, -value)])
    zeros = np.zeros(2 * len(weights))
    warm = np.concatenate([push_pull, zeros])
    cold = np.concatenate([zeros, push_pull])
    return np.stack([warm, cold], axis=1)


@dataclass(frozen=True)
class SignedProductConnection:
    feature_name: str
    cell_index: int
    q1_index: int
    q2_index: int
    q1_magnitude: float
    q2_magnitude: float


def signed_product_connections(
    population_size: int,
    feature_grid_size: int,
    epsilon: float = 1.0e-9,
) -> Tuple[SignedProductConnection, ...]:
    """Describe current source-to-grid connections without creating NEST nodes."""

    mappings = mapped_ring_indices(population_size, feature_grid_size)
    term_values = signed_product_term_values(population_size, feature_grid_size)
    result = []
    for term_name, (q1_values, q2_values) in term_values.items():
        q1_only = term_name in ("cos1", "sin1")
        q2_only = term_name in ("cos2", "sin2")
        for i in range(feature_grid_size):
            for j in range(feature_grid_size):
                a_value = q1_values[i]
                b_value = q2_values[j]
                product = a_value * b_value
                if product > epsilon:
                    sign = "pos"
                elif product < -epsilon:
                    sign = "neg"
                else:
                    continue
                result.append(
                    SignedProductConnection(
                        feature_name="{}_{}".format(term_name, sign),
                        cell_index=i * feature_grid_size + j,
                        q1_index=int(mappings[i]),
                        q2_index=int(mappings[j]),
                        q1_magnitude=0.0 if q2_only else float(abs(a_value)),
                        q2_magnitude=0.0 if q1_only else float(abs(b_value)),
                    )
                )
    return tuple(result)
