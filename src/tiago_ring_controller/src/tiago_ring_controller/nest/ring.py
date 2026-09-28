"""Exact backend-injected builders for the two ring-attractor variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

import numpy as np

from ..math.ring import (
    builder_ring_distances,
    builder_ring_weight,
    legacy_ring_distances,
    legacy_ring_weight,
    recurrent_connection_indices,
    stimulus_target_indices,
)
from .kernel import configure_kernel, spike_counts


@dataclass
class RingNetwork:
    neurons: List[Any]
    spike_recorders: List[Any]
    distances: np.ndarray
    population_size: int
    variant: str


def build_ring_network(
    backend: Any,
    population_size: int,
    neuron_parameters: Mapping[str, Any],
    variant: str = "legacy",
    reset_kernel: bool = True,
    max_distance: float = 50,
    excitation_std_dev: float = 10,
    inhibition_std_dev: float = 5,
    configure_backend: bool = True,
    distance_function: Optional[Callable[[], np.ndarray]] = None,
    weight_function: Optional[Callable[[float], Any]] = None,
) -> RingNetwork:
    """Build a ring while retaining legacy creation and connection order."""

    if configure_backend:
        configure_kernel(backend, reset_kernel=reset_kernel, verbosity="M_ERROR")
    neurons = []
    recorders = []
    for _ in range(population_size):
        neuron = backend.Create("iaf_psc_alpha", params=dict(neuron_parameters))
        recorder = backend.Create("spike_recorder")
        backend.Connect(neuron, recorder)
        neurons.append(neuron)
        recorders.append(recorder)

    if variant == "legacy":
        default_distances = legacy_ring_distances(population_size, max_distance)

        def default_weight(distance: float) -> Any:
            return legacy_ring_weight(
                distance,
                sd_1=excitation_std_dev,
                sd_2=inhibition_std_dev,
            )

    elif variant == "builder":
        default_distances = builder_ring_distances(population_size, max_distance)

        def default_weight(distance: float) -> Any:
            return builder_ring_weight(
                distance,
                population_size=population_size,
                excitation_std_dev=excitation_std_dev,
                inhibition_std_dev=inhibition_std_dev,
            )

    else:
        raise ValueError("Unknown ring variant: {}".format(variant))

    # Facades supply their bound methods so legacy subclassing/monkeypatching
    # still observes the same call graph.  Internal callers use the pure
    # defaults above.
    distances = (
        distance_function() if distance_function is not None else default_distances
    )
    compute_weight = weight_function if weight_function is not None else default_weight

    for pre_idx in range(population_size):
        for distance_idx in range(len(distances)):
            if variant == "legacy":
                shifted = pre_idx + distance_idx
                post_idx = shifted if shifted < population_size else shifted - population_size
            else:
                post_idx = (pre_idx + distance_idx) % population_size
            backend.Connect(
                neurons[pre_idx],
                neurons[post_idx],
                syn_spec={"weight": compute_weight(distances[distance_idx])},
            )
    return RingNetwork(
        neurons=neurons,
        spike_recorders=recorders,
        distances=distances,
        population_size=int(population_size),
        variant=variant,
    )


def build_legacy_ring(
    backend: Any,
    population_size: int,
    neuron_parameters: Mapping[str, Any],
    reset_kernel: bool = True,
) -> RingNetwork:
    return build_ring_network(
        backend,
        population_size,
        neuron_parameters,
        variant="legacy",
        reset_kernel=reset_kernel,
    )


def build_builder_ring(
    backend: Any,
    population_size: int,
    neuron_parameters: Mapping[str, Any],
    reset_kernel: bool = True,
) -> RingNetwork:
    return build_ring_network(
        backend,
        population_size,
        neuron_parameters,
        variant="builder",
        reset_kernel=reset_kernel,
    )


def inject_stimulus(
    backend: Any,
    network: RingNetwork,
    center_index: int = 0,
    half_width: int = 5,
    rate_hz: float = 200.0,
    weight: float = 4.5e3,
    duration_ms: float = 50.0,
) -> Any:
    stimulus = backend.Create("poisson_generator", params={"rate": rate_hz})
    for target_idx in stimulus_target_indices(
        network.population_size, center_index, half_width
    ):
        backend.Connect(
            stimulus,
            network.neurons[int(target_idx)],
            syn_spec={"weight": weight},
        )
    backend.Simulate(duration_ms)
    backend.SetStatus(stimulus, {"rate": 0.0})
    return stimulus


def ring_spike_counts(backend: Any, network: RingNetwork) -> np.ndarray:
    return spike_counts(backend, network.spike_recorders)
