"""Backend-injected homeostasis decision-circuit construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..contracts import HomeostasisSpec


@dataclass
class DecisionCircuit:
    neurons: Dict[str, Any]
    spike_recorders: Dict[str, Any]


def build_decision_circuit(
    backend: Any,
    neuron_parameters: Mapping[str, Any],
    spec: HomeostasisSpec,
) -> DecisionCircuit:
    warm = backend.Create("iaf_psc_alpha", params=dict(neuron_parameters))
    cold = backend.Create("iaf_psc_alpha", params=dict(neuron_parameters))
    left = backend.Create("iaf_psc_alpha", params=dict(neuron_parameters))
    right = backend.Create("iaf_psc_alpha", params=dict(neuron_parameters))

    warm_recorder = backend.Create("spike_recorder")
    cold_recorder = backend.Create("spike_recorder")
    left_recorder = backend.Create("spike_recorder")
    right_recorder = backend.Create("spike_recorder")

    backend.Connect(cold, right, syn_spec={"weight": spec.warm_exc_weight})
    backend.Connect(cold, left, syn_spec={"weight": spec.warm_inh_weight})
    backend.Connect(warm, left, syn_spec={"weight": spec.cold_exc_weight})
    backend.Connect(warm, right, syn_spec={"weight": spec.cold_inh_weight})
    backend.Connect(left, right, syn_spec={"weight": spec.decision_lateral_weight})
    backend.Connect(right, left, syn_spec={"weight": spec.decision_lateral_weight})

    backend.Connect(warm, warm_recorder)
    backend.Connect(cold, cold_recorder)
    backend.Connect(left, left_recorder)
    backend.Connect(right, right_recorder)
    return DecisionCircuit(
        neurons={"warm": warm, "cold": cold, "left": left, "right": right},
        spike_recorders={
            "warm_spike": warm_recorder,
            "cold_spike": cold_recorder,
            "left_spike": left_recorder,
            "right_spike": right_recorder,
        },
    )


def connect_ring_features(
    backend: Any,
    circuit: DecisionCircuit,
    ring_1_decoded_neurons: Mapping[str, Any],
    ring_2_decoded_neurons: Mapping[str, Any],
    feature_names: Sequence[str],
    weight_matrix: np.ndarray,
    weight_scale: float,
) -> None:
    weights = np.asarray(weight_matrix)
    connect_ring_feature_vectors(
        backend,
        circuit,
        ring_1_decoded_neurons,
        ring_2_decoded_neurons,
        feature_names,
        weights[:, 0],
        weights[:, 1],
        weight_scale,
    )


def connect_ring_feature_vectors(
    backend: Any,
    circuit: DecisionCircuit,
    ring_1_decoded_neurons: Mapping[str, Any],
    ring_2_decoded_neurons: Mapping[str, Any],
    feature_names: Sequence[str],
    warm_weights: Sequence[float],
    cold_weights: Sequence[float],
    weight_scale: float,
) -> None:
    """Connect separate legacy vectors in their partial-failure order."""

    n_features = len(feature_names)
    cold_offset = n_features
    for feature_idx, name in enumerate(feature_names):
        backend.Connect(
            ring_1_decoded_neurons[name],
            circuit.neurons["warm"],
            syn_spec={"weight": float(weight_scale * warm_weights[feature_idx])},
        )
        backend.Connect(
            ring_2_decoded_neurons[name],
            circuit.neurons["cold"],
            syn_spec={
                "weight": float(weight_scale * cold_weights[cold_offset + feature_idx])
            },
        )
