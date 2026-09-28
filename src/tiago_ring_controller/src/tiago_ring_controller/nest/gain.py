"""Backend-injected opponent gain-modulation topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

from ..contracts import GainSpec
from .comparator import DecisionCircuit


@dataclass
class GainNetwork:
    left_neurons: List[Any]
    right_neurons: List[Any]
    left_spike_recorders: List[Any]
    right_spike_recorders: List[Any]


def _create_population(
    backend: Any, population_size: int, neuron_parameters: Mapping[str, Any]
):
    neurons = []
    recorders = []
    for _ in range(population_size):
        neuron = backend.Create("iaf_psc_alpha", params=dict(neuron_parameters))
        recorder = backend.Create("spike_recorder")
        backend.Connect(neuron, recorder)
        neurons.append(neuron)
        recorders.append(recorder)
    return neurons, recorders


def build_gain_network(
    backend: Any,
    ring_neurons: Sequence[Any],
    decision_circuit: DecisionCircuit,
    neuron_parameters: Mapping[str, Any],
    spec: GainSpec,
) -> GainNetwork:
    population_size = len(ring_neurons)
    network = create_gain_network(
        backend,
        population_size,
        neuron_parameters,
    )
    connect_gain_network(
        backend,
        network,
        ring_neurons,
        decision_circuit,
        spec,
    )
    return network


def create_gain_network(
    backend: Any,
    population_size: int,
    neuron_parameters: Mapping[str, Any],
) -> GainNetwork:
    left, left_recorders = _create_population(
        backend, population_size, neuron_parameters
    )
    right, right_recorders = _create_population(
        backend, population_size, neuron_parameters
    )
    return GainNetwork(left, right, left_recorders, right_recorders)


def connect_gain_network(
    backend: Any,
    gain_network: GainNetwork,
    ring_neurons: Sequence[Any],
    decision_circuit: DecisionCircuit,
    spec: GainSpec,
) -> None:
    population_size = len(ring_neurons)
    for idx in range(population_size):
        backend.Connect(
            decision_circuit.neurons["left"],
            gain_network.left_neurons[idx],
            syn_spec={"weight": spec.left_homeostasis_gain_weight},
        )
        backend.Connect(
            decision_circuit.neurons["right"],
            gain_network.right_neurons[idx],
            syn_spec={"weight": spec.right_homeostasis_gain_weight},
        )
        backend.Connect(
            decision_circuit.neurons["left"],
            gain_network.right_neurons[idx],
            syn_spec={"weight": spec.cross_inhibition_weight},
        )
        backend.Connect(
            decision_circuit.neurons["right"],
            gain_network.left_neurons[idx],
            syn_spec={"weight": spec.cross_inhibition_weight},
        )
        backend.Connect(
            ring_neurons[idx],
            gain_network.left_neurons[idx],
            syn_spec={"weight": spec.ring_to_gain_weight},
        )
        backend.Connect(
            ring_neurons[idx],
            gain_network.right_neurons[idx],
            syn_spec={"weight": spec.ring_to_gain_weight},
        )


def connect_gain_feedback(
    backend: Any,
    gain_network: GainNetwork,
    ring_neurons: Sequence[Any],
    weight: float,
) -> None:
    population_size = len(ring_neurons)
    for idx in range(5, population_size - 5):
        backend.Connect(
            gain_network.left_neurons[idx],
            ring_neurons[(idx + 1) % population_size],
            syn_spec={"weight": weight},
        )
        backend.Connect(
            gain_network.right_neurons[idx],
            ring_neurons[(idx - 1) % population_size],
            syn_spec={"weight": weight},
        )
