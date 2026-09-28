"""Backend-injected positional Fourier readout builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .kernel import recorder_events


@dataclass
class FourierReadout:
    neurons: Dict[str, Any]
    spike_recorders: Dict[str, Any]
    all_neurons: Any
    all_recorders: Any
    dc_generator: Any
    weights: np.ndarray


def build_fourier_readout(
    backend: Any,
    ring_neurons: Sequence[Any],
    weights: np.ndarray,
    num_fourier_k: int,
    output_weight_scale: float,
    output_dc_baseline: float,
    validate_shape: bool = True,
    population_factory: Optional[Callable[[int], Tuple[Any, Any]]] = None,
    connection_builder: Optional[Callable[[Any, np.ndarray], None]] = None,
) -> FourierReadout:
    matrix = np.asarray(weights)
    expected_shape = (len(ring_neurons), 2 * num_fourier_k)
    if validate_shape and matrix.shape != expected_shape:
        raise ValueError(
            "Fourier weights shape mismatch: got {}, expected {}".format(
                matrix.shape, expected_shape
            )
        )

    if population_factory is None:
        nodes = backend.Create("iaf_psc_alpha", 2 * num_fourier_k)
        recorders = backend.Create("spike_recorder", 2 * num_fourier_k)
        dc = backend.Create("dc_generator", params={"amplitude": output_dc_baseline})
        backend.Connect(dc, nodes)
        for idx in range(2 * num_fourier_k):
            backend.Connect(nodes[idx], recorders[idx])
    else:
        nodes, recorders = population_factory(2 * num_fourier_k)
        dc = None

    named_nodes: Dict[str, Any] = {}
    named_recorders: Dict[str, Any] = {}
    for k in range(num_fourier_k):
        i0 = 2 * k
        for offset, name in enumerate(("sin_pos", "sin_neg")):
            key = "{}_k{}".format(name, k + 1)
            output_neuron = nodes[i0 + offset]
            if connection_builder is None:
                for source_idx, source in enumerate(ring_neurons):
                    backend.Connect(
                        source,
                        output_neuron,
                        syn_spec={
                            "weight": float(
                                output_weight_scale * matrix[source_idx, i0 + offset]
                            )
                        },
                    )
            else:
                connection_builder(output_neuron, matrix[:, i0 + offset])
            named_nodes[key] = output_neuron
            named_recorders[key + "_recs"] = recorders[i0 + offset]
    return FourierReadout(
        neurons=named_nodes,
        spike_recorders=named_recorders,
        all_neurons=nodes,
        all_recorders=recorders,
        dc_generator=dc,
        weights=matrix,
    )


def decoder_spike_count(backend: Any, readout: FourierReadout, recorder_name: str) -> int:
    events = recorder_events(backend, readout.spike_recorders[recorder_name])
    return len(events.get("times", []))
