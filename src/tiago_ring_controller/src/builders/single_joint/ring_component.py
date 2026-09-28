import json
import os
import sys

import numpy as np
import nest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from builders.ring_attractor import RingAttractor
from tiago_ring_controller.artifacts import load_numpy_legacy
from tiago_ring_controller.config import load_json
from tiago_ring_controller.nest.readout import build_fourier_readout


class SingleJointRingDecoder:
    """Decoder layer for ring attractor with Fourier readout neurons."""

    def __init__(self, ring_attractor: RingAttractor, num_fourier_k: int,
                 output_weight_scale: float, output_dc_baseline: float, weights: np.ndarray):
        self.ring_attractor = ring_attractor
        self.num_fourier_k = num_fourier_k
        self.output_weight_scale = output_weight_scale
        self.output_dc_baseline = output_dc_baseline
        self._weights = weights

        self.decoded_neurons = {}
        self.decoded_neuron_recorders = {}
        self._build_decoded_layer()

    def _create_output_neurons(self, n: int):
        """Create n output neurons with spike recorders and DC baseline."""
        nodes = nest.Create('iaf_psc_alpha', n)
        recs = nest.Create('spike_recorder', n)
        dc = nest.Create('dc_generator', params={'amplitude': self.output_dc_baseline})
        nest.Connect(dc, nodes)
        for i in range(n):
            nest.Connect(nodes[i], recs[i])
        return nodes, recs

    def _connect_from_ring_with_weights(self, out_neuron, weights: np.ndarray):
        for i, src in enumerate(self.ring_attractor.neurons):
            w = float(self.output_weight_scale * weights[i])
            nest.Connect(src, out_neuron, syn_spec={'weight': w})

    def _build_decoded_layer(self):
        """Build decoded output neurons with Fourier weights."""
        readout = build_fourier_readout(
            nest,
            self.ring_attractor.neurons,
            self._weights,
            self.num_fourier_k,
            self.output_weight_scale,
            self.output_dc_baseline,
            validate_shape=False,
            population_factory=self._create_output_neurons,
            connection_builder=self._connect_from_ring_with_weights,
        )
        self.decoded_neurons = readout.neurons
        self.decoded_neuron_recorders = readout.spike_recorders

    def get_spike_counts(self, recorder_name: str) -> int:
        """Get spike count for a decoded neuron by recorder name."""
        rec = self.decoded_neuron_recorders[recorder_name]
        ev = rec.get("events")
        return len(ev.get("times", []))


class SingleJointRingComponent:
    """Ring attractor with decoded output neurons."""

    def __init__(self, params_file: str = None,
                 weights_dir: str = None,
                 weights: np.ndarray = None,
                 reset_kernel: bool = True):
        if params_file is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            params_file = os.path.join(script_dir, "config/model_params/ring_params.json")
        
        if weights_dir is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            weights_dir = os.path.join(script_dir, "config/ring_decoding_weights")
        
        params = load_json(params_file)

        population_size = params["population_size"]
        neuron_params_file = os.path.join(os.path.dirname(params_file), "neuron_params.json")
        
        self.ring_attractor = RingAttractor(
            population_size=population_size,
            reset_kernel=reset_kernel,
            params_file=neuron_params_file,
        )

        if weights is None:
            weights_path = os.path.join(weights_dir, f"N_{population_size}_fourier_weights.npy")
            weights = load_numpy_legacy(weights_path)

        self.decoder = SingleJointRingDecoder(
            ring_attractor=self.ring_attractor,
            num_fourier_k=params["num_fourier_k"],
            output_weight_scale=params["readout_weight_scale"],
            output_dc_baseline=params["output_dc_baseline"],
            weights=weights,
        )
