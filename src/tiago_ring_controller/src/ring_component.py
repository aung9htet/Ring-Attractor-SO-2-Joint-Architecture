import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest
from ring_attractor import Ring_Attractor

from tiago_ring_controller.artifacts import load_numpy_legacy
from tiago_ring_controller.config import load_json
from tiago_ring_controller.math.circular import index_to_phase
from tiago_ring_controller.math.ring import generate_center_indices as _generate_center_indices
from tiago_ring_controller.nest.readout import build_fourier_readout


class RingAttractorComponent:
    """
    Sets up the ring attractor with output neurons.
    """

    @staticmethod
    def center_index_to_phase(center_idx: int, N: int) -> float:
        """Map a neuron index to its angular phase in [0, 2π)."""
        return index_to_phase(center_idx, N)

    @staticmethod
    def generate_center_indices(N: int, num_positions: int) -> np.ndarray:
        """Return num_positions evenly-spaced non-zero neuron indices around the ring.
        Generates num_positions+1 points and drops the first (index 0)."""
        return _generate_center_indices(N, num_positions)

    def __init__(self, params_file: str = "./config/model_params/ring_params.json",
                 weights_dir: str = "./config/ring_decoding_weights",
                 reset_kernel: bool = True):
        params = load_json(params_file)
        self.params_file = params_file
        self.num_fourier_k = params["num_fourier_k"]
        self.output_weight_scale = params["readout_weight_scale"]
        self.output_dc_baseline = params["output_dc_baseline"]
        self.harmonic_index = params.get("harmonic_index", 0)
        self.population_size = params["population_size"]
        N = self.population_size
        neuron_params_file = os.path.join(os.path.dirname(params_file), "neuron_params.json")
        self.ring_attractor = Ring_Attractor(population_size=N, reset_kernel=reset_kernel, params_file=neuron_params_file)

        weights_path = os.path.join(weights_dir, f"N_{N}_fourier_weights.npy")
        self._weights = load_numpy_legacy(weights_path)  # shape (N, 2K): sin_pos/sin_neg for each harmonic k=1..K
        expected_shape = (N, 2 * self.num_fourier_k)
        if self._weights.shape != expected_shape:
            raise ValueError(
                f"Fourier weights shape mismatch for N={N}: got {self._weights.shape}, expected {expected_shape}. "
                "Check ring_params num_fourier_k and retrain with train_ring_model.py."
            )

        self._build_decoded_layer()

    def _create_output_neurons(self, n: int):
        nodes = nest.Create('iaf_psc_alpha', n)
        recs = nest.Create('spike_recorder', n)
        dc = nest.Create('dc_generator', params={'amplitude': self.output_dc_baseline})
        nest.Connect(dc, nodes)
        for i in range(n):
            nest.Connect(nodes[i], recs[i])
        return nodes, recs

    def _connect_from_ring_with_weights(self, out_neuron, weights: np.ndarray):
        for i, src in enumerate(self.ring_attractor.ring_neurons):
            w = float(self.output_weight_scale * weights[i])
            nest.Connect(src, out_neuron, syn_spec={'weight': w})

    def _build_decoded_layer(self):
        """
        Builds the layer of 2K decoded neurons (sin_pos/sin_neg per harmonic),
        connecting them to the ring attractor with the appropriate Fourier weights.
        """
        readout = build_fourier_readout(
            nest,
            self.ring_attractor.ring_neurons,
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

    def _get_decoder_spike_counts(self, recorder_name: str) -> int:
        """Get spike count for a single decoded neuron by recorder name.

        Args:
            recorder_name: one of 'cos_pos_recs', 'cos_neg_recs',
                           'sin_pos_recs', 'sin_neg_recs'
        """
        rec = self.decoded_neuron_recorders[recorder_name]
        ev = rec.get("events")
        return len(ev.get("times", []))

    def _get_ring_spike_counts(self) -> np.ndarray:
        return self.ring_attractor._get_spike_counts()

    def _inject_bump(self, center_idx: int, half_width: int):
        self.ring_attractor.inject_stimulus(center_index=center_idx, half_width=half_width)
