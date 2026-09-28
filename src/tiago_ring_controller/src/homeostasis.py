import json
import os
import nest
import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")

from ring_component import RingAttractorComponent
from tiago_ring_controller.artifacts import load_json_legacy, load_numpy_legacy
from tiago_ring_controller.config import (
	load_homeostasis_spec,
	load_neuron_parameters,
)
from tiago_ring_controller.nest.comparator import (
	DecisionCircuit,
	build_decision_circuit,
	connect_ring_feature_vectors,
)

class HomeostasisModel:
	"""
	Decision circuit: compares two ring attractor phases and drives left/right output.
	Ring 1 = current state, Ring 2 = reference target.
	Train weights first with HomeostasisTrainer (train_homeostasis.py).
	"""

	def __init__(
		self,
		ring_1: RingAttractorComponent,
		ring_2: RingAttractorComponent,
	):
		src_dir = os.path.dirname(__file__)

		params = load_homeostasis_spec(
			os.path.join(src_dir, "config", "model_params", "homeostasis_params.json")
		)
		self._homeostasis_spec = params

		self._ring_1 = ring_1
		self._ring_2 = ring_2
		self.population_size = ring_1.population_size
		self.config_dir = os.path.join(src_dir, params.config_dir)
		self.tie_epsilon = params.tie_epsilon
		self.warm_cold_weight_scale = params.warm_cold_weight_scale
		self.warm_exc_weight = params.warm_exc_weight
		self.warm_inh_weight = params.warm_inh_weight
		self.cold_exc_weight = params.cold_exc_weight
		self.cold_inh_weight = params.cold_inh_weight
		self.decision_lateral_weight = params.decision_lateral_weight
		self.left_label = params.left_label
		self.right_label = params.right_label

		self.neuron_params = load_neuron_parameters(
			os.path.join(src_dir, "config", "model_params", "neuron_params.json"),
			"homeostasis",
		)

		# Trained weights shape (4K, 2): col 0 = warm, col 1 = cold
		N = self.population_size
		weights_path = os.path.join(self.config_dir, f"N_{N}_homeostasis_weights.npy")
		W = load_numpy_legacy(weights_path)
		self._weight_matrix = W
		self.warm_weights = W[:, 0]
		self.cold_weights = W[:, 1]

		metadata_path = os.path.join(self.config_dir, f"N_{N}_homeostasis_metadata.json")
		meta = load_json_legacy(metadata_path)
		self.warm_bias = float(meta["warm_bias"])
		self.cold_bias = float(meta["cold_bias"])
		self.feature_names = meta["feature_names"]

		n_feat = len(self.feature_names)  # 2K
		expected_w_shape = (2 * n_feat, 2)  # 4K x 2
		if W.shape != expected_w_shape:
			raise ValueError(
				f"Homeostasis weights shape mismatch for N={N}: got {W.shape}, expected {expected_w_shape}. "
				"Ensure ring_params num_fourier_k matches training and rerun train_homeostasis.py."
			)

		self._build_homeostasis()
		self._connect_ring_to_homeostasis()

	def _build_homeostasis(self):
		"""Build the 4-neuron warm/cold/left/right decision circuit."""
		spec = replace(
			self._homeostasis_spec,
			warm_exc_weight=self.warm_exc_weight,
			warm_inh_weight=self.warm_inh_weight,
			cold_exc_weight=self.cold_exc_weight,
			cold_inh_weight=self.cold_inh_weight,
			decision_lateral_weight=self.decision_lateral_weight,
		)
		circuit = build_decision_circuit(
			nest,
			self.neuron_params,
			spec,
		)
		self._decision_circuit = circuit
		self.homeostasis_neurons = circuit.neurons
		self.homeostasis_recorders = circuit.spike_recorders

	def _connect_ring_to_homeostasis(self):
		"""Connect ring_1 decoded neurons → warm and ring_2 decoded neurons → cold."""
		circuit = DecisionCircuit(
			neurons=self.homeostasis_neurons,
			spike_recorders=self.homeostasis_recorders,
		)
		connect_ring_feature_vectors(
			nest,
			circuit,
			self._ring_1.decoded_neurons,
			self._ring_2.decoded_neurons,
			self.feature_names,
			self.warm_weights,
			self.cold_weights,
			self.warm_cold_weight_scale,
		)

	def evaluate(self, duration_ms: float) -> dict:
		"""Simulate for duration_ms and return {'left_spike_count': int, 'right_spike_count': int}."""
		nest.Simulate(duration_ms)

		left_events = self.homeostasis_recorders["left_spike"].get("events")
		right_events = self.homeostasis_recorders["right_spike"].get("events")

		return {
			"left_spike_count": len(left_events.get("times", [])),
			"right_spike_count": len(right_events.get("times", [])),
		}
