import json
import os
import nest
import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")

from ring_component import RingAttractorComponent
from homeostasis import HomeostasisModel
from tiago_ring_controller.config import load_gain_spec, load_neuron_parameters
from tiago_ring_controller.nest.gain import (
	GainNetwork,
	connect_gain_feedback,
	connect_gain_network,
)
from tiago_ring_controller.nest.comparator import DecisionCircuit

class GainModulationModel:
	"""
	AND-gate gain neurons: fire only when the ring bump AND the winning homeostasis
	neuron (left or right) are both active. Cross-inhibition silences the losing side.
	"""

	def __init__(
		self,
		ring_attractor: RingAttractorComponent,
		homeostasis: HomeostasisModel,
	):
		src_dir = os.path.dirname(__file__)

		params = load_gain_spec(
			os.path.join(src_dir, "config", "model_params", "gain_modulation_params.json")
		)
		self._gain_spec = params

		self.gain_neuron_params = load_neuron_parameters(
			os.path.join(src_dir, "config", "model_params", "neuron_params.json"),
			"gain",
		)

		self.ring_attractor = ring_attractor
		self.homeostasis = homeostasis
		self.population_size = ring_attractor.population_size

		self.left_homeostasis_gain_weight  = params.left_homeostasis_gain_weight
		self.right_homeostasis_gain_weight = params.right_homeostasis_gain_weight
		self.ring_to_gain_weight           = params.ring_to_gain_weight
		self.gain_to_ring_weight           = params.gain_to_ring_weight
		self.cross_inhibition_weight       = params.cross_inhibition_weight

		self._build_gain_modulation()
		self._connect_gain_modulation()

	def _build_gain_modulation(self):
		"""Create left and right gain neuron populations with spike recorders."""
		self.left_gain_neurons, self.left_gain_spike_recorders = (
			self._create_population()
		)
		self.right_gain_neurons, self.right_gain_spike_recorders = (
			self._create_population()
		)
		self._gain_network = GainNetwork(
			self.left_gain_neurons,
			self.right_gain_neurons,
			self.left_gain_spike_recorders,
			self.right_gain_spike_recorders,
		)

	def _create_population(self):
		"""Create one population of gain neurons and attach a recorder to each."""
		neurons, recorders = [], []
		for _ in range(self.population_size):
			neuron   = nest.Create("iaf_psc_alpha", params=self.gain_neuron_params)
			recorder = nest.Create("spike_recorder")
			nest.Connect(neuron, recorder)
			neurons.append(neuron)
			recorders.append(recorder)
		return neurons, recorders

	def _connect_gain_modulation(self):
		"""Wire homeostasis → gain and ring → gain connections."""
		gain_network = GainNetwork(
			self.left_gain_neurons,
			self.right_gain_neurons,
			self.left_gain_spike_recorders,
			self.right_gain_spike_recorders,
		)
		decision_circuit = DecisionCircuit(
			neurons=self.homeostasis.homeostasis_neurons,
			spike_recorders=getattr(self.homeostasis, "homeostasis_recorders", {}),
		)
		spec = replace(
			self._gain_spec,
			left_homeostasis_gain_weight=self.left_homeostasis_gain_weight,
			right_homeostasis_gain_weight=self.right_homeostasis_gain_weight,
			ring_to_gain_weight=self.ring_to_gain_weight,
			gain_to_ring_weight=self.gain_to_ring_weight,
			cross_inhibition_weight=self.cross_inhibition_weight,
		)
		connect_gain_network(
			nest,
			gain_network,
			self.ring_attractor.ring_attractor.ring_neurons,
			decision_circuit,
			spec,
		)

	def _connect_gain_modulation_to_ring(self):
		"""Feed gain output back into the ring attractor (optional)."""
		gain_network = GainNetwork(
			self.left_gain_neurons,
			self.right_gain_neurons,
			self.left_gain_spike_recorders,
			self.right_gain_spike_recorders,
		)
		connect_gain_feedback(
			nest,
			gain_network,
			self.ring_attractor.ring_attractor.ring_neurons,
			self.gain_to_ring_weight,
		)
