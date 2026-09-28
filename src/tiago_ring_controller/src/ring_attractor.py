#!/usr/bin/env python3

import nest
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiago_ring_controller.math.ring import (
    legacy_ring_distances,
    legacy_ring_weight,
    stimulus_target_indices,
)
from tiago_ring_controller.config import load_neuron_parameters
from tiago_ring_controller.nest.ring import (
    build_ring_network,
    inject_stimulus as inject_ring_stimulus,
)

class Ring_Attractor:

    def __init__(self, population_size=100, reset_kernel=True,
                 params_file="./config/model_params/neuron_params.json"):
        if reset_kernel:
            nest.ResetKernel()
        nest.set_verbosity("M_ERROR")

        self.population_size = population_size
        self.max_distance = 50
        self.sd_1 = 10
        self.sd_2 = 5

        self.init_parameters(params_file)
        self.init_ring_attractor()

    def init_parameters(self, params_file):
        """
            Initializes the neuron parameters for the ring attractor from a JSON file
        """
        self.ring_parameters = load_neuron_parameters(params_file, "ring")

    def init_ring_attractor(self):
        """
            Initializes the ring attractor
        """

        network = build_ring_network(
            nest,
            self.population_size,
            self.ring_parameters,
            variant="legacy",
            reset_kernel=False,
            max_distance=self.max_distance,
            excitation_std_dev=self.sd_1,
            inhibition_std_dev=self.sd_2,
            configure_backend=False,
            distance_function=self.neuron_distance,
            weight_function=self.get_weight,
        )
        self.ring_neurons = network.neurons
        self.ring_spike_recorders = network.spike_recorders

    def neuron_distance(self):
        """
            The following method determines how the distance should be distributed between
            each neurons and returns an array of these distances.
        """
        return legacy_ring_distances(self.population_size, self.max_distance)

    def get_weight(self, distance):
        """
            The following method calculates the weight where n1 is pre-synaptic and n2 is
            post-synaptic.
        """
        return legacy_ring_weight(distance, self.sd_1, self.sd_2)

    def inject_stimulus(self, center_index=0, half_width=5):
        network = type("_LegacyRingView", (), {})()
        network.population_size = self.population_size
        network.neurons = self.ring_neurons
        inject_ring_stimulus(
            nest,
            network,
            center_index=center_index,
            half_width=half_width,
        )

    def _get_spike_counts(self):
        return np.array(
            [len(nest.GetStatus(sr, "events")[0].get("times", [])) for sr in self.ring_spike_recorders],
            dtype=float,
        )

class RingAttractorAnalysis:

    def __init__(self):
        self.ring_attractor = Ring_Attractor()
        self.ring_attractor.inject_stimulus(center_index=0, half_width=5)
        self.run_time = 10000
    
    def evaluate(self):
        nest.Simulate(self.run_time)
        results = self.ring_attractor._get_spike_counts()
        self.plot(results)
    
    def plot(self, results):
        output_dir = "outputs/ring_attractor"
        os.makedirs(output_dir, exist_ok=True)

        # Raster plot
        fig, ax = plt.subplots(figsize=(14, 6))
        for index, sr in enumerate(self.ring_attractor.ring_spike_recorders):
            times = np.array(nest.GetStatus(sr, "events")[0].get("times", []))
            ax.plot(times, np.full(len(times), index), ".", color="#1f77b4", markersize=2)
        ax.set_ylim([0, self.ring_attractor.population_size])
        ax.set_xlabel("Time (ms)", fontsize=20)
        ax.set_ylabel("Neuron Index", fontsize=20)
        ax.set_title("Ring Attractor Raster", fontsize=22)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "raster.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    analysis = RingAttractorAnalysis()
    analysis.evaluate()
    
