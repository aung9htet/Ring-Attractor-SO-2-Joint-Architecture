#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import nest

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([os.path.join(src_dir, "builders"), src_dir])

from builders.ring_attractor import RingAttractor
from helpers import FileManager


class RingAttractorAnalysis:
    """Test and analyze ring attractor network."""

    def __init__(self):
        """Initialize ring attractor and inject stimulus."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "model_params",
            "neuron_params.json",
        )
        self.ring_attractor = RingAttractor(params_file=config_path)
        self.ring_attractor.inject_stimulus(center_index=0, half_width=5)
        self.simulation_time_ms = 10000
        self.file_manager = FileManager()

    def evaluate(self):
        """Run simulation and generate plot."""
        nest.Simulate(self.simulation_time_ms)
        output_dir = self.file_manager.get_next_results_dir("ring_attractor", __file__)
        self.plot(output_dir)

    def plot(self, output_dir):
        """Create and save raster plot of neural activity."""
        fig, ax = plt.subplots(figsize=(14, 6))
        for neuron_idx, recorder in enumerate(self.ring_attractor.spike_recorders):
            spike_times = self.ring_attractor.get_spike_times(recorder)
            ax.plot(spike_times, [neuron_idx] * len(spike_times), ".", color="#1f77b4", markersize=2)

        ax.set_ylim([0, self.ring_attractor.population_size])
        ax.set_xlabel("Time (ms)", fontsize=20)
        ax.set_ylabel("Neuron Index", fontsize=20)
        ax.set_title("Ring Attractor Raster Plot", fontsize=22)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "raster_plot.png"), dpi=150)
        plt.close()

        print(f"Saved raster plot to: {os.path.join(output_dir, 'raster_plot.png')}")


if __name__ == "__main__":
    analysis = RingAttractorAnalysis()
    analysis.evaluate()
    
