#!/usr/bin/env python3

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

from helpers import FileManager
from tiago_ring_controller.artifacts import load_numpy_legacy


class DecoderAnalysis:
    """Analyze and visualize decoder training results."""

    def __init__(self, results_file: str):
        """Load training results from file."""
        data = load_numpy_legacy(results_file, allow_pickle=True)
        self.phi_true = data['phi_true']
        self.phi_est = data['phi_est']
        self.results_file = results_file
        self.file_manager = FileManager()

    def visualize(self):
        """Generate all analysis plots."""
        save_dir = self.file_manager.get_analysis_results_dir(self.results_file, "single_joint_ring_component")
        self._plot_readout_activity(save_dir)
        self._plot_error_distribution(save_dir)

    def _plot_readout_activity(self, save_dir: str):
        """Plot sin readout neuron activity."""
        order = np.argsort(self.phi_true)
        rc = {
            "font.family": "sans-serif", "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 12,
            "axes.linewidth": 1.2,
            "axes.spines.top": False, "axes.spines.right": False,
            "xtick.direction": "out", "ytick.direction": "out",
            "legend.frameon": False, "legend.fontsize": 10,
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.phi_true[order], label="Ground truth")
            ax.plot(self.phi_est[order], label="Decoded")
            ax.set_xlabel(r"Position")
            ax.set_ylabel(r"Phase (rad)")
            ax.legend()
            ax.grid(axis="y", alpha=0.25, lw=0.8)
            fig.savefig(os.path.join(save_dir, "readout_comparison.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _plot_error_distribution(self, save_dir: str):
        """Plot phase decoding error."""
        def circ_err(est, true):
            e = np.angle(np.exp(1j * (est - true)))
            return e, np.abs(e)

        _, abs_err = circ_err(self.phi_est, self.phi_true)

        rc = {
            "font.family": "sans-serif", "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 12,
            "axes.linewidth": 1.2,
            "axes.spines.top": False, "axes.spines.right": False,
            "xtick.direction": "out", "ytick.direction": "out",
            "legend.frameon": False, "legend.fontsize": 10,
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(abs_err, bins=20, color="#2980b9", alpha=0.7, edgecolor="black")
            ax.axvline(np.mean(abs_err), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(abs_err):.3f}")
            ax.set_xlabel(r"Absolute phase error $|\Delta\varphi|$ (rad)")
            ax.set_ylabel("Frequency")
            ax.set_title("Phase decoding error distribution")
            ax.legend()
            ax.grid(axis="y", alpha=0.25, lw=0.8)
            fig.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        print(f"Mean error: {np.mean(abs_err):.3f} rad")
        print(f"Max error: {np.max(abs_err):.3f} rad")


if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(src_dir)
    outputs_dir = os.path.join(project_root, "outputs/train/single_joint_ring_component")
    
    if os.path.exists(outputs_dir):
        results_dirs = sorted([d for d in os.listdir(outputs_dir) if d.startswith("results_")])
        for results_dir_name in results_dirs:
            results_file = os.path.join(outputs_dir, results_dir_name, "fourier_results.npz")
            if os.path.exists(results_file):
                analysis = DecoderAnalysis(results_file)
                analysis.visualize()
                print(f"Analysis completed for {results_dir_name}")
