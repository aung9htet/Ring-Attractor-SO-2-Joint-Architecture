#!/usr/bin/env python3

import os
import json
import time
import shutil
import resource
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import nest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.ndimage import gaussian_filter

from ring_component import RingAttractorComponent
from homeostasis import HomeostasisModel
from gain_modulation import GainModulationModel
from tiago_ring_controller.config import load_json
from tiago_ring_controller.evaluation.metrics import (
    centroid_change,
    goal_error as evaluation_goal_error,
    relative_goal_error_after_time,
)
from tiago_ring_controller.evaluation.plots import break_circular_wraps_for_plot
from tiago_ring_controller.math.circular import (
    circular_signed_difference,
    decode_circular_centroid,
)
from tiago_ring_controller.nest.kernel import configure_kernel


# =====================================================================================
# BENCHMARK PARAMETERS -- edit these to change what is swept/reported.
# -------------------------------------------------------------------------------------
# Methodology follows NEST's official HPC benchmark
# (https://nest-simulator.readthedocs.io/en/v3.5/auto_examples/hpc_benchmark.html):
# separate wall-clock timers for build/presim/sim, peak memory sampled at each phase,
# and a real-time factor (wall-clock / simulated time). We ADD a spike-based "energy"
# proxy on top, since NEST/CPU simulation cannot report real Joules -- see
# ENERGY_PER_SYNOP_PJ below. Every sweep point's settings are ALSO written to disk
# (benchmark_settings.json) so any plot can be traced back to the exact parameters used.
# =====================================================================================

# --- Scale sweep (population size) ---------------------------------------------------
# Only N=100 and N=200 have pretrained Fourier readout weights under
# ./config/ring_decoding_weights/ (N_100_fourier_weights.npy, N_200_fourier_weights.npy).
# Testing other N requires retraining via train_ring_model.py first -- out of scope here.
# Do NOT add sizes without a matching pretrained weights file; RingAttractorComponent
# will raise a shape-mismatch error.
BENCHMARK_POPULATION_SIZES = [100, 200]

# --- Thread-count sweep (NEST local_num_threads) --------------------------------------
BENCHMARK_THREAD_COUNTS = [1, 2, 4, 8, 16]  # edit freely; keep <= os.cpu_count()

# --- Repeats -----------------------------------------------------------------------
# Repeats run SEQUENTIALLY (never in a process pool): parallel workers would contend
# for CPU cores and corrupt the wall-clock / thread-scaling timing signal we're
# measuring, which is the whole point of this benchmark.
BENCHMARK_N_RUNS_PER_CONDITION = 5

# --- Stimulus placement (as a fraction of population_size, so indices scale with N) ---
BENCHMARK_R1_FRACTION = 0.2      # start bump position, as a fraction of N
BENCHMARK_GOAL_FRACTION = 0.7    # goal bump position, as a fraction of N
BENCHMARK_STIMULUS_HALF_WIDTH = 5

# --- Timed windows ---------------------------------------------------------------------
BENCHMARK_PRESIM_MS = 300.0          # settle time before the timed window (matches this model's own settle_t)
BENCHMARK_SIMTIME_MS = 100000.0      # main timed simulate window; edit to shorten benchmark runtime
BENCHMARK_AVG_AFTER_T_MS = 30000.0   # transient cutoff for the task-performance metric

# --- Energy proxy ----------------------------------------------------------------------
# NEST/CPU simulation cannot report real Joules. As a proxy, we estimate the number of
# synaptic operations (SynOps) triggered by spiking activity -- total_spikes * mean
# fan-out per neuron -- and scale it by an assumed energy-per-SynOp constant, a common
# approach in neuromorphic-hardware energy estimates (e.g. ~23.6 pJ/SynOp is a commonly
# cited figure for Intel Loihi). EDIT this constant to match whatever hardware
# assumption you want to compare against; it does not reflect this CPU's actual power draw.
ENERGY_PER_SYNOP_PJ = 23.6

BENCHMARK_OUTPUT_ROOT = "./outputs/benchmark_single_ring"
BENCHMARK_RING_PARAMS_FILE = "./config/model_params/ring_params.json"
BENCHMARK_WEIGHTS_DIR = "./config/ring_decoding_weights"


def _run_goal_distance_worker(args):
    (
        run_idx,
        seed,
        r1_inject,
        goal_idx,
        sim_t,
        sample_ms,
        ring_params_file,
        weights_dir,
        font_size,
        font_weight,
        stimulus_half_width,
    ) = args

    model = SingleRingModel(
        ring_params_file=ring_params_file,
        weights_dir=weights_dir,
        font_size=font_size,
        font_weight=font_weight,
        seed=seed,
    )

    model.setup(
        r1_inject=r1_inject,
        r2_inject=goal_idx,
        stimulus_half_width=stimulus_half_width,
    )

    nest.Simulate(sim_t)

    bin_centers, goal_distance = model.get_goal_distance_trace(
        goal_idx=goal_idx,
        bin_ms=sample_ms,
        t_max=sim_t,
    )

    return run_idx, seed, bin_centers, goal_distance


def _run_relative_goal_error_worker(args):
    (
        goal_i,
        run_idx,
        goal_idx,
        seed,
        r1_inject,
        sim_t,
        avg_after_t,
        sample_ms,
        ring_params_file,
        weights_dir,
        font_size,
        font_weight,
        stimulus_half_width,
    ) = args

    model = SingleRingModel(
        ring_params_file=ring_params_file,
        weights_dir=weights_dir,
        font_size=font_size,
        font_weight=font_weight,
        seed=seed,
    )

    model.setup(
        r1_inject=r1_inject,
        r2_inject=int(goal_idx),
        stimulus_half_width=stimulus_half_width,
    )

    nest.Simulate(sim_t)

    avg_relative_error = model.get_relative_goal_error_after_time(
        goal_idx=int(goal_idx),
        bin_ms=sample_ms,
        t_max=sim_t,
        avg_after_t=avg_after_t,
    )

    return goal_i, run_idx, int(goal_idx), seed, float(avg_relative_error)


def _style_axis_inline(ax, font_size, font_weight, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, fontsize=font_size, fontweight=font_weight)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=font_size, fontweight=font_weight)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=font_size, fontweight=font_weight)
    ax.tick_params(axis="both", labelsize=font_size)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(font_size)
        tick.set_fontweight(font_weight)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(font_size)
        tick.set_fontweight(font_weight)


def _style_legend_inline(ax, font_size, font_weight, loc="upper right"):
    leg = ax.legend(loc=loc, fontsize=font_size)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(font_size)
            txt.set_fontweight(font_weight)


def _safe_positive_percentile(values, percentile, fallback):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float(fallback)
    p = float(np.percentile(valid, percentile))
    if not np.isfinite(p) or p <= 0.0:
        return float(fallback)
    return p


class SingleRingModel:

    def __init__(
        self,
        ring_params_file="./config/model_params/ring_params.json",
        weights_dir="./config/ring_decoding_weights",
        font_size=25,
        font_weight="bold",
        seed=None,
        local_num_threads=1,
    ):
        configure_kernel(
            nest,
            reset_kernel=True,
            verbosity="M_ERROR",
            local_num_threads=local_num_threads,
            rng_seed=seed,
        )

        self.r1 = RingAttractorComponent(
            params_file=ring_params_file,
            weights_dir=weights_dir,
            reset_kernel=False,
        )

        self.r2 = RingAttractorComponent(
            params_file=ring_params_file,
            weights_dir=weights_dir,
            reset_kernel=False,
        )

        self.homeostasis = HomeostasisModel(self.r1, self.r2)
        self.gain_modulation = GainModulationModel(self.r1, self.homeostasis)

        self.gain_modulation._connect_gain_modulation_to_ring()

        self.population_size = self.r1.population_size

        self.font_size = font_size
        self.font_weight = font_weight

    def setup(self, r1_inject, r2_inject, stimulus_half_width=5):
        self.r1._inject_bump(r1_inject, stimulus_half_width)
        self.r2._inject_bump(r2_inject, stimulus_half_width)

    def _collect_spikes(self, recorders):
        return [
            np.array(nest.GetStatus(sr, "events")[0].get("times", []))
            for sr in recorders
        ]

    def _style_axis(self, ax, title=None, xlabel=None, ylabel=None):
        if title is not None:
            ax.set_title(
                title,
                fontsize=self.font_size,
                fontweight=self.font_weight,
            )

        if xlabel is not None:
            ax.set_xlabel(
                xlabel,
                fontsize=self.font_size,
                fontweight=self.font_weight,
            )

        if ylabel is not None:
            ax.set_ylabel(
                ylabel,
                fontsize=self.font_size,
                fontweight=self.font_weight,
            )

        ax.tick_params(axis="both", labelsize=self.font_size)

        for tick in ax.get_xticklabels():
            tick.set_fontsize(self.font_size)
            tick.set_fontweight(self.font_weight)

        for tick in ax.get_yticklabels():
            tick.set_fontsize(self.font_size)
            tick.set_fontweight(self.font_weight)

    def _style_legend(self, ax, loc="upper right"):
        leg = ax.legend(loc=loc, fontsize=self.font_size)
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(self.font_size)
                txt.set_fontweight(self.font_weight)

    def _style_colorbar(self, cbar, label):
        cbar.set_label(
            label,
            fontsize=self.font_size,
            fontweight=self.font_weight,
        )

        cbar.ax.tick_params(labelsize=self.font_size)

        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(self.font_size)
            tick.set_fontweight(self.font_weight)

    def plot(
        self,
        r1_inject,
        r2_inject,
        output_dir="./outputs/single_ring",
        settle_t=300.0,
    ):
        os.makedirs(output_dir, exist_ok=True)

        ring_spikes = self._collect_spikes(
            self.r1.ring_attractor.ring_spike_recorders
        )

        left_spikes = self._collect_spikes(
            self.gain_modulation.left_gain_spike_recorders
        )

        right_spikes = self._collect_spikes(
            self.gain_modulation.right_gain_spike_recorders
        )

        homeo_times = {
            label: np.array(
                self.homeostasis.homeostasis_recorders[key]
                .get("events")
                .get("times", [])
            )
            for label, key in [
                ("warm", "warm_spike"),
                ("cold", "cold_spike"),
                ("LEFT", "left_spike"),
                ("RIGHT", "right_spike"),
            ]
        }

        fig, axes = plt.subplots(4, 1, figsize=(16, 22), sharex=True)
        ax_ring, ax_homeo, ax_left, ax_right = axes

        for idx, times in enumerate(ring_spikes):
            ax_ring.plot(
                times,
                np.full(len(times), idx),
                ".",
                color="#1f77b4",
                markersize=1.5,
            )

        ax_ring.axhline(
            r1_inject,
            color="gray",
            lw=0.8,
            ls="--",
            alpha=0.6,
            label="start idx",
        )

        ax_ring.axhline(
            r2_inject,
            color="orange",
            lw=0.8,
            ls="--",
            alpha=0.8,
            label="goal idx",
        )

        ax_ring.axvline(
            settle_t,
            color="black",
            lw=1.2,
            ls=":",
            alpha=0.7,
            label="settle boundary",
        )

        ax_ring.set_ylim([-1, self.population_size - 1])

        self._style_axis(
            ax_ring,
            title=f"Ring-1 Attractor (start idx {r1_inject}, goal idx {r2_inject})",
            ylabel="Neuron Index",
        )
        self._style_legend(ax_ring)

        palette = {
            "warm": "#ff7f0e",
            "cold": "#2ca02c",
            "LEFT": "#d62728",
            "RIGHT": "#9467bd",
        }

        y_pos = {
            "warm": 0,
            "cold": 1,
            "LEFT": 2,
            "RIGHT": 3,
        }

        for label, times in homeo_times.items():
            ax_homeo.plot(
                times,
                np.full(len(times), y_pos[label]),
                "|",
                color=palette[label],
                markersize=8,
                label=label,
            )

        ax_homeo.axvline(
            settle_t,
            color="black",
            lw=1.2,
            ls=":",
            alpha=0.7,
        )

        ax_homeo.set_yticks([0, 1, 2, 3])
        ax_homeo.set_yticklabels(["warm", "cold", "LEFT", "RIGHT"])
        ax_homeo.set_ylim([-0.5, 3.5])

        self._style_axis(
            ax_homeo,
            title="Homeostasis Decision Neurons",
            ylabel="Unit",
        )
        self._style_legend(ax_homeo)

        for idx, times in enumerate(left_spikes):
            ax_left.plot(
                times,
                np.full(len(times), idx),
                ".",
                color="#d62728",
                markersize=1.5,
            )

        ax_left.axhline(
            r1_inject,
            color="gray",
            lw=0.8,
            ls="--",
            alpha=0.6,
        )

        ax_left.axhline(
            r2_inject,
            color="orange",
            lw=0.8,
            ls="--",
            alpha=0.8,
        )

        ax_left.axvline(
            settle_t,
            color="black",
            lw=1.2,
            ls=":",
            alpha=0.7,
        )

        ax_left.set_ylim([-1, self.population_size - 1])

        self._style_axis(
            ax_left,
            title="Left Gain (AND: ring bump ∩ left homeostasis)",
            ylabel="Neuron Index",
        )

        for idx, times in enumerate(right_spikes):
            ax_right.plot(
                times,
                np.full(len(times), idx),
                ".",
                color="#9467bd",
                markersize=1.5,
            )

        ax_right.axhline(
            r1_inject,
            color="gray",
            lw=0.8,
            ls="--",
            alpha=0.6,
        )

        ax_right.axhline(
            r2_inject,
            color="orange",
            lw=0.8,
            ls="--",
            alpha=0.8,
        )

        ax_right.axvline(
            settle_t,
            color="black",
            lw=1.2,
            ls=":",
            alpha=0.7,
        )

        ax_right.set_ylim([-1, self.population_size - 1])

        self._style_axis(
            ax_right,
            title="Right Gain (AND: ring bump ∩ right homeostasis)",
            xlabel="Time (ms)",
            ylabel="Neuron Index",
        )

        fig.suptitle(
            f"Single Ring | Start idx={r1_inject} · Goal idx={r2_inject}\n"
            f"(dotted line = settled at t={settle_t:.0f} ms)",
            fontsize=self.font_size,
            fontweight=self.font_weight,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = os.path.join(output_dir, "raster.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        n_left = sum(1 for t in left_spikes if np.any(t > settle_t))
        n_right = sum(1 for t in right_spikes if np.any(t > settle_t))

        print(f"  Left  gain active (t>{settle_t:.0f}ms): {n_left}/{self.population_size}")
        print(f"  Right gain active (t>{settle_t:.0f}ms): {n_right}/{self.population_size}")
        print(f"  Raster saved → {save_path}")

    def _spikes_to_rate_matrix(self, spikes, bin_ms=100.0, t_max=None):
        if t_max is None:
            nonempty = [t for t in spikes if len(t) > 0]
            if len(nonempty) > 0:
                t_max = max(np.max(t) for t in nonempty)
            else:
                t_max = bin_ms

        bins = np.arange(0.0, t_max + bin_ms, bin_ms)

        rate_matrix = np.zeros((len(spikes), len(bins) - 1), dtype=float)

        for neuron_idx, times in enumerate(spikes):
            counts, _ = np.histogram(times, bins=bins)
            rate_matrix[neuron_idx, :] = counts / (bin_ms / 1000.0)

        return rate_matrix, bins

    def _decode_circular_centroid(self, rate_matrix):
        return decode_circular_centroid(rate_matrix)

    def _circular_signed_difference(self, target_idx, current_idx):
        return circular_signed_difference(
            target_idx,
            current_idx,
            self.population_size,
        )

    def _break_wraps_for_plot(self, centroid_idx):
        return break_circular_wraps_for_plot(
            centroid_idx,
            self.population_size,
            preserve_input_type=True,
        )

    def _compute_centroid_change(self, centroid_idx):
        return centroid_change(
            centroid_idx,
            self.population_size,
            difference_function=self._circular_signed_difference,
            preserve_input_type=True,
        )

    def _compute_goal_error(self, centroid_idx, goal_idx):
        return evaluation_goal_error(
            centroid_idx,
            goal_idx,
            self.population_size,
            difference_function=self._circular_signed_difference,
            preserve_input_type=True,
        )

    def get_goal_distance_trace(self, goal_idx, bin_ms=5000.0, t_max=None):
        ring_spikes = self._collect_spikes(
            self.r1.ring_attractor.ring_spike_recorders
        )

        rate_matrix, bins = self._spikes_to_rate_matrix(
            ring_spikes,
            bin_ms=bin_ms,
            t_max=t_max,
        )

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        centroid_idx, _ = self._decode_circular_centroid(rate_matrix)

        _, goal_distance = self._compute_goal_error(
            centroid_idx=centroid_idx,
            goal_idx=goal_idx,
        )

        return bin_centers, goal_distance

    def get_relative_goal_error_after_time(
        self,
        goal_idx,
        bin_ms=5000.0,
        t_max=None,
        avg_after_t=30000.0,
    ):
        bin_centers, goal_distance = self.get_goal_distance_trace(
            goal_idx=goal_idx,
            bin_ms=bin_ms,
            t_max=t_max,
        )

        return relative_goal_error_after_time(
            bin_centers,
            goal_distance,
            self.population_size,
            avg_after_ms=avg_after_t,
            preserve_input_type=True,
        )

    def plot_ring_activity_heatmap(
        self,
        bins,
        bin_centers,
        rate_matrix,
        centroid_idx,
        r1_inject,
        goal_idx,
        output_dir,
    ):
        centroid_for_plot = self._break_wraps_for_plot(centroid_idx)

        fig, ax = plt.subplots(figsize=(16, 7))

        smoothed = gaussian_filter(rate_matrix, sigma=(1.5, 2.0))

        # Keep color scaling stable against isolated outlier bins.
        vmax = _safe_positive_percentile(
            smoothed,
            percentile=99.5,
            fallback=float(np.max(smoothed)) if smoothed.size > 0 else 1.0,
        )

        im = ax.imshow(
            smoothed,
            aspect="auto",
            origin="lower",
            extent=[bins[0], bins[-1], 0, self.population_size - 1],
            interpolation="nearest",
            cmap=cc.cm.fire,
            vmin=0.0,
            vmax=vmax,
        )

        ax.plot(
            bin_centers,
            centroid_for_plot,
            color="white",
            lw=1.8,
            label="circular centroid",
        )

        ax.axhline(r1_inject, color="cyan", lw=1.0, ls="--", alpha=0.8)
        ax.axhline(goal_idx, color="orange", lw=1.2, ls="--", alpha=0.9)

        ax.set_ylim(0, self.population_size - 1)
        ax.set_xlim(bins[0], bins[-1])

        self._style_axis(
            ax,
            title="Ring Attractor Activity Over Time",
            xlabel="Time (ms)",
            ylabel="Neuron Index",
        )

        self._style_legend(ax)

        cbar = plt.colorbar(im, ax=ax)
        self._style_colorbar(cbar, "Firing Rate (Hz)")

        plt.tight_layout()

        save_path = os.path.join(output_dir, "ring_activity_heatmap.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Ring activity heatmap saved → {save_path}")

    def plot_ring_centroid_change_over_time(
        self,
        bins,
        bin_centers,
        centroid_change,
        output_dir,
    ):
        fig, ax = plt.subplots(figsize=(16, 5))

        ax.plot(
            bin_centers,
            centroid_change,
            lw=1.8,
            label="Δ centroid",
        )

        ax.axhline(
            0.0,
            color="gray",
            lw=1.0,
            ls="--",
            alpha=0.8,
            label="zero change",
        )

        ax.set_xlim(bins[0], bins[-1])

        # Circular signed change is bounded by +/- N/2; use robust symmetric limits
        # to keep useful variation visible when a few bins are noisy.
        robust = _safe_positive_percentile(
            np.abs(centroid_change),
            percentile=99.0,
            fallback=1.0,
        )
        y_abs = min(self.population_size / 2.0, max(1.0, 1.1 * robust))
        ax.set_ylim(-y_abs, y_abs)

        self._style_axis(
            ax,
            title="Change in Ring Centroid Over Time",
            xlabel="Time (ms)",
            ylabel="Δ Centroid Index / Bin",
        )

        self._style_legend(ax)

        plt.tight_layout()

        save_path = os.path.join(output_dir, "ring_centroid_change_over_time.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Ring centroid change over time saved → {save_path}")

    def plot_ring_dynamics_separate(
        self,
        r1_inject,
        goal_idx,
        output_dir="./outputs/single_ring",
        bin_ms=500.0,
        t_max=None,
    ):
        os.makedirs(output_dir, exist_ok=True)

        ring_spikes = self._collect_spikes(
            self.r1.ring_attractor.ring_spike_recorders
        )

        rate_matrix, bins = self._spikes_to_rate_matrix(
            ring_spikes,
            bin_ms=bin_ms,
            t_max=t_max,
        )

        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        centroid_idx, _ = self._decode_circular_centroid(rate_matrix)

        centroid_change = self._compute_centroid_change(centroid_idx=centroid_idx)

        self.plot_ring_activity_heatmap(
            bins=bins,
            bin_centers=bin_centers,
            rate_matrix=rate_matrix,
            centroid_idx=centroid_idx,
            r1_inject=r1_inject,
            goal_idx=goal_idx,
            output_dir=output_dir,
        )

        self.plot_ring_centroid_change_over_time(
            bins=bins,
            bin_centers=bin_centers,
            centroid_change=centroid_change,
            output_dir=output_dir,
        )


# =====================================================================================
# BENCHMARK IMPLEMENTATION
# -------------------------------------------------------------------------------------
# Each individual run executes in a freshly spawned subprocess (see
# _run_one_benchmark_isolated), one at a time -- never in parallel. This matters for
# two reasons: (1) resource.getrusage(...).ru_maxrss is a whole-process high-water mark
# that never resets, so measuring several runs inside one long-lived process would just
# report cumulative peak memory, not per-run memory; a fresh process per run fixes that.
# (2) parallel workers would contend for CPU cores and corrupt the wall-clock /
# thread-scaling timing signal this benchmark exists to measure.
# =====================================================================================

def _make_scaled_ring_params_file(population_size, base_ring_params_file, weights_dir, tmp_dir):
    """Write a scratch copy of ring_params.json with population_size overridden.

    Never modifies base_ring_params_file. The pretrained Fourier weights for a given N
    were trained with their own num_fourier_k / readout scale (see the matching
    N_{N}_fourier_metadata.json in weights_dir) -- those fields are pulled from there so
    the scaled config always matches what the weight file actually expects, instead of
    assuming only population_size differs across scale points. RingAttractorComponent
    also looks up neuron_params.json as a sibling of the params file, so that file is
    copied alongside the scaled copy too.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    base_params = load_json(base_ring_params_file)
    scaled_params = dict(base_params)
    scaled_params["population_size"] = int(population_size)

    metadata_path = os.path.join(weights_dir, f"N_{int(population_size)}_fourier_metadata.json")
    if os.path.exists(metadata_path):
        metadata = load_json(metadata_path)
        for key in ("num_fourier_k", "readout_weight_scale", "output_dc_baseline", "harmonic_index"):
            if key in metadata:
                scaled_params[key] = metadata[key]

    scaled_path = os.path.join(tmp_dir, f"ring_params_N{int(population_size)}.json")
    with open(scaled_path, "w") as f:
        json.dump(scaled_params, f, indent=2)

    neuron_params_src = os.path.join(os.path.dirname(base_ring_params_file), "neuron_params.json")
    neuron_params_dst = os.path.join(tmp_dir, "neuron_params.json")
    if not os.path.exists(neuron_params_dst):
        shutil.copyfile(neuron_params_src, neuron_params_dst)

    return scaled_path


def _get_kernel_network_counts():
    """Return (num_kernel_nodes, num_connections), with fallbacks for differing NEST builds."""
    ks = nest.GetKernelStatus()

    num_nodes = ks.get("network_size")
    if num_nodes is None:
        try:
            num_nodes = len(nest.GetNodes())
        except Exception:
            num_nodes = 0

    num_connections = ks.get("num_connections")
    if num_connections is None:
        try:
            num_connections = len(nest.GetConnections())
        except Exception:
            num_connections = 0

    return int(num_nodes), int(num_connections)


def _run_one_benchmark(
    population_size,
    num_threads,
    run_idx,
    seed,
    ring_params_file=BENCHMARK_RING_PARAMS_FILE,
    weights_dir=BENCHMARK_WEIGHTS_DIR,
):
    """Build + presim + sim one benchmark condition, returning a flat metrics dict.

    Runs inside its own freshly spawned process (see _run_one_benchmark_isolated).
    """
    tmp_dir = os.path.join(BENCHMARK_OUTPUT_ROOT, "tmp_configs")
    scaled_params_file = _make_scaled_ring_params_file(population_size, ring_params_file, weights_dir, tmp_dir)

    r1_inject = int(round(BENCHMARK_R1_FRACTION * population_size))
    goal_idx = int(round(BENCHMARK_GOAL_FRACTION * population_size))

    tic = time.time()
    model = SingleRingModel(
        ring_params_file=scaled_params_file,
        weights_dir=weights_dir,
        seed=seed,
        local_num_threads=num_threads,
    )
    model.setup(
        r1_inject=r1_inject,
        r2_inject=goal_idx,
        stimulus_half_width=BENCHMARK_STIMULUS_HALF_WIDTH,
    )
    build_time_s = time.time() - tic
    mem_after_build_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    tic = time.time()
    nest.Simulate(BENCHMARK_PRESIM_MS)
    presim_time_s = time.time() - tic

    tic = time.time()
    nest.Simulate(BENCHMARK_SIMTIME_MS)
    sim_time_s = time.time() - tic
    mem_after_sim_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    total_elapsed_ms = BENCHMARK_PRESIM_MS + BENCHMARK_SIMTIME_MS
    real_time_factor = sim_time_s / (BENCHMARK_SIMTIME_MS / 1000.0)

    num_kernel_nodes, num_connections = _get_kernel_network_counts()

    ring_spikes = model._collect_spikes(model.r1.ring_attractor.ring_spike_recorders)
    total_spike_count = 0
    for times in ring_spikes:
        times = np.asarray(times, dtype=float)
        total_spike_count += int(np.sum(times >= BENCHMARK_PRESIM_MS))

    num_ring_neurons = max(1, len(ring_spikes))
    mean_firing_rate_hz = total_spike_count / (num_ring_neurons * (BENCHMARK_SIMTIME_MS / 1000.0))

    mean_fan_out = (num_connections / num_kernel_nodes) if num_kernel_nodes > 0 else 0.0
    synaptic_operations = total_spike_count * mean_fan_out
    energy_proxy_nJ = synaptic_operations * ENERGY_PER_SYNOP_PJ / 1000.0
    energy_proxy_per_second_nJ = energy_proxy_nJ / (BENCHMARK_SIMTIME_MS / 1000.0)

    avg_relative_goal_error = model.get_relative_goal_error_after_time(
        goal_idx=goal_idx,
        bin_ms=5000.0,
        t_max=total_elapsed_ms,
        avg_after_t=BENCHMARK_PRESIM_MS + BENCHMARK_AVG_AFTER_T_MS,
    )

    return {
        "population_size": int(population_size),
        "num_threads": int(num_threads),
        "run_idx": int(run_idx),
        "seed": int(seed),
        "r1_inject": int(r1_inject),
        "goal_idx": int(goal_idx),
        "build_time_s": float(build_time_s),
        "presim_time_s": float(presim_time_s),
        "sim_time_s": float(sim_time_s),
        "real_time_factor": float(real_time_factor),
        "mem_after_build_kb": float(mem_after_build_kb),
        "mem_after_sim_kb": float(mem_after_sim_kb),
        "num_kernel_nodes": int(num_kernel_nodes),
        "num_connections": int(num_connections),
        "total_spike_count": int(total_spike_count),
        "mean_firing_rate_hz": float(mean_firing_rate_hz),
        "mean_fan_out": float(mean_fan_out),
        "synaptic_operations": float(synaptic_operations),
        "energy_proxy_nJ": float(energy_proxy_nJ),
        "energy_proxy_per_second_nJ": float(energy_proxy_per_second_nJ),
        "avg_relative_goal_error": float(avg_relative_goal_error),
    }


def _benchmark_worker_entry(queue, population_size, num_threads, run_idx, seed, ring_params_file, weights_dir):
    try:
        result = _run_one_benchmark(
            population_size=population_size,
            num_threads=num_threads,
            run_idx=run_idx,
            seed=seed,
            ring_params_file=ring_params_file,
            weights_dir=weights_dir,
        )
        queue.put(("ok", result))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _run_one_benchmark_isolated(
    population_size,
    num_threads,
    run_idx,
    seed,
    ring_params_file=BENCHMARK_RING_PARAMS_FILE,
    weights_dir=BENCHMARK_WEIGHTS_DIR,
):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_benchmark_worker_entry,
        args=(queue, population_size, num_threads, run_idx, seed, ring_params_file, weights_dir),
    )
    proc.start()
    status, payload = queue.get()
    proc.join()
    if status == "error":
        raise RuntimeError(
            f"Benchmark run failed (N={population_size}, threads={num_threads}, "
            f"run={run_idx}): {payload}"
        )
    return payload


def _nan_mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


_BENCHMARK_SUMMARY_METRIC_KEYS = [
    "build_time_s",
    "presim_time_s",
    "sim_time_s",
    "real_time_factor",
    "mem_after_build_kb",
    "mem_after_sim_kb",
    "num_kernel_nodes",
    "num_connections",
    "total_spike_count",
    "mean_firing_rate_hz",
    "synaptic_operations",
    "energy_proxy_nJ",
    "energy_proxy_per_second_nJ",
    "avg_relative_goal_error",
]


def _summarize_conditions(rows):
    conditions = sorted({(r["population_size"], r["num_threads"]) for r in rows})
    summary_rows = []
    for population_size, num_threads in conditions:
        condition_rows = [
            r for r in rows
            if r["population_size"] == population_size and r["num_threads"] == num_threads
        ]
        summary = {
            "population_size": population_size,
            "num_threads": num_threads,
            "n_runs": len(condition_rows),
        }
        for key in _BENCHMARK_SUMMARY_METRIC_KEYS:
            mean_val, std_val = _nan_mean_std([r[key] for r in condition_rows])
            summary[f"{key}_mean"] = mean_val
            summary[f"{key}_std"] = std_val
        summary_rows.append(summary)
    return summary_rows


def _write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fieldnames) + "\n")


def _plot_timing_breakdown_bar(summary_rows, save_path, font_size=25, font_weight="bold"):
    baseline_threads = min(BENCHMARK_THREAD_COUNTS)
    baseline_rows = sorted(
        [r for r in summary_rows if r["num_threads"] == baseline_threads],
        key=lambda r: r["population_size"],
    )
    if not baseline_rows:
        return

    labels = [f"N={r['population_size']}" for r in baseline_rows]
    build_vals = [r["build_time_s_mean"] for r in baseline_rows]
    presim_vals = [r["presim_time_s_mean"] for r in baseline_rows]
    sim_vals = [r["sim_time_s_mean"] for r in baseline_rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x - width, build_vals, width, label="build")
    ax.bar(x, presim_vals, width, label="presim")
    ax.bar(x + width, sim_vals, width, label="sim")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title=f"Timing Breakdown (threads={baseline_threads})",
        xlabel="Network Scale",
        ylabel="Wall-Clock Time (s)",
    )
    _style_legend_inline(ax, font_size=font_size, font_weight=font_weight)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Timing breakdown plot saved → {save_path}")


def _plot_wallclock_vs_threads(summary_rows, save_path, font_size=25, font_weight="bold"):
    fig, ax = plt.subplots(figsize=(10, 7))

    for population_size in sorted({r["population_size"] for r in summary_rows}):
        cond_rows = sorted(
            [r for r in summary_rows if r["population_size"] == population_size],
            key=lambda r: r["num_threads"],
        )
        threads = [r["num_threads"] for r in cond_rows]
        sim_times = [r["sim_time_s_mean"] for r in cond_rows]
        sim_stds = [r["sim_time_s_std"] for r in cond_rows]

        ax.errorbar(threads, sim_times, yerr=sim_stds, marker="o", lw=2.0, label=f"N={population_size}")
        if sim_times:
            ideal = [sim_times[0] * threads[0] / t for t in threads]
            ax.plot(threads, ideal, ls="--", alpha=0.6, label=f"N={population_size} ideal scaling")

    ax.set_xscale("log", base=2)

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title="Wall-Clock Simulate Time vs Thread Count",
        xlabel="Threads (local_num_threads)",
        ylabel="Sim Wall-Clock Time (s)",
    )
    _style_legend_inline(ax, font_size=font_size, font_weight=font_weight)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Wall-clock vs threads plot saved → {save_path}")


def _plot_real_time_factor_vs_threads(summary_rows, save_path, font_size=25, font_weight="bold"):
    fig, ax = plt.subplots(figsize=(10, 7))

    for population_size in sorted({r["population_size"] for r in summary_rows}):
        cond_rows = sorted(
            [r for r in summary_rows if r["population_size"] == population_size],
            key=lambda r: r["num_threads"],
        )
        threads = [r["num_threads"] for r in cond_rows]
        rtf = [r["real_time_factor_mean"] for r in cond_rows]
        rtf_std = [r["real_time_factor_std"] for r in cond_rows]
        ax.errorbar(threads, rtf, yerr=rtf_std, marker="o", lw=2.0, label=f"N={population_size}")

    ax.axhline(1.0, color="gray", lw=1.5, ls=":", alpha=0.8, label="real-time (=1.0)")
    ax.set_xscale("log", base=2)

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title="Real-Time Factor vs Thread Count",
        xlabel="Threads (local_num_threads)",
        ylabel="Real-Time Factor",
    )
    _style_legend_inline(ax, font_size=font_size, font_weight=font_weight)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Real-time factor plot saved → {save_path}")


def _plot_memory_vs_scale(summary_rows, save_path, font_size=25, font_weight="bold"):
    baseline_threads = min(BENCHMARK_THREAD_COUNTS)
    baseline_rows = sorted(
        [r for r in summary_rows if r["num_threads"] == baseline_threads],
        key=lambda r: r["population_size"],
    )
    if not baseline_rows:
        return

    labels = [f"N={r['population_size']}" for r in baseline_rows]
    mem_build = [r["mem_after_build_kb_mean"] / 1024.0 for r in baseline_rows]
    mem_sim = [r["mem_after_sim_kb_mean"] / 1024.0 for r in baseline_rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x - width / 2, mem_build, width, label="after build")
    ax.bar(x + width / 2, mem_sim, width, label="after sim")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title=f"Peak Process Memory vs Scale (threads={baseline_threads})",
        xlabel="Network Scale",
        ylabel="Peak Memory (MB)",
    )
    _style_legend_inline(ax, font_size=font_size, font_weight=font_weight)
    ax.set_ylim(top=max(mem_sim + mem_build) * 1.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Memory vs scale plot saved → {save_path}")


def _plot_energy_proxy_vs_threads(summary_rows, save_path, font_size=25, font_weight="bold"):
    fig, ax = plt.subplots(figsize=(10, 7))

    for population_size in sorted({r["population_size"] for r in summary_rows}):
        cond_rows = sorted(
            [r for r in summary_rows if r["population_size"] == population_size],
            key=lambda r: r["num_threads"],
        )
        threads = [r["num_threads"] for r in cond_rows]
        energy = [r["energy_proxy_per_second_nJ_mean"] for r in cond_rows]
        energy_std = [r["energy_proxy_per_second_nJ_std"] for r in cond_rows]
        ax.errorbar(threads, energy, yerr=energy_std, marker="o", lw=2.0, label=f"N={population_size}")

    ax.set_xscale("log", base=2)

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title="Energy Proxy vs Thread Count",
        xlabel="Threads (local_num_threads)",
        ylabel="Energy Proxy (nJ/s)",
    )
    _style_legend_inline(ax, font_size=font_size, font_weight=font_weight)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Energy proxy vs threads plot saved → {save_path}")


def _plot_energy_vs_task_error_scatter(rows, save_path, font_size=25, font_weight="bold"):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    markers = ["o", "s", "^", "D", "v", "P"]
    for i, population_size in enumerate(sorted({r["population_size"] for r in rows})):
        cond_rows = [r for r in rows if r["population_size"] == population_size]
        x_vals = [r["avg_relative_goal_error"] for r in cond_rows]
        y_vals = [r["energy_proxy_per_second_nJ"] for r in cond_rows]
        ax.scatter(x_vals, y_vals, marker=markers[i % len(markers)], s=60, alpha=0.75, label=f"N={population_size}")

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title="Energy Proxy vs Task Error",
        xlabel="Avg. Relative Goal Error (distance / N)",
        ylabel="Energy Proxy (nJ/s)",
    )
    _style_legend_inline(ax, font_size=font_size, font_weight=font_weight)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Energy vs task-error scatter saved → {save_path}")


def run_benchmark_suite():
    os.makedirs(BENCHMARK_OUTPUT_ROOT, exist_ok=True)
    plots_dir = os.path.join(BENCHMARK_OUTPUT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    settings = {
        "population_sizes": list(BENCHMARK_POPULATION_SIZES),
        "thread_counts": list(BENCHMARK_THREAD_COUNTS),
        "n_runs_per_condition": int(BENCHMARK_N_RUNS_PER_CONDITION),
        "r1_fraction": float(BENCHMARK_R1_FRACTION),
        "goal_fraction": float(BENCHMARK_GOAL_FRACTION),
        "stimulus_half_width": int(BENCHMARK_STIMULUS_HALF_WIDTH),
        "presim_ms": float(BENCHMARK_PRESIM_MS),
        "simtime_ms": float(BENCHMARK_SIMTIME_MS),
        "avg_after_t_ms": float(BENCHMARK_AVG_AFTER_T_MS),
        "energy_per_synop_pJ": float(ENERGY_PER_SYNOP_PJ),
        "methodology": (
            "Wall-clock timers around build/presim/sim mirror NEST's official HPC "
            "benchmark (nest-simulator.readthedocs.io hpc_benchmark.py). Peak memory "
            "measured via resource.getrusage(RUSAGE_SELF).ru_maxrss in a freshly "
            "spawned subprocess per run (isolated, strictly sequential) so timing and "
            "memory are not confounded by CPU contention or cross-run memory "
            "accumulation. Energy is a synaptic-operations proxy "
            "(spikes x mean fan-out x energy_per_synop_pJ), not a measured value -- "
            "NEST/CPU simulation cannot report real Joules."
        ),
    }
    with open(os.path.join(BENCHMARK_OUTPUT_ROOT, "benchmark_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    rows = []
    total_conditions = len(BENCHMARK_POPULATION_SIZES) * len(BENCHMARK_THREAD_COUNTS)
    condition_i = 0
    for population_size in BENCHMARK_POPULATION_SIZES:
        for num_threads in BENCHMARK_THREAD_COUNTS:
            condition_i += 1
            print(f"\n[{condition_i}/{total_conditions}] N={population_size} threads={num_threads}")

            seeds = np.random.randint(0, 2**31, size=BENCHMARK_N_RUNS_PER_CONDITION, dtype=np.int64)
            for run_idx in range(BENCHMARK_N_RUNS_PER_CONDITION):
                row = _run_one_benchmark_isolated(
                    population_size=population_size,
                    num_threads=num_threads,
                    run_idx=run_idx,
                    seed=int(seeds[run_idx]),
                    ring_params_file=BENCHMARK_RING_PARAMS_FILE,
                    weights_dir=BENCHMARK_WEIGHTS_DIR,
                )
                rows.append(row)
                print(
                    f"  run {run_idx + 1}/{BENCHMARK_N_RUNS_PER_CONDITION} | "
                    f"build={row['build_time_s']:.2f}s sim={row['sim_time_s']:.2f}s "
                    f"rtf={row['real_time_factor']:.3f} "
                    f"energy/s={row['energy_proxy_per_second_nJ']:.3f}nJ "
                    f"goal_err={row['avg_relative_goal_error']:.4f}"
                )

    raw_csv_path = os.path.join(BENCHMARK_OUTPUT_ROOT, "raw_results.csv")
    _write_csv(rows, raw_csv_path)

    summary_rows = _summarize_conditions(rows)
    summary_csv_path = os.path.join(BENCHMARK_OUTPUT_ROOT, "condition_summary.csv")
    _write_csv(summary_rows, summary_csv_path)

    _plot_timing_breakdown_bar(summary_rows, os.path.join(plots_dir, "timing_breakdown_bar.png"))
    _plot_wallclock_vs_threads(summary_rows, os.path.join(plots_dir, "wallclock_vs_threads.png"))
    _plot_real_time_factor_vs_threads(summary_rows, os.path.join(plots_dir, "real_time_factor_vs_threads.png"))
    _plot_memory_vs_scale(summary_rows, os.path.join(plots_dir, "memory_vs_scale.png"))
    _plot_energy_proxy_vs_threads(summary_rows, os.path.join(plots_dir, "energy_proxy_vs_threads.png"))
    _plot_energy_vs_task_error_scatter(rows, os.path.join(plots_dir, "energy_vs_task_error_scatter.png"))

    print(f"\nBenchmark suite finished. Outputs under {BENCHMARK_OUTPUT_ROOT}")
    print(f"  Raw per-run CSV: {raw_csv_path}")
    print(f"  Condition summary CSV: {summary_csv_path}")
    print(f"  Settings saved to {os.path.join(BENCHMARK_OUTPUT_ROOT, 'benchmark_settings.json')}")


def analyze_goal_distance_across_runs(
    n_runs,
    r1_inject,
    goal_idx,
    sim_t,
    sample_ms=5000.0,
    output_dir="./outputs/single_ring",
    ring_params_file="./config/model_params/ring_params.json",
    weights_dir="./config/ring_decoding_weights",
    font_size=25,
    font_weight="bold",
    stimulus_half_width=5,
    save_individual_runs=False,
    max_workers=None,
):
    os.makedirs(output_dir, exist_ok=True)

    population_size = int(load_json(ring_params_file)["population_size"])
    max_goal_distance = population_size / 2.0

    worker_count = max_workers if max_workers is not None else os.cpu_count()
    if worker_count is None:
        worker_count = 1
    worker_count = max(1, min(int(worker_count), int(n_runs)))

    seeds = np.random.randint(0, 2**31, size=n_runs, dtype=np.int64)

    tasks = [
        (
            run_idx,
            int(seeds[run_idx]),
            r1_inject,
            goal_idx,
            sim_t,
            sample_ms,
            ring_params_file,
            weights_dir,
            font_size,
            font_weight,
            stimulus_half_width,
        )
        for run_idx in range(n_runs)
    ]

    results = [None] * n_runs

    if worker_count == 1:
        for args in tasks:
            run_idx, seed, bin_centers, goal_distance = _run_goal_distance_worker(args)
            results[run_idx] = (seed, bin_centers, goal_distance)
            print(f"  Finished run {run_idx + 1}/{n_runs} (seed={seed})")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as ex:
            futures = [ex.submit(_run_goal_distance_worker, args) for args in tasks]
            for fut in as_completed(futures):
                run_idx, seed, bin_centers, goal_distance = fut.result()
                results[run_idx] = (seed, bin_centers, goal_distance)
                print(f"  Finished run {run_idx + 1}/{n_runs} (seed={seed})")

    all_goal_distances = []
    common_bin_centers = None
    for run_idx, item in enumerate(results):
        seed, bin_centers, goal_distance = item
        if common_bin_centers is None:
            common_bin_centers = bin_centers
        all_goal_distances.append(goal_distance)

        if save_individual_runs:
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.plot(bin_centers, goal_distance, lw=2.0, label=f"run {run_idx + 1}")
            ax.axhline(
                0.0,
                color="gray",
                lw=1.2,
                ls="--",
                alpha=0.8,
                label="goal reached",
            )
            _style_axis_inline(
                ax,
                font_size=font_size,
                font_weight=font_weight,
                title=f"Distance to Goal - Run {run_idx + 1}",
                xlabel="Time (ms)",
                ylabel="Distance to Goal",
            )
            _style_legend_inline(
                ax,
                font_size=font_size,
                font_weight=font_weight,
            )
            ax.set_ylim(0.0, max_goal_distance)
            plt.tight_layout()
            run_path = os.path.join(output_dir, f"goal_distance_run_{run_idx + 1:02d}.png")
            plt.savefig(run_path, dpi=150, bbox_inches="tight")
            plt.close()

    all_goal_distances = np.vstack(all_goal_distances)

    mean_goal_distance = np.nanmean(all_goal_distances, axis=0)
    std_goal_distance = np.nanstd(all_goal_distances, axis=0)

    csv_path = os.path.join(output_dir, "goal_distance_mean_std_across_runs.csv")

    summary = np.column_stack(
        [common_bin_centers, mean_goal_distance, std_goal_distance]
    )

    np.savetxt(
        csv_path,
        summary,
        delimiter=",",
        header="time_ms,mean_goal_distance,std_goal_distance",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        common_bin_centers,
        mean_goal_distance,
        lw=3.0,
        label=f"mean ({n_runs} runs)",
    )

    ax.fill_between(
        common_bin_centers,
        mean_goal_distance - std_goal_distance,
        mean_goal_distance + std_goal_distance,
        alpha=0.25,
        label=f"std ({n_runs} runs)",
    )

    ax.axhline(
        0.0,
        color="gray",
        lw=1.5,
        ls="--",
        alpha=0.8,
        label="goal reached",
    )

    ax.set_xlim(common_bin_centers[0], common_bin_centers[-1])
    ax.set_ylim(0.0, max_goal_distance)

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title=f"Distance to Goal Over Time (mean ± std, {n_runs} runs, goal idx = {goal_idx})",
        xlabel="Time (ms)",
        ylabel="Distance to Goal",
    )

    _style_legend_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        loc="upper right",
    )

    plt.tight_layout()

    save_path = os.path.join(output_dir, "goal_distance_mean_std_across_runs.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Mean/std goal-distance plot saved → {save_path}")
    print(f"  Mean/std goal-distance CSV saved → {csv_path}")

    return common_bin_centers, mean_goal_distance, std_goal_distance


def analyze_relative_goal_error_across_goals(
    n_runs=10,
    n_goals=10,
    r1_inject=10,
    sim_t=100000.0,
    avg_after_t=30000.0,
    sample_ms=5000.0,
    output_dir="./outputs/single_ring",
    ring_params_file="./config/model_params/ring_params.json",
    weights_dir="./config/ring_decoding_weights",
    font_size=25,
    font_weight="bold",
    stimulus_half_width=5,
    max_workers=None,
):
    os.makedirs(output_dir, exist_ok=True)

    population_size = int(load_json(ring_params_file)["population_size"])

    edge_margin = max(1, int(round(0.1 * population_size)))

    goal_indices = np.linspace(
        edge_margin,
        population_size - edge_margin,
        n_goals,
        endpoint=True,
        dtype=int,
    )

    goal_means = []
    goal_stds = []
    per_run_rows = []

    print("\nStarting multi-goal relative-error analysis")
    print(f"  population size N = {population_size}")
    print(f"  goals = {goal_indices}")
    print(f"  n_runs per goal = {n_runs}")
    print(f"  relative error = distance_to_goal / N")
    print(f"  averaging over t > {avg_after_t:.0f} ms\n")

    worker_count = max_workers if max_workers is not None else os.cpu_count()
    if worker_count is None:
        worker_count = 1
    total_runs = int(n_runs) * len(goal_indices)
    worker_count = max(1, min(int(worker_count), total_runs))

    seeds = np.random.randint(0, 2**31, size=total_runs, dtype=np.int64)
    tasks = []
    seed_idx = 0
    for goal_i, goal_idx in enumerate(goal_indices):
        for run_idx in range(n_runs):
            tasks.append(
                (
                    goal_i,
                    run_idx,
                    int(goal_idx),
                    int(seeds[seed_idx]),
                    r1_inject,
                    sim_t,
                    avg_after_t,
                    sample_ms,
                    ring_params_file,
                    weights_dir,
                    font_size,
                    font_weight,
                    stimulus_half_width,
                )
            )
            seed_idx += 1

    run_error_by_goal = {int(g): [] for g in goal_indices}

    def _consume_result(res):
        goal_i, run_idx, goal_idx, seed, avg_relative_error = res
        run_error_by_goal[int(goal_idx)].append(avg_relative_error)
        per_run_rows.append([int(goal_idx), int(run_idx + 1), float(avg_relative_error)])
        print(
            f"  Goal {int(goal_idx):>3} "
            f"| run {run_idx + 1:>2}/{n_runs} "
            f"| mean relative error t>{avg_after_t:.0f} ms = {avg_relative_error:.5f} "
            f"(seed={seed})"
        )

    if worker_count == 1:
        for args in tasks:
            _consume_result(_run_relative_goal_error_worker(args))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as ex:
            futures = [ex.submit(_run_relative_goal_error_worker, args) for args in tasks]
            for fut in as_completed(futures):
                _consume_result(fut.result())

    goal_means = []
    goal_stds = []
    for goal_idx in goal_indices:
        run_avg_relative_errors = np.array(run_error_by_goal[int(goal_idx)], dtype=float)
        finite = run_avg_relative_errors[np.isfinite(run_avg_relative_errors)]
        if finite.size == 0:
            goal_means.append(np.nan)
            goal_stds.append(np.nan)
        else:
            goal_means.append(float(np.mean(finite)))
            goal_stds.append(float(np.std(finite)))

    goal_means = np.array(goal_means)
    goal_stds = np.array(goal_stds)

    summary_csv_path = os.path.join(
        output_dir,
        "relative_goal_error_summary.csv",
    )

    with open(summary_csv_path, "w") as f:
        f.write("goal_idx,mean_relative_error,std_relative_error\n")

        for goal_idx, mean_val, std_val in zip(goal_indices, goal_means, goal_stds):
            f.write(f"{int(goal_idx)},{mean_val},{std_val}\n")

    per_run_csv_path = os.path.join(
        output_dir,
        "relative_goal_error_per_run.csv",
    )

    with open(per_run_csv_path, "w") as f:
        f.write("goal_idx,run_idx,avg_relative_error_after_threshold\n")

        for row in per_run_rows:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")

    per_run_array = np.array(per_run_rows)
    box_data = [
        per_run_array[per_run_array[:, 0] == g, 2].tolist() for g in goal_indices
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    box_color = "#4a90d9"
    xs = np.arange(1, len(goal_indices) + 1)

    for x, data in zip(xs, box_data):
        if not data:
            continue
        arr = np.asarray(data, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        m = float(np.mean(finite))
        s = float(np.std(finite))
        if not np.isfinite(m) or not np.isfinite(s):
            continue
        ax.bar(x, 2 * s, bottom=m - s, width=0.5, color=box_color, alpha=0.8)
        ax.hlines(m, x - 0.25, x + 0.25, colors="white", linewidth=2.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(g)) for g in goal_indices])
    # Relative circular distance is in [0, 0.5] when normalized by N.
    dynamic_upper = 1.2 * _safe_positive_percentile(
        goal_means + goal_stds,
        percentile=100.0,
        fallback=0.05,
    )
    ax.set_ylim(0.0, min(0.5, max(0.05, dynamic_upper)))

    _style_axis_inline(
        ax,
        font_size=font_size,
        font_weight=font_weight,
        title=(
            f"Relative Goal Error Across Goal Locations\n"
            f"{n_runs} runs per goal, averaged over t > {avg_after_t:.0f} ms"
        ),
        xlabel="Goal Index",
        ylabel="Relative Error (distance / N)",
    )

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "relative_goal_error_boxplot.png",
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Relative goal error box plot saved → {save_path}")
    print(f"  Summary CSV saved → {summary_csv_path}")
    print(f"  Per-run CSV saved → {per_run_csv_path}")

    return goal_indices, goal_means, goal_stds


if __name__ == "__main__":
    run_benchmark_suite()
