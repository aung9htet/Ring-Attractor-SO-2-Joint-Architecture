#!/usr/bin/env python3

import os
import json
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
    r1 = 20
    r2 = 70
    sim_t = 100000.0

    output_dir = "./outputs/single_ring"
    font_size = 25

    model = SingleRingModel(
        font_size=font_size,
        seed=12345,
    )

    model.setup(
        r1_inject=r1,
        r2_inject=r2,
    )

    nest.Simulate(sim_t)

    model.plot(
        r1_inject=r1,
        r2_inject=r2,
        output_dir=output_dir,
    )

    model.plot_ring_dynamics_separate(
        r1_inject=r1,
        goal_idx=r2,
        output_dir=output_dir,
        bin_ms=500.0,
        t_max=sim_t,
    )

    analyze_goal_distance_across_runs(
        n_runs=10,
        r1_inject=r1,
        goal_idx=r2,
        sim_t=sim_t,
        sample_ms=5000.0,
        output_dir=output_dir,
        font_size=font_size,
        save_individual_runs=False,
    )

    analyze_relative_goal_error_across_goals(
        n_runs=10,
        n_goals=10,
        r1_inject=r1,
        sim_t=sim_t,
        avg_after_t=30000.0,
        sample_ms=5000.0,
        output_dir=output_dir,
        font_size=font_size,
    )
