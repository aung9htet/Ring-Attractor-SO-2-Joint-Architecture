#!/usr/bin/env python3

import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional
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


TEST_NEURON_LOSS = "neuron_loss"
TEST_NOISE_LOCALIZED = "noise_localized"
TEST_NOISE_GLOBAL = "noise_global"

# On-disk output folder name for each test_type (kept distinct from the internal
# test_type strings above, which are still used unchanged in CSV/JSON fields and logs).
_OUTPUT_FOLDER_NAME = {
    TEST_NEURON_LOSS: "neuron_loss",
    TEST_NOISE_GLOBAL: "global_noise",
    TEST_NOISE_LOCALIZED: "localized_noise",
}


# =====================================================================================
# ROBUSTNESS EXPERIMENT PARAMETERS -- edit these to change what is swept/reported.
# -------------------------------------------------------------------------------------
# Every run's exact settings are ALSO written to disk (experiment_settings.json per
# condition folder, plus perturbation_manifest.json per individual run) so any figure
# can always be traced back to the exact parameters used to produce it.
# =====================================================================================

# --- Neuron loss ---------------------------------------------------------------------
# Reported/swept indicator: PERCENTAGE OF RING NEURONS LOST (0-100).
# Mechanism: randomly choose that percentage of ring neurons, then remove every
# connection touching them (incoming + outgoing; ring<->ring recurrence AND
# ring<->gain-population feedback) by zeroing the synaptic weight. A "lost" neuron
# therefore has zero influence on, and receives zero influence from, the rest of the
# network -- equivalent to deleting the neuron and its synapses ("random connections
# chosen" for removal).
NEURON_LOSS_PERCENTAGES = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]  # percent of N neurons removed; edit freely

# --- Noise injected into a random percentage of connections (global test) ------------
# Reported/swept indicator: NOISE STRENGTH == the Poisson input RATE in Hz (synaptic
# weight is held fixed at GLOBAL_NOISE_WEIGHT_PA so rate is the only free variable).
# Mechanism: randomly choose GLOBAL_NOISE_CONNECTION_FRACTION of ring neurons and wire
# each selected neuron to its own poisson_generator with weight GLOBAL_NOISE_WEIGHT_PA.
GLOBAL_NOISE_CONNECTION_FRACTION = 0.5   # fraction (0-1) of ring neurons wired to noise; edit to test more/less coverage
GLOBAL_NOISE_STRENGTH_LEVELS_HZ = [5, 10, 15, 20]  # swept Poisson rate (Hz); EDIT THIS to adjust the noise rate
GLOBAL_NOISE_WEIGHT_PA = 0.00001          # fixed synaptic weight (pA-equivalent) for every global-noise synapse; edit freely

# --- Localized noise (small, random regions) ------------------------------------------
# Reported/swept indicator: NOISE STRENGTH == the Poisson input RATE in Hz, kept SMALL
# since it only targets a small population (weight fixed at LOCAL_NOISE_WEIGHT_PA).
# Mechanism: every run samples LOCALIZED_NOISE_NUM_REGIONS random center indices on the
# ring; each region spans LOCALIZED_NOISE_REGION_HALF_WIDTH neurons on either side of
# its center and is wired to its own poisson_generator with weight LOCAL_NOISE_WEIGHT_PA.
LOCALIZED_NOISE_NUM_REGIONS = 10          # number of random localized regions sampled per run; edit freely
LOCALIZED_NOISE_REGION_HALF_WIDTH = 2     # neurons on each side of every region center; edit freely
LOCAL_NOISE_STRENGTH_LEVELS_HZ = [5, 10, 15, 20]  # small swept Poisson rate (Hz); EDIT THIS to adjust the noise rate
LOCAL_NOISE_WEIGHT_PA = 0.00001            # fixed synaptic weight (pA-equivalent) for every localized-noise synapse; edit freely

# --- Shared execution settings ---------------------------------------------------------
ROBUSTNESS_N_RUNS_PER_CONDITION = 10      # independent seeded runs per sweep point (matches baseline n_runs=10)
ROBUSTNESS_R1_INJECT = 20                 # start bump index (matches baseline __main__ r1)
ROBUSTNESS_GOAL_IDX = 70                  # goal bump index (matches baseline __main__ r2)
ROBUSTNESS_SIM_T_MS = 100000.0            # simulation duration per run (matches baseline sim_t)
ROBUSTNESS_AVG_AFTER_T_MS = 30000.0       # transient cutoff for averaged error metrics
ROBUSTNESS_SAMPLE_MS = 5000.0             # bin width for goal-distance traces
ROBUSTNESS_STIMULUS_HALF_WIDTH = 5        # bump injection half-width (matches baseline default)
ROBUSTNESS_NOISE_SCHEDULES = ["persistent"]  # add "pulsed" here to also test noise on/off windows
ROBUSTNESS_PULSE_ON_MS = 100.0           # pulsed schedule: noise-on duration per cycle
ROBUSTNESS_PULSE_OFF_MS = 100.0          # pulsed schedule: noise-off duration per cycle
ROBUSTNESS_MAX_WORKERS = None             # None = os.cpu_count(); set an int to cap parallel workers


@dataclass
class RobustnessConfig:
    test_type: str
    rng_seed: int
    schedule_type: str = "persistent"
    pulse_on_ms: float = 1000.0
    pulse_off_ms: float = 1000.0
    start_ms: float = 0.0
    end_ms: Optional[float] = None

    # neuron loss
    neuron_loss_percentage: float = 0.0     # percent of neurons removed (0-100); the reported indicator

    # noise (shared fields for global + localized tests)
    noise_rate_hz: float = 0.0              # noise STRENGTH indicator (Poisson rate, Hz)
    noise_weight: float = 0.0               # fixed synaptic weight (pA-equivalent)
    noise_target_indices: tuple = ()        # exact neuron indices wired to noise (for reporting)
    noise_connection_fraction: float = 0.0  # global test: fraction of neurons wired to noise
    noise_region_centers: tuple = ()        # localized test: sampled region center indices
    noise_region_half_width: int = 0        # localized test: half-width per region


def _circular_window_indices(center_idx, half_width, population_size):
    center = int(center_idx) % int(population_size)
    hw = int(max(0, half_width))
    return sorted(
        {
            int((center + delta) % population_size)
            for delta in range(-hw, hw + 1)
        }
    )


def _sample_robustness_config(
    test_type,
    param_value,
    population_size,
    rng_seed,
    schedule_type="persistent",
):
    """Sample a fully-specified RobustnessConfig for one run.

    `param_value` is the single swept/reported indicator for the given test_type:
      - TEST_NEURON_LOSS: percentage of neurons lost (0-100)
      - TEST_NOISE_GLOBAL / TEST_NOISE_LOCALIZED: noise strength (Poisson rate, Hz)
    """
    rng = np.random.default_rng(int(rng_seed))

    cfg = RobustnessConfig(
        test_type=str(test_type),
        rng_seed=int(rng_seed),
        schedule_type=str(schedule_type),
        pulse_on_ms=float(ROBUSTNESS_PULSE_ON_MS),
        pulse_off_ms=float(ROBUSTNESS_PULSE_OFF_MS),
    )

    if test_type == TEST_NEURON_LOSS:
        cfg.neuron_loss_percentage = float(param_value)
        return cfg

    if test_type == TEST_NOISE_GLOBAL:
        cfg.noise_rate_hz = float(param_value)
        cfg.noise_weight = float(GLOBAL_NOISE_WEIGHT_PA)
        cfg.noise_connection_fraction = float(GLOBAL_NOISE_CONNECTION_FRACTION)
        n_targets = int(round(cfg.noise_connection_fraction * population_size))
        n_targets = max(0, min(int(population_size), n_targets))
        target_indices = (
            sorted(int(x) for x in rng.choice(int(population_size), size=n_targets, replace=False))
            if n_targets > 0
            else []
        )
        cfg.noise_target_indices = tuple(target_indices)
        return cfg

    if test_type == TEST_NOISE_LOCALIZED:
        cfg.noise_rate_hz = float(param_value)
        cfg.noise_weight = float(LOCAL_NOISE_WEIGHT_PA)
        cfg.noise_region_half_width = int(LOCALIZED_NOISE_REGION_HALF_WIDTH)
        centers = [
            int(rng.integers(0, int(population_size)))
            for _ in range(int(LOCALIZED_NOISE_NUM_REGIONS))
        ]
        cfg.noise_region_centers = tuple(centers)
        target_indices = set()
        for center in centers:
            target_indices.update(
                _circular_window_indices(center, cfg.noise_region_half_width, population_size)
            )
        cfg.noise_target_indices = tuple(sorted(target_indices))
        return cfg

    raise ValueError(f"Unknown robustness test_type: {test_type}")


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

    model.simulate(sim_t)

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

    model.simulate(sim_t)

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
        robustness_config=None,
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

        self.robustness_config = robustness_config
        self.perturbation_report = {
            "enabled": robustness_config is not None,
            "test_type": None,
            "rng_seed": None,
            "schedule_type": None,
            "neuron_loss_percentage": 0.0,
            "neuron_loss_count": 0,
            "lost_indices": [],
            "disconnected_connection_count": 0,
            "noise_rate_hz": 0.0,
            "noise_weight": 0.0,
            "noise_connection_fraction": 0.0,
            "noise_region_centers": [],
            "noise_region_half_width": 0,
            "noise_target_indices": [],
            "noise_coverage_fraction": 0.0,
            "noise_window_count": 0,
            "noise_active_duration_ms": 0.0,
            "noise_integrated_drive_per_neuron": 0.0,
            "noise_integrated_drive_total": 0.0,
        }

        if robustness_config is not None:
            self._apply_robustness_perturbation(robustness_config)

    def _as_robustness_config(self, robustness_config):
        if isinstance(robustness_config, RobustnessConfig):
            return robustness_config
        if isinstance(robustness_config, dict):
            return RobustnessConfig(**robustness_config)
        raise ValueError("robustness_config must be RobustnessConfig or dict")

    def _get_noise_windows(self, cfg, sim_t):
        start_ms = float(max(0.0, cfg.start_ms))
        end_ms = float(sim_t if cfg.end_ms is None else max(start_ms, cfg.end_ms))

        if cfg.schedule_type != "pulsed":
            return [(start_ms, end_ms)]

        pulse_on_ms = float(max(1.0, cfg.pulse_on_ms))
        pulse_off_ms = float(max(0.0, cfg.pulse_off_ms))
        windows = []
        t = start_ms
        while t < end_ms:
            t_end = min(end_ms, t + pulse_on_ms)
            if t_end > t:
                windows.append((t, t_end))
            t = t_end + pulse_off_ms
        return windows

    def _set_connections_weight_to_zero(self, connections):
        if len(connections) == 0:
            return 0
        try:
            nest.SetStatus(connections, {"weight": 0.0})
            return int(len(connections))
        except Exception:
            return 0

    def _disconnect_lost_neurons_structural(self, lost_indices):
        ring_neurons = self.r1.ring_attractor.ring_neurons
        left_gain = self.gain_modulation.left_gain_neurons
        right_gain = self.gain_modulation.right_gain_neurons
        gain_nodes = left_gain + right_gain

        zeroed = 0
        for idx in lost_indices:
            ring_node = ring_neurons[int(idx)]

            for peer in ring_neurons:
                zeroed += self._set_connections_weight_to_zero(
                    nest.GetConnections(source=ring_node, target=peer)
                )
                zeroed += self._set_connections_weight_to_zero(
                    nest.GetConnections(source=peer, target=ring_node)
                )

            for gain_node in gain_nodes:
                zeroed += self._set_connections_weight_to_zero(
                    nest.GetConnections(source=ring_node, target=gain_node)
                )
                zeroed += self._set_connections_weight_to_zero(
                    nest.GetConnections(source=gain_node, target=ring_node)
                )

        return int(zeroed)

    def _apply_neuron_loss(self, cfg):
        loss_fraction = float(np.clip(cfg.neuron_loss_percentage, 0.0, 100.0)) / 100.0
        n_loss = int(round(loss_fraction * self.population_size))
        n_loss = max(0, min(self.population_size, n_loss))

        rng = np.random.default_rng(int(cfg.rng_seed))
        if n_loss > 0:
            lost_indices = sorted(
                [int(x) for x in rng.choice(self.population_size, size=n_loss, replace=False)]
            )
        else:
            lost_indices = []

        self.perturbation_report["neuron_loss_percentage"] = float(cfg.neuron_loss_percentage)
        self.perturbation_report["neuron_loss_count"] = int(n_loss)
        self.perturbation_report["lost_indices"] = lost_indices

        if not lost_indices:
            self.perturbation_report["disconnected_connection_count"] = 0
            return

        zeroed = self._disconnect_lost_neurons_structural(lost_indices)
        self.perturbation_report["disconnected_connection_count"] = int(zeroed)

    def _apply_ring_noise(self, cfg):
        ring_neurons = self.r1.ring_attractor.ring_neurons
        target_indices = list(cfg.noise_target_indices)

        self.perturbation_report["noise_rate_hz"] = float(cfg.noise_rate_hz)
        self.perturbation_report["noise_weight"] = float(cfg.noise_weight)
        self.perturbation_report["noise_connection_fraction"] = float(cfg.noise_connection_fraction)
        self.perturbation_report["noise_region_centers"] = [int(c) for c in cfg.noise_region_centers]
        self.perturbation_report["noise_region_half_width"] = int(cfg.noise_region_half_width)
        self.perturbation_report["noise_target_indices"] = [int(i) for i in target_indices]
        self.perturbation_report["noise_coverage_fraction"] = (
            float(len(target_indices)) / float(self.population_size) if self.population_size else 0.0
        )

        self._noise_target_nodes = [ring_neurons[idx] for idx in target_indices]

    def _apply_robustness_perturbation(self, robustness_config):
        cfg = self._as_robustness_config(robustness_config)
        self.robustness_config = cfg

        self.perturbation_report["test_type"] = cfg.test_type
        self.perturbation_report["rng_seed"] = int(cfg.rng_seed)
        self.perturbation_report["schedule_type"] = str(cfg.schedule_type)

        if cfg.test_type == TEST_NEURON_LOSS:
            self._apply_neuron_loss(cfg)
        elif cfg.test_type in (TEST_NOISE_LOCALIZED, TEST_NOISE_GLOBAL):
            self._apply_ring_noise(cfg)
        else:
            raise ValueError(f"Unknown robustness test_type: {cfg.test_type}")

    def _connect_noise_generators(self, sim_t):
        if not hasattr(self, "_noise_target_nodes"):
            return
        cfg = self.robustness_config
        windows = self._get_noise_windows(cfg, sim_t)
        resolution = float(nest.GetKernelStatus("resolution"))

        def _align_to_resolution(t_ms):
            return float(round(float(t_ms) / resolution) * resolution)

        total_active_ms = 0.0
        for t_start, t_end in windows:
            t_start_aligned = _align_to_resolution(t_start)
            t_end_aligned = _align_to_resolution(t_end)
            if t_end_aligned <= t_start_aligned:
                t_end_aligned = t_start_aligned + resolution
            if t_end_aligned <= t_start_aligned:
                continue
            total_active_ms += (t_end_aligned - t_start_aligned)
            pg = nest.Create(
                "poisson_generator",
                params={
                    "rate": float(cfg.noise_rate_hz),
                    "start": float(t_start_aligned),
                    "stop": float(t_end_aligned),
                },
            )
            for node in self._noise_target_nodes:
                nest.Connect(
                    pg,
                    node,
                    syn_spec={"weight": float(cfg.noise_weight)},
                )

        n_targets = float(len(self.perturbation_report["noise_target_indices"]))
        drive_per_neuron = (
            float(cfg.noise_rate_hz) * (total_active_ms / 1000.0) * abs(float(cfg.noise_weight))
        )
        self.perturbation_report["noise_window_count"] = int(len(windows))
        self.perturbation_report["noise_active_duration_ms"] = float(total_active_ms)
        self.perturbation_report["noise_integrated_drive_per_neuron"] = float(drive_per_neuron)
        self.perturbation_report["noise_integrated_drive_total"] = float(drive_per_neuron * n_targets)

    def get_perturbation_report(self):
        return dict(self.perturbation_report)

    def simulate(self, sim_t):
        if self.robustness_config is not None:
            cfg = self.robustness_config
            if cfg.test_type in (TEST_NOISE_LOCALIZED, TEST_NOISE_GLOBAL):
                self._connect_noise_generators(sim_t)
        nest.Simulate(sim_t)

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


def _run_robustness_worker(args):
    (
        run_idx,
        seed,
        r1_inject,
        goal_idx,
        sim_t,
        sample_ms,
        avg_after_t,
        ring_params_file,
        weights_dir,
        font_size,
        font_weight,
        stimulus_half_width,
        robustness_config,
        representative_output_dir,
    ) = args

    model = SingleRingModel(
        ring_params_file=ring_params_file,
        weights_dir=weights_dir,
        font_size=font_size,
        font_weight=font_weight,
        seed=seed,
        robustness_config=robustness_config,
    )

    model.setup(
        r1_inject=r1_inject,
        r2_inject=goal_idx,
        stimulus_half_width=stimulus_half_width,
    )

    model.simulate(sim_t)

    if representative_output_dir is not None:
        os.makedirs(representative_output_dir, exist_ok=True)
        model.plot(
            r1_inject=r1_inject,
            r2_inject=goal_idx,
            output_dir=representative_output_dir,
        )
        model.plot_ring_dynamics_separate(
            r1_inject=r1_inject,
            goal_idx=goal_idx,
            output_dir=representative_output_dir,
            bin_ms=500.0,
            t_max=sim_t,
        )

    bin_centers, goal_distance = model.get_goal_distance_trace(
        goal_idx=goal_idx,
        bin_ms=sample_ms,
        t_max=sim_t,
    )

    avg_relative_error = model.get_relative_goal_error_after_time(
        goal_idx=goal_idx,
        bin_ms=sample_ms,
        t_max=sim_t,
        avg_after_t=avg_after_t,
    )

    after_mask = np.asarray(bin_centers) >= float(avg_after_t)
    if np.any(after_mask):
        mean_goal_after = float(np.nanmean(np.asarray(goal_distance)[after_mask]))
    else:
        mean_goal_after = float(np.nanmean(np.asarray(goal_distance)))

    return {
        "run_idx": int(run_idx),
        "seed": int(seed),
        "bin_centers": np.asarray(bin_centers, dtype=float),
        "goal_distance": np.asarray(goal_distance, dtype=float),
        "avg_relative_error": float(avg_relative_error),
        "mean_goal_distance_after_t": float(mean_goal_after),
        "perturbation": model.get_perturbation_report(),
    }


def _save_run_metrics_csv(run_results, output_path):
    with open(output_path, "w") as f:
        f.write(
            "run_idx,seed,avg_relative_error,mean_goal_distance_after_t,"
            "neuron_loss_percentage,neuron_loss_count,disconnected_connection_count,"
            "noise_rate_hz,noise_weight,noise_connection_fraction,noise_region_half_width,"
            "noise_target_count,noise_coverage_fraction,noise_active_duration_ms,"
            "noise_integrated_drive_per_neuron,noise_integrated_drive_total\n"
        )
        for res in run_results:
            p = res["perturbation"]
            f.write(
                f"{res['run_idx']},{res['seed']},{res['avg_relative_error']},"
                f"{res['mean_goal_distance_after_t']},"
                f"{p.get('neuron_loss_percentage', 0.0)},"
                f"{p.get('neuron_loss_count', 0)},"
                f"{p.get('disconnected_connection_count', 0)},"
                f"{p.get('noise_rate_hz', 0.0)},{p.get('noise_weight', 0.0)},"
                f"{p.get('noise_connection_fraction', 0.0)},"
                f"{p.get('noise_region_half_width', 0)},"
                f"{len(p.get('noise_target_indices', []))},"
                f"{p.get('noise_coverage_fraction', 0.0)},"
                f"{p.get('noise_active_duration_ms', 0.0)},"
                f"{p.get('noise_integrated_drive_per_neuron', 0.0)},"
                f"{p.get('noise_integrated_drive_total', 0.0)}\n"
            )


def _save_perturbation_manifest_json(run_results, output_path):
    payload = [
        {
            "run_idx": int(res["run_idx"]),
            "seed": int(res["seed"]),
            "perturbation": res["perturbation"],
        }
        for res in run_results
    ]
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def _plot_goal_distance_mean_std(bin_centers, all_goal_distances, save_path, title, n_runs):
    if all_goal_distances.size == 0:
        return
    mean_goal_distance = np.nanmean(all_goal_distances, axis=0)
    std_goal_distance = np.nanstd(all_goal_distances, axis=0)

    finite_vals = all_goal_distances[np.isfinite(all_goal_distances)]
    if finite_vals.size == 0:
        max_goal_distance = 1.0
    else:
        max_goal_distance = float(max(1.0, np.max(finite_vals)))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(bin_centers, mean_goal_distance, lw=3.0, label=f"mean ({n_runs} runs)")
    ax.fill_between(
        bin_centers,
        mean_goal_distance - std_goal_distance,
        mean_goal_distance + std_goal_distance,
        alpha=0.25,
        label=f"std ({n_runs} runs)",
    )
    ax.axhline(0.0, color="gray", lw=1.5, ls="--", alpha=0.8, label="goal reached")
    ax.set_xlim(bin_centers[0], bin_centers[-1])
    ax.set_ylim(0.0, max_goal_distance)

    _style_axis_inline(
        ax,
        font_size=20,
        font_weight="bold",
        title=title,
        xlabel="Time (ms)",
        ylabel="Distance to Goal",
    )
    _style_legend_inline(ax, font_size=18, font_weight="bold", loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_robustness_curve(summary_rows, save_path, metric_key, ylabel, title, xlabel="Parameter"):
    if len(summary_rows) == 0:
        return

    xs = np.array([row["param_value"] for row in summary_rows], dtype=float)
    means = np.array([row[metric_key + "_mean"] for row in summary_rows], dtype=float)
    stds = np.array([row[metric_key + "_std"] for row in summary_rows], dtype=float)

    order = np.argsort(xs)
    xs = xs[order]
    means = means[order]
    stds = stds[order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, means, "-o", lw=2.4, label="mean")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.25, label="std")
    _style_axis_inline(
        ax,
        font_size=18,
        font_weight="bold",
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    _style_legend_inline(ax, font_size=16, font_weight="bold", loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _nan_stats(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _condition_folder_name(test_type, param_value):
    if test_type == TEST_NEURON_LOSS:
        return f"loss_pct_{float(param_value):05.1f}"
    return f"noise_hz_{float(param_value):08.1f}"


def _save_experiment_settings_json(variant_dir, settings):
    with open(os.path.join(variant_dir, "experiment_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)


def run_robustness_condition(
    test_type,
    param_values,
    param_label,
    n_runs_per_condition=ROBUSTNESS_N_RUNS_PER_CONDITION,
    r1_inject=ROBUSTNESS_R1_INJECT,
    goal_idx=ROBUSTNESS_GOAL_IDX,
    sim_t=ROBUSTNESS_SIM_T_MS,
    avg_after_t=ROBUSTNESS_AVG_AFTER_T_MS,
    sample_ms=ROBUSTNESS_SAMPLE_MS,
    ring_params_file="./config/model_params/ring_params.json",
    weights_dir="./config/ring_decoding_weights",
    font_size=25,
    font_weight="bold",
    stimulus_half_width=ROBUSTNESS_STIMULUS_HALF_WIDTH,
    schedule_type="persistent",
    max_workers=None,
):
    output_root = os.path.join("./outputs", _OUTPUT_FOLDER_NAME[test_type])
    if test_type == TEST_NEURON_LOSS:
        variant_name = "random_connection_removal"
        xlabel = "Neurons Lost (%)"
    else:
        variant_name = f"schedule_{schedule_type}"
        xlabel = "Noise Strength (Hz)"
    variant_dir = os.path.join(output_root, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    settings = {
        "test_type": test_type,
        "variant": variant_name,
        "param_label": param_label,
        "param_values": [float(v) for v in param_values],
        "n_runs_per_condition": int(n_runs_per_condition),
        "sim_t_ms": float(sim_t),
        "avg_after_t_ms": float(avg_after_t),
        "sample_ms": float(sample_ms),
        "r1_inject": int(r1_inject),
        "goal_idx": int(goal_idx),
        "stimulus_half_width": int(stimulus_half_width),
        "schedule_type": schedule_type,
    }
    if test_type == TEST_NEURON_LOSS:
        settings["mechanism"] = (
            "Randomly choose param_value percent of ring neurons, then remove every "
            "connection touching them (ring<->ring recurrence and ring<->gain feedback, "
            "both directions) by zeroing the synaptic weight."
        )
    elif test_type == TEST_NOISE_GLOBAL:
        settings["noise_connection_fraction"] = float(GLOBAL_NOISE_CONNECTION_FRACTION)
        settings["noise_weight_pA"] = float(GLOBAL_NOISE_WEIGHT_PA)
        settings["mechanism"] = (
            "Randomly choose noise_connection_fraction of ring neurons and connect each "
            "to an independent poisson_generator with fixed weight; the swept/reported "
            "indicator is the Poisson rate (noise strength, Hz)."
        )
    else:
        settings["num_regions"] = int(LOCALIZED_NOISE_NUM_REGIONS)
        settings["region_half_width"] = int(LOCALIZED_NOISE_REGION_HALF_WIDTH)
        settings["noise_weight_pA"] = float(LOCAL_NOISE_WEIGHT_PA)
        settings["mechanism"] = (
            "Every run samples num_regions random region centers on the ring; each region "
            "(region_half_width neurons either side) is connected to an independent "
            "poisson_generator with fixed weight; the swept/reported indicator is the "
            "Poisson rate (noise strength, Hz, kept small)."
        )
    _save_experiment_settings_json(variant_dir, settings)

    worker_count = max_workers if max_workers is not None else os.cpu_count()
    if worker_count is None:
        worker_count = 1

    summary_rows = []

    for param_value in param_values:
        param_value = float(param_value)
        point_dir = os.path.join(variant_dir, _condition_folder_name(test_type, param_value))
        os.makedirs(point_dir, exist_ok=True)

        seeds = np.random.randint(0, 2**31, size=int(n_runs_per_condition), dtype=np.int64)

        tasks = []
        for run_idx in range(int(n_runs_per_condition)):
            robustness_cfg = _sample_robustness_config(
                test_type=test_type,
                param_value=param_value,
                population_size=int(load_json(ring_params_file)["population_size"]),
                rng_seed=int(seeds[run_idx]),
                schedule_type=schedule_type,
            )
            if test_type in (TEST_NOISE_LOCALIZED, TEST_NOISE_GLOBAL):
                robustness_cfg.end_ms = float(sim_t)

            representative_output_dir = None
            if run_idx == 0:
                representative_output_dir = os.path.join(point_dir, "representative_run")

            tasks.append(
                (
                    run_idx,
                    int(seeds[run_idx]),
                    r1_inject,
                    goal_idx,
                    sim_t,
                    sample_ms,
                    avg_after_t,
                    ring_params_file,
                    weights_dir,
                    font_size,
                    font_weight,
                    stimulus_half_width,
                    asdict(robustness_cfg),
                    representative_output_dir,
                )
            )

        run_results = [None] * int(n_runs_per_condition)
        local_worker_count = max(1, min(int(worker_count), int(n_runs_per_condition)))
        if local_worker_count == 1:
            for args in tasks:
                res = _run_robustness_worker(args)
                run_results[res["run_idx"]] = res
                print(
                    f"  {test_type} | {param_label}={param_value:g} "
                    f"| run {res['run_idx'] + 1}/{n_runs_per_condition}"
                )
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=local_worker_count, mp_context=ctx) as ex:
                futures = [ex.submit(_run_robustness_worker, args) for args in tasks]
                for fut in as_completed(futures):
                    res = fut.result()
                    run_results[res["run_idx"]] = res
                    print(
                        f"  {test_type} | {param_label}={param_value:g} "
                        f"| run {res['run_idx'] + 1}/{n_runs_per_condition}"
                    )

        run_metrics_csv = os.path.join(point_dir, "run_level_metrics.csv")
        _save_run_metrics_csv(run_results, run_metrics_csv)

        manifest_json = os.path.join(point_dir, "perturbation_manifest.json")
        _save_perturbation_manifest_json(run_results, manifest_json)

        common_bin_centers = np.asarray(run_results[0]["bin_centers"], dtype=float)
        all_goal_distances = np.vstack(
            [np.asarray(res["goal_distance"], dtype=float) for res in run_results]
        )
        np.savetxt(
            os.path.join(point_dir, "goal_distance_all_runs.csv"),
            all_goal_distances,
            delimiter=",",
        )

        _plot_goal_distance_mean_std(
            common_bin_centers,
            all_goal_distances,
            save_path=os.path.join(point_dir, "goal_distance_mean_std_across_runs.png"),
            title=(
                f"{test_type} ({variant_name}) | {param_label}={param_value:g} "
                f"(goal idx={goal_idx})"
            ),
            n_runs=int(n_runs_per_condition),
        )

        rel_errors = np.asarray(
            [res["avg_relative_error"] for res in run_results],
            dtype=float,
        )
        mean_goal_after = np.asarray(
            [res["mean_goal_distance_after_t"] for res in run_results],
            dtype=float,
        )

        rel_mean, rel_std = _nan_stats(rel_errors)
        goal_mean, goal_std = _nan_stats(mean_goal_after)

        summary_row = {
            "test_type": test_type,
            "variant": variant_name,
            "param_label": param_label,
            "param_value": param_value,
            "avg_relative_error_mean": rel_mean,
            "avg_relative_error_std": rel_std,
            "mean_goal_distance_after_t_mean": goal_mean,
            "mean_goal_distance_after_t_std": goal_std,
            "n_runs": int(n_runs_per_condition),
        }
        summary_rows.append(summary_row)

    summary_csv = os.path.join(variant_dir, "condition_summary.csv")
    with open(summary_csv, "w") as f:
        f.write(
            "test_type,variant,param_label,param_value,avg_relative_error_mean,"
            "avg_relative_error_std,mean_goal_distance_after_t_mean,"
            "mean_goal_distance_after_t_std,n_runs\n"
        )
        for row in summary_rows:
            f.write(
                f"{row['test_type']},{row['variant']},{row['param_label']},{row['param_value']},"
                f"{row['avg_relative_error_mean']},{row['avg_relative_error_std']},"
                f"{row['mean_goal_distance_after_t_mean']},"
                f"{row['mean_goal_distance_after_t_std']},{row['n_runs']}\n"
            )

    _plot_robustness_curve(
        summary_rows,
        save_path=os.path.join(variant_dir, "robustness_curve_relative_error.png"),
        metric_key="avg_relative_error",
        ylabel="Relative Error (distance / N)",
        title=f"Robustness Curve ({test_type}, {variant_name})",
        xlabel=xlabel,
    )
    _plot_robustness_curve(
        summary_rows,
        save_path=os.path.join(variant_dir, "robustness_curve_goal_distance_after_t.png"),
        metric_key="mean_goal_distance_after_t",
        ylabel="Mean Goal Distance (after threshold)",
        title=f"Post-Transient Goal Distance ({test_type}, {variant_name})",
        xlabel=xlabel,
    )

    print(f"\nFinished robustness condition: {test_type} ({variant_name})")
    print(f"  Outputs saved under {variant_dir}")
    print(f"  Settings saved to {os.path.join(variant_dir, 'experiment_settings.json')}")


def run_neuron_loss_robustness():
    print("\nStarting robustness experiment: neuron_loss")
    print(f"  neuron loss percentages: {NEURON_LOSS_PERCENTAGES}")
    print(f"  runs per condition point: {ROBUSTNESS_N_RUNS_PER_CONDITION}")

    run_robustness_condition(
        test_type=TEST_NEURON_LOSS,
        param_values=NEURON_LOSS_PERCENTAGES,
        param_label="neuron_loss_percentage",
        max_workers=ROBUSTNESS_MAX_WORKERS,
    )


def run_global_noise_robustness():
    print("\nStarting robustness experiment: noise_global")
    print(
        f"  global noise strengths (Hz): {GLOBAL_NOISE_STRENGTH_LEVELS_HZ} "
        f"(connection fraction={GLOBAL_NOISE_CONNECTION_FRACTION})"
    )
    print(f"  runs per condition point: {ROBUSTNESS_N_RUNS_PER_CONDITION}")
    print(f"  noise schedules: {ROBUSTNESS_NOISE_SCHEDULES}")

    for schedule in ROBUSTNESS_NOISE_SCHEDULES:
        run_robustness_condition(
            test_type=TEST_NOISE_GLOBAL,
            param_values=GLOBAL_NOISE_STRENGTH_LEVELS_HZ,
            param_label="noise_strength_hz",
            schedule_type=schedule,
            max_workers=ROBUSTNESS_MAX_WORKERS,
        )


def run_localized_noise_robustness():
    print("\nStarting robustness experiment: noise_localized")
    print(
        f"  localized noise strengths (Hz): {LOCAL_NOISE_STRENGTH_LEVELS_HZ} "
        f"(num_regions={LOCALIZED_NOISE_NUM_REGIONS}, half_width={LOCALIZED_NOISE_REGION_HALF_WIDTH})"
    )
    print(f"  runs per condition point: {ROBUSTNESS_N_RUNS_PER_CONDITION}")
    print(f"  noise schedules: {ROBUSTNESS_NOISE_SCHEDULES}")

    for schedule in ROBUSTNESS_NOISE_SCHEDULES:
        run_robustness_condition(
            test_type=TEST_NOISE_LOCALIZED,
            param_values=LOCAL_NOISE_STRENGTH_LEVELS_HZ,
            param_label="noise_strength_hz",
            schedule_type=schedule,
            max_workers=ROBUSTNESS_MAX_WORKERS,
        )


def run_all_robustness_experiments():
    run_neuron_loss_robustness()
    run_global_noise_robustness()
    run_localized_noise_robustness()


if __name__ == "__main__":
    run_all_robustness_experiments()
