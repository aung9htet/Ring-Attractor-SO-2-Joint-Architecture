#!/usr/bin/env python3

import os
import sys
import json
import contextlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest

from ring_component import RingAttractorComponent
from homeostasis import HomeostasisModel
from gain_modulation import GainModulationModel
from tiago_ring_controller.config import load_json

# --- Config ---
PARAMS_FILE         = "./config/model_params/ring_params.json"
NUM_POSITIONS       = 10        # positions per ring sampled from interior indices
SIM_MS              = 2000.0     # simulation duration per trial
SETTLE_MS           = 300.0     # steady-state window starts here
STIMULUS_HALF_WIDTH = 5
SWEEP_OUTPUT_DIR    = "./outputs/sweep_gain_modulation"
SAVE_RASTERS        = True
NUM_WORKERS         = min(mp.cpu_count(), NUM_POSITIONS * NUM_POSITIONS)


@contextlib.contextmanager
def _suppress_stdout():
    """Silence NEST's stdout during build/run."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old

class GainModulationSimulation:
    """Single-trial AND-gate gain modulation demo."""

    def __init__(
        self,
        params_file:         str = PARAMS_FILE,
        ring1_center_idx:    int = 40,
        ring2_center_idx:    int = 10,
        stimulus_half_width: int = STIMULUS_HALF_WIDTH,
    ):
        self.ring1_center_idx = ring1_center_idx
        self.ring2_center_idx = ring2_center_idx

        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")

        self.ring_1 = RingAttractorComponent(params_file=params_file, reset_kernel=False)
        self.ring_2 = RingAttractorComponent(params_file=params_file, reset_kernel=False)
        self.population_size = self.ring_1.population_size

        self.homeostasis = HomeostasisModel(ring_1=self.ring_1, ring_2=self.ring_2)
        self.gain_model  = GainModulationModel(
            ring_attractor=self.ring_1,
            homeostasis=self.homeostasis,
        )

        self.ring_1._inject_bump(ring1_center_idx, stimulus_half_width)
        self.ring_2._inject_bump(ring2_center_idx, stimulus_half_width)

    def simulate(self, duration_ms: float = 1000.0):
        nest.Simulate(duration_ms)

    def _collect_spikes(self, recorders: list) -> list:
        return [
            np.array(nest.GetStatus(sr, "events")[0].get("times", []))
            for sr in recorders
        ]

    def plot(self, output_dir: str = "./outputs/gain_modulation", settle_t: float = SETTLE_MS):
        os.makedirs(output_dir, exist_ok=True)

        ring_spikes  = self._collect_spikes(self.ring_1.ring_attractor.ring_spike_recorders)
        left_spikes  = self._collect_spikes(self.gain_model.left_gain_spike_recorders)
        right_spikes = self._collect_spikes(self.gain_model.right_gain_spike_recorders)
        homeo_times  = {
            label: np.array(
                self.homeostasis.homeostasis_recorders[key].get("events").get("times", [])
            )
            for label, key in [
                ("warm",  "warm_spike"), ("cold", "cold_spike"),
                ("LEFT",  "left_spike"), ("RIGHT", "right_spike"),
            ]
        }

        fig, axes = plt.subplots(4, 1, figsize=(16, 22), sharex=True)
        ax_ring, ax_homeo, ax_left, ax_right = axes

        for idx, times in enumerate(ring_spikes):
            ax_ring.plot(times, np.full(len(times), idx), ".", color="#1f77b4", markersize=1.5)
        ax_ring.set_ylabel("Neuron Index", fontsize=14)
        ax_ring.set_title(f"Ring-1 Attractor  (bump idx {self.ring1_center_idx})", fontsize=16)
        ax_ring.set_ylim([-1, self.population_size - 1])
        ax_ring.axhline(self.ring1_center_idx, color="gray", lw=0.8, ls="--", alpha=0.6)

        palette = {"warm": "#ff7f0e", "cold": "#2ca02c", "LEFT": "#d62728", "RIGHT": "#9467bd"}
        y_pos   = {"warm": 0, "cold": 1, "LEFT": 2, "RIGHT": 3}
        for label, times in homeo_times.items():
            ax_homeo.plot(times, np.full(len(times), y_pos[label]), "|",
                          color=palette[label], markersize=8, label=label)
        ax_homeo.set_yticks([0, 1, 2, 3])
        ax_homeo.set_yticklabels(["warm", "cold", "LEFT", "RIGHT"], fontsize=11)
        ax_homeo.set_ylabel("Unit", fontsize=14)
        ax_homeo.set_title("Homeostasis Decision Neurons", fontsize=16)
        ax_homeo.legend(loc="upper right", fontsize=10)
        ax_homeo.set_ylim([-0.5, 3.5])

        for idx, times in enumerate(left_spikes):
            ax_left.plot(times, np.full(len(times), idx), ".", color="#d62728", markersize=1.5)
        ax_left.set_ylabel("Neuron Index", fontsize=14)
        ax_left.set_title("Left Gain  (AND: ring bump ∩ left homeostasis)", fontsize=16)
        ax_left.set_ylim([-1, self.population_size - 1])
        ax_left.axhline(self.ring1_center_idx, color="gray", lw=0.8, ls="--", alpha=0.6)

        for idx, times in enumerate(right_spikes):
            ax_right.plot(times, np.full(len(times), idx), ".", color="#9467bd", markersize=1.5)
        ax_right.set_ylabel("Neuron Index", fontsize=14)
        ax_right.set_title("Right Gain  (AND: ring bump ∩ right homeostasis)", fontsize=16)
        ax_right.set_ylim([-1, self.population_size - 1])
        ax_right.axhline(self.ring1_center_idx, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax_right.set_xlabel("Time (ms)", fontsize=14)

        for ax in axes:
            ax.axvline(settle_t, color="black", lw=1.2, ls=":", alpha=0.7)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
        axes[0].legend(["settle boundary"], loc="upper right", fontsize=9)

        fig.suptitle(
            f"Gain Modulation  |  Ring-1 idx={self.ring1_center_idx}  ·  Ring-2 idx={self.ring2_center_idx}\n"
            f"(dotted line = settled at t={settle_t:.0f} ms)",
            fontsize=16,
        )
        plt.tight_layout()
        save_path = os.path.join(output_dir, "raster.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        n_left  = sum(1 for t in left_spikes  if np.any(t > settle_t))
        n_right = sum(1 for t in right_spikes if np.any(t > settle_t))
        print(f"  Left  gain active (t>{settle_t:.0f}ms): {n_left}/{self.population_size}")
        print(f"  Right gain active (t>{settle_t:.0f}ms): {n_right}/{self.population_size}")
        print(f"  Raster saved → {save_path}")

def build_and_run(r1_idx: int, r2_idx: int, collect_spikes: bool = False) -> dict:
    """Build a fresh network, simulate, and return spike arrays + steady-state counts."""
    with _suppress_stdout():
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")

        ring_1 = RingAttractorComponent(params_file=PARAMS_FILE, reset_kernel=False)
        ring_2 = RingAttractorComponent(params_file=PARAMS_FILE, reset_kernel=False)
        N = ring_1.population_size

        homeostasis = HomeostasisModel(ring_1=ring_1, ring_2=ring_2)
        gain_model  = GainModulationModel(ring_attractor=ring_1, homeostasis=homeostasis)

        ring_1._inject_bump(r1_idx, STIMULUS_HALF_WIDTH)
        ring_2._inject_bump(r2_idx, STIMULUS_HALF_WIDTH)
        nest.Simulate(SIM_MS)

    def _spikes(recorders):
        return [np.array(nest.GetStatus(sr, "events")[0].get("times", [])) for sr in recorders]

    left_events = [np.array(nest.GetStatus(sr, "events")[0].get("times", [])) for sr in gain_model.left_gain_spike_recorders]
    right_events = [np.array(nest.GetStatus(sr, "events")[0].get("times", [])) for sr in gain_model.right_gain_spike_recorders]
    left_ss  = [i for i, t in enumerate(left_events) if np.any(t > SETTLE_MS)]
    right_ss = [i for i, t in enumerate(right_events) if np.any(t > SETTLE_MS)]

    result = {
        "N": N,
        "left_ss_count": len(left_ss),
        "right_ss_count": len(right_ss),
        "left_ss_active": left_ss,
        "right_ss_active": right_ss,
    }

    if collect_spikes:
        result.update({
            "ring_spikes": _spikes(ring_1.ring_attractor.ring_spike_recorders),
            "left_spikes": left_events,
            "right_spikes": right_events,
            "homeo_times": {
                label: np.array(homeostasis.homeostasis_recorders[key].get("events").get("times", []))
                for label, key in [
                    ("warm", "warm_spike"), ("cold", "cold_spike"),
                    ("LEFT", "left_spike"), ("RIGHT", "right_spike"),
                ]
            },
        })

    return result


def save_raster(result: dict, r1_idx: int, r2_idx: int, out_dir: str):
    """Save a 4-panel raster for one sweep trial."""
    N = result["N"]
    fig, axes = plt.subplots(4, 1, figsize=(16, 22), sharex=True)
    ax_ring, ax_homeo, ax_left, ax_right = axes

    for idx, times in enumerate(result["ring_spikes"]):
        ax_ring.plot(times, np.full(len(times), idx), ".", color="#1f77b4", markersize=1.5)
    ax_ring.set_ylabel("Neuron Index", fontsize=13)
    ax_ring.set_title(f"Ring-1 Attractor  (bump idx {r1_idx})", fontsize=15)
    ax_ring.set_ylim([-1, N])
    ax_ring.axhline(r1_idx, color="gray", lw=0.8, ls="--", alpha=0.6)

    palette = {"warm": "#ff7f0e", "cold": "#2ca02c", "LEFT": "#d62728", "RIGHT": "#9467bd"}
    y_pos   = {"warm": 0, "cold": 1, "LEFT": 2, "RIGHT": 3}
    for label, times in result["homeo_times"].items():
        ax_homeo.plot(times, np.full(len(times), y_pos[label]), "|",
                      color=palette[label], markersize=8, label=label)
    ax_homeo.set_yticks([0, 1, 2, 3])
    ax_homeo.set_yticklabels(["warm", "cold", "LEFT", "RIGHT"], fontsize=10)
    ax_homeo.set_ylabel("Unit", fontsize=13)
    ax_homeo.set_title("Homeostasis Decision Neurons", fontsize=15)
    ax_homeo.legend(loc="upper right", fontsize=9)
    ax_homeo.set_ylim([-0.5, 3.5])

    for idx, times in enumerate(result["left_spikes"]):
        ax_left.plot(times, np.full(len(times), idx), ".", color="#d62728", markersize=1.5)
    ax_left.set_ylabel("Neuron Index", fontsize=13)
    ax_left.set_title(f"Left Gain  (active: {result['left_ss_count']})", fontsize=15)
    ax_left.set_ylim([-1, N])
    ax_left.axhline(r1_idx, color="gray", lw=0.8, ls="--", alpha=0.6)

    for idx, times in enumerate(result["right_spikes"]):
        ax_right.plot(times, np.full(len(times), idx), ".", color="#9467bd", markersize=1.5)
    ax_right.set_ylabel("Neuron Index", fontsize=13)
    ax_right.set_title(f"Right Gain  (active: {result['right_ss_count']})", fontsize=15)
    ax_right.set_ylim([-1, N])
    ax_right.axhline(r1_idx, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_right.set_xlabel("Time (ms)", fontsize=13)

    for ax in axes:
        ax.axvline(SETTLE_MS, color="black", lw=1.2, ls=":", alpha=0.7)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)

    fig.suptitle(
        f"Gain Modulation  |  Ring-1 idx={r1_idx}  ·  Ring-2 idx={r2_idx}\n"
        f"(dotted line = settled at t={SETTLE_MS:.0f} ms)",
        fontsize=16,
    )
    plt.tight_layout()
    fname = os.path.join(out_dir, f"raster_r1{r1_idx:03d}_r2{r2_idx:03d}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()


def _worker(args):
    """Top-level picklable worker for ProcessPoolExecutor."""
    ri, rj, r1_idx, r2_idx = args
    result = build_and_run(r1_idx, r2_idx, collect_spikes=SAVE_RASTERS)
    return ri, rj, r1_idx, r2_idx, result["left_ss_count"], result["right_ss_count"], result


def run_sweep() -> tuple:
    ring_params = load_json(PARAMS_FILE)
    N = ring_params["population_size"]
    edge_margin = 10
    if N <= (2 * edge_margin):
        raise ValueError(f"population_size={N} is too small for edge_margin={edge_margin}")
    positions = np.linspace(edge_margin, N - edge_margin, NUM_POSITIONS, dtype=int).tolist()

    left_map  = np.zeros((NUM_POSITIONS, NUM_POSITIONS), dtype=float)
    right_map = np.zeros((NUM_POSITIONS, NUM_POSITIONS), dtype=float)

    raster_dir = os.path.join(SWEEP_OUTPUT_DIR, "rasters")
    if SAVE_RASTERS:
        os.makedirs(raster_dir, exist_ok=True)

    jobs = [
        (ri, rj, positions[ri], positions[rj])
        for ri in range(NUM_POSITIONS)
        for rj in range(NUM_POSITIONS)
    ]
    total = len(jobs)
    done  = 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as pool:
        futures = {pool.submit(_worker, job): job for job in jobs}
        for fut in as_completed(futures):
            ri, rj, r1_idx, r2_idx, left_cnt, right_cnt, result = fut.result()
            left_map[ri, rj]  = left_cnt
            right_map[ri, rj] = right_cnt
            done += 1
            pct    = done / total
            filled = int(pct * 40)
            bar    = "█" * filled + "░" * (40 - filled)
            sys.stdout.write(f"\r  [{bar}] {done:3d}/{total}  ({pct*100:.0f}%)")
            sys.stdout.flush()
            if SAVE_RASTERS:
                save_raster(result, r1_idx, r2_idx, raster_dir)

    print()
    return positions, left_map, right_map

def _base_heatmap(ax, data, cmap, title, positions, vmax=None):
    ticks = list(range(NUM_POSITIONS))
    labels = [str(p) for p in positions]
    im = ax.imshow(data, origin="lower", aspect="auto",
                    cmap=cmap, vmin=0, vmax=vmax or data.max() or 1)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Ring-1 (current) bump index",   fontsize=12)
    ax.set_ylabel("Ring-2 (reference) bump index", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.plot(ticks, ticks, "w--", lw=1.2, alpha=0.7, label="r1=r2")
    ax.legend(fontsize=8, loc="upper left")
    return im

def plot_heatmaps(positions: list, left_map: np.ndarray, right_map: np.ndarray):
    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)
    labels = [str(p) for p in positions]
    ticks  = list(range(NUM_POSITIONS))

    vmax = max(left_map.max(), right_map.max(), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = _base_heatmap(ax, left_map, "Reds", "Left Gain — steady-state active neurons", positions, vmax)
    fig.colorbar(im, ax=ax, label="active neuron count")
    plt.tight_layout()
    plt.savefig(os.path.join(SWEEP_OUTPUT_DIR, "left_heatmap.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = _base_heatmap(ax, right_map, "Blues", "Right Gain — steady-state active neurons", positions, vmax)
    fig.colorbar(im, ax=ax, label="active neuron count")
    plt.tight_layout()
    plt.savefig(os.path.join(SWEEP_OUTPUT_DIR, "right_heatmap.png"), dpi=150)
    plt.close()

    dom_map = left_map - right_map
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = _base_heatmap(axes[0], left_map,  "Reds",  "Left Gain active count",  positions, vmax)
    fig.colorbar(im0, ax=axes[0], label="count", fraction=0.046)
    im1 = _base_heatmap(axes[1], right_map, "Blues", "Right Gain active count", positions, vmax)
    fig.colorbar(im1, ax=axes[1], label="count", fraction=0.046)

    vlim = max(abs(dom_map).max(), 1)
    im2 = axes[2].imshow(dom_map, origin="lower", aspect="auto",
                         cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[2].set_xticks(ticks); axes[2].set_xticklabels(labels, fontsize=10)
    axes[2].set_yticks(ticks); axes[2].set_yticklabels(labels, fontsize=10)
    axes[2].set_xlabel("Ring-1 (current) bump index",   fontsize=12)
    axes[2].set_ylabel("Ring-2 (reference) bump index", fontsize=12)
    axes[2].set_title("Left − Right  (red=left dominant, blue=right dominant)", fontsize=13)
    axes[2].plot(ticks, ticks, "k--", lw=1.2, alpha=0.5, label="r1=r2")
    axes[2].legend(fontsize=8, loc="upper left")
    fig.colorbar(im2, ax=axes[2], label="left − right", fraction=0.046)

    fig.suptitle(
        f"Gain Modulation Sweep  |  {NUM_POSITIONS}×{NUM_POSITIONS} positions  "
        f"(steady-state t > {SETTLE_MS:.0f} ms)",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(SWEEP_OUTPUT_DIR, "combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmaps saved → {SWEEP_OUTPUT_DIR}/")

if __name__ == "__main__":
    # --- Single demo (ring-1 ahead of ring-2 → LEFT wins) ---
    print("=== Single-trial demo ===")
    sim = GainModulationSimulation(ring1_center_idx=40, ring2_center_idx=10)
    sim.simulate(duration_ms=1000.0)
    sim.plot()

    # --- Sweep all 10×10 position pairs ---
    print(f"\n=== Sweep {NUM_POSITIONS}×{NUM_POSITIONS} = {NUM_POSITIONS**2} trials ===")
    print(f"  {SIM_MS} ms/trial  |  settle cutoff: {SETTLE_MS} ms\n")
    positions, left_map, right_map = run_sweep()
    print("\nLeft map:\n",  left_map.astype(int))
    print("\nRight map:\n", right_map.astype(int))
    plot_heatmaps(positions, left_map, right_map)
    print("Done.")
