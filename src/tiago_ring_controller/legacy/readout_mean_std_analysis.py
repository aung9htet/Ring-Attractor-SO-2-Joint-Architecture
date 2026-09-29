#!/usr/bin/env python3

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest

from ring_component import RingAttractorComponent
from homeostasis import HomeostasisModel
from tiago_ring_controller.config import load_json


@contextlib.contextmanager
def _suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = os.dup(1)
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.close(old_stdout)


def _collect_repeat_worker(params_file, center_idx, settle_ms, evaluate_ms, stimulus_half_width):
    with _suppress_stdout():
        seed = np.random.randint(0, 2**31)
        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")
        try:
            nest.rng_seed = int(seed)
        except Exception:
            nest.SetKernelStatus({"rng_seed": int(seed)})

        ring_1 = RingAttractorComponent(params_file=params_file, reset_kernel=False)
        ring_2 = RingAttractorComponent(params_file=params_file, reset_kernel=False)
        model = HomeostasisModel(ring_1, ring_2)

        ring_1._inject_bump(center_idx=int(center_idx), half_width=stimulus_half_width)
        ring_2._inject_bump(center_idx=int(center_idx), half_width=stimulus_half_width)

        nest.Simulate(settle_ms)
        model.evaluate(duration_ms=evaluate_ms)

        warm_count = len(model.homeostasis_recorders["warm_spike"].get("events").get("times", []))
        cold_count = len(model.homeostasis_recorders["cold_spike"].get("events").get("times", []))

    return int(center_idx), float((warm_count + cold_count) / 2.0)


class ReadoutMeanStdAnalysis:

    _TIC_VALS = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    _TIC_LBLS = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    def __init__(self, params_file="./config/model_params/ring_params.json",
                 homeostasis_params_file="./config/model_params/homeostasis_params.json"):
        ring_params = load_json(params_file)
        self.population_size = ring_params["population_size"]
        self.params_file = params_file
        self.homeostasis_params_file = homeostasis_params_file

    def collect(self, center_indices, settle_ms=300.0, evaluate_ms=100.0,
                stimulus_half_width=5, num_repeats=10, max_workers=None):
        N = self.population_size
        results = []
        counts_by_center = {int(center_idx): [] for center_idx in center_indices}
        jobs = [
            (
                self.params_file,
                int(center_idx),
                float(settle_ms),
                float(evaluate_ms),
                int(stimulus_half_width),
            )
            for center_idx in center_indices
            for _ in range(int(num_repeats))
        ]
        total = len(jobs)
        if max_workers is None:
            max_workers = min(total, max(1, min(os.cpu_count() or 1, 8)))
        else:
            max_workers = min(total, max(1, int(max_workers)))
        ctx = mp.get_context("spawn")

        print(f"  launching {max_workers} workers for {total} repeats...", flush=True)

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futures = [pool.submit(_collect_repeat_worker, *job) for job in jobs]
            for done, future in enumerate(as_completed(futures), start=1):
                center_idx, value = future.result()
                counts_by_center[center_idx].append(float(value))
                bar_len = 30
                bar_filled = int(done / total * bar_len)
                bar = "\u2588" * bar_filled + "\u2591" * (bar_len - bar_filled)
                print(f"\r  collecting [{bar}] {done}/{total}", end="", flush=True)

        for center_idx in center_indices:
            phase = RingAttractorComponent.center_index_to_phase(int(center_idx), N)
            trial_counts = np.array(counts_by_center[int(center_idx)], dtype=float)
            results.append({
                "center_idx": int(center_idx),
                "phase": float(phase),
                "mean": float(np.mean(trial_counts)),
                "std": float(np.std(trial_counts, ddof=1)) if num_repeats > 1 else 0.0,
            })

        print()
        return results

    def visualize(self, results, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        phases = np.array([r["phase"] for r in results])
        means = np.array([r["mean"] for r in results])
        stds = np.array([r["std"] for r in results])

        order = np.argsort(phases)
        phases, means, stds = phases[order], means[order], stds[order]

        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        ax.errorbar(phases, means, yerr=stds, fmt="o", linestyle="none",
                    color="#1f77b4", ecolor="#333333", elinewidth=0.95,
                    capsize=2.7, capthick=0.95, markersize=5.3,
                    markeredgecolor="white", markeredgewidth=0.55, zorder=3)

        ax.set_title("Homeostasis Comparator Readout", fontsize=12, fontweight="bold", pad=9)
        ax.set_xlabel("Ring phase (rad)", fontsize=12, fontweight="bold", labelpad=7)
        ax.set_ylabel("Comparator spike count", fontsize=12, fontweight="bold", labelpad=7)
        ax.set_xticks(self._TIC_VALS)
        ax.set_xticklabels(self._TIC_LBLS, fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(max(0, np.min(means - stds) - 6), np.max(means + stds) + 6)
        ax.tick_params(axis="x", pad=5)
        ax.tick_params(axis="y", labelsize=10.5, pad=4)
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.22)
        ax.grid(axis="x", visible=False)
        fig.subplots_adjust(left=0.17, bottom=0.21, right=0.98, top=0.88)

        path = os.path.join(save_dir, "homeostasis_comparator_readout.png")
        fig.savefig(path, dpi=500, facecolor="white")
        plt.close(fig)
        print(f"Plot saved to {path}")

    def analyze(self, num_positions=25, settle_ms=300.0, evaluate_ms=100.0,
                stimulus_half_width=5, num_repeats=10, save_dir="./outputs/homeostasis",
                max_workers=None):
        N = self.population_size
        center_indices = RingAttractorComponent.generate_center_indices(N, num_positions)
        print(f"ReadoutMeanStdAnalysis — {num_positions} positions × {num_repeats} repeats, N={N}...")
        results = self.collect(
            center_indices,
            settle_ms,
            evaluate_ms,
            stimulus_half_width,
            num_repeats,
            max_workers=max_workers,
        )
        self.visualize(results, save_dir)
        return results


if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(src_dir, "config", "model_params", "ring_params.json")
    h_params_file = os.path.join(src_dir, "config", "model_params", "homeostasis_params.json")

    analysis = ReadoutMeanStdAnalysis(
        params_file=params_file,
        homeostasis_params_file=h_params_file,
    )
    analysis.analyze(
        num_positions=25,
        settle_ms=300.0,
        evaluate_ms=100.0,
        stimulus_half_width=5,
        num_repeats=10,
        save_dir=os.path.join(src_dir, "outputs", "homeostasis"),
    )
