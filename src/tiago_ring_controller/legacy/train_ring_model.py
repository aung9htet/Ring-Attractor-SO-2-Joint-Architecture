#!/usr/bin/env python3

import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest
from ring_attractor import Ring_Attractor

from tiago_ring_controller.config import load_json
from tiago_ring_controller.artifacts import (
    save_json_legacy,
    save_npz_legacy,
    save_numpy_legacy,
)
from tiago_ring_controller.math.circular import preferred_angles
from tiago_ring_controller.math.fourier import (
    build_push_pull_sine_weights,
    preprocess_activity,
)


def _run_fourier_trial_worker(args):
    (
        idx,
        rep,
        params_file,
        population_size,
        num_positions,
        stimulus_half_width,
        sim_settle_ms,
    ) = args

    center_index = (idx * population_size) // num_positions
    phi_true = 2 * np.pi * center_index / population_size

    model = RingAttractorSetup(params_file=params_file)
    theta = 2 * np.pi * np.arange(population_size) / population_size
    layer = model.build_fourier_layer(theta)

    before = {
        "sin_pos": model._get_spike_counts(layer["sin_pos_recs"]),
        "sin_neg": model._get_spike_counts(layer["sin_neg_recs"]),
    }

    model._inject_bump(center_idx=center_index, half_width=stimulus_half_width)
    nest.Simulate(sim_settle_ms)

    after = {
        "sin_pos": model._get_spike_counts(layer["sin_pos_recs"]),
        "sin_neg": model._get_spike_counts(layer["sin_neg_recs"]),
    }

    sin_pos_delta = after["sin_pos"] - before["sin_pos"]
    sin_neg_delta = after["sin_neg"] - before["sin_neg"]
    sin_counts = sin_pos_delta - sin_neg_delta
    phi_est_fourier = np.arcsin(np.clip(sin_counts[0], -1.0, 1.0)) if len(sin_counts) > 0 else np.nan

    return idx, rep, phi_true, phi_est_fourier, sin_counts, sin_pos_delta, sin_neg_delta


class RingAttractorSetup:

    def __init__(self, params_file: str = "./config/model_params/ring_params.json"):
        params = load_json(params_file)
        self.num_fourier_k = params["num_fourier_k"]
        self.output_weight_scale = params["readout_weight_scale"]
        self.output_dc_baseline = params["output_dc_baseline"]
        neuron_params_file = os.path.join(os.path.dirname(params_file), "neuron_params.json")
        self.ring_attractor = Ring_Attractor(population_size=params["population_size"], reset_kernel=True, params_file=neuron_params_file)

    def _create_output_neurons(self, n: int):
        nodes = nest.Create('iaf_psc_alpha', n)
        recs = nest.Create('spike_recorder', n)
        dc = nest.Create('dc_generator', params={'amplitude': self.output_dc_baseline})
        nest.Connect(dc, nodes)
        for i in range(n):
            nest.Connect(nodes[i], recs[i])
        return nodes, recs

    def _connect_from_ring_with_weights(self, out_neuron, weights: np.ndarray):
        for i, src in enumerate(self.ring_attractor.ring_neurons):
            w = float(self.output_weight_scale * weights[i])
            nest.Connect(src, out_neuron, syn_spec={'weight': w})

    def build_fourier_layer(self, theta: np.ndarray):
        K = self.num_fourier_k
        nodes, recs = self._create_output_neurons(2 * K)
        sin_pos_recs, sin_neg_recs = [], []
        for k in range(1, K + 1):
            w_sin = np.sin(k * theta)

            w_sin_pos = np.maximum(w_sin, 0.0)
            w_sin_neg = np.maximum(-w_sin, 0.0)

            i0 = 2 * (k - 1)
            node_sin_pos, node_sin_neg = nodes[i0:i0+2]
            rec_sin_pos,  rec_sin_neg  = recs[i0:i0+2]

            self._connect_from_ring_with_weights(node_sin_pos, w_sin_pos)
            self._connect_from_ring_with_weights(node_sin_neg, w_sin_neg)

            sin_pos_recs.append(rec_sin_pos)
            sin_neg_recs.append(rec_sin_neg)

        return {
            'sin_pos_recs': sin_pos_recs,
            'sin_neg_recs': sin_neg_recs,
        }

    def _get_spike_counts(self, recorders) -> np.ndarray:
        counts = []
        for sr in recorders:
            ev = sr.get("events")
            counts.append(len(ev.get("times", [])))
        return np.array(counts, dtype=float)

    def _get_ring_spike_counts(self) -> np.ndarray:
        return self.ring_attractor._get_spike_counts()

    def _inject_bump(self, center_idx: int, half_width: int):
        self.ring_attractor.inject_stimulus(center_index=center_idx, half_width=half_width)

class DecoderTrainer:

    def __init__(self, params_file: str = "./config/model_params/ring_params.json"):
        params = load_json(params_file)
        self.params_file = params_file
        self.num_positions = params["num_positions"]
        self.sim_settle_ms = params["sim_settle_ms"]
        self.stimulus_half_width = params["stimulus_half_width"]
        self.repeats_per_position = params["samples_per_position"]
        self.num_fourier_k = params["num_fourier_k"]
        self.population_size = params["population_size"]
        self.config_dir = params["config_dir"]
        self.results = []

    def _preferred_angles(self, N: int) -> np.ndarray:
        return preferred_angles(N)

    def _preprocess_activity(self, r: np.ndarray) -> np.ndarray:
        # zero-mean, L2-normalized
        return preprocess_activity(r)

    def run(self, save_dir: str = "./outputs/ring_decoding", max_workers: int = None):
        os.makedirs(save_dir, exist_ok=True)
        Npos = self.num_positions
        print(f"Running Fourier decoding across {Npos} ring positions...")

        total_tasks = Npos * self.repeats_per_position
        worker_count = max_workers if max_workers is not None else (os.cpu_count() or 1)
        worker_count = max(1, min(int(worker_count), int(total_tasks)))

        tasks = [
            (
                idx,
                rep,
                self.params_file,
                self.population_size,
                self.num_positions,
                self.stimulus_half_width,
                self.sim_settle_ms,
            )
            for idx in range(Npos)
            for rep in range(self.repeats_per_position)
        ]

        by_idx = {
            idx: {
                "phi_true": np.nan,
                "repeats": [],
            }
            for idx in range(Npos)
        }

        completed = 0
        if worker_count == 1:
            for task in tasks:
                idx, rep, phi_true, phi_est_fourier, sin_counts, sin_pos_delta, sin_neg_delta = _run_fourier_trial_worker(task)
                by_idx[idx]["phi_true"] = phi_true
                by_idx[idx]["repeats"].append((rep, phi_est_fourier, sin_counts, sin_pos_delta, sin_neg_delta))
                completed += 1
                ratio = completed / total_tasks
                bar_n = min(Npos, max(1, int(np.ceil(ratio * Npos))))
                bar = "\u2588" * bar_n + "\u2591" * (Npos - bar_n)
                print(f"\r[{bar}] {completed}/{total_tasks}", end="", flush=True)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as ex:
                futures = [ex.submit(_run_fourier_trial_worker, task) for task in tasks]
                for fut in as_completed(futures):
                    idx, rep, phi_true, phi_est_fourier, sin_counts, sin_pos_delta, sin_neg_delta = fut.result()
                    by_idx[idx]["phi_true"] = phi_true
                    by_idx[idx]["repeats"].append((rep, phi_est_fourier, sin_counts, sin_pos_delta, sin_neg_delta))
                    completed += 1
                    ratio = completed / total_tasks
                    bar_n = min(Npos, max(1, int(np.ceil(ratio * Npos))))
                    bar = "\u2588" * bar_n + "\u2591" * (Npos - bar_n)
                    print(f"\r[{bar}] {completed}/{total_tasks}", end="", flush=True)

        self.results = []
        for idx in range(Npos):
            pos_result = {
                "phi_true": by_idx[idx]["phi_true"],
                "phi_est_fourier_list": [],
            }
            repeats = sorted(by_idx[idx]["repeats"], key=lambda x: x[0])
            for _rep, phi_est_fourier, sin_counts, sin_pos_delta, sin_neg_delta in repeats:
                pos_result["phi_est_fourier_list"].append(phi_est_fourier)
                pos_result["fourier_counts_sin"] = sin_counts
                pos_result["sin_pos_delta"] = sin_pos_delta
                pos_result["sin_neg_delta"] = sin_neg_delta
            pos_result["phi_est_fourier"] = pos_result["phi_est_fourier_list"][-1] if pos_result["phi_est_fourier_list"] else np.nan
            self.results.append(pos_result)
        print()

        self._visualize(save_dir)
        self._save_summary(save_dir)

    def _visualize(self, save_dir: str):
        phi_true = np.array([d['phi_true'] for d in self.results])
        phi_est_fourier = np.array([d['phi_est_fourier'] for d in self.results])

        def circ_err(est, true):
            e = np.angle(np.exp(1j * (est - true)))
            return e, np.abs(e)

        errF, abserrF = circ_err(phi_est_fourier, phi_true)

        # sin component activity vs true phase
        y_sin_pos = np.array([d['sin_pos_delta'][0] for d in self.results])
        y_sin_neg = np.array([d['sin_neg_delta'][0] for d in self.results])
        y_sin_diff = np.array([d['fourier_counts_sin'][0] for d in self.results])

        order = np.argsort(phi_true)
        rc = {
            "font.family": "sans-serif", "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 12,
            "axes.linewidth": 1.2,
            "axes.spines.top": False, "axes.spines.right": False,
            "xtick.direction": "out", "ytick.direction": "out",
            "legend.frameon": False, "legend.fontsize": 10,
        }
        with plt.rc_context(rc):
            fig1, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(8, 5.5), sharex=True,
                gridspec_kw={"hspace": 0.08},
            )
            ax_top.plot(phi_true[order], y_sin_pos[order],
                        color="#2980b9", lw=1.8, marker="o", ms=4,
                        markeredgewidth=0)
            ax_top.plot(phi_true[order], y_sin_neg[order],
                        color="#27ae60", lw=1.8, marker="s", ms=4,
                        markeredgewidth=0)
            ax_top.set_ylabel("Spike count")
            ax_top.set_title(r"Push-pull $\sin$ readout neurons ($k=1$)")
            ax_top.grid(axis="y", alpha=0.25, lw=0.8)
            # label at peak of each curve, offset upward
            idx_sp = int(np.argmax(y_sin_pos[order]))
            idx_sn = int(np.argmax(y_sin_neg[order]))
            ax_top.annotate("sine⁺",
                            xy=(phi_true[order][idx_sp], y_sin_pos[order][idx_sp]),
                            xytext=(0, -20), textcoords="offset points",
                            color="#2980b9", fontsize=12, va="bottom",
                            fontweight="bold", ha="center")
            ax_top.annotate("sine⁻",
                            xy=(phi_true[order][idx_sn], y_sin_neg[order][idx_sn]),
                            xytext=(0, -20), textcoords="offset points",
                            color="#27ae60", fontsize=12, va="bottom",
                            fontweight="bold", ha="center")

            ax_bot.plot(phi_true[order], y_sin_diff[order],
                        color="#c0392b", lw=1.8, marker="o", ms=4,
                        markeredgewidth=0)
            ax_bot.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
            ax_bot.set_xlabel(r"Stimulus phase $\varphi$ (rad)")
            ax_bot.set_ylabel("Net spike count")
            ax_bot.grid(axis="y", alpha=0.25, lw=0.8)
            idx_sd = int(np.argmax(y_sin_diff[order]))
            ax_bot.annotate("sine\u207a \u2212 sine\u207b",
                            xy=(phi_true[order][idx_sd], y_sin_diff[order][idx_sd]),
                            xytext=(0, -20), textcoords="offset points",
                            color="#c0392b", fontsize=12, va="bottom",
                            fontweight="bold", ha="center")
            ax_bot.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax_bot.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

            fig1.savefig(os.path.join(save_dir, "sin_readout_activity.png"), dpi=200, bbox_inches="tight")
            plt.close(fig1)

        # per-position error
        order = np.argsort(phi_true)
        with plt.rc_context(rc):
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            box_data = [
                np.abs(np.angle(np.exp(1j*(np.array(d['phi_est_fourier_list']) - d['phi_true']))))
                for d in np.array(self.results)[order]
            ]
            positions = phi_true[order]
            bp = ax2.boxplot(
                box_data, positions=positions,
                widths=(positions[1] - positions[0]) * 0.6 if len(positions) > 1 else 0.2,
                patch_artist=True,
                medianprops=dict(color="#c0392b", lw=2),
                boxprops=dict(facecolor="#2980b9", alpha=0.4, linewidth=1.2),
                whiskerprops=dict(lw=1.2, color="#2c3e50"),
                capprops=dict(lw=1.2, color="#2c3e50"),
                flierprops=dict(marker="o", ms=3, markerfacecolor="#7f8c8d",
                                markeredgewidth=0, alpha=0.5),
                manage_ticks=False,
            )
            ax2.set_xlabel(r"Stimulus phase $\varphi$ (rad)")
            ax2.set_ylabel(r"Absolute phase error $|\Delta\varphi|$ (rad)")
            ax2.set_title("Phase decoding error across ring positions")
            ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax2.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
            ax2.set_xlim(-0.2, 2*np.pi + 0.2)
            ax2.set_ylim(bottom=0)
            ax2.grid(axis="y", alpha=0.25, lw=0.8)
            fig2.savefig(os.path.join(save_dir, 'decoder_error_per_position.png'), dpi=200, bbox_inches="tight")
            plt.close(fig2)

        print(f"\nFourier mean error: {np.mean(abserrF):.3f} rad  max: {np.max(abserrF):.3f} rad\n")

    def _save_summary(self, save_dir: str):
        N = self.population_size
        K = self.num_fourier_k
        theta = self._preferred_angles(N)

        # weight matrix (N x 2K)
        W = np.zeros((N, 2 * K))
        for k in range(1, K + 1):
            w_sin = np.sin(k * theta)
            i0 = 2 * (k - 1)
            W[:, i0] = np.maximum(w_sin, 0.0)
            W[:, i0 + 1] = np.maximum(-w_sin, 0.0)

        weights_dir = "./config/ring_decoding_weights"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path  = os.path.join(weights_dir, f"N_{N}_fourier_weights.npy")
        metadata_path = os.path.join(weights_dir, f"N_{N}_fourier_metadata.json")
        save_numpy_legacy(weights_path, W)

        metadata = {
            "population_size":      N,
            "num_fourier_k":        K,
            "readout_weight_scale": 0.0,
            "output_dc_baseline":   200.0,
            "num_positions":        self.num_positions,
            "sim_settle_ms":        self.sim_settle_ms,
            "stimulus_half_width":  self.stimulus_half_width,
            "samples_per_position": self.repeats_per_position,
        }
        p = load_json(self.params_file)
        metadata["readout_weight_scale"] = p.get("readout_weight_scale", 200.0)
        metadata["output_dc_baseline"]   = p.get("output_dc_baseline",   200.0)
        save_json_legacy(metadata_path, metadata)

        phi_true = np.array([d['phi_true'] for d in self.results])
        phi_est  = np.array([d['phi_est_fourier'] for d in self.results])
        save_npz_legacy(
            os.path.join(save_dir, "fourier_results.npz"),
            phi_true=phi_true,
            phi_est=phi_est,
        )

        print(f"Weights saved  → {weights_path}")
        print(f"Metadata saved → {metadata_path}")


if __name__ == "__main__":
    PARAMS_FILE = "./config/model_params/ring_params.json"
    trainer = DecoderTrainer(params_file=PARAMS_FILE)
    trainer.run(save_dir="./outputs/ring_decoding")
