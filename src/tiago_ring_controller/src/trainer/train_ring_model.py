#!/usr/bin/env python3

import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

os.environ.setdefault("PYNEST_QUIET", "1")
import nest

nest.set_verbosity("M_ERROR")

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([os.path.join(src_dir, "builders/single_joint"), src_dir])

from builders.single_joint.ring_component import SingleJointRingComponent
from helpers import FileManager
from tiago_ring_controller.artifacts import (
    save_json_legacy,
    save_npz_legacy,
    save_numpy_legacy,
)
from tiago_ring_controller.config import load_json
from tiago_ring_controller.training.analytic import build_fourier_weights


class DecoderTrainer:
    """Train Fourier decoder weights by sampling ring positions."""
    
    def __init__(self, params_file: str = None):
        """Load parameters and initialize."""
        if params_file is None:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            params_file = os.path.join(script_dir, "config/model_params/ring_params.json")
        
        p = load_json(params_file)
        self.params_file = params_file
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.num_positions = p["num_positions"]
        self.sim_settle_ms = p["sim_settle_ms"]
        self.stimulus_half_width = p["stimulus_half_width"]
        self.repeats_per_position = p["samples_per_position"]
        self.num_fourier_k = p["num_fourier_k"]
        self.population_size = p["population_size"]
        self.decoder_weights = self._build_fourier_weights(self.population_size, self.num_fourier_k)
        self.results = []
        self.file_manager = FileManager()

    def _build_fourier_weights(self, population_size: int, num_fourier_k: int) -> np.ndarray:
        """Build push-pull sine Fourier decoder weights (N x 2K)."""
        return build_fourier_weights(population_size, num_fourier_k)

    def _run_fourier_trial_worker(self, idx, rep):
        """Run one trial: stimulus, simulate, measure spike counts."""
        center_index = (idx * self.population_size) // self.num_positions
        phi_true = 2 * np.pi * center_index / self.population_size
        component = SingleJointRingComponent(params_file=self.params_file, weights=self.decoder_weights)

        before_sin_pos = [component.decoder.get_spike_counts(f"sin_pos_k{k}_recs") 
                          for k in range(1, self.num_fourier_k + 1)]
        before_sin_neg = [component.decoder.get_spike_counts(f"sin_neg_k{k}_recs") 
                          for k in range(1, self.num_fourier_k + 1)]

        component.ring_attractor.inject_stimulus(center_index=center_index, half_width=self.stimulus_half_width)
        nest.Simulate(self.sim_settle_ms)

        after_sin_pos = [component.decoder.get_spike_counts(f"sin_pos_k{k}_recs") 
                         for k in range(1, self.num_fourier_k + 1)]
        after_sin_neg = [component.decoder.get_spike_counts(f"sin_neg_k{k}_recs") 
                         for k in range(1, self.num_fourier_k + 1)]

        sin_pos_delta = np.array(after_sin_pos) - np.array(before_sin_pos)
        sin_neg_delta = np.array(after_sin_neg) - np.array(before_sin_neg)
        sin_counts = sin_pos_delta - sin_neg_delta
        phi_est = np.arcsin(np.clip(sin_counts[0], -1.0, 1.0)) if sin_counts.size > 0 else np.nan

        return idx, rep, phi_true, phi_est, sin_counts, sin_pos_delta, sin_neg_delta

    def _build_tasks(self):
        """Create task list: (position_idx, repeat) for each trial."""
        return [(idx, rep) for idx in range(self.num_positions) for rep in range(self.repeats_per_position)]

    def _run_tasks(self, tasks, max_workers: int):
        """Execute trials sequentially or in parallel, collect results."""
        by_idx = {idx: {"phi_true": np.nan, "repeats": []} for idx in range(self.num_positions)}
        worker_count = max_workers if max_workers is not None else (os.cpu_count() or 1)
        worker_count = max(1, min(int(worker_count), int(len(tasks))))
        
        if worker_count == 1:
            for i, (idx, rep) in enumerate(tasks, 1):
                idx, rep, phi_true, phi_est, sin_counts, sin_pos_delta, sin_neg_delta = self._run_fourier_trial_worker(idx, rep)
                by_idx[idx]["phi_true"] = phi_true
                by_idx[idx]["repeats"].append((rep, phi_est, sin_counts, sin_pos_delta, sin_neg_delta))
                ratio = i / len(tasks)
                filled = min(self.num_positions, max(1, int(np.ceil(ratio * self.num_positions))))
                print(f"\r[{'█' * filled}{'░' * (self.num_positions - filled)}] {i}/{len(tasks)} trials", end="", flush=True)
        else:
            completed = 0
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
                for result in as_completed(executor.submit(self._run_fourier_trial_worker, idx, rep) for idx, rep in tasks):
                    idx, rep, phi_true, phi_est, sin_counts, sin_pos_delta, sin_neg_delta = result.result()
                    by_idx[idx]["phi_true"] = phi_true
                    by_idx[idx]["repeats"].append((rep, phi_est, sin_counts, sin_pos_delta, sin_neg_delta))
                    completed += 1
                    ratio = completed / len(tasks)
                    filled = min(self.num_positions, max(1, int(np.ceil(ratio * self.num_positions))))
                    print(f"\r[{'█' * filled}{'░' * (self.num_positions - filled)}] {completed}/{len(tasks)} trials", end="", flush=True)
        return by_idx

    def run(self, max_workers: int = None):
        """Run all trials and save weights, metadata, and results."""
        save_dir = self.file_manager.get_next_results_dir("train/single_joint_ring_component", __file__, parent_dir="outputs")

        by_idx = self._run_tasks(self._build_tasks(), max_workers)
        
        self.results = []
        for idx in range(self.num_positions):
            repeats = sorted(by_idx[idx]["repeats"], key=lambda x: x[0])
            phi_list = [r[1] for r in repeats]
            last = repeats[-1] if repeats else (None, np.nan, np.array([]), np.array([]), np.array([]))
            self.results.append({
                "phi_true": by_idx[idx]["phi_true"],
                "phi_est_fourier_list": phi_list,
                "phi_est_fourier": phi_list[-1] if phi_list else np.nan,
                "fourier_counts_sin": last[2],
                "sin_pos_delta": last[3],
                "sin_neg_delta": last[4],
            })

        print()
        self._save_summary(save_dir)

    def _save_summary(self, save_dir: str):
        """Build weight matrix, save weights and metadata."""
        N, K = self.population_size, self.num_fourier_k
        W = self.decoder_weights

        weights_dir = os.path.join(self.script_dir, "config/ring_decoding_weights")
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f"N_{N}_fourier_weights.npy")
        metadata_path = os.path.join(weights_dir, f"N_{N}_fourier_metadata.json")
        save_numpy_legacy(weights_path, W)

        p = load_json(self.params_file)
        
        metadata = {
            "population_size": N,
            "num_fourier_k": K,
            "readout_weight_scale": p.get("readout_weight_scale", 200.0),
            "output_dc_baseline": p.get("output_dc_baseline", 200.0),
            "num_positions": self.num_positions,
            "sim_settle_ms": self.sim_settle_ms,
            "stimulus_half_width": self.stimulus_half_width,
            "samples_per_position": self.repeats_per_position,
        }
        
        save_json_legacy(metadata_path, metadata)

        phi_true = np.array([d['phi_true'] for d in self.results])
        phi_est = np.array([d['phi_est_fourier'] for d in self.results])
        results_path = os.path.join(save_dir, "fourier_results.npz")
        save_npz_legacy(
            results_path,
            phi_true=phi_true,
            phi_est=phi_est,
        )

        print(f"Weights saved  -> {weights_path}")
        print(f"Metadata saved -> {metadata_path}")
        print(f"Results saved  -> {results_path}")

if __name__ == "__main__":
    trainer = DecoderTrainer()
    trainer.run()
