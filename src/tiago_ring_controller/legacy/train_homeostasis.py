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
from tiago_ring_controller.config import load_homeostasis_spec, load_json, load_ring_spec
from tiago_ring_controller.artifacts import save_json_legacy, save_numpy_legacy
from tiago_ring_controller.features import (
	embed_homeostasis_weights,
	homeostasis_dataset,
)
from tiago_ring_controller.math.fourier import fourier_feature_names
from tiago_ring_controller.training.ridge import fit_normalized_ridge


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


def _collect_state_worker(params_file: str, feature_names: list, center_idx: int, settle_ms: float, stimulus_half_width: int):
	"""Worker for collecting one ring state in an isolated NEST process."""
	with _suppress_stdout():
		nest.set_verbosity("M_WARNING")
		component = RingAttractorComponent(params_file=params_file, reset_kernel=True)

		before = {
			name: component._get_decoder_spike_counts(name + "_recs")
			for name in feature_names
		}

		component._inject_bump(center_idx=int(center_idx), half_width=stimulus_half_width)
		nest.Simulate(settle_ms)

		features = np.array([
			component._get_decoder_spike_counts(name + "_recs") - before[name]
			for name in feature_names
		], dtype=float)

	return {
		"center_idx": int(center_idx),
		"features": features,
	}


def _evaluate_pair_worker(
	params_file: str,
	r1_idx: int,
	r2_idx: int,
	settle_ms: float,
	evaluate_ms: float,
	stimulus_half_width: int,
):
	"""Worker for evaluating one homeostasis pair in an isolated NEST process."""
	from homeostasis import HomeostasisModel

	with _suppress_stdout():
		nest.set_verbosity("M_WARNING")
		ring_1 = RingAttractorComponent(params_file=params_file, reset_kernel=True)
		ring_2 = RingAttractorComponent(params_file=params_file, reset_kernel=False)
		model = HomeostasisModel(ring_1, ring_2)

		ring_1._inject_bump(center_idx=int(r1_idx), half_width=stimulus_half_width)
		ring_2._inject_bump(center_idx=int(r2_idx), half_width=stimulus_half_width)
		nest.Simulate(settle_ms)

		eval_result = model.evaluate(duration_ms=evaluate_ms)
		warm_ev = model.homeostasis_recorders["warm_spike"].get("events")
		cold_ev = model.homeostasis_recorders["cold_spike"].get("events")

	return {
		"idx_1": int(r1_idx),
		"idx_2": int(r2_idx),
		"warm": len(warm_ev.get("times", [])),
		"cold": len(cold_ev.get("times", [])),
		"left": eval_result["left_spike_count"],
		"right": eval_result["right_spike_count"],
	}

class HomeostasisTrainer:
	"""
	Trains the warm/cold linear decoder via ridge regression.
	Uses all K Fourier harmonics as K signed sin features.
	Target: phase / (2π)  — sawtooth ramp, linear 0→1 over the full circle.
	"""

	@staticmethod
	def signed_phase_error(phase_1: float, phase_2: float) -> float:
		"""Signed arc from phase_1 to phase_2 in [-π, π]. Positive = clockwise."""
		return float(np.arctan2(np.sin(phase_2 - phase_1), np.cos(phase_2 - phase_1)))

	def __init__(
		self,
		params_file: str = "./config/model_params/ring_params.json",
		homeostasis_params_file: str = "./config/model_params/homeostasis_params.json",
	):
		ring_params = load_ring_spec(params_file)
		h_params = load_homeostasis_spec(homeostasis_params_file)

		self.population_size = ring_params.population_size
		self.params_file = params_file
		self.feature_names = fourier_feature_names(ring_params.num_fourier_k)
		self.config_dir = h_params.config_dir
		self.warm_cold_weight_scale = h_params.warm_cold_weight_scale

	def _collect_state_library(self, center_indices: np.ndarray, settle_ms: float, stimulus_half_width: int) -> list:
		"""Simulate the ring at each position and record decoded spike-count deltas."""
		states = []
		N = self.population_size
		n = len(center_indices)

		max_workers = min(n, max(1, os.cpu_count() or 1))
		ctx = mp.get_context("spawn")
		future_to_idx = {}

		with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
			for center_idx in center_indices:
				future = pool.submit(
					_collect_state_worker,
					self.params_file,
					self.feature_names,
					int(center_idx),
					float(settle_ms),
					int(stimulus_half_width),
				)
				future_to_idx[future] = int(center_idx)

			k = 0
			for future in as_completed(future_to_idx):
				item = future.result()
				states.append({
					"center_idx": item["center_idx"],
					"phase": RingAttractorComponent.center_index_to_phase(item["center_idx"], N),
					"features": item["features"],
				})
				k += 1
				bar = "\u2588" * k + "\u2591" * (n - k)
				print(f"\r  collecting states [{bar}] {k}/{n}", end="", flush=True)

		states.sort(key=lambda s: s["center_idx"])
		print()
		return states

	def _build_dataset(self, states: list):
		"""Build signed sin features (K-dim) and sawtooth targets: phase/(2π)."""
		return homeostasis_dataset(states)

	def train(
		self,
		num_positions: int = 32,
		settle_ms: float = 300.0,
		stimulus_half_width: int = 5,
		ridge_lambda: float = 1e-6,
		save_dir: str = "./outputs/homeostasis",
	) -> float:
		"""Collect states, fit ridge regression, save weights, plot results. Returns R²."""
		N = self.population_size
		print(f"Collecting ring state library — {num_positions} positions, N={N}...")
		center_indices = RingAttractorComponent.generate_center_indices(N, num_positions)
		states = self._collect_state_library(center_indices, settle_ms, stimulus_half_width)

		print("Building magnitude dataset...")
		X, Y, phases = self._build_dataset(states)

		fit = fit_normalized_ridge(X, Y, ridge_lambda)
		w2 = fit.weights
		b = fit.bias
		r2 = fit.r2
		rmse = fit.rmse

		print(f"Train R²: {r2:.4f}   RMSE: {rmse:.4f}")

		# Embed weights into 4K-dim NEST format: [+w, -w] per harmonic per ring
		warm_bias = b
		cold_bias = b

		metadata = {
			"population_size": N,
			"feature_names": self.feature_names,
			"num_positions": num_positions,
			"settle_ms": settle_ms,
			"stimulus_half_width": stimulus_half_width,
			"ridge_lambda": ridge_lambda,
			"train_r2": r2,
			"train_rmse": rmse,
			"warm_cold_weight_scale": self.warm_cold_weight_scale,
		}
		metadata["warm_bias"] = float(warm_bias)
		metadata["cold_bias"] = float(cold_bias)
		os.makedirs(self.config_dir, exist_ok=True)
		W = embed_homeostasis_weights(w2)
		save_numpy_legacy(
			os.path.join(self.config_dir, f"N_{N}_homeostasis_weights.npy"),
			W,
		)
		save_json_legacy(
			os.path.join(self.config_dir, f"N_{N}_homeostasis_metadata.json"),
			metadata,
		)
		print(f"Weights saved to {self.config_dir}/N_{N}_homeostasis_weights.npy")

		return r2


class HomeostasisAnalysis:
	"""
	Evaluates the trained HomeostasisModel at sampled positions using actual NEST spike counts.
	Records warm, cold, left, and right spikes and produces diagnostic plots.
	"""

	_TIC_VALS = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
	_TIC_LBLS = ["0", "\u03c0/2", "\u03c0", "3\u03c0/2", "2\u03c0"]

	def __init__(self, params_file: str, homeostasis_params_file: str):
		ring_params = load_json(params_file)
		self.population_size = ring_params["population_size"]
		self.params_file = params_file
		self.homeostasis_params_file = homeostasis_params_file

	def collect(
		self,
		center_indices: np.ndarray,
		settle_ms: float = 300.0,
		evaluate_ms: float = 100.0,
		stimulus_half_width: int = 5,
	) -> list:
		"""
		Nested loop: ring_1 sweeps 0→2π (x-axis) × ring_2 sweeps 0→2π (y-axis).
		For every (ring_1, ring_2) pair, run a full NEST simulation and record
		warm, cold, left, and right spike counts.
		"""
		N = self.population_size
		n = len(center_indices)
		total = n * n
		results = []
		jobs = [
			(int(r1_idx), int(r2_idx))
			for r1_idx in center_indices
			for r2_idx in center_indices
		]
		max_workers = min(total, max(1, os.cpu_count() or 1))
		ctx = mp.get_context("spawn")

		with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
			futures = {
				pool.submit(
					_evaluate_pair_worker,
					self.params_file,
					r1_idx,
					r2_idx,
					float(settle_ms),
					float(evaluate_ms),
					int(stimulus_half_width),
				): (r1_idx, r2_idx)
				for r1_idx, r2_idx in jobs
			}

			for k, future in enumerate(as_completed(futures), start=1):
				item = future.result()
				results.append({
					"idx_1": item["idx_1"],
					"idx_2": item["idx_2"],
					"phase_1": RingAttractorComponent.center_index_to_phase(item["idx_1"], N),
					"phase_2": RingAttractorComponent.center_index_to_phase(item["idx_2"], N),
					"warm": item["warm"],
					"cold": item["cold"],
					"left": item["left"],
					"right": item["right"],
				})

				bar_filled = int(k / total * n)
				bar = "\u2588" * bar_filled + "\u2591" * (n - bar_filled)
				print(f"\r  evaluating [{bar}] {k}/{total}", end="", flush=True)

		print()
		results.sort(key=lambda r: (r["idx_1"], r["idx_2"]))
		return results

	@staticmethod
	def _phase_ticks(ax_obj, p_arr, axis):
		positions, labels = [], []
		for v, lbl in zip(HomeostasisAnalysis._TIC_VALS, HomeostasisAnalysis._TIC_LBLS):
			idx = int(np.argmin(np.abs(p_arr - v)))
			positions.append(idx)
			labels.append(lbl)
		if axis == "x":
			ax_obj.set_xticks(positions)
			ax_obj.set_xticklabels(labels)
		else:
			ax_obj.set_yticks(positions)
			ax_obj.set_yticklabels(labels)

	def visualize(self, results: list, save_dir: str) -> None:
		os.makedirs(save_dir, exist_ok=True)

		# ── warm/cold spike count: diagonal entries (ring_1 phase == ring_2 phase) ──
		diag = [r for r in results if r["idx_1"] == r["idx_2"]]
		diag.sort(key=lambda r: r["phase_1"])
		phases = np.array([r["phase_1"] for r in diag])
		warm_counts = np.array([r["warm"] for r in diag], dtype=float)
		cold_counts = np.array([r["cold"] for r in diag], dtype=float)

		fig, ax = plt.subplots(figsize=(8, 5))
		ax.scatter(phases, warm_counts, s=40, alpha=0.8, color="tab:blue", label="warm")
		ax.scatter(phases, cold_counts, s=40, alpha=0.8, color="tab:red", label="cold")
		ax.set_xlabel("Ring phase (rad)")
		ax.set_ylabel("Spike count")
		ax.set_title("Warm/Cold neuron spike count vs ring phase")
		ax.set_xticks(self._TIC_VALS)
		ax.set_xticklabels(self._TIC_LBLS)
		ax.legend()
		fig.tight_layout()
		fig.savefig(os.path.join(save_dir, "warm_cold_spikes.png"), dpi=120)
		plt.close(fig)

		# ── left/right heatmaps: x = ring_1 phase (warm), y = ring_2 phase (cold) ──
		p1_sorted = sorted(set(r["phase_1"] for r in results))
		p2_sorted = sorted(set(r["phase_2"] for r in results))
		n1, n2 = len(p1_sorted), len(p2_sorted)
		p1_idx = {p: i for i, p in enumerate(p1_sorted)}
		p2_idx = {p: i for i, p in enumerate(p2_sorted)}

		left_map  = np.zeros((n2, n1))
		right_map = np.zeros((n2, n1))
		for r in results:
			xi = p1_idx[r["phase_1"]]
			yi = p2_idx[r["phase_2"]]
			left_map[yi, xi]  = r["left"]
			right_map[yi, xi] = r["right"]

		p1_arr = np.array(p1_sorted)
		p2_arr = np.array(p2_sorted)

		fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

		im_l = ax_l.imshow(left_map, origin="lower", aspect="auto", cmap="Blues")
		self._phase_ticks(ax_l, p1_arr, "x")
		self._phase_ticks(ax_l, p2_arr, "y")
		ax_l.set_xlabel("Ring 1 phase — warm (rad)")
		ax_l.set_ylabel("Ring 2 phase — cold (rad)")
		ax_l.set_title("LEFT neuron spike count")
		fig2.colorbar(im_l, ax=ax_l, label="spike count")

		im_r = ax_r.imshow(right_map, origin="lower", aspect="auto", cmap="Reds")
		self._phase_ticks(ax_r, p1_arr, "x")
		self._phase_ticks(ax_r, p2_arr, "y")
		ax_r.set_xlabel("Ring 1 phase — warm (rad)")
		ax_r.set_ylabel("Ring 2 phase — cold (rad)")
		ax_r.set_title("RIGHT neuron spike count")
		fig2.colorbar(im_r, ax=ax_r, label="spike count")

		fig2.suptitle("Left/Right spike count heatmap — ring 1 \u00d7 ring 2 phase sweep")
		fig2.tight_layout()
		fig2.savefig(os.path.join(save_dir, "left_right_activity.png"), dpi=120)
		plt.close(fig2)

		print(f"Analysis plots saved to {save_dir}/")

	def analyze(
		self,
		num_positions: int = 32,
		settle_ms: float = 300.0,
		evaluate_ms: float = 100.0,
		stimulus_half_width: int = 5,
		save_dir: str = "./outputs/homeostasis",
	) -> list:
		"""Collect spike counts from the full circuit, visualize, and return results."""
		N = self.population_size
		center_indices = RingAttractorComponent.generate_center_indices(N, num_positions)
		print(f"Analysing HomeostasisModel \u2014 {num_positions}\u00d7{num_positions} grid, N={N}...")
		results = self.collect(center_indices, settle_ms, evaluate_ms, stimulus_half_width)
		self.visualize(results, save_dir)
		return results


if __name__ == "__main__":
	RUN_ANALYSIS = True

	src_dir = os.path.dirname(os.path.abspath(__file__))
	params_file = os.path.join(src_dir, "config", "model_params", "ring_params.json")
	h_params_file = os.path.join(src_dir, "config", "model_params", "homeostasis_params.json")

	trainer = HomeostasisTrainer(
		params_file=params_file,
		homeostasis_params_file=h_params_file,
	)

	r2 = trainer.train(
		num_positions=32,
		settle_ms=300.0,
		stimulus_half_width=5,
		ridge_lambda=1e-6,
		save_dir=os.path.join(src_dir, "outputs", "homeostasis"),
	)

	print(f"\nTraining complete — magnitude decoder R²: {r2:.4f}")

	if RUN_ANALYSIS:
		analyzer = HomeostasisAnalysis(
			params_file=params_file,
			homeostasis_params_file=h_params_file,
		)
		analyzer.analyze(
			num_positions=8,
			settle_ms=300.0,
			evaluate_ms=100.0,
			stimulus_half_width=5,
			save_dir=os.path.join(src_dir, "outputs", "homeostasis"),
		)
