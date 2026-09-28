#!/usr/bin/env python3
"""Spiking decoder based on an analytic signed Fourier-product grid."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest
import numpy as np

from ring_attractor import Ring_Attractor
from tiago_ring_controller.artifacts import save_json_legacy, save_npz_legacy
from tiago_ring_controller.config import load_json
from tiago_ring_controller.features import (
    flatten_signed_counts,
    mapped_ring_indices,
    signed_product_feature_dimension,
    signed_product_feature_order,
    signed_product_term_names,
    signed_product_term_values,
    signed_vector_term_activity,
    signed_vector_term_sums,
    unflatten_signed_vector,
)
from tiago_ring_controller.math.circular import (
    angle_to_neuron_index,
    angle_to_ring_index,
    circular_angle_error,
    circular_signed_difference,
    decode_sawtooth_profile,
    preferred_angles,
)
from tiago_ring_controller.math.fourier import target_cosine_modulation
from tiago_ring_controller.math.kinematics import (
    lift_pitch_yaw_from_rotation,
    resample_profile,
    rotation_matrix,
)
from tiago_ring_controller.nest.multi_ring import (
    build_output_rings,
    build_signed_product_layer,
)
from tiago_ring_controller.training.ridge import fit_ridge


class MultiRingAttractorSetup:
    """Builds rings, signed feature grid, and output rings in one NEST kernel."""

    def __init__(self, params_file: str = "./config/model_params/multi_ring_params.json"):
        p = load_json(params_file)

        self.population_size = p["population_size"]
        self.num_joints = p.get("num_joints", 2)
        if self.num_joints != 2:
            raise ValueError("This signed Fourier-product decoder currently supports exactly 2 joints: q1 and q2.")
        self.joint_axes = p.get("joint_axes", ["x", "y"])
        self.output_ring_size = p.get("output_ring_size", 100)
        self.sim_settle_ms = p["sim_settle_ms"]
        self.stimulus_half_width = p["stimulus_half_width"]

        self.output_dc_baseline = p.get("output_dc_baseline", 200.0)

        self.feature_grid_size = p.get(
            "signed_product_population_size",
            p.get("product_population_size", min(self.population_size, 32)),
        )
        self.signed_product_dc_baseline = p.get("signed_product_dc_baseline", 180.0)
        self.signed_product_input_weight = p.get("signed_product_input_weight", 120.0)
        self.signed_product_output_weight_scale = p.get(
            "signed_product_output_weight_scale",
            p.get("sfp_output_weight_scale", 1.0),
        )

        neuron_params_file = os.path.join(os.path.dirname(params_file), "neuron_params.json")
        self.rings = []
        for i in range(self.num_joints):
            self.rings.append(
                Ring_Attractor(
                    population_size=self.population_size,
                    reset_kernel=(i == 0),
                    params_file=neuron_params_file,
                )
            )

    def _preferred_angles(self, n: int) -> np.ndarray:
        return preferred_angles(n)

    def _get_spike_counts(self, recorders) -> np.ndarray:
        counts = []
        for sr in (recorders if hasattr(recorders, "__iter__") else [recorders]):
            ev = sr.get("events")
            counts.append(len(ev.get("times", [])))
        return np.array(counts, dtype=float)

    def _get_joint_ring_spike_counts(self, joint_idx: int) -> np.ndarray:
        return self.rings[joint_idx]._get_spike_counts()

    def _inject_joint_bump(self, joint_idx: int, center_idx: int, half_width: int):
        self.rings[joint_idx].inject_stimulus(center_index=center_idx, half_width=half_width)

    def build_signed_product_grid_layer(self, theta: np.ndarray) -> dict:
        """Builds analytic signed Fourier-product populations."""
        del theta

        n_ring = self.population_size
        n_side = self.feature_grid_size
        n_cells = n_side * n_side
        mapped_idxs = mapped_ring_indices(n_ring, n_side)
        eps = 1e-9

        feature_order = signed_product_feature_order()
        term_values = None

        def term_values_factory():
            nonlocal term_values
            if term_values is None:
                term_values = signed_product_term_values(n_ring, n_side)
            return term_values

        built = build_signed_product_layer(
            nest,
            self.rings[0].ring_neurons,
            self.rings[1].ring_neurons,
            n_ring,
            n_side,
            self.signed_product_dc_baseline,
            self.signed_product_input_weight,
            epsilon=eps,
            mapped_indices=mapped_idxs,
            term_values_factory=term_values_factory,
            feature_order=feature_order,
        )
        terms = {}
        for feature_name in feature_order:
            term_name, sign = feature_name.rsplit("_", 1)
            terms.setdefault(term_name, {})[sign] = built.populations[feature_name]
        layer = {
            "feature_grid_size": n_side,
            "n_cells": n_cells,
            "mapped_ring_idxs": mapped_idxs,
            "feature_order": feature_order,
            "terms": terms,
            "flat_nodes": built.flat_nodes,
        }

        self._signed_product_grid_layer = layer
        return layer

    def _get_signed_product_grid_counts(self) -> dict:
        """Returns feature grid spike counts for each term/sign."""
        counts = {}
        for feature_name in self._signed_product_grid_layer["feature_order"]:
            term_name, sign = feature_name.rsplit("_", 1)
            recs = self._signed_product_grid_layer["terms"][term_name][sign]["recs"]
            counts[feature_name] = self._get_spike_counts(recs)
        return counts

    def build_output_rings_from_signed_product_grid(self, w_lift, w_pitch, w_yaw) -> dict:
        """Wires signed product features to output rings."""
        n_out = self.output_ring_size
        scale = self.signed_product_output_weight_scale
        src_nodes = self._signed_product_grid_layer["flat_nodes"]

        self._output_rings_signed = {}

        def register_population(name, nodes, recs):
            self._output_rings_signed[name] = {"nodes": nodes, "recs": recs}

        build_output_rings(
            nest,
            self._signed_product_grid_layer,
            {"lift": w_lift, "pitch": w_pitch, "yaw": w_yaw},
            n_out,
            self.output_dc_baseline,
            scale,
            on_population_built=register_population,
            source_nodes=src_nodes,
            matrix_orientation_getter=lambda: getattr(
                self, "output_weight_matrix_is_target_by_source", True
            ),
        )

        return self._output_rings_signed

    def _get_signed_product_output_ring_counts(self, name: str) -> np.ndarray:
        return self._get_spike_counts(self._output_rings_signed[name]["recs"])


class CompositionalFourierDecoderTrainer:
    """Trains output-ring decoder weights from signed product features."""

    _RC = {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 10,
    }

    def __init__(self, params_file: str = "./config/model_params/multi_ring_params.json"):
        p = load_json(params_file)

        self.params_file = params_file
        self.population_size = p["population_size"]
        self.sim_settle_ms = p["sim_settle_ms"]
        self.stimulus_half_width = p["stimulus_half_width"]
        self.num_positions = p["num_positions"]
        self.num_joints = p.get("num_joints", 2)
        if self.num_joints != 2:
            raise ValueError("This signed Fourier-product decoder currently supports exactly 2 joints: q1 and q2.")
        self.joint_axes = p.get("joint_axes", ["x", "y"])
        self.output_ring_size = p.get("output_ring_size", 100)

        self.output_rate_amplitude = p.get("output_rate_amplitude", 40.0)
        self.ridge_lambda = p.get("ridge_lambda", 1e-3)
        self.max_train_samples = p.get("max_train_samples", 256)
        self.n_vis_points = p.get("n_vis_points", 20)
        self.baseline_burnin_ms = p.get("baseline_burnin_ms", self.sim_settle_ms)

        self.feature_grid_size = p.get(
            "signed_product_population_size",
            p.get("product_population_size", min(self.population_size, 32)),
        )
        self.signed_product_dc_baseline = p.get("signed_product_dc_baseline", 180.0)
        self.signed_product_input_weight = p.get("signed_product_input_weight", 120.0)
        self.signed_product_output_weight_scale = p.get(
            "signed_product_output_weight_scale",
            p.get("sfp_output_weight_scale", 1.0),
        )

        self._W_signed_lift = None
        self._W_signed_pitch = None
        self._W_signed_yaw = None

        self._signed_product_active = False
        self._signed_feature_stats = {}
        self._calibrated_signed_product_params = {}
        self._active_signed_product_terms = []
        self._q_joint_contribution_diagnostic = []
        self._output_weight_matrix_is_target_by_source = True

        self._train_configs = []
        self._signed_feature_order = self._signed_feature_order_list()

    def _preferred_angles(self, n: int) -> np.ndarray:
        return preferred_angles(n)

    def _rot(self, axis: str, q: float) -> np.ndarray:
        return rotation_matrix(axis, q)

    def _forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        rot = np.eye(3)
        for axis, qi in zip(self.joint_axes, q):
            rot = rot @ self._rot(axis, qi)
        return rot

    def _lift_pitch_yaw_from_angles(self, q: np.ndarray):
        return lift_pitch_yaw_from_rotation(self._forward_kinematics(q))

    def _resample_profile(self, profile: np.ndarray, n_out: int) -> np.ndarray:
        return resample_profile(profile, n_out)

    def _angle_to_ring_index(self, angle: float, n: int) -> float:
        # This facade historically returned a NumPy scalar for scalar input.
        result = angle_to_ring_index(angle, n)
        return np.float64(result) if np.ndim(result) == 0 else result

    def _decode_angle_sawtooth_from_profile(self, profile: np.ndarray) -> float:
        profile = np.asarray(profile, dtype=float)
        activity = np.maximum(profile - np.min(profile), 0.0)
        theta = self._preferred_angles(len(profile))
        return np.arctan2(
            np.dot(activity, np.sin(theta)),
            np.dot(activity, np.cos(theta)),
        )

    def _fit_ridge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return fit_ridge(x, y, self.ridge_lambda)

    def _build_joint_configs(self) -> list:
        total_grid = self.num_positions ** self.num_joints
        if total_grid <= self.max_train_samples:
            axes = [np.linspace(-np.pi, np.pi, self.num_positions, endpoint=False)] * self.num_joints
            grids = np.meshgrid(*axes, indexing="ij")
            return list(np.column_stack([g.ravel() for g in grids]))

        rng = np.random.default_rng(seed=42)
        return list(rng.uniform(-np.pi, np.pi, size=(self.max_train_samples, self.num_joints)))

    def _angle_to_center_idx(self, qi: float) -> int:
        return angle_to_neuron_index(qi, self.population_size)

    def _target_output_modulation(self, angle: float) -> np.ndarray:
        theta_out = self._preferred_angles(self.output_ring_size)
        return self.output_rate_amplitude * np.cos(theta_out - angle)

    def _signed_feature_order_list(self):
        order = []
        for term_name in self._signed_term_names():
            order.append("{}_pos".format(term_name))
            order.append("{}_neg".format(term_name))
        return order

    def _signed_term_names(self):
        return signed_product_term_names()

    def _signed_feature_dim(self) -> int:
        return (
            2
            * len(self._signed_term_names())
            * self.feature_grid_size
            * self.feature_grid_size
        )

    def _signed_counts_to_vector(self, counts: dict) -> np.ndarray:
        return flatten_signed_counts(counts, self._signed_feature_order)

    def _signed_vector_to_maps(self, vector: np.ndarray):
        return unflatten_signed_vector(
            vector, self.feature_grid_size, self._signed_feature_order
        )

    def _signed_vector_term_sums(self, vector: np.ndarray) -> dict:
        maps = self._signed_vector_to_maps(vector)
        out = {}
        for term_name in self._signed_term_names():
            out[term_name] = float(
                np.sum(maps[f"{term_name}_pos"])
                - np.sum(maps[f"{term_name}_neg"])
            )
        return out

    def _signed_vector_term_activity(self, vector: np.ndarray) -> dict:
        maps = self._signed_vector_to_maps(vector)
        out = {}
        for term_name in self._signed_term_names():
            pos = np.asarray(maps[f"{term_name}_pos"], dtype=float)
            neg = np.asarray(maps[f"{term_name}_neg"], dtype=float)
            out[term_name] = float(
                np.sum(np.abs(pos)) + np.sum(np.abs(neg))
            )
        return out

    def _print_progress_bar(self, done: int, total: int, width: int = 40, prefix: str = ""):
        total = max(int(total), 1)
        done = min(max(int(done), 0), total)
        filled = int(width * done / total)
        bar = f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"
        print(f"\r{prefix}{bar}", end="", flush=True)
        if done >= total:
            print()

    def _make_signed_model(self) -> MultiRingAttractorSetup:
        model = MultiRingAttractorSetup(params_file=self.params_file)
        model.feature_grid_size = self.feature_grid_size
        model.signed_product_dc_baseline = self.signed_product_dc_baseline
        model.signed_product_input_weight = self.signed_product_input_weight
        model.signed_product_output_weight_scale = self.signed_product_output_weight_scale
        model.output_weight_matrix_is_target_by_source = self._output_weight_matrix_is_target_by_source
        return model

    def _inject_q(self, model: MultiRingAttractorSetup, q: np.ndarray):
        for j, qi in enumerate(q):
            model._inject_joint_bump(j, self._angle_to_center_idx(qi), model.stimulus_half_width)

    def _inject_selected_q(self, model: MultiRingAttractorSetup, q: np.ndarray, joint_indices):
        for j in joint_indices:
            model._inject_joint_bump(j, self._angle_to_center_idx(q[j]), model.stimulus_half_width)

    def _counts_sub(self, a, b):
        if isinstance(a, dict):
            return {k: self._counts_sub(a[k], b[k]) for k in a}
        if isinstance(a, list):
            return [self._counts_sub(x, y) for x, y in zip(a, b)]
        if isinstance(a, tuple):
            return tuple(self._counts_sub(x, y) for x, y in zip(a, b))
        return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)

    def _measure_window_delta(self, get_counts_fn, pre_window_action=None):
        before = get_counts_fn()
        if pre_window_action is not None:
            pre_window_action()
        nest.Simulate(self.sim_settle_ms)
        after = get_counts_fn()
        return self._counts_sub(after, before)

    def _measure_baseline_delta(self, get_counts_fn):
        return self._measure_window_delta(get_counts_fn=get_counts_fn, pre_window_action=None)

    def _measure_stimulated_delta(self, model: MultiRingAttractorSetup, q: np.ndarray, get_counts_fn):
        return self._measure_window_delta(
            get_counts_fn=get_counts_fn,
            pre_window_action=lambda: self._inject_q(model, q),
        )

    def _measure_net_stimulus_delta(self, model: MultiRingAttractorSetup, q: np.ndarray, get_counts_fn, pre_stim_action=None):
        nest.Simulate(self.baseline_burnin_ms)
        baseline_delta = self._measure_baseline_delta(get_counts_fn)
        if pre_stim_action is None:
            pre_stim_action = lambda: self._inject_q(model, q)
        stimulated_delta = self._measure_window_delta(get_counts_fn=get_counts_fn, pre_window_action=pre_stim_action)
        return self._counts_sub(stimulated_delta, baseline_delta)

    def _run_q_joint_contribution_diagnostic(self, verbose: bool = True) -> bool:
        test_qs = [
            np.array([0.0, 0.0]),
            np.array([np.pi / 4, -np.pi / 4]),
            np.array([-np.pi / 3, np.pi / 6]),
        ]
        product_terms = ["cos1cos2", "cos1sin2", "sin1cos2", "sin1sin2"]
        diag_rows = []
        diag_ok = True

        if verbose:
            print("\nq1-only / q2-only / q1+q2 signed feature diagnostic")
        for q in test_qs:
            if verbose:
                print(f"  q=({q[0]:.3f}, {q[1]:.3f})")

            def measure_term_metrics(joint_indices):
                model = self._make_signed_model()
                theta = self._preferred_angles(self.population_size)
                model.build_signed_product_grid_layer(theta)
                get_grid_counts = model._get_signed_product_grid_counts
                delta_counts = self._measure_net_stimulus_delta(
                    model=model,
                    q=q,
                    get_counts_fn=get_grid_counts,
                    pre_stim_action=lambda: self._inject_selected_q(model, q, joint_indices),
                )
                vec = self._signed_counts_to_vector(delta_counts)
                return self._signed_vector_term_sums(vec), self._signed_vector_term_activity(vec)

            q1_only, q1_only_activity = measure_term_metrics([0])
            q2_only, q2_only_activity = measure_term_metrics([1])
            both, both_activity = measure_term_metrics([0, 1])
            expected = {
                "cos1cos2": np.cos(q[0]) * np.cos(q[1]),
                "cos1sin2": np.cos(q[0]) * np.sin(q[1]),
                "sin1cos2": np.sin(q[0]) * np.cos(q[1]),
                "sin1sin2": np.sin(q[0]) * np.sin(q[1]),
            }
            row = {
                "q": [float(q[0]), float(q[1])],
                "q1_only": {k: float(v) for k, v in q1_only.items()},
                "q2_only": {k: float(v) for k, v in q2_only.items()},
                "both": {k: float(v) for k, v in both.items()},
                "q1_only_activity": {k: float(v) for k, v in q1_only_activity.items()},
                "q2_only_activity": {k: float(v) for k, v in q2_only_activity.items()},
                "both_activity": {k: float(v) for k, v in both_activity.items()},
                "expected_product_terms": {k: float(v) for k, v in expected.items()},
                "warnings": [],
            }

            if verbose:
                print("    q1-only: " + ", ".join(f"{t}={q1_only[t]:.1f}" for t in self._signed_term_names()))
                print("    q2-only: " + ", ".join(f"{t}={q2_only[t]:.1f}" for t in self._signed_term_names()))
                print("    both:    " + ", ".join(f"{t}={both[t]:.1f}" for t in self._signed_term_names()))

            for t in product_terms:
                expected_mag = abs(float(expected[t]))
                if expected_mag < 0.15:
                    continue
                both_abs = abs(float(both_activity[t]))
                single_abs = max(abs(float(q1_only_activity[t])), abs(float(q2_only_activity[t])))
                if both_abs <= (1.2 * single_abs + 1e-9):
                    msg = (
                        f"product term {t} not clearly stronger with both joints "
                        f"(both_activity={both_abs:.2f}, single_activity={single_abs:.2f}, expected_abs={expected_mag:.2f})"
                    )
                    row["warnings"].append(msg)
                    diag_ok = False
                    if verbose:
                        print(f"    WARNING: {msg}")
            diag_rows.append(row)

        self._q_joint_contribution_diagnostic = diag_rows
        return diag_ok

    def _verify_all_to_all_weight_orientation(self) -> bool:
        desired = np.array([[11.0, 12.0], [21.0, 22.0], [31.0, 32.0]], dtype=float)

        def run_trial(weight_arg):
            nest.ResetKernel()
            pre = nest.Create("parrot_neuron", 2)
            post = nest.Create("iaf_psc_alpha", 3)
            nest.Connect(pre, post, conn_spec={"rule": "all_to_all"}, syn_spec={"weight": weight_arg})
            conns = nest.GetConnections(source=pre, target=post)
            src_ids = pre.get("global_id")
            tgt_ids = post.get("global_id")
            src = conns.get("source")
            tgt = conns.get("target")
            wts = conns.get("weight")
            src_idx = {gid: i for i, gid in enumerate(src_ids)}
            tgt_idx = {gid: i for i, gid in enumerate(tgt_ids)}
            mat = np.zeros((len(tgt_ids), len(src_ids)), dtype=float)
            for s, t, w in zip(src, tgt, wts):
                mat[tgt_idx[t], src_idx[s]] = float(w)
            return mat

        err_t_by_s = float("inf")
        err_s_by_t = float("inf")
        try:
            trial_t_by_s = run_trial(desired.tolist())
            err_t_by_s = float(np.max(np.abs(trial_t_by_s - desired)))
        except Exception as exc:
            print(f"NEST all_to_all orientation trial target_by_source failed: {exc}")
        try:
            trial_s_by_t = run_trial(desired.T.tolist())
            err_s_by_t = float(np.max(np.abs(trial_s_by_t - desired)))
        except Exception as exc:
            print(f"NEST all_to_all orientation trial source_by_target failed: {exc}")
        if np.isinf(err_t_by_s) and np.isinf(err_s_by_t):
            raise RuntimeError("Failed both NEST all_to_all orientation trials.")
        orient_target_by_source = err_t_by_s <= err_s_by_t
        print(
            "NEST all_to_all orientation test: "
            f"target_by_source={orient_target_by_source} "
            f"(err_t_by_s={err_t_by_s:.3e}, err_s_by_t={err_s_by_t:.3e})"
        )
        nest.ResetKernel()
        return orient_target_by_source

    def _calibrate_signed_product_grid_layer(self):
        dc_candidates = [120.0, 160.0, 180.0, 220.0, 260.0, 300.0, 340.0]
        w_candidates = [150.0, 250.0, 350.0, 500.0, 700.0, 900.0, 1200.0, 1500.0]
        test_qs = [
            np.array([0.0, 0.0]),
            np.array([np.pi / 6, -np.pi / 6]),
            np.array([np.pi / 4, -np.pi / 4]),
            np.array([np.pi / 2, np.pi / 3]),
        ]

        best_three = None
        total_candidates = len(dc_candidates) * len(w_candidates)
        done_candidates = 0

        print("\nSigned product grid calibration")
        for dc in dc_candidates:
            for w in w_candidates:
                self.signed_product_dc_baseline = dc
                self.signed_product_input_weight = w

                all_vectors = []
                term_signed_sums = {name: 0.0 for name in self._signed_term_names()}
                term_traces = {name: [] for name in self._signed_term_names()}

                for q in test_qs:
                    model = self._make_signed_model()
                    theta = self._preferred_angles(self.population_size)
                    model.build_signed_product_grid_layer(theta)

                    get_grid_counts = model._get_signed_product_grid_counts
                    delta_counts = self._measure_net_stimulus_delta(model, q, get_grid_counts)
                    vector = self._signed_counts_to_vector(delta_counts)
                    all_vectors.append(vector)

                    signed_sums = self._signed_vector_term_sums(vector)
                    for term_name, val in signed_sums.items():
                        term_signed_sums[term_name] += val
                        term_traces[term_name].append(float(val))

                arr = np.concatenate(all_vectors)
                mean_abs = float(np.mean(np.abs(arr)))
                max_abs = float(np.max(np.abs(arr)))
                active_count = int(np.sum(np.abs(arr) > 1e-9))
                silent_pct = 100.0 * (len(arr) - active_count) / len(arr)
                term_variations = {
                    name: float(np.ptp(np.array(term_traces[name], dtype=float)))
                    for name in self._signed_term_names()
                }
                active_terms = [name for name in self._signed_term_names() if term_variations[name] > 1.0]
                n_active_terms = len(active_terms)
                print(
                    f"  dc={dc:.0f} w={w:.0f} mean_abs={mean_abs:.3f} max_abs={max_abs:.1f} "
                    f"active={active_count} silent={silent_pct:.1f}% active_terms={n_active_terms}/8"
                )
                print(
                    "  term signed sums: "
                    + ", ".join(f"{name}={term_signed_sums[name]:.1f}" for name in self._signed_term_names())
                )
                print(
                    "  term variation: "
                    + ", ".join(f"{name}={term_variations[name]:.1f}" for name in self._signed_term_names())
                )
                done_candidates += 1
                self._print_progress_bar(done_candidates, total_candidates, prefix="  calibration ")

                basic_ok = (max_abs > 0.0) and (mean_abs > 0.02) and (active_count > 0)
                if basic_ok and n_active_terms >= 8:
                    self._active_signed_product_terms = active_terms
                    self._calibrated_signed_product_params = {
                        "signed_product_dc_baseline": dc,
                        "signed_product_input_weight": w,
                        "active_signed_product_terms": active_terms,
                        "active_terms_count": n_active_terms,
                        "activity_criterion": "term variation across q-test configurations",
                    }
                    print("  active_terms=8/8")
                    print(f"  -> Using dc={dc:.0f}, w={w:.0f}")
                    self._run_q_joint_contribution_diagnostic(verbose=True)
                    return

                if basic_ok and n_active_terms >= 6 and best_three is None:
                    best_three = (dc, w, active_terms, n_active_terms)

        if best_three is not None:
            dc, w, active_terms, n_active_terms = best_three
            self.signed_product_dc_baseline = dc
            self.signed_product_input_weight = w
            self._active_signed_product_terms = active_terms
            self._calibrated_signed_product_params = {
                "signed_product_dc_baseline": dc,
                "signed_product_input_weight": w,
                "active_signed_product_terms": active_terms,
                "active_terms_count": n_active_terms,
                "activity_criterion": "term variation across q-test configurations",
            }
            print("  WARNING: using fallback active_terms>=6/8")
            print(f"  active_terms={n_active_terms}/8")
            print(f"  -> Using dc={dc:.0f}, w={w:.0f}")
            self._run_q_joint_contribution_diagnostic(verbose=True)
            return

        raise RuntimeError("Signed product grid is silent (no bump-driven spikes). Increase signed_product_dc_baseline or signed_product_input_weight.")

    def _collect_signed_product_sample(self, q: np.ndarray) -> tuple:
        model = self._make_signed_model()
        theta = self._preferred_angles(self.population_size)
        model.build_signed_product_grid_layer(theta)

        get_grid_counts = model._get_signed_product_grid_counts
        delta_counts = self._measure_net_stimulus_delta(model, q, get_grid_counts)
        feature_vec = self._signed_counts_to_vector(delta_counts)
        lift, pitch, yaw = self._lift_pitch_yaw_from_angles(q)
        return feature_vec, lift, pitch, yaw

    def _calibrate_signed_product_output_scale(self):
        scale_candidates = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
        test_qs = [
            np.array([0.0, 0.0]),
            np.array([np.pi / 4, -np.pi / 4]),
            np.array([-np.pi / 3, np.pi / 6]),
        ]

        print("\nCalibrating signed product output weight scale")
        hdr = f"  {'scale':>6}  {'sig_cov':>7}  {'cont_mean':>9}  {'cont_min':>8}  {'MAE_lift':>8}  {'MAE_pitch':>9}  {'MAE_yaw':>7}  {'MAE_mean':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        results = []
        for idx_scale, scale in enumerate(scale_candidates, start=1):
            all_contrasts = []
            n_profiles = 0
            n_with_signal = 0
            mae_errs: dict = {"lift": [], "pitch": [], "yaw": []}

            for q in test_qs:
                model = self._make_signed_model()
                model.signed_product_output_weight_scale = scale
                theta = self._preferred_angles(self.population_size)
                model.build_signed_product_grid_layer(theta)
                model.build_output_rings_from_signed_product_grid(
                    self._W_signed_lift,
                    self._W_signed_pitch,
                    self._W_signed_yaw,
                )
                lift_t, pitch_t, yaw_t = self._lift_pitch_yaw_from_angles(q)

                def get_output_counts(m=model):
                    return {
                        "lift": m._get_signed_product_output_ring_counts("lift"),
                        "pitch": m._get_signed_product_output_ring_counts("pitch"),
                        "yaw": m._get_signed_product_output_ring_counts("yaw"),
                    }

                output_delta = self._measure_net_stimulus_delta(model, q, get_output_counts)

                for name, cnt, true_a in [
                    ("lift", output_delta["lift"], lift_t),
                    ("pitch", output_delta["pitch"], pitch_t),
                    ("yaw", output_delta["yaw"], yaw_t),
                ]:
                    n_profiles += 1
                    all_contrasts.append(float(np.max(cnt) - np.min(cnt)))
                    if self._profile_has_signal(cnt):
                        n_with_signal += 1
                        pred = self._decode_angle_sawtooth_from_profile(cnt)
                        mae_errs[name].append(abs(circular_angle_error(pred, true_a)))

            contrast_mean = float(np.mean(all_contrasts)) if all_contrasts else 0.0
            contrast_min = float(np.min(all_contrasts)) if all_contrasts else 0.0
            sig_cov = 100.0 * n_with_signal / n_profiles if n_profiles else 0.0
            mae_lift = float(np.degrees(np.mean(mae_errs["lift"]))) if mae_errs["lift"] else float("nan")
            mae_pitch = float(np.degrees(np.mean(mae_errs["pitch"]))) if mae_errs["pitch"] else float("nan")
            mae_yaw = float(np.degrees(np.mean(mae_errs["yaw"]))) if mae_errs["yaw"] else float("nan")
            valid_maes = [m for m in [mae_lift, mae_pitch, mae_yaw] if not np.isnan(m)]
            mae_mean = float(np.mean(valid_maes)) if valid_maes else float("nan")

            print(
                f"  {scale:>6}  {sig_cov:>6.1f}%  {contrast_mean:>9.2f}  {contrast_min:>8.2f}"
                f"  {mae_lift:>8.2f}  {mae_pitch:>9.2f}  {mae_yaw:>7.2f}  {mae_mean:>8.2f}"
            )
            self._print_progress_bar(idx_scale, len(scale_candidates), prefix="  output-scale ")

            results.append({
                "scale": scale,
                "sig_cov": sig_cov,
                "contrast_mean": contrast_mean,
                "contrast_min": contrast_min,
                "mae_lift": mae_lift,
                "mae_pitch": mae_pitch,
                "mae_yaw": mae_yaw,
                "mae_mean": mae_mean,
            })

        acceptable = [
            r for r in results
            if r["sig_cov"] >= 50.0
            and r["contrast_mean"] > 2.0
            and r["contrast_min"] > 0.5
            and not np.isnan(r["mae_mean"])
        ]
        if acceptable:
            best = min(acceptable, key=lambda r: r["mae_mean"])
        else:
            with_signal = [r for r in results if r["sig_cov"] > 0 and not np.isnan(r["mae_mean"])]
            if not with_signal:
                raise RuntimeError(
                    "Signed product output rings are silent. Increase signed_product_output_weight_scale."
                )
            best = min(with_signal, key=lambda r: r["mae_mean"])
            print("  WARNING: no scale passed coverage/contrast thresholds; using best available")

        self.signed_product_output_weight_scale = float(best["scale"])
        self._calibrated_signed_product_params["signed_product_output_weight_scale"] = float(best["scale"])
        self._calibrated_signed_product_params["output_scale_calibration_table"] = results
        print(
            f"  -> Chosen scale={best['scale']} "
            f"(MAE_mean={best['mae_mean']:.2f}°, sig_cov={best['sig_cov']:.1f}%)"
        )

    def _run_full_signed_product_sim(self, q: np.ndarray) -> dict:
        model = self._make_signed_model()
        theta = self._preferred_angles(self.population_size)
        model.build_signed_product_grid_layer(theta)
        model.build_output_rings_from_signed_product_grid(
            self._W_signed_lift,
            self._W_signed_pitch,
            self._W_signed_yaw,
        )

        def get_all_counts():
            return {
                "grid": model._get_signed_product_grid_counts(),
                "output": {
                    "lift": model._get_signed_product_output_ring_counts("lift"),
                    "pitch": model._get_signed_product_output_ring_counts("pitch"),
                    "yaw": model._get_signed_product_output_ring_counts("yaw"),
                },
                "q_rings": [model._get_joint_ring_spike_counts(j) for j in range(model.num_joints)],
            }

        net_delta = self._measure_net_stimulus_delta(model, q, get_all_counts)

        return {
            "q_ring_counts": net_delta["q_rings"],
            "signed_feature_counts": self._signed_counts_to_vector(net_delta["grid"]),
            "lift_counts": net_delta["output"]["lift"],
            "pitch_counts": net_delta["output"]["pitch"],
            "yaw_counts": net_delta["output"]["yaw"],
        }

    def _run_vis_simulations(self, sim_fn, label: str, n_vis: int) -> dict:
        q_train = np.array(self._train_configs)
        q1_fixed = float(np.median(q_train[:, 0]))
        q2_fixed = float(np.median(q_train[:, 1]))
        vary_angles = np.linspace(-np.pi, np.pi, n_vis, endpoint=False)

        slices = [
            (0, q1_fixed, 1, "q1_fixed"),
            (1, q2_fixed, 0, "q2_fixed"),
        ]

        vis_data = {}
        total, done = 2 * n_vis, 0
        print(f"\nRunning {total} {label} vis simulations ({n_vis} per slice) ...")

        for fixed_jt, fixed_val, vary_jt, key in slices:
            lists = {k: [] for k in ("lift", "pitch", "yaw")}
            ring_list = [[] for _ in range(self.num_joints)]
            extra_list = []

            for angle in vary_angles:
                q = np.zeros(self.num_joints)
                q[fixed_jt] = fixed_val
                q[vary_jt] = angle
                res = sim_fn(q)

                lists["lift"].append(res["lift_counts"])
                lists["pitch"].append(res["pitch_counts"])
                lists["yaw"].append(res["yaw_counts"])
                for j in range(self.num_joints):
                    ring_list[j].append(res["q_ring_counts"][j])
                extra_list.append(res["signed_feature_counts"])

                done += 1
                filled = int(40 * done / total)
                print(f"\r  [{'#' * filled}{'.' * (40 - filled)}] {done}/{total}", end="", flush=True)

            vis_data[key] = {
                "vary_angles": vary_angles,
                "vary_jt": vary_jt,
                "fixed_jt": fixed_jt,
                "fixed_val": fixed_val,
                "lift_counts": np.array(lists["lift"]),
                "pitch_counts": np.array(lists["pitch"]),
                "yaw_counts": np.array(lists["yaw"]),
                "q_ring_counts": [np.array(rc) for rc in ring_list],
                "extra": np.array(extra_list),
            }

        print()
        return vis_data

    def _decode_traces(self, sd: dict, cnt_key: str, angle_sel: int) -> tuple:
        vary_angles = sd["vary_angles"]
        fixed_jt = sd["fixed_jt"]
        fixed_val = sd["fixed_val"]
        vary_jt = sd["vary_jt"]
        counts = sd[cnt_key]

        saws = []
        for i in range(len(vary_angles)):
            if self._profile_has_signal(counts[i]):
                saws.append(
                    float(
                        self._angle_to_ring_index(
                            self._decode_angle_sawtooth_from_profile(counts[i]),
                            self.output_ring_size,
                        )
                    )
                )
            else:
                saws.append(np.nan)
        saws = np.array(saws, dtype=float)

        true_idxs = []
        for angle in vary_angles:
            q = np.zeros(self.num_joints)
            q[fixed_jt] = fixed_val
            q[vary_jt] = angle
            tgt = self._lift_pitch_yaw_from_angles(q)
            true_idxs.append(self._angle_to_ring_index(tgt[angle_sel], self.output_ring_size))

        return saws, np.array(true_idxs), vary_angles

    def _profile_has_signal(self, profile: np.ndarray, eps: float = 1e-9) -> bool:
        p = np.asarray(profile, dtype=float)
        return bool(np.max(p) - np.min(p) > eps)

    def _sawtooth_mae_per_slice(self, vis: dict) -> dict:
        """Compute per-slice sawtooth circular MAE and signal coverage."""
        result = {}
        total_profiles = 0
        total_with_signal = 0
        all_errs: dict = {"lift": [], "pitch": [], "yaw": []}
        for slice_key, sd in vis.items():
            vary_angles = sd["vary_angles"]
            fixed_jt = sd["fixed_jt"]
            fixed_val = sd["fixed_val"]
            vary_jt = sd["vary_jt"]
            errs: dict = {"lift": [], "pitch": [], "yaw": []}
            n_prof = 0
            n_sig = 0
            for i, angle in enumerate(vary_angles):
                q = np.zeros(self.num_joints)
                q[fixed_jt] = fixed_val
                q[vary_jt] = angle
                lift_t, pitch_t, yaw_t = self._lift_pitch_yaw_from_angles(q)
                for name, true_a in [("lift", lift_t), ("pitch", pitch_t), ("yaw", yaw_t)]:
                    counts = sd[f"{name}_counts"][i]
                    n_prof += 1
                    if self._profile_has_signal(counts):
                        n_sig += 1
                        pred = self._decode_angle_sawtooth_from_profile(counts)
                        err = abs(circular_angle_error(pred, true_a))
                        errs[name].append(err)
                        all_errs[name].append(err)
            total_profiles += n_prof
            total_with_signal += n_sig
            result[slice_key] = {
                "sawtooth_mae_deg": {
                    name: float(np.degrees(np.mean(errs[name]))) if errs[name] else float("nan")
                    for name in ("lift", "pitch", "yaw")
                },
                "signal_coverage_pct": 100.0 * n_sig / n_prof if n_prof else 0.0,
            }
        result["overall"] = {
            "sawtooth_mae_deg": {
                name: float(np.degrees(np.mean(all_errs[name]))) if all_errs[name] else float("nan")
                for name in ("lift", "pitch", "yaw")
            },
            "signal_coverage_pct": 100.0 * total_with_signal / total_profiles if total_profiles else 0.0,
        }
        return result

    def _print_output_ring_stats(self, vis: dict):
        all_lift = []
        all_pitch = []
        all_yaw = []
        for sd in vis.values():
            all_lift.append(sd["lift_counts"].ravel())
            all_pitch.append(sd["pitch_counts"].ravel())
            all_yaw.append(sd["yaw_counts"].ravel())

        print("\n[Signed output rings]")
        for name, arr in [
            ("Lift", np.concatenate(all_lift)),
            ("Pitch", np.concatenate(all_pitch)),
            ("Yaw", np.concatenate(all_yaw)),
        ]:
            n_total = len(arr)
            n_silent = int(np.sum(arr == 0))
            silent_pct = 100.0 * n_silent / n_total if n_total else 0.0
            all_silent = bool(np.all(arr == 0))
            mean_abs = float(np.mean(np.abs(arr)))
            max_abs = float(np.max(np.abs(arr)))
            contrast = float(np.max(arr) - np.min(arr))
            print(
                f"  {name}: mean={np.mean(arr):.3f} max={np.max(arr):.0f} "
                f"mean_abs={mean_abs:.3f} max_abs={max_abs:.3f} contrast={contrast:.3f} "
                f"silent={silent_pct:.1f}% all_silent={all_silent}"
            )

    def _expected_term_curve(self, term_name: str, vary_angles: np.ndarray, q_fixed: float, vary_joint: int) -> np.ndarray:
        if vary_joint == 1:
            q1 = q_fixed
            q2 = vary_angles
            if term_name == "cos1":
                return np.cos(q1) * np.ones_like(q2)
            if term_name == "sin1":
                return np.sin(q1) * np.ones_like(q2)
            if term_name == "cos2":
                return np.ones_like(q2) * np.cos(q2)
            if term_name == "sin2":
                return np.ones_like(q2) * np.sin(q2)
            if term_name == "cos1cos2":
                return np.cos(q1) * np.cos(q2)
            if term_name == "cos1sin2":
                return np.cos(q1) * np.sin(q2)
            if term_name == "sin1cos2":
                return np.sin(q1) * np.cos(q2)
            return np.sin(q1) * np.sin(q2)

        q1 = vary_angles
        q2 = q_fixed
        if term_name == "cos1":
            return np.cos(q1) * np.ones_like(q1)
        if term_name == "sin1":
            return np.sin(q1) * np.ones_like(q1)
        if term_name == "cos2":
            return np.ones_like(q1) * np.cos(q2)
        if term_name == "sin2":
            return np.ones_like(q1) * np.sin(q2)
        if term_name == "cos1cos2":
            return np.cos(q1) * np.cos(q2)
        if term_name == "cos1sin2":
            return np.cos(q1) * np.sin(q2)
        if term_name == "sin1cos2":
            return np.sin(q1) * np.cos(q2)
        return np.sin(q1) * np.sin(q2)

    def _visualize_signed_product(self, vis: dict, save_dir: str):
        n_out = self.output_ring_size
        n_idx = np.arange(n_out)
        colors = {
            "lift": "#2980b9",
            "pitch": "#e67e22",
            "yaw": "#8e44ad",
            "q1": "#e74c3c",
            "q2": "#16a085",
            "true": "#c0392b",
            "saw": "#27ae60",
        }

        s = vis["q1_fixed"]
        sample_idx = len(s["vary_angles"]) // 2
        feature_maps = self._signed_vector_to_maps(s["extra"][sample_idx])
        term_names = self._signed_term_names()

        with plt.rc_context(self._RC):
            fig, axes = plt.subplots(len(term_names), 2, figsize=(10, 3 * len(term_names)))
            for row, term_name in enumerate(term_names):
                for col, sign in enumerate(("pos", "neg")):
                    ax = axes[row, col]
                    hmap = feature_maps[f"{term_name}_{sign}"]
                    im = ax.imshow(hmap, aspect="auto", origin="lower", cmap="plasma", interpolation="nearest")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f"{term_name} {sign}")
                    ax.set_xlabel("q2 grid index")
                    ax.set_ylabel("q1 grid index")
            fig.suptitle("Signed product feature activity", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "signed_product_feature_activity.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        with plt.rc_context(self._RC):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax, term_name in zip(axes.ravel(), term_names):
                signed_map = feature_maps[f"{term_name}_pos"] - feature_maps[f"{term_name}_neg"]
                im = ax.imshow(signed_map, aspect="auto", origin="lower", cmap="coolwarm", interpolation="nearest")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(term_name)
                ax.set_xlabel("q2 grid index")
                ax.set_ylabel("q1 grid index")
            fig.suptitle("Signed product term maps (pos - neg)", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "signed_product_term_maps.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        with plt.rc_context(self._RC):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for ax, term_name in zip(axes.ravel(), term_names):
                for key, label in [("q1_fixed", "q1 fixed - q2 varies"), ("q2_fixed", "q2 fixed - q1 varies")]:
                    sd = vis[key]
                    vary_angles = sd["vary_angles"]
                    measured = np.array([self._signed_vector_term_sums(vec)[term_name] for vec in sd["extra"]])
                    expected = self._expected_term_curve(
                        term_name=term_name,
                        vary_angles=vary_angles,
                        q_fixed=float(sd["fixed_val"]),
                        vary_joint=int(sd["vary_jt"]),
                    )
                    exp_max = float(np.max(np.abs(expected)))
                    mea_max = float(np.max(np.abs(measured)))
                    if exp_max > 0.0:
                        expected_scaled = expected * (mea_max / exp_max)
                    else:
                        expected_scaled = expected
                    ax.plot(vary_angles, measured, lw=1.8, marker="o", ms=3.0, label=f"Measured ({label})")
                    ax.plot(vary_angles, expected_scaled, lw=1.2, ls="--", label=f"Expected ({label})")
                ax.axhline(0, color="gray", lw=0.8, ls=":")
                ax.set_title(term_name)
                ax.set_xlabel("Angle (rad)")
                ax.set_ylabel("Signed spike count delta")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
            fig.suptitle("Signed product term traces", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "signed_product_term_traces.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        sample_idxs = np.linspace(0, len(s["vary_angles"]) - 1, 5, dtype=int)
        with plt.rc_context(self._RC):
            fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
            for ax, si in zip(axes, sample_idxs):
                angle = s["vary_angles"][si]
                ax.plot(n_idx, s["lift_counts"][si], color=colors["lift"], lw=1.8, label="Lift")
                ax.plot(n_idx, s["pitch_counts"][si], color=colors["pitch"], lw=1.8, label="Pitch")
                ax.plot(n_idx, s["yaw_counts"][si], color=colors["yaw"], lw=1.8, label="Yaw")
                ax.plot(
                    n_idx,
                    self._resample_profile(s["q_ring_counts"][0][si], n_out),
                    color=colors["q1"],
                    lw=1.4,
                    ls="--",
                    label="q1 ring",
                )
                ax.plot(
                    n_idx,
                    self._resample_profile(s["q_ring_counts"][1][si], n_out),
                    color=colors["q2"],
                    lw=1.4,
                    ls="--",
                    label="q2 ring",
                )
                ax.set_title(f"q2 = {angle:.2f} rad")
                ax.set_xlabel("Neuron index")
                ax.set_ylabel("Spike count delta")
                ax.grid(axis="y", alpha=0.25)
                if si == sample_idxs[0]:
                    ax.legend(fontsize=8)
            fig.suptitle("Signed product output rings - spike count delta (NEST)", y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "signed_product_output_ring_samples.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        slice_meta = [
            ("q1_fixed", "q1 fixed - q2 varies", "q2 (rad)"),
            ("q2_fixed", "q2 fixed - q1 varies", "q1 (rad)"),
        ]
        output_cols = [("Lift", "lift_counts", 0), ("Pitch", "pitch_counts", 1), ("Yaw", "yaw_counts", 2)]

        all_traces = {}
        for key, _, _ in slice_meta:
            for lbl, cnt_key, angle_sel in output_cols:
                all_traces[(key, lbl)] = self._decode_traces(vis[key], cnt_key, angle_sel)

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            for row, (key, row_title, xlabel) in enumerate(slice_meta):
                for col, (lbl, _cnt_key, _angle_sel) in enumerate(output_cols):
                    saws, true, vary = all_traces[(key, lbl)]
                    ax = axs[row, col]
                    ax.plot(vary, saws, color=colors["saw"], lw=2.4, marker="^", ms=3.0, label="Sawtooth")
                    ax.plot(vary, true, color=colors["true"], lw=2.0, ls="--", label="True")
                    ax.set_title(f"{lbl}\n{row_title}")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Ring neuron index")
                    ax.set_ylim(0, n_out)
                    ax.grid(alpha=0.2)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=8)
            fig.suptitle("Signed product sawtooth decoded index vs target (NEST spike count delta)", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "signed_product_sawtooth_vs_target.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

        half_n = n_out / 2.0

        def _circ_err(decoded, true):
            return circular_signed_difference(decoded, true, n_out)

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            for row, (key, row_title, xlabel) in enumerate(slice_meta):
                for col, (lbl, _cnt_key, _angle_sel) in enumerate(output_cols):
                    saws, true, vary = all_traces[(key, lbl)]
                    ax = axs[row, col]
                    ax.axhline(0, color="gray", lw=1.0, ls=":")
                    ax.plot(vary, _circ_err(saws, true), color=colors["saw"], lw=2.4, marker="^", ms=3.0, label="Sawtooth")
                    ax.set_title(f"{lbl}\n{row_title}")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Circular index error (neurons)")
                    ax.set_ylim(-half_n, half_n)
                    ax.grid(alpha=0.2)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=8)
            fig.suptitle("Signed product sawtooth circular index error (NEST spike count delta)", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "signed_product_sawtooth_circular_index_error.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _save_summary(self, vis_signed, save_dir: str):
        def _output_ring_stats(vis):
            all_counts = []
            for sd in vis.values():
                for key in ("lift_counts", "pitch_counts", "yaw_counts"):
                    all_counts.append(sd[key].ravel())
            arr = np.concatenate(all_counts)
            n_total = len(arr)
            n_silent = int(np.sum(arr == 0))
            return {
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "mean_abs": float(np.mean(np.abs(arr))),
                "max_abs": float(np.max(np.abs(arr))),
                "contrast": float(np.max(arr) - np.min(arr)),
                "silent_pct": 100.0 * n_silent / n_total if n_total else 0.0,
                "all_silent": bool(np.all(arr == 0)),
            }

        sawtooth_metrics = self._sawtooth_mae_per_slice(vis_signed)

        metrics = {
            "note": "All inference and plotting use NEST spike_recorder count deltas.",
            "decoder": "sawtooth",
            "population_size": self.population_size,
            "num_joints": self.num_joints,
            "joint_axes": self.joint_axes,
            "output_ring_size": self.output_ring_size,
            "feature_grid_size": self.feature_grid_size,
            "sim_settle_ms": self.sim_settle_ms,
            "baseline_burnin_ms": self.baseline_burnin_ms,
            "ridge_lambda": self.ridge_lambda,
            "signed_product_dc_baseline": self.signed_product_dc_baseline,
            "signed_product_input_weight": self.signed_product_input_weight,
            "signed_product_output_weight_scale": self.signed_product_output_weight_scale,
            "output_weight_matrix_is_target_by_source": bool(self._output_weight_matrix_is_target_by_source),
            "calibrated_signed_product_params": self._calibrated_signed_product_params,
            "signed_product_feature_stats": self._signed_feature_stats,
            "signed_product_output_stats": _output_ring_stats(vis_signed),
            "q_joint_contribution_diagnostic": self._q_joint_contribution_diagnostic,
            "sawtooth_mae_per_slice": sawtooth_metrics,
            "active_signed_product_terms": self._active_signed_product_terms,
            "n_vis_points": self.n_vis_points,
        }

        weights_dir = "./config/ring_decoding_weights"
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        n = self.population_size
        j = self.num_joints
        weights_path = os.path.join(weights_dir, f"N_{n}_J_{j}_multi_ring_sawtooth_weights.npz")
        metadata_path = os.path.join(weights_dir, f"N_{n}_J_{j}_multi_ring_sawtooth_metadata.json")

        save_npz_legacy(
            weights_path,
            W_signed_lift=self._W_signed_lift,
            W_signed_pitch=self._W_signed_pitch,
            W_signed_yaw=self._W_signed_yaw,
            signed_product_feature_order=np.array(self._signed_feature_order),
            signed_product_population_size=np.array(self.feature_grid_size),
            population_size=np.array(self.population_size),
            num_joints=np.array(self.num_joints),
            joint_axes=np.array(self.joint_axes),
            output_ring_size=np.array(self.output_ring_size),
            signed_product_output_weight_scale=np.array(self.signed_product_output_weight_scale),
            output_weight_matrix_is_target_by_source=np.array(bool(self._output_weight_matrix_is_target_by_source)),
        )

        save_json_legacy(metadata_path, metrics)

        print(f"\nWeights  -> {weights_path}")
        print(f"Metadata -> {metadata_path}")
        print(f"Plots    -> {save_dir}/")

    def run(self, save_dir: str = "./outputs/multi_ring_sawtooth_decoding"):
        os.makedirs(save_dir, exist_ok=True)

        configs = self._build_joint_configs()
        self._train_configs = configs

        n_out = self.output_ring_size
        n_signed = self._signed_feature_dim()
        n_train = len(configs)

        print(
            f"Joints: {self.num_joints}  axes: {self.joint_axes}  "
            f"population: {self.population_size}  output ring: {n_out}"
        )
        print(
            f"Signed product grid: N_side={self.feature_grid_size}  "
            f"features={n_signed}"
        )
        self._output_weight_matrix_is_target_by_source = self._verify_all_to_all_weight_orientation()
        if self.feature_grid_size > 24:
            print("WARNING: feature_grid_size > 24 may make all-to-all output wiring heavy.")

        self._calibrate_signed_product_grid_layer()

        x_signed = np.zeros((n_train, n_signed))
        y_lift = np.zeros((n_train, n_out))
        y_pitch = np.zeros((n_train, n_out))
        y_yaw = np.zeros((n_train, n_out))

        print(f"\nCollecting training signed product features ({n_train} configs) ...")
        for idx, q in enumerate(configs):
            signed_vec, lift, pitch, yaw = self._collect_signed_product_sample(q)
            x_signed[idx] = signed_vec
            y_lift[idx] = self._target_output_modulation(lift)
            y_pitch[idx] = self._target_output_modulation(pitch)
            y_yaw[idx] = self._target_output_modulation(yaw)

            filled = int(40 * (idx + 1) / n_train)
            print(f"\r  [{'#' * filled}{'.' * (40 - filled)}] {idx + 1}/{n_train}", end="", flush=True)
        print()

        mean_abs_signed = float(np.mean(np.abs(x_signed)))
        max_abs_signed = float(np.max(np.abs(x_signed)))
        active_signed = int(np.sum(np.abs(x_signed) > 1e-9))
        total_signed = int(x_signed.size)
        silent_signed = 100.0 * (total_signed - active_signed) / total_signed if total_signed else 0.0
        self._signed_feature_stats = {
            "mean_abs": mean_abs_signed,
            "max_abs": max_abs_signed,
            "active": active_signed,
            "silent_pct": silent_signed,
            "feature_dim": n_signed,
        }
        print(
            f"Signed feature activity: mean_abs={mean_abs_signed:.3f} max_abs={max_abs_signed:.0f} "
            f"active={active_signed}/{total_signed} silent={silent_signed:.1f}% feature_dim={n_signed}"
        )
        print("Active signed product terms:", self._active_signed_product_terms)

        if max_abs_signed == 0:
            raise RuntimeError(
                "Signed product grid is silent. Increase signed_product_dc_baseline or signed_product_input_weight."
            )

        print("Fitting output decoder (lift / pitch / yaw) ...")
        self._W_signed_lift = self._fit_ridge(x_signed, y_lift)
        self._W_signed_pitch = self._fit_ridge(x_signed, y_pitch)
        self._W_signed_yaw = self._fit_ridge(x_signed, y_yaw)
        for name, w in [
            ("lift", self._W_signed_lift),
            ("pitch", self._W_signed_pitch),
            ("yaw", self._W_signed_yaw),
        ]:
            if not np.all(np.isfinite(w)):
                raise RuntimeError(f"Non-finite values found in W_signed_{name}")
            print(
                f"W_signed_{name}: min={np.min(w):.3f} max={np.max(w):.3f} "
                f"mean_abs={np.mean(np.abs(w)):.3f}"
            )
        self._signed_product_active = True

        self._calibrate_signed_product_output_scale()
        print(f"Chosen signed_product_output_weight_scale={self.signed_product_output_weight_scale}")

        vis_signed = self._run_vis_simulations(
            sim_fn=self._run_full_signed_product_sim,
            label="signed product grid (full spiking)",
            n_vis=self.n_vis_points,
        )

        print("\nNEST spike-count diagnostics")
        self._print_output_ring_stats(vis_signed)
        mae_report = self._sawtooth_mae_per_slice(vis_signed)
        print("\n[Sawtooth MAE (circular) from NEST spike count deltas]")
        for skey in ("q1_fixed", "q2_fixed", "overall"):
            m = mae_report[skey]
            s = m["sawtooth_mae_deg"]
            print(
                f"  {skey}: Lift={s['lift']:.2f}\u00b0  Pitch={s['pitch']:.2f}\u00b0  Yaw={s['yaw']:.2f}\u00b0"
                f"  sig_cov={m['signal_coverage_pct']:.1f}%"
            )

        self._visualize_signed_product(vis_signed, save_dir)
        self._save_summary(vis_signed, save_dir)


if __name__ == "__main__":
    params_file = "./config/model_params/multi_ring_params.json"
    trainer = CompositionalFourierDecoderTrainer(params_file=params_file)
    trainer.run(save_dir="./outputs/multi_ring_sawtooth_decoding")
