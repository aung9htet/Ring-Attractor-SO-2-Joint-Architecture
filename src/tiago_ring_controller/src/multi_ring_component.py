#!/usr/bin/env python3

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest
import numpy as np

from ring_attractor import Ring_Attractor
from tiago_ring_controller.artifacts import load_json_legacy, load_numpy_legacy
from tiago_ring_controller.config import load_json, resolve_legacy_read_path
from tiago_ring_controller.features import (
    mapped_ring_indices,
    signed_product_term_values,
)
from tiago_ring_controller.math.circular import (
    angle_to_neuron_index,
    angle_to_ring_index,
    decode_sawtooth_profile,
    preferred_angles,
    ring_index_to_angle,
)
from tiago_ring_controller.math.kinematics import lift_pitch_yaw_from_angles
from tiago_ring_controller.evaluation.metrics import (
    circular_index_error,
    circular_index_nearest,
)
from tiago_ring_controller.nest.multi_ring import (
    build_output_rings,
    build_signed_product_layer,
)


DEFAULT_PARAMS_FILE = "./config/model_params/multi_ring_params.json"


class MultiRingDecode:
    """Usage-only multi-ring decoder component exposing all neuron populations."""

    def __init__(self, params_file: str = DEFAULT_PARAMS_FILE, reset_kernel: bool = False):
        self.params_file = params_file
        self.reset_kernel = bool(reset_kernel)

        p = load_json(params_file)

        self.population_size = int(p["population_size"])
        self.num_joints = int(p.get("num_joints", 2))
        if self.num_joints != 2:
            raise ValueError("This decoder supports exactly 2 joints: q1 and q2.")

        self.joint_axes = list(p.get("joint_axes", ["x", "y"]))
        self.output_ring_size = int(p.get("output_ring_size", 100))
        self.stimulus_half_width = int(p["stimulus_half_width"])
        self.output_dc_baseline = float(p.get("output_dc_baseline", 200.0))

        self.feature_grid_size = int(
            p.get("signed_product_population_size", p.get("product_population_size", min(self.population_size, 32)))
        )
        self.signed_product_dc_baseline = float(p.get("signed_product_dc_baseline", 180.0))
        self.signed_product_input_weight = float(p.get("signed_product_input_weight", 120.0))
        self.signed_product_output_weight_scale = float(
            p.get("signed_product_output_weight_scale", p.get("sfp_output_weight_scale", 1.0))
        )
        self.output_weight_matrix_is_target_by_source = True

        self._W_signed_lift = None
        self._W_signed_pitch = None
        self._W_signed_yaw = None
        self._signed_feature_order = []

        self.joint_rings = {}
        self.signed_product_layer = {}
        self.signed_product_populations = {}
        self.output_rings = {}
        self.populations = {}

    def _preferred_angles(self, n: int) -> np.ndarray:
        return preferred_angles(n)

    def _angle_to_center_idx(self, angle: float) -> int:
        return angle_to_neuron_index(angle, self.population_size)

    def angle_to_ring_index(self, angle, n=None):
        size = int(self.output_ring_size if n is None else n)
        return angle_to_ring_index(angle, size)

    def ring_index_to_angle(self, index, n=None):
        size = int(self.output_ring_size if n is None else n)
        return ring_index_to_angle(index, size)

    def _decode_angle_sawtooth_from_profile(self, profile: np.ndarray) -> float:
        profile = np.asarray(profile, dtype=float)
        a = np.maximum(profile - np.min(profile), 0.0)
        theta = self._preferred_angles(len(profile))
        return float(np.arctan2(np.dot(a, np.sin(theta)), np.dot(a, np.cos(theta))))

    def _as_node_collection(self, nodes):
        if isinstance(nodes, nest.NodeCollection):
            return nodes
        gids = []
        for node in nodes:
            if isinstance(node, nest.NodeCollection):
                node_gids = node.get("global_id")
                if np.isscalar(node_gids):
                    gids.append(int(node_gids))
                else:
                    gids.extend([int(g) for g in node_gids])
            else:
                gids.append(int(node))
        return nest.NodeCollection(gids)

    def _get_spike_counts(self, recorders) -> np.ndarray:
        counts = []
        for sr in (recorders if hasattr(recorders, "__iter__") else [recorders]):
            ev = sr.get("events")
            counts.append(len(ev.get("times", [])))
        return np.array(counts, dtype=float)

    def _joint_index_from_name_or_idx(self, joint_idx_or_name):
        if isinstance(joint_idx_or_name, str):
            key = joint_idx_or_name.lower().strip()
            if key in ("q1", "joint1", "0"):
                return 0
            if key in ("q2", "joint2", "1"):
                return 1
            raise KeyError(f"Unknown joint name: {joint_idx_or_name}")
        idx = int(joint_idx_or_name)
        if idx < 0 or idx >= self.num_joints:
            raise IndexError(f"Joint index out of range: {idx}")
        return idx

    def _load_trained_weights(self):
        n = self.population_size
        j = self.num_joints
        weights_path = f"./config/ring_decoding_weights/N_{n}_J_{j}_multi_ring_sawtooth_weights.npz"
        metadata_path = f"./config/ring_decoding_weights/N_{n}_J_{j}_multi_ring_sawtooth_metadata.json"

        weights_path = resolve_legacy_read_path(weights_path)
        metadata_path = resolve_legacy_read_path(metadata_path)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                "Pre-trained multi-ring sawtooth weights not found. "
                f"Expected: {weights_path}"
            )

        data = load_numpy_legacy(weights_path, allow_pickle=True)
        self._W_signed_lift = data["W_signed_lift"]
        self._W_signed_pitch = data["W_signed_pitch"]
        self._W_signed_yaw = data["W_signed_yaw"]
        self._signed_feature_order = [str(name) for name in data["signed_product_feature_order"]]

        self.feature_grid_size = int(data["signed_product_population_size"])
        self.population_size = int(data["population_size"])
        self.num_joints = int(data["num_joints"])
        self.joint_axes = [str(axis) for axis in data["joint_axes"]]
        self.output_ring_size = int(data["output_ring_size"])
        self.signed_product_output_weight_scale = float(data["signed_product_output_weight_scale"])
        self.output_weight_matrix_is_target_by_source = bool(data["output_weight_matrix_is_target_by_source"])

        if os.path.exists(metadata_path):
            metadata = load_json_legacy(metadata_path)

            calibrated = metadata.get("calibrated_signed_product_params", {})
            if "signed_product_dc_baseline" in calibrated:
                self.signed_product_dc_baseline = float(calibrated["signed_product_dc_baseline"])
            if "signed_product_input_weight" in calibrated:
                self.signed_product_input_weight = float(calibrated["signed_product_input_weight"])

            if "signed_product_output_weight_scale" in calibrated:
                self.signed_product_output_weight_scale = float(calibrated["signed_product_output_weight_scale"])
            elif "signed_product_output_weight_scale" in metadata:
                self.signed_product_output_weight_scale = float(metadata["signed_product_output_weight_scale"])

            if "output_weight_matrix_is_target_by_source" in metadata:
                self.output_weight_matrix_is_target_by_source = bool(metadata["output_weight_matrix_is_target_by_source"])

    def _build_joint_rings(self):
        neuron_params_file = os.path.join(os.path.dirname(self.params_file), "neuron_params.json")

        rings = []
        for i in range(self.num_joints):
            ring = Ring_Attractor(
                population_size=self.population_size,
                reset_kernel=False,
                params_file=neuron_params_file,
            )
            rings.append(ring)

        q1_nodes = self._as_node_collection(rings[0].ring_neurons)
        q2_nodes = self._as_node_collection(rings[1].ring_neurons)
        q1_recs = self._as_node_collection(rings[0].ring_spike_recorders)
        q2_recs = self._as_node_collection(rings[1].ring_spike_recorders)

        self.joint_rings = {
            "q1": {"nodes": q1_nodes, "ring": rings[0]},
            "q2": {"nodes": q2_nodes, "ring": rings[1]},
        }

        self.populations["q1_ring"] = {"nodes": q1_nodes, "recs": q1_recs, "kind": "joint_ring"}
        self.populations["q2_ring"] = {"nodes": q2_nodes, "recs": q2_recs, "kind": "joint_ring"}

    def _build_signed_product_layer(self):
        n_ring = self.population_size
        n_side = self.feature_grid_size
        n_cells = n_side * n_side
        theta_full = self._preferred_angles(n_ring)
        mapped_idxs = mapped_ring_indices(n_ring, n_side)
        theta_side = theta_full[mapped_idxs]
        ones = np.ones_like(theta_side)
        eps = 1e-9

        feature_order = []

        term_values = None

        def term_values_factory():
            nonlocal term_values
            if term_values is None:
                term_values = {
                    "cos1": (np.cos(theta_side), ones),
                    "sin1": (np.sin(theta_side), ones),
                    "cos2": (ones, np.cos(theta_side)),
                    "sin2": (ones, np.sin(theta_side)),
                    "cos1cos2": (np.cos(theta_side), np.cos(theta_side)),
                    "cos1sin2": (np.cos(theta_side), np.sin(theta_side)),
                    "sin1cos2": (np.sin(theta_side), np.cos(theta_side)),
                    "sin1sin2": (np.sin(theta_side), np.sin(theta_side)),
                }
            return term_values

        def register_population(feature_name, nodes, recs):
            feature_order.append(feature_name)
            self.signed_product_populations[feature_name] = {
                "nodes": nodes,
                "recs": recs,
            }
            self.populations[feature_name] = {
                "nodes": nodes,
                "recs": recs,
                "kind": "signed_product",
            }

        q1_ring = self.joint_rings["q1"]["ring"]
        q2_ring = self.joint_rings["q2"]["ring"]
        built = build_signed_product_layer(
            nest,
            q1_ring.ring_neurons,
            q2_ring.ring_neurons,
            n_ring,
            n_side,
            self.signed_product_dc_baseline,
            self.signed_product_input_weight,
            epsilon=eps,
            mapped_indices=mapped_idxs,
            term_values_factory=term_values_factory,
            feature_order=feature_order,
            on_population_built=register_population,
        )

        terms = {}
        for feature_name in feature_order:
            term_name, sign = feature_name.rsplit("_", 1)
            terms.setdefault(term_name, {})[sign] = built.populations[feature_name]
        self.signed_product_layer = {
            "feature_grid_size": n_side,
            "n_cells": n_cells,
            "mapped_ring_idxs": mapped_idxs,
            "feature_order": feature_order,
            "terms": terms,
            "flat_nodes": built.flat_nodes,
        }

    def _build_output_rings(self):
        n_out = self.output_ring_size
        scale = self.signed_product_output_weight_scale
        src_nodes = self.signed_product_layer["flat_nodes"]

        weights = [
            ("lift", self._W_signed_lift),
            ("pitch", self._W_signed_pitch),
            ("yaw", self._W_signed_yaw),
        ]

        self.output_rings = {}

        def register_population(name, nodes, recs):
            self.output_rings[name] = {"nodes": nodes, "recs": recs}
            self.populations[f"{name}_ring"] = {"nodes": nodes, "recs": recs, "kind": "output_ring"}

        build_output_rings(
            nest,
            self.signed_product_layer,
            dict(weights),
            n_out,
            self.output_dc_baseline,
            scale,
            on_population_built=register_population,
            source_nodes=src_nodes,
            matrix_orientation_getter=lambda: self.output_weight_matrix_is_target_by_source,
        )

    def build(self):
        if self.reset_kernel:
            nest.ResetKernel()

        self.joint_rings = {}
        self.signed_product_layer = {}
        self.signed_product_populations = {}
        self.output_rings = {}
        self.populations = {}

        self._load_trained_weights()
        self._build_joint_rings()
        self._build_signed_product_layer()
        self._build_output_rings()
        return self

    def get_population(self, name: str):
        if name not in self.populations:
            raise KeyError(f"Unknown population: {name}")
        return self.populations[name]

    def get_population_nodes(self, name: str):
        return self.get_population(name)["nodes"]

    def get_population_recorders(self, name: str):
        return self.get_population(name)["recs"]

    def get_joint_nodes(self, joint_idx_or_name):
        j = self._joint_index_from_name_or_idx(joint_idx_or_name)
        key = "q1" if j == 0 else "q2"
        return self.joint_rings[key]["nodes"]

    def get_signed_product_nodes(self, feature_name: str):
        if feature_name not in self.signed_product_populations:
            raise KeyError(f"Unknown signed-product feature: {feature_name}")
        return self.signed_product_populations[feature_name]["nodes"]

    def get_output_nodes(self, name: str):
        key = name.lower().strip()
        if key not in self.output_rings:
            raise KeyError(f"Unknown output ring: {name}")
        return self.output_rings[key]["nodes"]

    def inject_joint_bump(self, joint_idx, angle_rad=None, center_idx=None, half_width=None):
        j = self._joint_index_from_name_or_idx(joint_idx)
        if center_idx is None:
            if angle_rad is None:
                raise ValueError("Provide either angle_rad or center_idx.")
            center_idx = self._angle_to_center_idx(float(angle_rad))

        width = int(self.stimulus_half_width if half_width is None else half_width)
        key = "q1" if j == 0 else "q2"
        self.joint_rings[key]["ring"].inject_stimulus(center_index=int(center_idx), half_width=width)

    def inject_q(self, q):
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape[0] != self.num_joints:
            raise ValueError(f"Expected q with {self.num_joints} joints, got shape {q_arr.shape}.")
        for j, angle in enumerate(q_arr):
            self.inject_joint_bump(j, angle_rad=float(angle))

    def get_joint_ring_counts(self, joint_idx_or_name):
        j = self._joint_index_from_name_or_idx(joint_idx_or_name)
        key = "q1" if j == 0 else "q2"
        return self.joint_rings[key]["ring"]._get_spike_counts()

    def get_signed_product_counts(self, feature_name=None):
        if feature_name is None:
            return {
                name: self._get_spike_counts(data["recs"])
                for name, data in self.signed_product_populations.items()
            }

        if feature_name not in self.signed_product_populations:
            raise KeyError(f"Unknown signed-product feature: {feature_name}")
        return self._get_spike_counts(self.signed_product_populations[feature_name]["recs"])

    def get_output_counts(self, name):
        key = name.lower().strip()
        if key not in self.output_rings:
            raise KeyError(f"Unknown output ring: {name}")
        return self._get_spike_counts(self.output_rings[key]["recs"])

    def decode_sawtooth_angle(self, name):
        counts = self.get_output_counts(name)
        return self._decode_angle_sawtooth_from_profile(counts)

    def decode_all_sawtooth_angles(self):
        return {
            "lift": self.decode_sawtooth_angle("lift"),
            "pitch": self.decode_sawtooth_angle("pitch"),
            "yaw": self.decode_sawtooth_angle("yaw"),
        }


class MultiRingDecodeAnalysis:
    """Analysis-only helper that uses MultiRingDecode and writes diagnostic plots."""

    _RC = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }

    def __init__(
        self,
        decoder: MultiRingDecode = None,
        params_file: str = DEFAULT_PARAMS_FILE,
        output_dir: str = "./outputs/multi_ring_component_analysis",
        reset_kernel: bool = True,
    ):
        self.params_file = params_file
        self.output_dir = output_dir
        self.decoder = decoder if decoder is not None else MultiRingDecode(params_file=params_file, reset_kernel=reset_kernel)

        p = load_json(params_file)
        self.sim_settle_ms = float(p.get("sim_settle_ms", 50.0))
        self.baseline_burnin_ms = float(p.get("baseline_burnin_ms", self.sim_settle_ms))
        self.n_vis_points = int(p.get("n_vis_points", 20))

    def _lift_pitch_yaw_from_angles(self, q: np.ndarray):
        return lift_pitch_yaw_from_angles(q, self.decoder.joint_axes)

    def _counts_sub(self, a, b):
        if isinstance(a, dict):
            return {k: self._counts_sub(a[k], b[k]) for k in a}
        return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)

    def _measure_window_delta(self, get_counts_fn, pre_window_action=None):
        before = get_counts_fn()
        if pre_window_action is not None:
            pre_window_action()
        nest.Simulate(self.sim_settle_ms)
        after = get_counts_fn()
        return self._counts_sub(after, before)

    def _circular_index_nearest(self, decoded, true, n):
        return circular_index_nearest(decoded, true, n)

    def _circular_index_error(self, decoded, true, n):
        return circular_index_error(decoded, true, n)

    def _print_progress_bar(self, done: int, total: int, width: int = 40, prefix: str = ""):
        total = max(int(total), 1)
        done = min(max(int(done), 0), total)
        filled = int(width * done / total)
        bar = f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"
        print(f"\r{prefix}{bar}", end="", flush=True)
        if done >= total:
            print()

    def _run_single_q(self, q):
        q = np.asarray(q, dtype=float)
        decoder = MultiRingDecode(params_file=self.params_file, reset_kernel=True).build()

        def get_counts():
            return {
                "q1": decoder.get_joint_ring_counts("q1"),
                "q2": decoder.get_joint_ring_counts("q2"),
                "signed": decoder.get_signed_product_counts(),
                "lift": decoder.get_output_counts("lift"),
                "pitch": decoder.get_output_counts("pitch"),
                "yaw": decoder.get_output_counts("yaw"),
            }

        nest.Simulate(self.baseline_burnin_ms)
        baseline = self._measure_window_delta(get_counts_fn=get_counts)
        stimulated = self._measure_window_delta(get_counts_fn=get_counts, pre_window_action=lambda: decoder.inject_q(q))
        delta = self._counts_sub(stimulated, baseline)
        return decoder, delta

    def _collect_vis_data(self, n_vis: int):
        vary_angles = np.linspace(-np.pi, np.pi, n_vis, endpoint=False)
        slices = [
            {"name": "q1_fixed", "fixed_joint": 0, "vary_joint": 1, "fixed_value": 0.0},
            {"name": "q2_fixed", "fixed_joint": 1, "vary_joint": 0, "fixed_value": 0.0},
        ]

        out = {}
        total = 2 * n_vis
        done = 0
        print(f"Running {total} analysis simulations ({n_vis} per slice) ...")
        for sl in slices:
            block = {
                "vary_angles": vary_angles.copy(),
                "fixed_joint": sl["fixed_joint"],
                "vary_joint": sl["vary_joint"],
                "fixed_value": sl["fixed_value"],
                "lift": [],
                "pitch": [],
                "yaw": [],
                "q1": [],
                "q2": [],
                "signed": [],
            }
            for angle in vary_angles:
                q = np.zeros(self.decoder.num_joints, dtype=float)
                q[sl["fixed_joint"]] = sl["fixed_value"]
                q[sl["vary_joint"]] = angle
                decoder, delta = self._run_single_q(q)
                block["q1"].append(delta["q1"])
                block["q2"].append(delta["q2"])
                block["lift"].append(delta["lift"])
                block["pitch"].append(delta["pitch"])
                block["yaw"].append(delta["yaw"])
                block["signed"].append(delta["signed"])
                done += 1
                self._print_progress_bar(done, total, prefix="  collect ")

            for key in ("q1", "q2", "lift", "pitch", "yaw"):
                block[key] = np.array(block[key], dtype=float)
            out[sl["name"]] = block
        return out

    def _plot_output_ring_samples(self, vis):
        s = vis["q1_fixed"]
        n_out = self.decoder.output_ring_size
        n_idx = np.arange(n_out)
        sample_idxs = np.linspace(0, len(s["vary_angles"]) - 1, 5, dtype=int)

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
            for ax, si in zip(axs, sample_idxs):
                ax.plot(n_idx, s["lift"][si], lw=1.8, label="Lift")
                ax.plot(n_idx, s["pitch"][si], lw=1.8, label="Pitch")
                ax.plot(n_idx, s["yaw"][si], lw=1.8, label="Yaw")
                ax.set_title(f"q2={s['vary_angles'][si]:.2f} rad")
                ax.set_xlabel("Neuron index")
                ax.set_ylabel("Spike-count delta")
                ax.grid(alpha=0.2)
            axs[0].legend()
            fig.suptitle("Output ring profiles (q1 fixed, q2 sweep)", y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, "output_ring_samples.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _plot_decoded_vs_true(self, vis):
        slice_specs = [("q1_fixed", "q2 (rad)"), ("q2_fixed", "q1 (rad)")]
        outputs = [("lift", 0), ("pitch", 1), ("yaw", 2)]

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            for row, (slice_name, xlabel) in enumerate(slice_specs):
                s = vis[slice_name]
                for col, (name, out_idx) in enumerate(outputs):
                    vary = s["vary_angles"]
                    decoded = []
                    true = []
                    for i, angle in enumerate(vary):
                        q = np.zeros(self.decoder.num_joints, dtype=float)
                        q[s["fixed_joint"]] = s["fixed_value"]
                        q[s["vary_joint"]] = angle
                        true_angle = self._lift_pitch_yaw_from_angles(q)[out_idx]
                        true.append(self.decoder.angle_to_ring_index(true_angle, self.decoder.output_ring_size))
                        decoded_angle = self.decoder._decode_angle_sawtooth_from_profile(s[name][i])
                        decoded.append(self.decoder.angle_to_ring_index(decoded_angle, self.decoder.output_ring_size))

                    decoded = np.asarray(decoded, dtype=float)
                    true = np.asarray(true, dtype=float)
                    valid_true = np.isfinite(true)
                    valid_dec = np.isfinite(decoded)
                    valid = valid_true & valid_dec
                    decoded_plot = self._circular_index_nearest(
                        decoded,
                        true,
                        self.decoder.output_ring_size,
                    )

                    ax = axs[row, col]
                    ax.plot(vary[valid], decoded_plot[valid], lw=2.0, marker="^", ms=3.0, label="Sawtooth")
                    ax.plot(vary[valid_true], true[valid_true], lw=1.8, ls="--", label="True")
                    ax.set_ylim(-5, self.decoder.output_ring_size + 5)
                    ax.set_title(f"{name} - {slice_name}")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Ring index")
                    ax.grid(alpha=0.2)
            axs[0, 0].legend()
            fig.suptitle("Sawtooth decoded index vs true target", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, "sawtooth_decoded_vs_true.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _plot_circular_index_error(self, vis):
        slice_specs = [("q1_fixed", "q2 (rad)"), ("q2_fixed", "q1 (rad)")]
        outputs = [("lift", 0), ("pitch", 1), ("yaw", 2)]
        half_n = self.decoder.output_ring_size / 2.0

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            for row, (slice_name, xlabel) in enumerate(slice_specs):
                s = vis[slice_name]
                for col, (name, out_idx) in enumerate(outputs):
                    vary = s["vary_angles"]
                    decoded = []
                    true = []
                    for i, angle in enumerate(vary):
                        q = np.zeros(self.decoder.num_joints, dtype=float)
                        q[s["fixed_joint"]] = s["fixed_value"]
                        q[s["vary_joint"]] = angle
                        true_angle = self._lift_pitch_yaw_from_angles(q)[out_idx]
                        true.append(self.decoder.angle_to_ring_index(true_angle, self.decoder.output_ring_size))
                        decoded_angle = self.decoder._decode_angle_sawtooth_from_profile(s[name][i])
                        decoded.append(self.decoder.angle_to_ring_index(decoded_angle, self.decoder.output_ring_size))

                    decoded = np.asarray(decoded, dtype=float)
                    true = np.asarray(true, dtype=float)
                    valid = np.isfinite(decoded) & np.isfinite(true)
                    err = self._circular_index_error(decoded, true, self.decoder.output_ring_size)

                    ax = axs[row, col]
                    ax.axhline(0, color="gray", lw=0.9, ls=":")
                    ax.plot(vary[valid], err[valid], lw=2.0, marker="^", ms=3.0)
                    ax.set_ylim(-half_n, half_n)
                    ax.set_title(f"{name} - {slice_name}")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Circular error (neurons)")
                    ax.grid(alpha=0.2)

            fig.suptitle("Sawtooth circular index error", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, "sawtooth_circular_index_error.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _plot_signed_term_traces(self, vis):
        s = vis["q1_fixed"]
        vary = s["vary_angles"]
        term_names = ["cos1", "sin1", "cos2", "sin2", "cos1cos2", "cos1sin2", "sin1cos2", "sin1sin2"]

        traces = {t: [] for t in term_names}
        for signed_counts in s["signed"]:
            for term in term_names:
                pos = np.asarray(signed_counts[f"{term}_pos"], dtype=float)
                neg = np.asarray(signed_counts[f"{term}_neg"], dtype=float)
                traces[term].append(float(np.sum(pos) - np.sum(neg)))

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(2, 4, figsize=(16, 8))
            for ax, term in zip(axs.ravel(), term_names):
                ax.plot(vary, np.array(traces[term], dtype=float), lw=1.8, marker="o", ms=3.0)
                ax.axhline(0, color="gray", lw=0.8, ls=":")
                ax.set_title(term)
                ax.set_xlabel("q2 (rad)")
                ax.set_ylabel("Signed activity")
                ax.grid(alpha=0.2)
            fig.suptitle("Signed-product term traces (q1 fixed, q2 sweep)", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, "signed_product_term_traces.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

    def run(self, n_vis: int = None):
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.decoder.populations:
            self.decoder.build()

        n_vis = self.n_vis_points if n_vis is None else int(n_vis)
        vis = self._collect_vis_data(n_vis=n_vis)
        self._plot_output_ring_samples(vis)
        self._plot_decoded_vs_true(vis)
        self._plot_circular_index_error(vis)
        self._plot_signed_term_traces(vis)

        print("Saved analysis plots:")
        print(f"  {os.path.join(self.output_dir, 'output_ring_samples.png')}")
        print(f"  {os.path.join(self.output_dir, 'sawtooth_decoded_vs_true.png')}")
        print(f"  {os.path.join(self.output_dir, 'sawtooth_circular_index_error.png')}")
        print(f"  {os.path.join(self.output_dir, 'signed_product_term_traces.png')}")
        return vis


if __name__ == "__main__":
    analysis = MultiRingDecodeAnalysis(reset_kernel=True)
    analysis.run(n_vis=20)
