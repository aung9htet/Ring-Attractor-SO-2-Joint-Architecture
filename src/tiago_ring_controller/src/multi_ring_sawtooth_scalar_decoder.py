#!/usr/bin/env python3

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nest
import numpy as np

from compositional_fourier_decoder import CompositionalFourierDecoderTrainer
from tiago_ring_controller.artifacts import (
    load_json_legacy,
    load_numpy_legacy,
    save_json_legacy,
    save_npz_legacy,
)
from tiago_ring_controller.config import load_json, resolve_legacy_read_path
from tiago_ring_controller.evaluation.metrics import scalar_readout_metrics
from tiago_ring_controller.math.circular import circular_angle_error
from tiago_ring_controller.training.analytic import build_scalar_ramp_weights


# constants
DEFAULT_PARAMS_FILE = "./config/model_params/multi_ring_params.json"
DEFAULT_OUTPUT_DIR = "./outputs/multi_ring_sawtooth_scalar_decoding"
ANALYSIS_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "analysis_templates",
    "multi_ring_sawtooth_scalar_analysis_notes_template.md",
)
JOINT_DEFS = (
    ("lift", "lift_counts", 0),
    ("pitch", "pitch_counts", 1),
    ("yaw", "yaw_counts", 2),
)


class MultiRingScalarReadoutInference(CompositionalFourierDecoderTrainer):
    """Add downstream scalar ramp readout neurons to the saved sawtooth decoder."""

    _ROW_LABELS = {
        "q1_fixed": "q1 fixed - q2 varies",
        "q2_fixed": "q2 fixed - q1 varies",
    }
    _X_LABELS = {
        "q1_fixed": "q2 (rad)",
        "q2_fixed": "q1 (rad)",
    }
    _SLICE_COLORS = {
        "q1_fixed": "#2980b9",
        "q2_fixed": "#e67e22",
    }
    _JOINT_COLORS = {
        "lift": "#2980b9",
        "pitch": "#e67e22",
        "yaw": "#8e44ad",
    }

    # class init + weight loading

    def __init__(self, params_file: str = DEFAULT_PARAMS_FILE):
        super().__init__(params_file=params_file)
        params = load_json(params_file)
        self.scalar_ramp_weight_scale = float(params.get("scalar_ramp_weight_scale", 100.0))
        self.scalar_dc_baseline = float(params.get("scalar_dc_baseline", 200.0))

    def _load_sawtooth_weights(self) -> str:
        """Load pre-trained multi-ring sawtooth weights and matching metadata."""
        n = self.population_size
        j = self.num_joints
        weights_path = f"./config/ring_decoding_weights/N_{n}_J_{j}_multi_ring_sawtooth_weights.npz"
        metadata_path = f"./config/ring_decoding_weights/N_{n}_J_{j}_multi_ring_sawtooth_metadata.json"

        weights_path = resolve_legacy_read_path(weights_path)
        metadata_path = resolve_legacy_read_path(metadata_path)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"\nPre-trained sawtooth weights not found at:\n  {weights_path}\n\n"
                "Please run the multi-ring sawtooth trainer first:\n"
                "  python3 compositional_fourier_decoder.py\n"
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
        self._output_weight_matrix_is_target_by_source = bool(
            data["output_weight_matrix_is_target_by_source"]
        )
        self._signed_product_active = True

        if os.path.exists(metadata_path):
            metadata = load_json_legacy(metadata_path)
            calibrated = metadata.get("calibrated_signed_product_params", {})
            if "signed_product_dc_baseline" in calibrated:
                self.signed_product_dc_baseline = float(calibrated["signed_product_dc_baseline"])
            if "signed_product_input_weight" in calibrated:
                self.signed_product_input_weight = float(calibrated["signed_product_input_weight"])
            self._calibrated_signed_product_params = calibrated
            self._active_signed_product_terms = metadata.get("active_signed_product_terms", [])
            print(
                f"  Calibrated grid params: dc={self.signed_product_dc_baseline:.0f},"
                f" w_in={self.signed_product_input_weight:.0f}"
            )

        print(f"Loaded sawtooth weights from: {weights_path}")
        return weights_path

    # scalar readout wiring

    def _decode_sawtooth_index(self, profile: np.ndarray) -> float:
        saw_angle = self._decode_angle_sawtooth_from_profile(profile)
        return float(self._angle_to_ring_index(saw_angle, self.output_ring_size))

    def _scalar_ramp_weights(self) -> np.ndarray:
        return build_scalar_ramp_weights(
            self.output_ring_size, self.scalar_ramp_weight_scale
        )

    def _scalar_weight_matrix(self) -> list:
        ramp_weights = self._scalar_ramp_weights()
        # NEST all_to_all expects either [n_target, n_source] or [n_source, n_target]
        # depending on the orientation verified by the base decoder.
        if self._output_weight_matrix_is_target_by_source:
            return ramp_weights.reshape(1, -1).tolist()
        return ramp_weights.reshape(-1, 1).tolist()

    def _simulate_scalar_readout(self, q: np.ndarray) -> dict:
        """Run one inference simulation with downstream scalar ramp readouts."""
        model = self._make_signed_model()
        theta = self._preferred_angles(self.population_size)
        model.build_signed_product_grid_layer(theta)
        model.build_output_rings_from_signed_product_grid(
            self._W_signed_lift,
            self._W_signed_pitch,
            self._W_signed_yaw,
        )

        scalar_weight_matrix = self._scalar_weight_matrix()
        scalar_readouts = {}
        for name in ("lift", "pitch", "yaw"):
            scalar_node = nest.Create("iaf_psc_alpha", 1)
            scalar_recorder = nest.Create("spike_recorder", 1)
            if self.scalar_dc_baseline > 0.0:
                dc = nest.Create("dc_generator", params={"amplitude": self.scalar_dc_baseline})
                nest.Connect(dc, scalar_node)
            nest.Connect(scalar_node, scalar_recorder)
            ring_nodes = model._output_rings_signed[name]["nodes"]
            nest.Connect(
                ring_nodes,
                scalar_node,
                conn_spec={"rule": "all_to_all"},
                syn_spec={"weight": scalar_weight_matrix},
            )
            scalar_readouts[name] = {"node": scalar_node, "rec": scalar_recorder}

        def get_all_counts():
            scalar_counts = {
                name: float(len(scalar_readouts[name]["rec"].get("events")["times"]))
                for name in ("lift", "pitch", "yaw")
            }
            return {
                "grid": model._get_signed_product_grid_counts(),
                "output": {
                    "lift": model._get_signed_product_output_ring_counts("lift"),
                    "pitch": model._get_signed_product_output_ring_counts("pitch"),
                    "yaw": model._get_signed_product_output_ring_counts("yaw"),
                },
                "scalar": scalar_counts,
                "q_rings": [model._get_joint_ring_spike_counts(idx) for idx in range(model.num_joints)],
            }

        net_delta = self._measure_net_stimulus_delta(model, q, get_all_counts)
        return {
            "q_ring_counts": net_delta["q_rings"],
            "signed_feature_counts": self._signed_counts_to_vector(net_delta["grid"]),
            "lift_counts": net_delta["output"]["lift"],
            "pitch_counts": net_delta["output"]["pitch"],
            "yaw_counts": net_delta["output"]["yaw"],
            "lift_scalar_count": float(net_delta["scalar"]["lift"]),
            "pitch_scalar_count": float(net_delta["scalar"]["pitch"]),
            "yaw_scalar_count": float(net_delta["scalar"]["yaw"]),
        }

    def _run_smoke_check(self):
        if self.num_joints != 2:
            raise RuntimeError(
                f"Smoke check expects 2 input joints but found num_joints={self.num_joints}."
            )

        q = np.array([0.0, 0.0])
        result = self._simulate_scalar_readout(q)

        contrasts = {}
        scalar_deltas = {}
        for joint_name in ("lift", "pitch", "yaw"):
            profile = np.array(result[f"{joint_name}_counts"], dtype=float)
            contrast = float(np.max(profile) - np.min(profile))
            scalar_delta = float(result[f"{joint_name}_scalar_count"])
            contrasts[joint_name] = contrast
            scalar_deltas[joint_name] = scalar_delta

        print("\n[One-sample smoke check @ q=[0, 0]]")
        print(
            "  Output-ring contrasts: "
            + ", ".join(f"{joint}={contrasts[joint]:.3f}" for joint in ("lift", "pitch", "yaw"))
        )
        print(
            "  Scalar deltas: "
            + ", ".join(f"{joint}={scalar_deltas[joint]:.3f}" for joint in ("lift", "pitch", "yaw"))
        )

        all_scalar_zero = all(np.isclose(scalar_deltas[joint], 0.0) for joint in ("lift", "pitch", "yaw"))
        all_contrast_zero = all(np.isclose(contrasts[joint], 0.0) for joint in ("lift", "pitch", "yaw"))
        if all_scalar_zero or all_contrast_zero:
            raise RuntimeError(
                "Smoke check failed: scalar readout or output-ring activity is flat. "
                "Check downstream scalar wiring and saved sawtooth output-ring weights."
            )

    # simulation collection

    def _collect_scalar_vis_data(self, n_vis: int) -> dict:
        """Collect output-ring and scalar traces for the two validation slices."""
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
        print(f"\nRunning {total} scalar-readout vis simulations ({n_vis} per slice) ...")

        for fixed_joint, fixed_value, vary_joint, slice_key in slices:
            ring_lists = {name: [] for name in ("lift", "pitch", "yaw")}
            scalar_lists = {name: [] for name in ("lift", "pitch", "yaw")}
            q_ring_lists = [[] for _ in range(self.num_joints)]
            feature_vectors = []

            for angle in vary_angles:
                q = np.zeros(self.num_joints)
                q[fixed_joint] = fixed_value
                q[vary_joint] = angle
                result = self._simulate_scalar_readout(q)

                for joint_name in ("lift", "pitch", "yaw"):
                    ring_lists[joint_name].append(result[f"{joint_name}_counts"])
                    scalar_lists[joint_name].append(result[f"{joint_name}_scalar_count"])
                for joint_idx in range(self.num_joints):
                    q_ring_lists[joint_idx].append(result["q_ring_counts"][joint_idx])
                feature_vectors.append(result["signed_feature_counts"])

                done += 1
                filled = int(40 * done / total)
                print(f"\r  [{'#' * filled}{'.' * (40 - filled)}] {done}/{total}", end="", flush=True)

            vis_data[slice_key] = {
                "vary_angles": vary_angles.copy(),
                "vary_jt": vary_joint,
                "fixed_jt": fixed_joint,
                "fixed_val": fixed_value,
                "lift_counts": np.array(ring_lists["lift"]),
                "pitch_counts": np.array(ring_lists["pitch"]),
                "yaw_counts": np.array(ring_lists["yaw"]),
                "q_ring_counts": [np.array(values) for values in q_ring_lists],
                "extra": np.array(feature_vectors),
                "lift_scalar": np.array(scalar_lists["lift"], dtype=float),
                "pitch_scalar": np.array(scalar_lists["pitch"], dtype=float),
                "yaw_scalar": np.array(scalar_lists["yaw"], dtype=float),
            }

        print()
        return vis_data

    # metrics

    def _compute_scalar_metrics(self, vis: dict) -> dict:
        """Compute sawtooth and scalar readout metrics per slice and output."""
        output_ring_size = self.output_ring_size
        result = {}

        for slice_key, slice_data in vis.items():
            vary_angles = slice_data["vary_angles"]
            fixed_joint = int(slice_data["fixed_jt"])
            fixed_value = float(slice_data["fixed_val"])
            vary_joint = int(slice_data["vary_jt"])
            joint_metrics = {}

            for joint_name, count_key, angle_sel in JOINT_DEFS:
                scalar_counts = np.array(slice_data[f"{joint_name}_scalar"], dtype=float)
                saw_indices = []
                true_indices = []
                sawtooth_errors_deg = []

                for idx, angle in enumerate(vary_angles):
                    q = np.zeros(self.num_joints)
                    q[fixed_joint] = fixed_value
                    q[vary_joint] = angle
                    true_angle = self._lift_pitch_yaw_from_angles(q)[angle_sel]
                    true_indices.append(self._angle_to_ring_index(true_angle, output_ring_size))

                    profile = slice_data[count_key][idx]
                    if self._profile_has_signal(profile):
                        saw_idx = self._decode_sawtooth_index(profile)
                        saw_indices.append(saw_idx)
                        saw_angle = 2.0 * np.pi * (saw_idx / output_ring_size)
                        circular_error = abs(circular_angle_error(saw_angle, true_angle))
                        sawtooth_errors_deg.append(float(np.degrees(circular_error)))
                    else:
                        saw_indices.append(float("nan"))

                saw_indices = np.array(saw_indices, dtype=float)
                true_indices = np.array(true_indices, dtype=float)
                def warn_constant_signal(scalar_min, scalar_max):
                    print(
                        f"WARNING: Scalar readout is near-constant for {slice_key}/{joint_name} "
                        f"(min={scalar_min:.3f}, max={scalar_max:.3f}); "
                        "correlations and scalar RMSE are set to n/a."
                    )

                joint_metrics[joint_name] = scalar_readout_metrics(
                    saw_indices,
                    true_indices,
                    scalar_counts,
                    output_ring_size,
                    sawtooth_errors_deg=sawtooth_errors_deg,
                    vary_angles=vary_angles,
                    constant_signal_callback=warn_constant_signal,
                    legacy_vary_angles_tolist=True,
                )

            result[slice_key] = joint_metrics

        return result

    def _format_metric(self, value: float) -> str:
        return f"{value:.3f}" if np.isfinite(value) else "n/a"

    def _print_scalar_metrics(self, metrics: dict):
        print("\n[Scalar ramp readout metrics]")
        for slice_key, joint_metrics in metrics.items():
            print(f"  Slice: {slice_key}")
            for joint_name in ("lift", "pitch", "yaw"):
                metric = joint_metrics[joint_name]
                print(
                    f"    {joint_name}: saw_MAE={metric['sawtooth_mae_deg']:.2f}°"
                    f"  corr_vs_saw={self._format_metric(metric['scalar_vs_sawtooth_correlation'])}"
                    f"  corr_vs_true={self._format_metric(metric['scalar_vs_true_correlation'])}"
                    f"  scalar_RMSE_norm={self._format_metric(metric['scalar_rmse_normalized'])}"
                    f"  sig_cov={metric['signal_coverage_pct']:.1f}%"
                )

    # plots

    def _plot_scalar_readout_vs_sawtooth(self, metrics: dict, save_dir: str):
        slice_keys = list(metrics.keys())
        joint_names = ["lift", "pitch", "yaw"]

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(len(slice_keys), len(joint_names), figsize=(14, 8))
            if len(slice_keys) == 1:
                axs = axs[np.newaxis, :]
            for row, slice_key in enumerate(slice_keys):
                for col, joint_name in enumerate(joint_names):
                    metric = metrics[slice_key][joint_name]
                    ax = axs[row, col]
                    saw_norm = np.array(metric["saw_norm"])
                    scalar_norm = np.array(metric["scalar_norm"])
                    valid = ~np.isnan(saw_norm)
                    ax.scatter(
                        saw_norm[valid],
                        scalar_norm[valid],
                        color=self._SLICE_COLORS.get(slice_key, "#555"),
                        s=30,
                        alpha=0.8,
                    )
                    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", alpha=0.5)
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_title(
                        f"{joint_name.capitalize()} - {self._ROW_LABELS.get(slice_key, slice_key)}\n"
                        f"r={self._format_metric(metric['scalar_vs_sawtooth_correlation'])}"
                    )
                    ax.set_xlabel("Sawtooth decoded index (norm)")
                    ax.set_ylabel("Scalar readout (norm)")
                    ax.grid(alpha=0.2)
            fig.suptitle("Downstream scalar ramp readout vs. sawtooth decoded index", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "scalar_readout_vs_sawtooth_index.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _plot_scalar_readout_vs_true(self, metrics: dict, save_dir: str):
        slice_keys = list(metrics.keys())
        joint_names = ["lift", "pitch", "yaw"]

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(len(slice_keys), len(joint_names), figsize=(14, 8))
            if len(slice_keys) == 1:
                axs = axs[np.newaxis, :]
            for row, slice_key in enumerate(slice_keys):
                for col, joint_name in enumerate(joint_names):
                    metric = metrics[slice_key][joint_name]
                    ax = axs[row, col]
                    true_norm = np.array(metric["true_norm"])
                    scalar_norm = np.array(metric["scalar_norm"])
                    valid = ~np.isnan(np.array(metric["saw_norm"]))
                    ax.scatter(
                        true_norm[valid],
                        scalar_norm[valid],
                        color=self._SLICE_COLORS.get(slice_key, "#555"),
                        s=30,
                        alpha=0.8,
                    )
                    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", alpha=0.5)
                    ax.set_xlim(-0.05, 1.05)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_title(
                        f"{joint_name.capitalize()} - {self._ROW_LABELS.get(slice_key, slice_key)}\n"
                        f"r={self._format_metric(metric['scalar_vs_true_correlation'])}"
                    )
                    ax.set_xlabel("True target index (norm)")
                    ax.set_ylabel("Scalar readout (norm)")
                    ax.grid(alpha=0.2)
            fig.suptitle("Downstream scalar ramp readout vs. true target index", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "scalar_readout_vs_true_index.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _plot_sawtooth_vs_true_index(self, metrics: dict, save_dir: str):
        slice_keys = list(metrics.keys())
        joint_names = ["lift", "pitch", "yaw"]
        output_ring_size = self.output_ring_size
        colors = {"true": "#c0392b", "saw": "#27ae60"}

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(len(slice_keys), len(joint_names), figsize=(14, 8))
            if len(slice_keys) == 1:
                axs = axs[np.newaxis, :]
            for row, slice_key in enumerate(slice_keys):
                for col, joint_name in enumerate(joint_names):
                    metric = metrics[slice_key][joint_name]
                    ax = axs[row, col]
                    vary_angles = np.array(metric["vary_angles"])
                    true_idx = np.array(metric["true_norm"]) * output_ring_size
                    saw_idx = np.array(metric["saw_norm"]) * output_ring_size
                    ax.plot(vary_angles, true_idx, color=colors["true"], lw=2.0, ls="--", label="True")
                    ax.plot(vary_angles, saw_idx, color=colors["saw"], lw=2.4, marker="^", ms=3.0, label="Sawtooth")
                    ax.set_ylim(0, output_ring_size)
                    ax.set_title(f"{joint_name.capitalize()} - {self._ROW_LABELS.get(slice_key, slice_key)}")
                    ax.set_xlabel(self._X_LABELS.get(slice_key, "angle (rad)"))
                    ax.set_ylabel("Ring neuron index")
                    ax.grid(alpha=0.2)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=8)
            fig.suptitle(
                "Sawtooth decoded index vs. true target index (100-neuron circular output rings)",
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "sawtooth_vs_true_index.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _plot_scalar_readout_traces(self, vis: dict, save_dir: str):
        slice_keys = list(vis.keys())

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(len(slice_keys), 1, figsize=(10, 4 * len(slice_keys)))
            if len(slice_keys) == 1:
                axs = [axs]
            for row, slice_key in enumerate(slice_keys):
                slice_data = vis[slice_key]
                ax = axs[row]
                vary_angles = slice_data["vary_angles"]
                for joint_name in ("lift", "pitch", "yaw"):
                    ax.plot(
                        vary_angles,
                        slice_data[f"{joint_name}_scalar"],
                        color=self._JOINT_COLORS[joint_name],
                        lw=1.8,
                        marker="o",
                        ms=3.0,
                        label=f"{joint_name.capitalize()} scalar",
                    )
                ax.set_title(
                    f"Downstream scalar ramp readout spike-count delta - {self._ROW_LABELS.get(slice_key, slice_key)}"
                )
                ax.set_xlabel(self._X_LABELS.get(slice_key, "angle (rad)"))
                ax.set_ylabel("Scalar spike-count delta")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=9)
            fig.suptitle("Downstream scalar ramp readout neuron output traces", y=1.01)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "scalar_readout_traces.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _plot_scalar_error_summary(self, metrics: dict, save_dir: str):
        slice_keys = list(metrics.keys())
        joint_names = ["lift", "pitch", "yaw"]

        with plt.rc_context(self._RC):
            fig, axs = plt.subplots(len(slice_keys), len(joint_names), figsize=(14, 8))
            if len(slice_keys) == 1:
                axs = axs[np.newaxis, :]
            for row, slice_key in enumerate(slice_keys):
                for col, joint_name in enumerate(joint_names):
                    metric = metrics[slice_key][joint_name]
                    ax = axs[row, col]
                    true_norm = np.array(metric["true_norm"])
                    scalar_norm = np.array(metric["scalar_norm"])
                    valid = ~np.isnan(np.array(metric["saw_norm"]))
                    error = scalar_norm - true_norm
                    ax.axhline(0, color="gray", lw=0.8, ls=":")
                    ax.scatter(
                        true_norm[valid],
                        error[valid],
                        color=self._SLICE_COLORS.get(slice_key, "#555"),
                        s=30,
                        alpha=0.8,
                    )
                    ax.set_title(
                        f"{joint_name.capitalize()} - {self._ROW_LABELS.get(slice_key, slice_key)}\n"
                        f"RMSE={self._format_metric(metric['scalar_rmse_normalized'])}"
                    )
                    ax.set_xlabel("True target index (norm)")
                    ax.set_ylabel("Scalar norm - true norm")
                    ax.set_ylim(-1.1, 1.1)
                    ax.grid(alpha=0.2)
            fig.suptitle(
                "Scalar ramp readout signed error vs. true target index\n"
                "(wrap-point errors are expected from the linear ramp)",
                y=1.01,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "scalar_error_summary.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

            # saving

    def _save_outputs(self, metrics: dict, sawtooth_mae: dict, source_weights_path: str, save_dir: str):
        weights_dir = "./config/ring_decoding_weights"
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        n = self.population_size
        j = self.num_joints
        scalar_weights_path = os.path.join(weights_dir, f"N_{n}_J_{j}_multi_ring_sawtooth_scalar_weights.npz")
        scalar_metadata_path = os.path.join(weights_dir, f"N_{n}_J_{j}_multi_ring_sawtooth_scalar_metadata.json")

        save_npz_legacy(
            scalar_weights_path,
            scalar_ramp_weights=self._scalar_ramp_weights(),
            scalar_ramp_weight_scale=np.array(self.scalar_ramp_weight_scale),
            scalar_dc_baseline=np.array(self.scalar_dc_baseline),
            output_ring_size=np.array(self.output_ring_size),
            population_size=np.array(n),
            num_joints=np.array(j),
            joint_axes=np.array(self.joint_axes),
            source_sawtooth_weights_path=np.array(source_weights_path),
        )

        array_keys = {"vary_angles", "true_norm", "saw_norm", "scalar_norm", "scalar_counts_raw"}
        scalar_metrics_summary = {
            slice_key: {
                joint_name: {key: value for key, value in joint_metrics.items() if key not in array_keys}
                for joint_name, joint_metrics in slice_metrics.items()
            }
            for slice_key, slice_metrics in metrics.items()
        }

        metadata = {
            "decoder": "sawtooth_plus_scalar_ramp_readout",
            "note": (
                "The 100-neuron output rings remain the circular 0-to-2pi representation. "
                "The scalar readout neurons are downstream ramp readouts for magnitude-style comparison."
            ),
            "scalar_ramp_weight_scale": self.scalar_ramp_weight_scale,
            "scalar_dc_baseline": self.scalar_dc_baseline,
            "output_ring_size": self.output_ring_size,
            "population_size": n,
            "num_joints": j,
            "joint_axes": self.joint_axes,
            "source_sawtooth_weights_path": source_weights_path,
            "sawtooth_mae_per_slice": sawtooth_mae,
            "scalar_readout_metrics": scalar_metrics_summary,
        }

        save_json_legacy(scalar_metadata_path, metadata)

        print(f"\nScalar weights  -> {scalar_weights_path}")
        print(f"Scalar metadata -> {scalar_metadata_path}")
        print(f"Plots           -> {save_dir}/")

    def _write_analysis_notes(self, save_dir: str):
        if not os.path.exists(ANALYSIS_TEMPLATE_PATH):
            print(
                "WARNING: analysis notes template not found; skipping notes copy:\n"
                f"  {ANALYSIS_TEMPLATE_PATH}"
            )
            return

        notes_path = os.path.join(save_dir, "analysis_notes.md")
        with open(ANALYSIS_TEMPLATE_PATH, "r", encoding="utf-8") as fh:
            notes_text = fh.read()
        with open(notes_path, "w", encoding="utf-8") as fh:
            fh.write(notes_text)
        print(f"Analysis notes  -> {notes_path}")

    # run entry point

    def run(self, save_dir: str = DEFAULT_OUTPUT_DIR):
        """Load saved sawtooth weights, run scalar inference, and save outputs."""
        os.makedirs(save_dir, exist_ok=True)

        print("=" * 60)
        print("Multi-ring sawtooth + scalar ramp readout decoder")
        print("=" * 60)

        print("\nLoading pre-trained multi-ring sawtooth weights ...")
        source_weights_path = self._load_sawtooth_weights()

        print("\nVerifying NEST all_to_all weight orientation ...")
        self._output_weight_matrix_is_target_by_source = self._verify_all_to_all_weight_orientation()

        self._run_smoke_check()

        self._train_configs = self._build_joint_configs()

        print("\nScalar ramp readout configuration:")
        print(f"  scalar_ramp_weight_scale = {self.scalar_ramp_weight_scale}")
        print(f"  scalar_dc_baseline       = {self.scalar_dc_baseline}")
        print(f"  output_ring_size         = {self.output_ring_size}")
        print(f"  n_vis_points             = {self.n_vis_points}")

        vis = self._collect_scalar_vis_data(n_vis=self.n_vis_points)

        print("\nNEST spike-count diagnostics (output rings)")
        self._print_output_ring_stats(vis)

        metrics = self._compute_scalar_metrics(vis)
        self._print_scalar_metrics(metrics)

        sawtooth_mae = self._sawtooth_mae_per_slice(vis)
        print("\n[Sawtooth MAE (circular) - verification that ring decoding is unchanged]")
        for slice_key in ("q1_fixed", "q2_fixed", "overall"):
            metric = sawtooth_mae[slice_key]
            saw = metric["sawtooth_mae_deg"]
            print(
                f"  {slice_key}: Lift={saw['lift']:.2f}°"
                f"  Pitch={saw['pitch']:.2f}°"
                f"  Yaw={saw['yaw']:.2f}°"
                f"  sig_cov={metric['signal_coverage_pct']:.1f}%"
            )

        print("\nGenerating plots ...")
        self._plot_scalar_readout_vs_sawtooth(metrics, save_dir)
        self._plot_scalar_readout_vs_true(metrics, save_dir)
        self._plot_sawtooth_vs_true_index(metrics, save_dir)
        self._plot_scalar_readout_traces(vis, save_dir)
        self._plot_scalar_error_summary(metrics, save_dir)

        self._save_outputs(metrics, sawtooth_mae, source_weights_path, save_dir)
        self._write_analysis_notes(save_dir)


if __name__ == "__main__":
    decoder = MultiRingScalarReadoutInference(params_file=DEFAULT_PARAMS_FILE)
    decoder.run(save_dir=DEFAULT_OUTPUT_DIR)
