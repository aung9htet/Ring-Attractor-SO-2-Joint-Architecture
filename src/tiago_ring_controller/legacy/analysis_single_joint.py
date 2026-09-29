#!/usr/bin/env python3

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nest
import numpy as np
import rospy

from tiago_controller import TiagoPublisher, TiagoSubscriber
from single_ring import SingleRingModel
from tiago_ring_controller.artifacts import atomic_json_dump_legacy, load_json_legacy
from tiago_ring_controller.control.controller import (
    DecoderParameters,
    DriveControlCore,
)
from tiago_ring_controller.control.profiles import ANALYSIS_PROFILE
from tiago_ring_controller.config import module_config_path
from tiago_ring_controller.math.control import (
    apply_delay,
    decode_velocity,
    decoder_features,
    exponential_filter,
    legacy_joint_to_ring_index,
)
from tiago_ring_controller.evaluation.metrics import analysis_region_statistics
from tiago_ring_controller.evaluation.serialization import json_safe, slug


class TiagoDecoderAnalysis:
    def __init__(self, target_joint, use_manual_limits=False):
        self.target_joint = target_joint
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.publisher = TiagoPublisher()
        self.subscriber = TiagoSubscriber()

        self.publisher.wait_for_joint_state(timeout=5.0)

        self.ring_model = SingleRingModel()
        self.control_profile = ANALYSIS_PROFILE

        # NEST / robot control step
        self.time_step = self.control_profile.time_step_ms

        # Loaded decoder parameters (fixed during analysis)
        self.decoder_gain_positive = 1e-4
        self.decoder_gain_negative = -1e-4
        self.decoder_tau = 0.3
        self.decoder_delay_steps = 0

        # Trial control parameters
        self.lookahead = self.control_profile.lookahead
        self.drive_threshold = self.control_profile.drive_threshold
        self.n_settle = self.control_profile.n_settle
        self.max_steps = self.control_profile.max_steps

        self.left_count = 0
        self.right_count = 0
        self.state_count = 0

        self.r1_delta_spike_counts = None
        self.goal_inject_idx = None
        self.ring_state_inject_idx = None

        self.config_path = module_config_path(
            __file__, "calibration", "velocity_calibration.json"
        )
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        limits_loaded = self.load_limits_from_config()

        if not limits_loaded or use_manual_limits:
            self.joint_min, self.joint_max = self.find_joint_limits_manually()
        else:
            rospy.loginfo(
                f"Joint {self.target_joint} limits loaded: "
                f"[{self.joint_min:.4f}, {self.joint_max:.4f}] rad"
            )

        self.load_decoder_from_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_limits_from_config(self):
        try:
            config = load_json_legacy(self.config_path)

            joint_key = str(self.target_joint)
            if (
                joint_key in config
                and "joint_min" in config[joint_key]
                and "joint_max" in config[joint_key]
            ):
                self.joint_min = float(config[joint_key]["joint_min"])
                self.joint_max = float(config[joint_key]["joint_max"])
                return True
        except Exception:
            pass

        return False

    def load_decoder_from_config(self):
        """Load decoder parameters from velocity_calibration.json."""
        try:
            config = load_json_legacy(self.config_path)

            joint_key = str(self.target_joint)

            if joint_key not in config:
                rospy.loginfo(
                    f"Joint {self.target_joint} has no decoder config. Using defaults."
                )
                return

            cfg = config[joint_key]

            required_decoder_keys = [
                "decoder_gain_positive",
                "decoder_gain_negative",
                "decoder_tau",
                "decoder_delay_steps",
            ]

            if all(k in cfg for k in required_decoder_keys):
                self.decoder_gain_positive = float(cfg["decoder_gain_positive"])
                self.decoder_gain_negative = float(cfg["decoder_gain_negative"])
                self.decoder_tau = float(cfg["decoder_tau"])
                self.decoder_delay_steps = int(cfg["decoder_delay_steps"])

                rospy.loginfo(
                    f"[DECODER LOADED] joint={self.target_joint} "
                    f"k_pos={self.decoder_gain_positive:.6e}, "
                    f"k_neg={self.decoder_gain_negative:.6e}, "
                    f"tau={self.decoder_tau:.3f}s, "
                    f"delay={self.decoder_delay_steps} steps"
                )
                return

            # Legacy fallback
            if "raw_drive_velocity_gain" in cfg:
                gain = float(cfg["raw_drive_velocity_gain"])
                self.decoder_gain_positive = gain
                self.decoder_gain_negative = -gain
                self.decoder_tau = 0.3
                self.decoder_delay_steps = 0

                rospy.logwarn(
                    "[DECODER LOADED] Using legacy raw_drive_velocity_gain. "
                    f"k_pos={gain:.6e}, k_neg={-gain:.6e}, tau=0.3, delay=0"
                )
                return

        except Exception as e:
            rospy.logwarn(f"[DECODER LOAD] Could not load decoder config: {e}")

        rospy.loginfo(
            f"[DECODER DEFAULT] joint={self.target_joint} "
            f"k_pos={self.decoder_gain_positive:.6e}, "
            f"k_neg={self.decoder_gain_negative:.6e}, "
            f"tau={self.decoder_tau:.3f}s, "
            f"delay={self.decoder_delay_steps} steps"
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _atomic_json_dump(self, data, path):
        atomic_json_dump_legacy(path, data)

    def _json_safe(self, obj):
        return json_safe(obj, recursive_converter=self._json_safe)

    def _slug(self, text):
        return slug(text)

    def _current_decoder_params(self):
        return {
            "decoder_gain_positive": float(self.decoder_gain_positive),
            "decoder_gain_negative": float(self.decoder_gain_negative),
            "decoder_tau": float(self.decoder_tau),
            "decoder_delay_steps": int(self.decoder_delay_steps),
        }

    def _analysis_settings(self):
        return {
            "session_id": self.session_id,
            "target_joint": int(self.target_joint),
            "joint_min_rad": float(self.joint_min),
            "joint_max_rad": float(self.joint_max),
            "time_step_ms": float(self.time_step),
            "lookahead": int(self.lookahead),
            "drive_threshold": float(self.drive_threshold),
            "n_settle": int(self.n_settle),
            "max_steps": int(self.max_steps),
            "decoder": self._current_decoder_params(),
        }

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------

    def _plot_dir(self, category):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results_plots",
            f"joint_{self.target_joint}",
            category,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _analysis_output_dir(self, category):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "experiment_results",
                "analysis",
                f"joint_{self.target_joint}",
                category,
            )
        )
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Save analysis data
    # ------------------------------------------------------------------

    def save_trial_data(self, trial, batch_idx, iteration_idx):
        output_dir = self._analysis_output_dir("trial_data")
        filename = (
            f"{self.session_id}_joint{self.target_joint}_"
            f"batch{batch_idx}_iter{iteration_idx}.json"
        )
        path = os.path.join(output_dir, filename)
        self._atomic_json_dump(self._json_safe(trial), path)
        rospy.loginfo(f"[SAVE] Trial data -> {path}")
        return path

    def save_batch_summary(self, batch_summary):
        output_dir = self._analysis_output_dir("batch_summaries")
        filename = (
            f"{self.session_id}_joint{self.target_joint}_"
            f"batch{batch_summary['batch_idx']}_summary.json"
        )
        path = os.path.join(output_dir, filename)
        self._atomic_json_dump(self._json_safe(batch_summary), path)
        rospy.loginfo(f"[SAVE] Batch summary -> {path}")
        return path

    def save_overall_summary(self, overall_summary):
        output_dir = self._analysis_output_dir("overall_summary")
        filename = f"{self.session_id}_joint{self.target_joint}_overall_summary.json"
        path = os.path.join(output_dir, filename)
        self._atomic_json_dump(self._json_safe(overall_summary), path)
        rospy.loginfo(f"[SAVE] Overall summary -> {path}")
        return path

    # ------------------------------------------------------------------
    # Recorder extraction helpers
    # ------------------------------------------------------------------

    def _maybe_get_attr_by_path(self, root_obj, path):
        current = root_obj
        for part in path.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current

    def _first_existing_path(self, root_obj, candidate_paths):
        for path in candidate_paths:
            value = self._maybe_get_attr_by_path(root_obj, path)
            if value is not None:
                return value, path
        return None, None

    def _normalize_devices(self, devices):
        if devices is None:
            return []

        if isinstance(devices, (list, tuple)):
            return list(devices)

        try:
            return list(devices)
        except Exception:
            return [devices]

    def _get_nest_events(self, device):
        try:
            return nest.GetStatus(device, "events")[0]
        except Exception:
            try:
                return nest.GetStatus([device], "events")[0]
            except Exception:
                return None

    def _collect_spike_recorder_group(self, recorders, source_path=None):
        recorder_list = self._normalize_devices(recorders)

        group = {
            "source_path": source_path,
            "n_recorders": len(recorder_list),
            "events": [],
        }

        for idx, rec in enumerate(recorder_list):
            events = self._get_nest_events(rec)
            if events is None:
                group["events"].append(
                    {
                        "recorder_index": idx,
                        "n_events": 0,
                        "times_ms": [],
                        "senders": [],
                    }
                )
                continue

            times = np.asarray(events.get("times", []), dtype=float)
            senders = np.asarray(events.get("senders", []), dtype=int)

            group["events"].append(
                {
                    "recorder_index": idx,
                    "n_events": int(len(times)),
                    "times_ms": times.tolist(),
                    "senders": senders.tolist(),
                }
            )

        return group

    def _collect_generic_device_group(self, devices, source_path=None):
        device_list = self._normalize_devices(devices)

        group = {
            "source_path": source_path,
            "n_devices": len(device_list),
            "events": [],
        }

        for idx, dev in enumerate(device_list):
            events = self._get_nest_events(dev)
            group["events"].append(
                {
                    "device_index": idx,
                    "events": self._json_safe(events if events is not None else {}),
                }
            )

        return group

    def collect_model_recordings(self):
        """Collect as much raw recorder data as possible.

        This includes:
        - ring attractor raster-related spike recorder data
        - gain modulation raster data
        - homeostasis-related recorder data (best effort / generic)
        """

        data = {
            "ring_raster": {},
            "gain_modulation": {},
            "homeostasis": {},
            "metadata": {
                "goal_inject_idx": self.goal_inject_idx,
                "ring_state_inject_idx": self.ring_state_inject_idx,
                "population_size": getattr(self.ring_model, "population_size", None),
            },
        }

        # Ring raster data
        ring_candidates = {
            "r1_ring_spikes": "r1.ring_attractor.ring_spike_recorders",
            "r2_ring_spikes": "r2.ring_attractor.ring_spike_recorders",
        }

        for name, path in ring_candidates.items():
            value = self._maybe_get_attr_by_path(self.ring_model, path)
            if value is not None:
                data["ring_raster"][name] = self._collect_spike_recorder_group(
                    value,
                    source_path=path,
                )

        # Gain modulation raster data
        gain_candidates = {
            "left_gain_spikes": "gain_modulation.left_gain_spike_recorders",
            "right_gain_spikes": "gain_modulation.right_gain_spike_recorders",
        }

        for name, path in gain_candidates.items():
            value = self._maybe_get_attr_by_path(self.ring_model, path)
            if value is not None:
                data["gain_modulation"][name] = self._collect_spike_recorder_group(
                    value,
                    source_path=path,
                )

        # Homeostasis / related adaptive variables
        # These are best-effort candidate paths. Extend if your model exposes
        # other recorder names.
        homeostasis_candidates = {
            "r1_homeostasis_multimeters": [
                "r1.ring_attractor.homeostasis_multimeters",
                "r1.homeostasis_multimeters",
                "r1.ring_attractor.multimeters",
            ],
            "r2_homeostasis_multimeters": [
                "r2.ring_attractor.homeostasis_multimeters",
                "r2.homeostasis_multimeters",
                "r2.ring_attractor.multimeters",
            ],
            "gain_homeostasis_multimeters": [
                "gain_modulation.homeostasis_multimeters",
                "gain_modulation.multimeters",
            ],
            "r1_homeostasis_spike_recorders": [
                "r1.ring_attractor.homeostasis_spike_recorders",
                "r1.homeostasis_spike_recorders",
            ],
            "r2_homeostasis_spike_recorders": [
                "r2.ring_attractor.homeostasis_spike_recorders",
                "r2.homeostasis_spike_recorders",
            ],
            "gain_homeostasis_spike_recorders": [
                "gain_modulation.homeostasis_spike_recorders",
            ],
        }

        for name, candidate_paths in homeostasis_candidates.items():
            value, used_path = self._first_existing_path(self.ring_model, candidate_paths)
            if value is None:
                continue

            if "spike" in name or "recorders" in name:
                data["homeostasis"][name] = self._collect_spike_recorder_group(
                    value,
                    source_path=used_path,
                )
            else:
                data["homeostasis"][name] = self._collect_generic_device_group(
                    value,
                    source_path=used_path,
                )

        return data

    # ------------------------------------------------------------------
    # Ring / robot helpers
    # ------------------------------------------------------------------

    def get_joint_position(self):
        return self.publisher.current_positions

    def _get_spike_counts(self, recorders):
        spikes = self.ring_model._collect_spikes(recorders)
        return np.array([len(spike_times) for spike_times in spikes], dtype=float)

    def _joint_to_ring_index(self, joint_position, stimulus_half_width):
        return legacy_joint_to_ring_index(
            joint_position,
            self.joint_min,
            self.joint_max,
            self.ring_model.population_size,
            stimulus_half_width,
        )

    def set_ring_goal(self, goal_position, stimulus_half_width=5):
        stimulus_half_width = self.control_profile.effective_half_width(
            stimulus_half_width
        )
        inject = self._joint_to_ring_index(goal_position, stimulus_half_width)

        self.goal_inject_idx = inject
        self.ring_model.r2._inject_bump(inject, stimulus_half_width)

        rospy.loginfo(
            f"[RING GOAL] goal={goal_position:.4f} rad -> r2 index={inject}"
        )

    def set_ring_state(self, stimulus_half_width=5):
        position = self.get_joint_position()
        current_joint_position = position[self.target_joint]

        stimulus_half_width = self.control_profile.effective_half_width(
            stimulus_half_width
        )
        inject = self._joint_to_ring_index(current_joint_position, stimulus_half_width)

        self.ring_state_inject_idx = inject
        self.ring_model.r1._inject_bump(inject, stimulus_half_width)

        rospy.loginfo(
            f"[RING STATE] current={current_joint_position:.4f} rad -> r1 index={inject}"
        )

    def get_ring_state(self):
        if self.r1_delta_spike_counts is None:
            return None

        spike_counts = self.r1_delta_spike_counts
        total_spikes = spike_counts.sum()
        self.state_count = float(total_spikes)

        if total_spikes == 0:
            return None

        n_neurons = self.ring_model.population_size
        activity = spike_counts / total_spikes
        angles = np.arange(n_neurons) * 2 * np.pi / n_neurons

        cos_mean = np.sum(activity * np.cos(angles))
        sin_mean = np.sum(activity * np.sin(angles))

        mean_angle = np.arctan2(sin_mean, cos_mean)
        if mean_angle < 0:
            mean_angle += 2 * np.pi

        bump_index_mean = (mean_angle / (2 * np.pi)) * n_neurons
        bump_idx = int(np.argmax(spike_counts))

        self.bump_index = bump_idx
        self.bump_index_mean = float(bump_index_mean)

        return bump_idx

    def run_ring_simulation(self):
        """Run NEST and return signed gain signal."""
        left_spikes_before = self.ring_model._collect_spikes(
            self.ring_model.gain_modulation.left_gain_spike_recorders
        )
        right_spikes_before = self.ring_model._collect_spikes(
            self.ring_model.gain_modulation.right_gain_spike_recorders
        )
        r1_counts_before = self._get_spike_counts(
            self.ring_model.r1.ring_attractor.ring_spike_recorders
        )

        left_count_before = sum(len(spike_times) for spike_times in left_spikes_before)
        right_count_before = sum(len(spike_times) for spike_times in right_spikes_before)

        nest.Simulate(self.time_step)

        left_spikes_after = self.ring_model._collect_spikes(
            self.ring_model.gain_modulation.left_gain_spike_recorders
        )
        right_spikes_after = self.ring_model._collect_spikes(
            self.ring_model.gain_modulation.right_gain_spike_recorders
        )
        r1_counts_after = self._get_spike_counts(
            self.ring_model.r1.ring_attractor.ring_spike_recorders
        )

        left_count_after = sum(len(spike_times) for spike_times in left_spikes_after)
        right_count_after = sum(len(spike_times) for spike_times in right_spikes_after)

        r1_delta_counts = np.maximum(r1_counts_after - r1_counts_before, 0.0)

        self.left_count = int(left_count_after - left_count_before)
        self.right_count = int(right_count_after - right_count_before)
        self.r1_delta_spike_counts = r1_delta_counts
        self.state_count = float(r1_delta_counts.sum())

        return self.right_count - self.left_count

    # ------------------------------------------------------------------
    # Decoder helpers
    # ------------------------------------------------------------------

    def _filter_spikes(self, spike_history, dt_s, tau):
        return exponential_filter(spike_history, dt_s, tau)

    def _apply_delay(self, drive_history, delay_steps):
        return apply_delay(drive_history, delay_steps)

    def _decode_velocity_from_drive(self, delayed_drive):
        return decode_velocity(
            delayed_drive,
            self.decoder_gain_positive,
            self.decoder_gain_negative,
        )

    def _decoder_features_from_spikes(self, spike_history, dt_s, tau, delay_steps):
        filtered = self._filter_spikes(spike_history, dt_s, tau)
        delayed = self._apply_delay(filtered, delay_steps)
        drive_pos = np.maximum(delayed, 0.0)
        drive_neg = np.maximum(-delayed, 0.0)
        return {
            "filtered_drive": filtered,
            "delayed_drive": delayed,
            "drive_pos": drive_pos,
            "drive_neg": drive_neg,
            "S_pos": float(np.sum(drive_pos) * dt_s),
            "S_neg": float(np.sum(drive_neg) * dt_s),
        }

    # ------------------------------------------------------------------
    # Single analysis trial
    # ------------------------------------------------------------------

    def run_trial(self, goal_state):
        """Run one analysis trial using the currently loaded decoder.

        No learning occurs here.
        """

        dt_s = self.time_step / 1000.0
        decoder_tau = self.decoder_tau
        alpha = np.exp(-dt_s / decoder_tau)

        self.publisher.prime_command_state(force=True)

        q_start = self.get_joint_position()[self.target_joint]
        q_start_all = list(self.publisher.current_positions)

        if self.publisher.current_velocities is None:
            qdot_start_all = [0.0] * len(q_start_all)
        else:
            qdot_start_all = list(self.publisher.current_velocities)

        dq_desired = goal_state - q_start
        movement_direction = np.sign(dq_desired) if abs(dq_desired) > 1e-9 else 0.0

        self.set_ring_goal(goal_state)
        self.set_ring_state()

        drive_core = DriveControlCore(
            dt_s,
            DecoderParameters(
                tau_s=decoder_tau,
                delay_steps=self.decoder_delay_steps,
            ),
            alpha=alpha,
        )

        consecutive_settled = 0
        stop_reason = "max_steps"

        sample_buffer = []

        # Histories
        wall_time_history = []
        time_history = []

        signed_spike_history = []
        left_gain_spike_count_history = []
        right_gain_spike_count_history = []
        r1_state_spike_count_history = []
        r1_bump_index_history = []

        filtered_drive_history = []
        delayed_drive_history = []
        decoded_velocity_history = []

        joint_position_history = []
        joint_velocity_history = []
        all_joint_position_history = []
        all_joint_velocity_history = []

        position_error_history = []

        # Pre-fill lookahead
        for _ in range(self.lookahead):
            signed_spike = self.run_ring_simulation()

            _, integrated_drive, delayed_drive = drive_core.advance(
                signed_spike,
                self.decoder_delay_steps,
            )

            velocity_cmd = self._decode_velocity_from_drive(delayed_drive)
            r1_bump_idx = self.get_ring_state()

            sample_buffer.append(
                {
                    "signed_spike": float(signed_spike),
                    "left_gain_spike_count": int(self.left_count),
                    "right_gain_spike_count": int(self.right_count),
                    "r1_state_spike_count": float(self.state_count),
                    "r1_bump_index": r1_bump_idx,
                    "filtered_drive": float(integrated_drive),
                    "delayed_drive": float(delayed_drive),
                    "decoded_velocity": float(velocity_cmd),
                }
            )

        self.publisher.publish_receding_trajectory(
            self.target_joint,
            [s["decoded_velocity"] for s in sample_buffer],
            dt_s,
        )

        for step in range(self.max_steps):
            rospy.sleep(dt_s)

            sample = sample_buffer[0]

            current_positions = list(self.publisher.current_positions)
            if self.publisher.current_velocities is None:
                current_velocities = [0.0] * len(current_positions)
            else:
                current_velocities = list(self.publisher.current_velocities)

            current_position = current_positions[self.target_joint]
            current_velocity = current_velocities[self.target_joint]

            position_error = goal_state - current_position
            t_now = (step + 1) * dt_s

            wall_time_history.append(float(rospy.Time.now().to_sec()))
            time_history.append(float(t_now))

            signed_spike_history.append(sample["signed_spike"])
            left_gain_spike_count_history.append(sample["left_gain_spike_count"])
            right_gain_spike_count_history.append(sample["right_gain_spike_count"])
            r1_state_spike_count_history.append(sample["r1_state_spike_count"])
            r1_bump_index_history.append(sample["r1_bump_index"])

            filtered_drive_history.append(sample["filtered_drive"])
            delayed_drive_history.append(sample["delayed_drive"])
            decoded_velocity_history.append(sample["decoded_velocity"])

            joint_position_history.append(float(current_position))
            joint_velocity_history.append(float(current_velocity))
            all_joint_position_history.append(current_positions)
            all_joint_velocity_history.append(current_velocities)

            position_error_history.append(float(position_error))

            rospy.loginfo(
                f"[STEP {step:03d}] "
                f"spike={sample['signed_spike']:.1f} "
                f"left={sample['left_gain_spike_count']} "
                f"right={sample['right_gain_spike_count']} "
                f"drive={sample['filtered_drive']:.2f} "
                f"d_delay={sample['delayed_drive']:.2f} "
                f"cmd_vel={sample['decoded_velocity']:.6f} rad/s "
                f"meas_vel={current_velocity:.6f} rad/s "
                f"q={current_position:.4f} rad "
                f"goal_err={position_error:.4f} rad"
            )

            if abs(sample["delayed_drive"]) < self.drive_threshold:
                consecutive_settled += 1
                if consecutive_settled >= self.n_settle:
                    stop_reason = "drive_settled"
                    rospy.loginfo(
                        f"[STEP {step:03d}] Delayed drive settled for "
                        f"{self.n_settle} steps."
                    )
                    break
            else:
                consecutive_settled = 0

            # Generate one new future sample
            signed_spike = self.run_ring_simulation()
            _, integrated_drive, delayed_drive = drive_core.advance(
                signed_spike,
                self.decoder_delay_steps,
            )

            velocity_cmd = self._decode_velocity_from_drive(delayed_drive)
            r1_bump_idx = self.get_ring_state()

            new_sample = {
                "signed_spike": float(signed_spike),
                "left_gain_spike_count": int(self.left_count),
                "right_gain_spike_count": int(self.right_count),
                "r1_state_spike_count": float(self.state_count),
                "r1_bump_index": r1_bump_idx,
                "filtered_drive": float(integrated_drive),
                "delayed_drive": float(delayed_drive),
                "decoded_velocity": float(velocity_cmd),
            }

            sample_buffer = sample_buffer[1:] + [new_sample]

            self.publisher.publish_receding_trajectory(
                self.target_joint,
                [s["decoded_velocity"] for s in sample_buffer],
                dt_s,
            )

        self.publisher.stop_joint(self.target_joint, dt_s)
        self.publisher.wait_for_settled(self.target_joint)

        q_final = self.get_joint_position()[self.target_joint]
        q_final_all = list(self.publisher.current_positions)

        if self.publisher.current_velocities is None:
            qdot_final_all = [0.0] * len(q_final_all)
        else:
            qdot_final_all = list(self.publisher.current_velocities)

        dq_actual = q_final - q_start

        raw_position_error_rad = goal_state - q_final
        abs_position_error_rad = abs(raw_position_error_rad)
        aligned_error_rad = raw_position_error_rad * movement_direction

        features = self._decoder_features_from_spikes(
            signed_spike_history,
            dt_s,
            self.decoder_tau,
            self.decoder_delay_steps,
        )

        S_raw = float(np.sum(signed_spike_history) * dt_s)
        S_filtered = float(np.sum(filtered_drive_history) * dt_s)
        S_delayed = float(np.sum(delayed_drive_history) * dt_s)
        S_pos = float(features["S_pos"])
        S_neg = float(features["S_neg"])
        S_vel = float(np.sum(decoded_velocity_history) * dt_s)

        decoder_predicted_dq_rad = (
            self.decoder_gain_positive * S_pos
            + self.decoder_gain_negative * S_neg
        )

        model_recordings = self.collect_model_recordings()

        trial = {
            "session_id": self.session_id,
            "joint_index": int(self.target_joint),

            "analysis_settings": self._analysis_settings(),
            "decoder_used": self._current_decoder_params(),

            "goal_data": {
                "goal_position_rad": float(goal_state),
                "goal_ring_index": self.goal_inject_idx,
                "initial_ring_index": self.ring_state_inject_idx,
            },

            "robot_start_state": {
                "joint_positions_rad": q_start_all,
                "joint_velocities_rad_s": qdot_start_all,
            },
            "robot_final_state": {
                "joint_positions_rad": q_final_all,
                "joint_velocities_rad_s": qdot_final_all,
            },

            "q_start": float(q_start),
            "q_goal": float(goal_state),
            "q_final": float(q_final),

            "dq_desired": float(dq_desired),
            "dq_actual": float(dq_actual),
            "decoder_predicted_dq_rad": float(decoder_predicted_dq_rad),

            "raw_position_error_rad": float(raw_position_error_rad),
            "abs_position_error_rad": float(abs_position_error_rad),
            "aligned_error_rad": float(aligned_error_rad),

            "wall_time_history": wall_time_history,
            "time_history": time_history,

            "signed_spike_history": signed_spike_history,
            "left_gain_spike_count_history": left_gain_spike_count_history,
            "right_gain_spike_count_history": right_gain_spike_count_history,
            "r1_state_spike_count_history": r1_state_spike_count_history,
            "r1_bump_index_history": r1_bump_index_history,

            "filtered_drive_history": filtered_drive_history,
            "delayed_drive_history": delayed_drive_history,
            "decoded_velocity_history": decoded_velocity_history,

            "joint_position_history": joint_position_history,
            "joint_velocity_history": joint_velocity_history,
            "all_joint_position_history": all_joint_position_history,
            "all_joint_velocity_history": all_joint_velocity_history,

            "position_error_history": position_error_history,

            "dt": float(dt_s),
            "n_steps": int(len(time_history)),
            "stop_reason": stop_reason,

            "S_raw": float(S_raw),
            "S_filtered": float(S_filtered),
            "S_delayed": float(S_delayed),
            "S_pos": float(S_pos),
            "S_neg": float(S_neg),
            "S_vel": float(S_vel),

            "model_recordings": model_recordings,
        }

        rospy.loginfo(
            f"[TRIAL] start={q_start:.4f} rad "
            f"goal={goal_state:.4f} rad "
            f"final={q_final:.4f} rad "
            f"dq_desired={dq_desired:.4f} rad "
            f"dq_actual={dq_actual:.4f} rad "
            f"dq_pred={decoder_predicted_dq_rad:.4f} rad "
            f"abs_error={abs_position_error_rad:.4f} rad "
            f"aligned_error={aligned_error_rad:.4f} rad "
            f"stop={stop_reason}"
        )

        return trial

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_region_statistics(self, region_trials):
        return analysis_region_statistics(region_trials)

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_spike_group(self, ax, group, title):
        if group is None or "events" not in group or len(group["events"]) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_ylabel("Neuron / recorder")
            ax.grid(True, alpha=0.3)
            return

        for ev in group["events"]:
            idx = ev.get("recorder_index", ev.get("device_index", 0))
            times = np.asarray(ev.get("times_ms", []), dtype=float)
            if len(times) > 0:
                ax.vlines(times, idx - 0.4, idx + 0.4, linewidth=0.5)

        ax.set_title(title)
        ax.set_ylabel("Neuron / recorder")
        ax.grid(True, alpha=0.3)

    def _max_time_from_spike_group(self, group):
        if group is None or "events" not in group:
            return self.time_step

        max_time = self.time_step
        for ev in group["events"]:
            times = ev.get("times_ms", [])
            if len(times) > 0:
                max_time = max(max_time, max(times))
        return max_time

    # ------------------------------------------------------------------
    # Trial plots
    # ------------------------------------------------------------------

    def plot_trial_diagnostics(self, trial, region_idx, iteration_idx):
        output_dir = self._plot_dir("trial_diagnostics")

        t = np.array(trial["time_history"], dtype=float)
        q = np.array(trial["joint_position_history"], dtype=float)
        q_goal = float(trial["q_goal"])

        filtered_drive = np.array(trial["filtered_drive_history"], dtype=float)
        delayed_drive = np.array(trial["delayed_drive_history"], dtype=float)

        v_cmd = np.array(trial["decoded_velocity_history"], dtype=float)
        v_meas = np.array(trial["joint_velocity_history"], dtype=float)
        e_pos = np.array(trial["position_error_history"], dtype=float)

        left_counts = np.array(trial["left_gain_spike_count_history"], dtype=float)
        right_counts = np.array(trial["right_gain_spike_count_history"], dtype=float)
        signed_spikes = np.array(trial["signed_spike_history"], dtype=float)

        fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)

        axes[0].plot(t, q, label="Measured joint position q(t)")
        axes[0].axhline(q_goal, linestyle="--", label="Goal position")
        axes[0].set_ylabel("Position (rad)")
        axes[0].set_title(
            f"Trial diagnostics | Joint {self.target_joint} | "
            f"Region {region_idx}, Iteration {iteration_idx}"
        )
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, filtered_drive, label="Filtered drive d(t)")
        axes[1].plot(t, delayed_drive, label="Delayed drive d(t-delay)")
        axes[1].axhline(0.0, linestyle="--", linewidth=1)
        axes[1].set_ylabel("Drive\n(spike units)")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, v_cmd, label="Decoded commanded velocity")
        axes[2].plot(t, v_meas, label="Measured joint velocity")
        axes[2].axhline(0.0, linestyle="--", linewidth=1)
        axes[2].set_ylabel("Velocity (rad/s)")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t, e_pos, label="Position error = goal - q")
        axes[3].axhline(0.0, linestyle="--", linewidth=1, label="Zero error")
        axes[3].set_ylabel("Error (rad)")
        axes[3].legend(loc="best")
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(t, left_counts, label="Left gain spikes / step")
        axes[4].plot(t, right_counts, label="Right gain spikes / step")
        axes[4].plot(t, signed_spikes, label="Signed spike = right - left")
        axes[4].set_xlabel("Time (s)")
        axes[4].set_ylabel("Spike count / step")
        axes[4].legend(loc="best")
        axes[4].grid(True, alpha=0.3)

        fig.tight_layout()

        path = os.path.join(
            output_dir,
            f"{self.session_id}_trial_diag_joint{self.target_joint}_"
            f"region{region_idx}_iter{iteration_idx}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Trial diagnostics -> {path}")

    def plot_ring_raster(self, trial, region_idx, iteration_idx):
        output_dir = self._plot_dir("raster_ring")

        recordings = trial["model_recordings"]["ring_raster"]
        r1_group = recordings.get("r1_ring_spikes")
        r2_group = recordings.get("r2_ring_spikes")

        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

        self._plot_spike_group(axes[0], r1_group, "r1 ring attractor spikes")
        self._plot_spike_group(axes[1], r2_group, "r2 ring attractor spikes")

        if r1_group is not None and trial["goal_data"]["initial_ring_index"] is not None:
            time_max = self._max_time_from_spike_group(r1_group)
            arrow_length = max(1.0, 0.1 * time_max)
            axes[0].arrow(
                0.2 * arrow_length,
                trial["goal_data"]["initial_ring_index"],
                arrow_length,
                0,
                head_width=2,
                head_length=0.3 * arrow_length,
                fc="blue",
                ec="blue",
                linewidth=2,
            )

        if r2_group is not None and trial["goal_data"]["goal_ring_index"] is not None:
            time_max = self._max_time_from_spike_group(r2_group)
            arrow_length = max(1.0, 0.1 * time_max)
            axes[1].arrow(
                0.2 * arrow_length,
                trial["goal_data"]["goal_ring_index"],
                arrow_length,
                0,
                head_width=2,
                head_length=0.3 * arrow_length,
                fc="green",
                ec="green",
                linewidth=2,
            )

        axes[0].set_xlabel("Time (ms)")
        axes[1].set_xlabel("Time (ms)")
        axes[0].set_title(
            f"Ring attractor raster | Joint {self.target_joint} | "
            f"Region {region_idx}, Iteration {iteration_idx}"
        )

        fig.tight_layout()

        path = os.path.join(
            output_dir,
            f"{self.session_id}_ring_raster_joint{self.target_joint}_"
            f"region{region_idx}_iter{iteration_idx}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Ring raster -> {path}")

    def plot_gain_modulation_raster(self, trial, region_idx, iteration_idx):
        output_dir = self._plot_dir("raster_gain_modulation")

        recordings = trial["model_recordings"]["gain_modulation"]
        left_group = recordings.get("left_gain_spikes")
        right_group = recordings.get("right_gain_spikes")

        fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

        self._plot_spike_group(axes[0], left_group, "Left gain modulation spikes")
        self._plot_spike_group(axes[1], right_group, "Right gain modulation spikes")

        axes[0].set_xlabel("Time (ms)")
        axes[1].set_xlabel("Time (ms)")
        axes[0].set_title(
            f"Gain modulation raster | Joint {self.target_joint} | "
            f"Region {region_idx}, Iteration {iteration_idx}"
        )

        fig.tight_layout()

        path = os.path.join(
            output_dir,
            f"{self.session_id}_gain_raster_joint{self.target_joint}_"
            f"region{region_idx}_iter{iteration_idx}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Gain modulation raster -> {path}")

    def plot_homeostasis_data(self, trial, region_idx, iteration_idx):
        output_dir = self._plot_dir("homeostasis")
        homeo = trial["model_recordings"].get("homeostasis", {})

        if not homeo:
            return

        for dataset_name, dataset in homeo.items():
            events_list = dataset.get("events", [])
            if not events_list:
                continue

            # Find numeric keys other than times/senders
            numeric_keys = set()
            for dev in events_list:
                ev = dev.get("events", {})
                for key, value in ev.items():
                    if key in ("times", "senders"):
                        continue
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], (int, float, np.integer, np.floating)):
                            numeric_keys.add(key)

            if not numeric_keys:
                continue

            numeric_keys = sorted(list(numeric_keys))[:4]

            fig, axes = plt.subplots(len(numeric_keys), 1, figsize=(13, 3.5 * len(numeric_keys)))
            if len(numeric_keys) == 1:
                axes = [axes]

            for ax, key in zip(axes, numeric_keys):
                plotted_any = False
                for dev in events_list[:6]:
                    ev = dev.get("events", {})
                    times = np.asarray(ev.get("times", []), dtype=float)
                    values = np.asarray(ev.get(key, []), dtype=float)

                    if len(times) > 0 and len(values) > 0 and len(times) == len(values):
                        label = f"device_{dev.get('device_index', 0)}"
                        ax.plot(times, values, label=label)
                        plotted_any = True

                ax.set_ylabel(key)
                ax.grid(True, alpha=0.3)
                if plotted_any:
                    ax.legend(loc="best")

            axes[-1].set_xlabel("Time (ms)")
            axes[0].set_title(
                f"Homeostasis / adaptive signals: {dataset_name} | "
                f"Joint {self.target_joint} | Region {region_idx}, Iteration {iteration_idx}"
            )

            fig.tight_layout()

            path = os.path.join(
                output_dir,
                f"{self.session_id}_homeostasis_{self._slug(dataset_name)}_"
                f"joint{self.target_joint}_region{region_idx}_iter{iteration_idx}.png",
            )
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            rospy.loginfo(f"[PLOT] Homeostasis -> {path}")

    # ------------------------------------------------------------------
    # Region / overall plots
    # ------------------------------------------------------------------

    def plot_batch_summary(self, batch_trials, batch_summary):
        batch_idx = batch_summary["batch_idx"]
        output_dir = self._plot_dir("batch_summary")

        trial_ids = np.arange(1, len(batch_trials) + 1)
        abs_errors = np.array([t["abs_position_error_rad"] for t in batch_trials], dtype=float)

        dq_desired = np.array([t["dq_desired"] for t in batch_trials], dtype=float)
        dq_actual = np.array([t["dq_actual"] for t in batch_trials], dtype=float)
        dq_pred = np.array([t["decoder_predicted_dq_rad"] for t in batch_trials], dtype=float)

        # Error trend
        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(trial_ids, abs_errors, marker="o")
        ax.set_ylabel("Absolute error (rad)")
        ax.set_title(f"Batch {batch_idx} absolute final position error")
        ax.set_xlabel("Trial within batch")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        path1 = os.path.join(
            output_dir,
            f"{self.session_id}_batch_error_joint{self.target_joint}_batch{batch_idx}.png",
        )
        fig.savefig(path1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rospy.loginfo(f"[PLOT] Batch error summary -> {path1}")

        # Displacement diagnostics
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        min1 = min(np.min(dq_desired), np.min(dq_actual))
        max1 = max(np.max(dq_desired), np.max(dq_actual))
        axes[0].scatter(dq_desired, dq_actual, label="Actual vs desired")
        axes[0].plot([min1, max1], [min1, max1], linestyle="--", label="Perfect")
        axes[0].set_xlabel("Desired displacement dq_desired (rad)")
        axes[0].set_ylabel("Actual displacement dq_actual (rad)")
        axes[0].set_title("Robot execution accuracy")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        min2 = min(np.min(dq_desired), np.min(dq_pred))
        max2 = max(np.max(dq_desired), np.max(dq_pred))
        axes[1].scatter(dq_desired, dq_pred, label="Decoder prediction vs desired")
        axes[1].plot([min2, max2], [min2, max2], linestyle="--", label="Perfect")
        axes[1].set_xlabel("Desired displacement dq_desired (rad)")
        axes[1].set_ylabel("Decoder-predicted displacement (rad)")
        axes[1].set_title("Decoder prediction accuracy")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()

        path2 = os.path.join(
            output_dir,
            f"{self.session_id}_batch_displacement_joint{self.target_joint}_batch{batch_idx}.png",
        )
        fig.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rospy.loginfo(f"[PLOT] Batch displacement summary -> {path2}")

    def plot_mean_std_by_batch(self, batch_summaries):
        output_dir = self._plot_dir("overall_summary")

        batch_labels = [f"B{r['batch_idx']}" for r in batch_summaries]
        x = np.arange(len(batch_labels))

        mean_abs = np.array(
            [r["statistics"]["mean_abs_error_rad"] for r in batch_summaries],
            dtype=float,
        )
        std_abs = np.array(
            [r["statistics"]["std_abs_error_rad"] for r in batch_summaries],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.bar(x, mean_abs, yerr=std_abs, capsize=5)
        ax.set_ylabel("Mean ± std absolute error (rad)")
        ax.set_title("Batch-wise absolute error mean and standard deviation")
        ax.set_xticks(x)
        ax.set_xticklabels(batch_labels)
        ax.set_xlabel("Batch")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()

        path = os.path.join(
            output_dir,
            f"{self.session_id}_mean_std_by_batch_joint{self.target_joint}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Mean/std by batch -> {path}")

    def plot_success_rate_by_batch(self, batch_summaries):
        output_dir = self._plot_dir("overall_summary")

        batch_labels = [f"B{r['batch_idx']}" for r in batch_summaries]
        x = np.arange(len(batch_labels))
        width = 0.25

        success_002 = np.array(
            [r["statistics"]["success_rate_abs_error_le_0p02_rad"] for r in batch_summaries],
            dtype=float,
        )
        success_005 = np.array(
            [r["statistics"]["success_rate_abs_error_le_0p05_rad"] for r in batch_summaries],
            dtype=float,
        )
        success_010 = np.array(
            [r["statistics"]["success_rate_abs_error_le_0p10_rad"] for r in batch_summaries],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - width, success_002, width=width, label="|error| ≤ 0.02 rad")
        ax.bar(x, success_005, width=width, label="|error| ≤ 0.05 rad")
        ax.bar(x + width, success_010, width=width, label="|error| ≤ 0.10 rad")

        ax.set_xticks(x)
        ax.set_xticklabels(batch_labels)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Success rate")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Success rate by batch at different absolute-error thresholds")
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()

        path = os.path.join(
            output_dir,
            f"{self.session_id}_success_rate_by_batch_joint{self.target_joint}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Success rate by batch -> {path}")

    def plot_abs_error_boxplot_by_batch(self, batch_summaries):
        output_dir = self._plot_dir("overall_summary")

        labels = []
        data = []

        for r in batch_summaries:
            labels.append(f"B{r['batch_idx']}")
            data.append(r["abs_error_list_rad"])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Absolute final error (rad)")
        ax.set_title("Distribution of absolute final error by batch")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()

        path = os.path.join(
            output_dir,
            f"{self.session_id}_abs_error_boxplot_by_batch_joint{self.target_joint}.png",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Absolute error boxplot by batch -> {path}")

    # ------------------------------------------------------------------
    # Ring reset
    # ------------------------------------------------------------------

    def reset_ring_model(self):
        try:
            nest.ResetKernel()
        except Exception as e:
            rospy.logwarn(f"[RING RESET] nest.ResetKernel() failed: {e}")

        self.ring_model = SingleRingModel()
        self.goal_inject_idx = None
        self.ring_state_inject_idx = None
        self.r1_delta_spike_counts = None

    # ------------------------------------------------------------------
    # Manual limits
    # ------------------------------------------------------------------

    def find_joint_limits_manually(self):
        rospy.loginfo(f"Finding limits for joint {self.target_joint}...")

        velocity = 1.0
        timeout = 5.0

        self.publisher.publish_velocity(self.target_joint, -velocity, timeout)
        rospy.sleep(timeout)
        joint_min = self.get_joint_position()[self.target_joint]

        self.publisher.publish_velocity(self.target_joint, velocity, timeout)
        rospy.sleep(timeout)
        joint_max = self.get_joint_position()[self.target_joint]

        self.publisher.publish_velocity(self.target_joint, 0.0, timeout)

        rospy.loginfo(
            f"Joint {self.target_joint} limits: "
            f"[{joint_min:.4f}, {joint_max:.4f}] rad"
        )

        return joint_min, joint_max

    # ------------------------------------------------------------------
    # Main analysis loop
    # ------------------------------------------------------------------

    def run_analysis(self, num_iterations=100, num_batches=5):
        """Run decoder analysis with fixed decoder parameters.

        num_iterations trials are run in total, goals sampled uniformly from
        the full joint range. Trials are grouped into num_batches equal batches
        to compute mean and std of the absolute position error per batch.
        """

        analysis_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        batch_size = max(1, num_iterations // num_batches)

        rospy.loginfo(
            f"[ANALYSIS START] joint={self.target_joint} "
            f"limits=[{self.joint_min:.4f}, {self.joint_max:.4f}] "
            f"k_pos={self.decoder_gain_positive:.6e} "
            f"k_neg={self.decoder_gain_negative:.6e} "
            f"tau={self.decoder_tau:.3f}s "
            f"delay={self.decoder_delay_steps} steps "
            f"num_iterations={num_iterations} num_batches={num_batches} batch_size={batch_size}"
        )

        batch_summaries = []
        all_trial_files = []
        current_batch_trials = []
        current_batch_trial_files = []

        for iteration_idx in range(1, num_iterations + 1):
            batch_idx = min((iteration_idx - 1) // batch_size + 1, num_batches)

            goal_state = np.random.uniform(self.joint_min, self.joint_max)
            current_pos = self.get_joint_position()[self.target_joint]

            rospy.loginfo(
                f"[Iter {iteration_idx}/{num_iterations} | Batch {batch_idx}/{num_batches}] "
                f"current={current_pos:.4f} rad "
                f"goal={goal_state:.4f} rad"
            )

            self.publisher.reset_pose()
            self.publisher.prime_command_state(force=True)

            trial = self.run_trial(goal_state)

            trial["batch_idx"] = int(batch_idx)
            trial["iteration_idx"] = int(iteration_idx)

            trial_path = self.save_trial_data(trial, batch_idx, iteration_idx)
            all_trial_files.append(trial_path)
            current_batch_trial_files.append(trial_path)

            self.plot_trial_diagnostics(trial, batch_idx, iteration_idx)
            self.plot_ring_raster(trial, batch_idx, iteration_idx)
            self.plot_gain_modulation_raster(trial, batch_idx, iteration_idx)
            self.plot_homeostasis_data(trial, batch_idx, iteration_idx)

            current_batch_trials.append(trial)

            rospy.loginfo(
                f"  -> final={trial['q_final']:.4f} rad "
                f"abs_error={trial['abs_position_error_rad']:.4f} rad "
                f"aligned_error={trial['aligned_error_rad']:.4f} rad"
            )

            self.reset_ring_model()

            # Batch is done when we've filled it, or on the last iteration
            batch_is_done = (
                iteration_idx == batch_idx * batch_size
                or (batch_idx == num_batches and iteration_idx == num_iterations)
            )

            if batch_is_done:
                batch_stats = self._compute_region_statistics(current_batch_trials)

                batch_summary = {
                    "session_id": self.session_id,
                    "joint_index": int(self.target_joint),
                    "batch_idx": int(batch_idx),
                    "n_trials_in_batch": int(len(current_batch_trials)),
                    "analysis_settings": self._analysis_settings(),
                    "statistics": batch_stats,
                    "trial_files": list(current_batch_trial_files),
                    "aligned_error_list_rad": [float(t["aligned_error_rad"]) for t in current_batch_trials],
                    "abs_error_list_rad": [float(t["abs_position_error_rad"]) for t in current_batch_trials],
                    "dq_desired_list_rad": [float(t["dq_desired"]) for t in current_batch_trials],
                    "dq_actual_list_rad": [float(t["dq_actual"]) for t in current_batch_trials],
                    "dq_pred_list_rad": [float(t["decoder_predicted_dq_rad"]) for t in current_batch_trials],
                }

                self.save_batch_summary(batch_summary)
                self.plot_batch_summary(current_batch_trials, batch_summary)

                batch_summaries.append(batch_summary)

                rospy.loginfo(
                    f"[BATCH DONE] batch={batch_idx}/{num_batches} "
                    f"n_trials={len(current_batch_trials)} "
                    f"mean_abs_error={batch_stats['mean_abs_error_rad']:.4f} rad "
                    f"std_abs_error={batch_stats['std_abs_error_rad']:.4f} rad"
                )

                current_batch_trials = []
                current_batch_trial_files = []

        self.plot_mean_std_by_batch(batch_summaries)
        self.plot_success_rate_by_batch(batch_summaries)
        self.plot_abs_error_boxplot_by_batch(batch_summaries)

        overall_summary = {
            "session_id": self.session_id,
            "analysis_start": analysis_start,
            "analysis_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "joint_index": int(self.target_joint),
            "analysis_settings": self._analysis_settings(),
            "decoder_used": self._current_decoder_params(),
            "num_iterations": int(num_iterations),
            "num_batches": int(num_batches),
            "batch_size": int(batch_size),
            "total_trials": int(num_iterations),
            "all_trial_files": all_trial_files,
            "batches": batch_summaries,
        }

        overall_summary_path = self.save_overall_summary(overall_summary)

        rospy.loginfo(
            f"[ANALYSIS COMPLETE] joint={self.target_joint} "
            f"total_trials={num_iterations} "
            f"overall_summary={overall_summary_path}"
        )

        return overall_summary


def main():
    rospy.init_node("tiago_decoder_analysis", anonymous=True)

    target_joint = 6

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "calibration",
        "velocity_calibration.json",
    )

    use_manual_limits = True

    try:
        config = load_json_legacy(config_path)

        joint_key = str(target_joint)

        if (
            joint_key in config
            and "joint_min" in config[joint_key]
            and "joint_max" in config[joint_key]
        ):
            use_manual_limits = False

    except Exception:
        pass

    analysis = TiagoDecoderAnalysis(
        target_joint=target_joint,
        use_manual_limits=use_manual_limits,
    )

    analysis.run_analysis(
        num_iterations=25,  # total trials
        num_batches=5,       # group into 5 batches for mean/std
    )


if __name__ == "__main__":
    main()
