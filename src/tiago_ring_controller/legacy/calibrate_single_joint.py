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
from tiago_ring_controller.control.profiles import CALIBRATION_PROFILE
from tiago_ring_controller.config import module_config_path
from tiago_ring_controller.math.control import (
    apply_delay,
    calibration_joint_to_ring_index,
    decode_velocity,
    decoder_features,
    exponential_filter,
)
from tiago_ring_controller.training.calibration import (
    fit_decoder_from_trials as fit_decoder_calibration,
    fit_signed_direction_gains,
)


class TiagoCalibration:
    def __init__(self, target_joint, use_manual_limits=False):
        self.target_joint = target_joint

        self.publisher = TiagoPublisher()
        self.subscriber = TiagoSubscriber()

        self.publisher.wait_for_joint_state(timeout=5.0)

        self.ring_model = SingleRingModel()
        self.control_profile = CALIBRATION_PROFILE
        # Keep spike-drive scale comparable across population sizes.
        # N=100 -> 1.0 (old behavior), N=200 -> 0.5.
        self.spike_scale = self.control_profile.spike_scale(
            self.ring_model.population_size
        )
        # Keep goal/state injections away from circular boundary effects.
        # Example: N=200 -> [20, 180], N=100 -> [10, 90].
        self.ring_edge_margin_fraction = self.control_profile.ring_edge_margin_fraction

        # NEST / robot control step
        self.time_step = self.control_profile.time_step_ms

        # Decoder:
        #
        # d_t = alpha * d_{t-1} + (1 - alpha) * spike_t
        #
        # v_cmd =
        #     k_pos * max(d_delayed, 0)
        #   + k_neg * max(-d_delayed, 0)
        #
        # k_pos should usually be positive.
        # k_neg should usually be negative.
        self.decoder_gain_positive = 1e-4
        self.decoder_gain_negative = -1e-4
        self.decoder_tau = 0.3
        self.decoder_delay_steps = 0

        # Decoder search grid
        self.tau_candidates = [0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20]
        self.delay_candidates = [0, 1, 2, 3, 4]

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

        self.iteration_errors = []

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
                    f"Joint {self.target_joint} has no decoder config. "
                    "Using defaults."
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
                    f"k_pos={gain:.6e}, k_neg={-gain:.6e}, "
                    "tau=0.3, delay=0"
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

    def _atomic_json_dump(self, data, path):
        atomic_json_dump_legacy(path, data)

    def _current_decoder_params(self):
        return {
            "decoder_gain_positive": float(self.decoder_gain_positive),
            "decoder_gain_negative": float(self.decoder_gain_negative),
            "decoder_tau": float(self.decoder_tau),
            "decoder_delay_steps": int(self.decoder_delay_steps),
        }

    def _calibration_settings(self):
        return {
            "time_step_ms": float(self.time_step),
            "lookahead": int(self.lookahead),
            "drive_threshold": float(self.drive_threshold),
            "n_settle": int(self.n_settle),
            "max_steps": int(self.max_steps),
            "tau_candidates": list(self.tau_candidates),
            "delay_candidates": list(self.delay_candidates),
        }

    def _compact_trial(self, trial, trial_idx=None):
        compact = {
            "q_start_rad": trial["q_start"],
            "q_goal_rad": trial["q_goal"],
            "q_final_rad": trial["q_final"],
            "dq_desired_rad": trial["dq_desired"],
            "dq_actual_rad": trial["dq_actual"],
            "raw_position_error_rad": trial["raw_position_error_rad"],
            "abs_position_error_rad": trial["abs_position_error_rad"],
            "aligned_error_rad": trial["aligned_error_rad"],
            "relative_error_percent": trial["relative_error_percent"],
            "S_raw_spike_seconds": trial["S_raw"],
            "S_filtered_drive_seconds": trial["S_filtered"],
            "S_delayed_drive_seconds": trial["S_delayed"],
            "S_pos_drive_seconds": trial["S_pos"],
            "S_neg_drive_seconds": trial["S_neg"],
            "S_vel_rad": trial["S_vel"],
            "decoder_gain_positive_used": trial["decoder_gain_positive_used"],
            "decoder_gain_negative_used": trial["decoder_gain_negative_used"],
            "decoder_tau_used": trial["decoder_tau_used"],
            "decoder_delay_steps_used": trial["decoder_delay_steps_used"],
            "stop_reason": trial["stop_reason"],
            "n_steps": trial["n_steps"],
        }

        if trial_idx is not None:
            compact["trial"] = int(trial_idx)

        return compact

    def _compact_fit_result(self, fit_result):
        if fit_result is None:
            return None

        return {
            "k_pos": float(fit_result["k_pos"]),
            "k_neg": float(fit_result["k_neg"]),
            "tau": float(fit_result["tau"]),
            "delay_steps": int(fit_result["delay_steps"]),
            "desired_mse": float(fit_result["desired_mse"]),
            "actual_mse": float(fit_result["actual_mse"]),
            "desired_mae": float(fit_result["desired_mae"]),
            "actual_mae": float(fit_result["actual_mae"]),
            "n_valid": int(fit_result["n_valid"]),
        }

    def save_decoder_to_config(
        self,
        region_idx=None,
        iteration_idx=None,
        phase="manual_save",
        latest_trial=None,
        fit_result=None,
        beta=None,
    ):
        """Save directly to config/calibration/velocity_calibration.json.

        This is called after every trial and after every decoder update.
        """

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        try:
            config = load_json_legacy(self.config_path)
        except Exception:
            config = {}

        joint_key = str(self.target_joint)

        if joint_key not in config:
            config[joint_key] = {}

        cfg = config[joint_key]

        cfg["joint_min"] = float(self.joint_min)
        cfg["joint_max"] = float(self.joint_max)

        cfg.update(self._current_decoder_params())

        # Legacy compatibility key
        cfg["raw_drive_velocity_gain"] = float(abs(self.decoder_gain_positive))

        cfg["decoder_model"] = (
            "d_t = exp(-dt/tau)*d_{t-1} + (1-exp(-dt/tau))*spike_t; "
            "v_cmd = k_pos*max(d_delay,0) + k_neg*max(-d_delay,0)"
        )

        cfg["calibration_settings"] = self._calibration_settings()

        cfg["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg["last_save_phase"] = phase

        cfg["calibration_progress"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "region_idx": region_idx,
            "iteration_idx": iteration_idx,
            "beta": beta,
            "current_decoder": self._current_decoder_params(),
            "fit_result": self._compact_fit_result(fit_result),
            "latest_trial": (
                self._compact_trial(latest_trial, trial_idx=iteration_idx)
                if latest_trial is not None
                else None
            ),
        }

        self._atomic_json_dump(config, self.config_path)

        rospy.loginfo(
            f"[SAVE CONFIG] phase={phase} joint={self.target_joint} "
            f"k_pos={self.decoder_gain_positive:.6e}, "
            f"k_neg={self.decoder_gain_negative:.6e}, "
            f"tau={self.decoder_tau:.3f}s, "
            f"delay={self.decoder_delay_steps} steps "
            f"-> {self.config_path}"
        )

    def _plot_dir(self, category):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "plots",
            f"joint_{self.target_joint}",
            category,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _calibration_output_dir(self, category):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "experiment_results",
                "calibration",
                f"joint_{self.target_joint}",
                category,
            )
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_joint_position(self):
        return self.publisher.current_positions

    def _get_spike_counts(self, recorders):
        spikes = self.ring_model._collect_spikes(recorders)
        return np.array([len(spike_times) for spike_times in spikes], dtype=float)

    def _joint_to_ring_index(self, joint_position, stimulus_half_width):
        return calibration_joint_to_ring_index(
            joint_position,
            self.joint_min,
            self.joint_max,
            int(self.ring_model.population_size),
            stimulus_half_width,
            self.ring_edge_margin_fraction,
        )

    def set_ring_goal(self, goal_position, stimulus_half_width=5):
        inject = self._joint_to_ring_index(goal_position, stimulus_half_width)

        self.goal_inject_idx = inject
        self.ring_model.r2._inject_bump(inject, stimulus_half_width)

        rospy.loginfo(
            f"[RING GOAL] goal={goal_position:.4f} rad -> r2 index={inject}"
        )

    def set_ring_state(self, stimulus_half_width=5):
        position = self.get_joint_position()
        current_joint_position = position[self.target_joint]

        inject = self._joint_to_ring_index(current_joint_position, stimulus_half_width)

        self.ring_state_inject_idx = inject
        self.ring_model.r1._inject_bump(inject, stimulus_half_width)

        self.r1_spikes_before = self._get_spike_counts(
            self.ring_model.r1.ring_attractor.ring_spike_recorders
        )

        rospy.loginfo(
            f"[RING STATE] current={current_joint_position:.4f} rad -> r1 index={inject}"
        )

    def run_ring_simulation(self):
        """Run NEST and return signed gain signal.

        signed_spike = right_gain_count - left_gain_count

        Positive means positive joint velocity.
        Negative means negative joint velocity.
        """

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

        self.left_count = left_count_after - left_count_before
        self.right_count = right_count_after - right_count_before

        self.r1_delta_spike_counts = r1_delta_counts
        self.state_count = float(r1_delta_counts.sum())

        signed_spike = self.right_count - self.left_count
        return signed_spike * self.spike_scale

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
            "S_pos": float(np.sum(drive_pos) * dt_s),
            "S_neg": float(np.sum(drive_neg) * dt_s),
        }

    def calibration_step(self, goal_state):
        """Run one Mode A trial.

        Uses current decoder parameters to move the robot.

        Records raw spike history so tau, delay, k_pos, and k_neg can be
        fitted offline after the batch.
        """

        dt_s = self.time_step / 1000.0
        decoder_tau = self.decoder_tau
        alpha = np.exp(-dt_s / decoder_tau)

        self.publisher.prime_command_state(force=True)

        q_start = self.get_joint_position()[self.target_joint]
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

        spike_buffer = []
        filtered_drive_buffer = []
        delayed_drive_buffer = []
        velocity_buffer = []

        time_history = []
        spike_history = []
        filtered_drive_history = []
        delayed_drive_history = []
        velocity_history = []
        joint_position_history = []
        joint_velocity_history = []
        position_error_history = []

        # Pre-fill lookahead
        for _ in range(self.lookahead):
            signed_spike = self.run_ring_simulation()

            _, integrated_drive, delayed_drive = drive_core.advance(
                signed_spike,
                self.decoder_delay_steps,
            )

            velocity_cmd = self._decode_velocity_from_drive(delayed_drive)

            spike_buffer.append(signed_spike)
            filtered_drive_buffer.append(integrated_drive)
            delayed_drive_buffer.append(delayed_drive)
            velocity_buffer.append(velocity_cmd)

        self.publisher.publish_receding_trajectory(
            self.target_joint,
            velocity_buffer,
            dt_s,
        )

        for step in range(self.max_steps):
            rospy.sleep(dt_s)

            consumed_spike = spike_buffer[0]
            consumed_filtered_drive = filtered_drive_buffer[0]
            consumed_delayed_drive = delayed_drive_buffer[0]
            consumed_velocity = velocity_buffer[0]

            current_position = self.get_joint_position()[self.target_joint]

            if self.publisher.current_velocities is None:
                current_velocity = 0.0
            else:
                current_velocity = self.publisher.current_velocities[self.target_joint]

            position_error = goal_state - current_position
            t_now = (step + 1) * dt_s

            time_history.append(t_now)
            spike_history.append(consumed_spike)
            filtered_drive_history.append(consumed_filtered_drive)
            delayed_drive_history.append(consumed_delayed_drive)
            velocity_history.append(consumed_velocity)
            joint_position_history.append(current_position)
            joint_velocity_history.append(current_velocity)
            position_error_history.append(position_error)

            rospy.loginfo(
                f"[STEP {step:03d}] "
                f"spike={consumed_spike:.1f} "
                f"drive={consumed_filtered_drive:.2f} "
                f"delayed_drive={consumed_delayed_drive:.2f} "
                f"cmd_vel={consumed_velocity:.6f} rad/s "
                f"meas_vel={current_velocity:.6f} rad/s "
                f"q={current_position:.4f} rad "
                f"goal_err={position_error:.4f} rad"
            )

            if abs(consumed_delayed_drive) < self.drive_threshold:
                consecutive_settled += 1

                if consecutive_settled >= self.n_settle:
                    stop_reason = "drive_settled"
                    rospy.loginfo(
                        f"[STEP {step:03d}] Drive settled for {self.n_settle} steps."
                    )
                    break
            else:
                consecutive_settled = 0

            signed_spike = self.run_ring_simulation()

            _, integrated_drive, delayed_drive = drive_core.advance(
                signed_spike,
                self.decoder_delay_steps,
            )

            velocity_cmd = self._decode_velocity_from_drive(delayed_drive)

            spike_buffer = spike_buffer[1:] + [signed_spike]
            filtered_drive_buffer = filtered_drive_buffer[1:] + [integrated_drive]
            delayed_drive_buffer = delayed_drive_buffer[1:] + [delayed_drive]
            velocity_buffer = velocity_buffer[1:] + [velocity_cmd]

            self.publisher.publish_receding_trajectory(
                self.target_joint,
                velocity_buffer,
                dt_s,
            )

        self.publisher.stop_joint(self.target_joint, dt_s)
        self.publisher.wait_for_settled(self.target_joint)

        q_final = self.get_joint_position()[self.target_joint]
        dq_actual = q_final - q_start

        raw_position_error_rad = goal_state - q_final
        abs_position_error_rad = abs(raw_position_error_rad)
        aligned_error_rad = raw_position_error_rad * movement_direction

        relative_error_ratio = aligned_error_rad / (abs(dq_desired) + 1e-9)
        relative_error_percent = 100.0 * relative_error_ratio

        S_raw = float(np.sum(spike_history) * dt_s)
        S_filtered = float(np.sum(filtered_drive_history) * dt_s)
        S_delayed = float(np.sum(delayed_drive_history) * dt_s)
        S_vel = float(np.sum(velocity_history) * dt_s)

        features = self._decoder_features_from_spikes(
            spike_history,
            dt_s,
            self.decoder_tau,
            self.decoder_delay_steps,
        )

        trial = {
            "q_start": q_start,
            "q_goal": goal_state,
            "q_final": q_final,

            "dq_desired": dq_desired,
            "dq_actual": dq_actual,

            "raw_position_error_rad": raw_position_error_rad,
            "abs_position_error_rad": abs_position_error_rad,
            "aligned_error_rad": aligned_error_rad,
            "relative_error_ratio": relative_error_ratio,
            "relative_error_percent": relative_error_percent,

            "time_history": time_history,
            "spike_history": spike_history,
            "filtered_drive_history": filtered_drive_history,
            "delayed_drive_history": delayed_drive_history,
            "velocity_history": velocity_history,
            "joint_position_history": joint_position_history,
            "joint_velocity_history": joint_velocity_history,
            "position_error_history": position_error_history,

            "dt": dt_s,

            "S_raw": S_raw,
            "S_filtered": S_filtered,
            "S_delayed": S_delayed,
            "S_pos": features["S_pos"],
            "S_neg": features["S_neg"],
            "S_vel": S_vel,

            "decoder_gain_positive_used": self.decoder_gain_positive,
            "decoder_gain_negative_used": self.decoder_gain_negative,
            "decoder_tau_used": self.decoder_tau,
            "decoder_delay_steps_used": self.decoder_delay_steps,

            "stop_reason": stop_reason,
            "n_steps": len(spike_history),
        }

        rospy.loginfo(
            f"[TRIAL] start={q_start:.4f} rad "
            f"goal={goal_state:.4f} rad "
            f"final={q_final:.4f} rad "
            f"dq_desired={dq_desired:.4f} rad "
            f"dq_actual={dq_actual:.4f} rad "
            f"S_pos={trial['S_pos']:.4f} "
            f"S_neg={trial['S_neg']:.4f} "
            f"S_vel={S_vel:.4f} rad "
            f"aligned_error={aligned_error_rad:.4f} rad "
            f"relative_error={relative_error_percent:.2f}% "
            f"stop={stop_reason}"
        )

        return trial

    def _fit_signed_direction_gains(self, X, y):
        """Fit:

            y ≈ k_pos*S_pos + k_neg*S_neg

        S_pos >= 0
        S_neg >= 0

        Expected:
            k_pos > 0
            k_neg < 0
        """

        return fit_signed_direction_gains(X, y)

    def fit_decoder_from_trials(self, trials):
        """Grid-search tau and delay, then fit k_pos/k_neg.

        For each tau and delay:
            1. Recompute filtered drive from raw spike_history.
            2. Apply candidate delay.
            3. Compute S_pos and S_neg.
            4. Fit:
                   dq_desired = k_pos*S_pos + k_neg*S_neg
            5. Choose lowest desired-displacement MSE.
        """

        fit = fit_decoder_calibration(
            trials,
            tau_candidates=self.tau_candidates,
            delay_candidates=self.delay_candidates,
            feature_extractor=self._decoder_features_from_spikes,
            gain_fitter=self._fit_signed_direction_gains,
        )
        best = None if fit is None else fit.as_legacy_dict()

        if best is None:
            rospy.logwarn("[DECODER FIT] No valid decoder fit found.")
            return None

        rospy.loginfo(
            f"[DECODER FIT] best tau={best['tau']:.3f}s "
            f"delay={best['delay_steps']} steps "
            f"k_pos={best['k_pos']:.6e} "
            f"k_neg={best['k_neg']:.6e} "
            f"desired_mae={best['desired_mae']:.6f} rad "
            f"actual_mae={best['actual_mae']:.6f} rad "
            f"n_valid={best['n_valid']}"
        )

        if best["k_pos"] < 0.0:
            rospy.logwarn(
                "[DECODER FIT] k_pos is negative. "
                "Positive drive may be inverted relative to joint direction."
            )

        if best["k_neg"] > 0.0:
            rospy.logwarn(
                "[DECODER FIT] k_neg is positive. "
                "Negative drive may be inverted relative to joint direction."
            )

        return best

    def update_decoder_from_fit(self, fit_result, beta=0.2):
        if fit_result is None:
            return False

        old_k_pos = self.decoder_gain_positive
        old_k_neg = self.decoder_gain_negative
        old_tau = self.decoder_tau
        old_delay = self.decoder_delay_steps

        self.decoder_gain_positive = (
            (1.0 - beta) * old_k_pos + beta * fit_result["k_pos"]
        )
        self.decoder_gain_negative = (
            (1.0 - beta) * old_k_neg + beta * fit_result["k_neg"]
        )
        self.decoder_tau = (1.0 - beta) * old_tau + beta * fit_result["tau"]

        # Delay is discrete. Use best candidate directly.
        self.decoder_delay_steps = int(fit_result["delay_steps"])

        rospy.loginfo(
            "[DECODER UPDATE] "
            f"k_pos: {old_k_pos:.6e} -> {self.decoder_gain_positive:.6e}, "
            f"k_neg: {old_k_neg:.6e} -> {self.decoder_gain_negative:.6e}, "
            f"tau: {old_tau:.3f}s -> {self.decoder_tau:.3f}s, "
            f"delay: {old_delay} -> {self.decoder_delay_steps} steps, "
            f"beta={beta}"
        )

        return True

    def save_trial_summary(self, trials, region_idx, fit_result=None):
        output_dir = self._calibration_output_dir("summaries")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"decoder_trials_joint{self.target_joint}_"
            f"region{region_idx}_{timestamp}.json"
        )
        path = os.path.join(output_dir, filename)

        summary = {
            "joint": self.target_joint,
            "joint_min_rad": self.joint_min,
            "joint_max_rad": self.joint_max,
            "decoder_after_region": self._current_decoder_params(),
            "calibration_settings": self._calibration_settings(),
            "fit_result": self._compact_fit_result(fit_result),
            "trials": [
                self._compact_trial(t, trial_idx=i + 1)
                for i, t in enumerate(trials)
            ],
        }

        self._atomic_json_dump(summary, path)

        rospy.loginfo(f"[SAVE] Trial summary → {path}")

    def plot_trial_diagnostics(self, trial, region_idx, iteration):
        output_dir = self._plot_dir("trial_diagnostics")

        t = np.array(trial["time_history"])
        q = np.array(trial["joint_position_history"])
        q_goal = trial["q_goal"]

        filtered_drive = np.array(trial["filtered_drive_history"])
        delayed_drive = np.array(trial["delayed_drive_history"])

        v_cmd = np.array(trial["velocity_history"])
        v_meas = np.array(trial["joint_velocity_history"])
        e_pos = np.array(trial["position_error_history"])

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        axes[0].plot(t, q, label="measured joint position q(t)")
        axes[0].axhline(q_goal, linestyle="--", label="goal position")
        axes[0].set_ylabel("Joint position q (rad)")
        axes[0].set_title(
            f"Trial diagnostics | Joint {self.target_joint} | "
            f"Region {region_idx}, Iteration {iteration}"
        )
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, filtered_drive, label="filtered drive d(t)")
        axes[1].plot(t, delayed_drive, label="delayed drive d(t-delay)")
        axes[1].axhline(0.0, linestyle="--", linewidth=1)
        axes[1].set_ylabel("Drive\n(spike-diff units)")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, v_cmd, label="commanded velocity v_cmd")
        axes[2].plot(t, v_meas, label="measured joint velocity")
        axes[2].axhline(0.0, linestyle="--", linewidth=1)
        axes[2].set_ylabel("Velocity (rad/s)")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t, e_pos, label="goal - joint position")
        axes[3].axhline(0.0, linestyle="--", linewidth=1, label="zero error")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Position error (rad)")
        axes[3].legend(loc="best")
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()

        plot_path = os.path.join(
            output_dir,
            f"trial_diag_joint{self.target_joint}_region{region_idx}_iter{iteration}.png",
        )

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Trial diagnostics saved: {plot_path}")

    def plot_region_summary(self, region_trials, region_idx, fit_result=None):
        output_dir = self._plot_dir("decoder_calibration_summary")

        trials = np.arange(1, len(region_trials) + 1)

        aligned_error_rad = np.array([t["aligned_error_rad"] for t in region_trials])
        abs_error_rad = np.array([t["abs_position_error_rad"] for t in region_trials])
        relative_error_percent = np.array(
            [t["relative_error_percent"] for t in region_trials]
        )

        # Error summary
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

        axes[0].plot(trials, aligned_error_rad, marker="o")
        axes[0].axhline(0.0, linestyle="--", linewidth=1)
        axes[0].set_ylabel("Aligned final error (rad)")
        axes[0].set_title(
            f"Region {region_idx} final error | "
            "positive = undershoot, negative = overshoot"
        )
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(trials, abs_error_rad, marker="o")
        axes[1].set_ylabel("|goal - final| (rad)")
        axes[1].set_title("Absolute final position error")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(trials, relative_error_percent, marker="o")
        axes[2].axhline(0.0, linestyle="--", linewidth=1)
        axes[2].set_xlabel("Trial")
        axes[2].set_ylabel("Aligned error (% of requested movement)")
        axes[2].set_title(
            "Relative final error: + undershoot, - overshoot, 0 target reached"
        )
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()

        error_plot_path = os.path.join(
            output_dir,
            f"region_error_joint{self.target_joint}_region{region_idx}.png",
        )

        fig.savefig(error_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Region error summary saved: {error_plot_path}")

        if fit_result is None:
            return

        y_desired = fit_result["y_desired"]
        y_actual = fit_result["y_actual"]
        y_pred = fit_result["y_pred"]

        fig, axes = plt.subplots(2, 1, figsize=(11, 10))

        axes[0].scatter(
            y_desired,
            y_pred,
            marker="o",
            label="desired vs decoder prediction",
        )
        min_y = min(float(np.min(y_desired)), float(np.min(y_pred)))
        max_y = max(float(np.max(y_desired)), float(np.max(y_pred)))
        axes[0].plot([min_y, max_y], [min_y, max_y], linestyle="--", label="perfect")
        axes[0].set_xlabel("Desired displacement dq_desired (rad)")
        axes[0].set_ylabel("Decoder-predicted displacement (rad)")
        axes[0].set_title(
            f"Decoder fit to desired displacement | "
            f"tau={fit_result['tau']:.3f}s, "
            f"delay={fit_result['delay_steps']} steps"
        )
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(
            y_actual,
            y_pred,
            marker="x",
            label="actual vs decoder prediction",
        )
        min_y = min(float(np.min(y_actual)), float(np.min(y_pred)))
        max_y = max(float(np.max(y_actual)), float(np.max(y_pred)))
        axes[1].plot([min_y, max_y], [min_y, max_y], linestyle="--", label="perfect")
        axes[1].set_xlabel("Actual displacement dq_actual (rad)")
        axes[1].set_ylabel("Decoder-predicted displacement (rad)")
        axes[1].set_title(
            "Diagnostic: whether robot execution matched decoder prediction"
        )
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()

        fit_plot_path = os.path.join(
            output_dir,
            f"decoder_fit_joint{self.target_joint}_region{region_idx}.png",
        )

        fig.savefig(fit_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rospy.loginfo(f"[PLOT] Decoder fit summary saved: {fit_plot_path}")

    def plot_ring_attractor_raster(self, iteration):
        spike_times_list = self.ring_model._collect_spikes(
            self.ring_model.r1.ring_attractor.ring_spike_recorders
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        for neuron_idx, spike_times in enumerate(spike_times_list):
            if len(spike_times) > 0:
                ax.vlines(
                    spike_times,
                    neuron_idx - 0.4,
                    neuron_idx + 0.4,
                    colors="black",
                    linewidth=0.5,
                )

        valid_times = [st for st in spike_times_list if len(st) > 0]

        if valid_times:
            all_spike_times = np.concatenate(valid_times)
            time_max = np.max(all_spike_times)
        else:
            time_max = self.time_step

        arrow_length = max(time_max * 0.1, 1.0)

        if self.goal_inject_idx is not None:
            ax.arrow(
                arrow_length * 0.2,
                self.goal_inject_idx,
                arrow_length,
                0,
                head_width=2,
                head_length=arrow_length * 0.3,
                fc="green",
                ec="green",
                linewidth=2,
                label="Goal ring position",
            )

        if self.ring_state_inject_idx is not None:
            ax.arrow(
                arrow_length * 0.2,
                self.ring_state_inject_idx,
                arrow_length,
                0,
                head_width=2,
                head_length=arrow_length * 0.3,
                fc="blue",
                ec="blue",
                linewidth=2,
                label="Initial state ring position",
            )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Ring neuron index")
        ax.set_title(f"Ring Attractor Raster Plot - Iteration {iteration}")
        ax.set_ylim(-1, self.ring_model.population_size)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        output_dir = self._plot_dir("raster")

        plot_path = os.path.join(
            output_dir,
            f"ring_raster_joint{self.target_joint}_iter{iteration}.png",
        )

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

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
    # Main calibration loop
    # ------------------------------------------------------------------

    def calibration(self, max_iterations=10, num_regions=10, beta=0.3, fit_every=5):
        """Run decoder-parameter calibration with mini-batch updates.

        Learns:
            k_pos
            k_neg
            tau
            delay_steps

        Difference from old version:
            Old: update decoder once after all iterations in a region.
            New: update decoder every `fit_every` trials.

        This should converge faster because the controller improves while the
        experiment is still running.
        """

        rospy.loginfo(
            f"Joint {self.target_joint} "
            f"limits=[{self.joint_min:.4f}, {self.joint_max:.4f}] "
            f"k_pos={self.decoder_gain_positive:.3e} "
            f"k_neg={self.decoder_gain_negative:.3e} "
            f"tau={self.decoder_tau:.3f}s "
            f"delay={self.decoder_delay_steps} "
            f"beta={beta} "
            f"fit_every={fit_every}"
        )

        self.save_decoder_to_config(
            phase="calibration_start",
            beta=beta,
        )

        region_size = (self.joint_max - self.joint_min) / num_regions

        for region_idx in range(1, num_regions + 1):
            region_min = self.joint_min + (region_idx - 1) * region_size
            region_max = self.joint_min + region_idx * region_size

            self.iteration_errors = []
            region_trials = []
            batch_trials = []
            last_fit_result = None

            rospy.loginfo(
                f"[REGION START] Region {region_idx}/{num_regions} "
                f"range=[{region_min:.4f}, {region_max:.4f}] rad"
            )

            for iteration in range(max_iterations):
                goal_state = np.random.uniform(region_min, region_max)
                self.goal_state = goal_state

                current_pos = self.get_joint_position()[self.target_joint]

                rospy.loginfo(
                    f"[Region {region_idx}/{num_regions} | "
                    f"Iter {iteration + 1}/{max_iterations}] "
                    f"current={current_pos:.4f} rad "
                    f"goal={goal_state:.4f} rad "
                    f"k_pos={self.decoder_gain_positive:.3e} "
                    f"k_neg={self.decoder_gain_negative:.3e} "
                    f"tau={self.decoder_tau:.3f}s "
                    f"delay={self.decoder_delay_steps}"
                )

                self.publisher.reset_pose()
                self.publisher.prime_command_state(force=True)

                trial = self.calibration_step(goal_state)

                region_trials.append(trial)
                batch_trials.append(trial)
                self.iteration_errors.append(trial["relative_error_percent"])

                final_pos = self.get_joint_position()[self.target_joint]

                rospy.loginfo(
                    f"  → final={final_pos:.4f} rad "
                    f"aligned_error={trial['aligned_error_rad']:.4f} rad "
                    f"rel_error={trial['relative_error_percent']:.2f}%"
                )

                self.save_decoder_to_config(
                    region_idx=region_idx,
                    iteration_idx=iteration + 1,
                    phase="after_trial",
                    latest_trial=trial,
                    fit_result=None,
                    beta=beta,
                )

                self.plot_trial_diagnostics(trial, region_idx, iteration + 1)
                self.plot_ring_attractor_raster(iteration + 1)

                should_fit_now = (
                    len(batch_trials) >= fit_every
                    or iteration == max_iterations - 1
                )

                if should_fit_now and len(batch_trials) >= 2:
                    rospy.loginfo(
                        f"[MINI-BATCH FIT] Region {region_idx}, "
                        f"iteration {iteration + 1}: fitting decoder from "
                        f"{len(batch_trials)} recent trials"
                    )

                    fit_result = self.fit_decoder_from_trials(batch_trials)

                    if fit_result is not None:
                        old_params = self._current_decoder_params()

                        self.update_decoder_from_fit(fit_result, beta=beta)

                        last_fit_result = fit_result

                        rospy.loginfo(
                            f"[MINI-BATCH UPDATE] "
                            f"old_k_pos={old_params['decoder_gain_positive']:.6e}, "
                            f"old_k_neg={old_params['decoder_gain_negative']:.6e}, "
                            f"old_tau={old_params['decoder_tau']:.3f}, "
                            f"old_delay={old_params['decoder_delay_steps']} | "
                            f"new_k_pos={self.decoder_gain_positive:.6e}, "
                            f"new_k_neg={self.decoder_gain_negative:.6e}, "
                            f"new_tau={self.decoder_tau:.3f}, "
                            f"new_delay={self.decoder_delay_steps}"
                        )

                        self.save_decoder_to_config(
                            region_idx=region_idx,
                            iteration_idx=iteration + 1,
                            phase="mini_batch_decoder_update",
                            latest_trial=trial,
                            fit_result=fit_result,
                            beta=beta,
                        )

                    else:
                        rospy.logwarn(
                            f"[MINI-BATCH FIT FAILED] Region {region_idx}, "
                            f"iteration {iteration + 1}. Decoder unchanged."
                        )

                        self.save_decoder_to_config(
                            region_idx=region_idx,
                            iteration_idx=iteration + 1,
                            phase="mini_batch_fit_failed",
                            latest_trial=trial,
                            fit_result=None,
                            beta=beta,
                        )

                    batch_trials = []

                self.ring_model = SingleRingModel()

            if last_fit_result is None and len(region_trials) >= 2:
                rospy.loginfo(
                    f"[REGION FINAL FIT] Region {region_idx}: "
                    "no successful mini-batch fit yet, trying full-region fit"
                )

                last_fit_result = self.fit_decoder_from_trials(region_trials)

                if last_fit_result is not None:
                    self.update_decoder_from_fit(last_fit_result, beta=beta)

                    self.save_decoder_to_config(
                        region_idx=region_idx,
                        iteration_idx=max_iterations,
                        phase="after_region_fit",
                        latest_trial=region_trials[-1],
                        fit_result=last_fit_result,
                        beta=beta,
                    )
                else:
                    self.save_decoder_to_config(
                        region_idx=region_idx,
                        iteration_idx=max_iterations,
                        phase="after_region_fit_failed",
                        latest_trial=region_trials[-1] if region_trials else None,
                        fit_result=None,
                        beta=beta,
                    )

            n_max_step = sum(
                1 for t in region_trials if t["stop_reason"] == "max_steps"
            )

            abs_errors = np.array(
                [t["abs_position_error_rad"] for t in region_trials],
                dtype=float,
            )
            aligned_errors = np.array(
                [t["aligned_error_rad"] for t in region_trials],
                dtype=float,
            )
            rel_errors = np.array(
                [t["relative_error_percent"] for t in region_trials],
                dtype=float,
            )

            rospy.loginfo(
                f"[REGION SUMMARY] Joint {self.target_joint} | Region {region_idx}/{num_regions} "
                f"trials={len(region_trials)} "
                f"max_steps={n_max_step}/{len(region_trials)} "
                f"mean_abs_error={np.mean(abs_errors):.6f} rad "
                f"std_abs_error={np.std(abs_errors):.6f} rad "
                f"mean_aligned_error={np.mean(aligned_errors):.6f} rad "
                f"std_aligned_error={np.std(aligned_errors):.6f} rad "
                f"mean_rel_error={np.mean(rel_errors):.2f}% "
                f"std_rel_error={np.std(rel_errors):.2f}%"
            )

            self.save_trial_summary(region_trials, region_idx, last_fit_result)
            self.plot_region_summary(region_trials, region_idx, last_fit_result)

            rospy.loginfo(f"Region {region_idx}/{num_regions} done.")

        self.save_decoder_to_config(
            phase="calibration_complete",
            beta=beta,
        )

        rospy.loginfo(
            f"Calibration complete. "
            f"k_pos={self.decoder_gain_positive:.6e}, "
            f"k_neg={self.decoder_gain_negative:.6e}, "
            f"tau={self.decoder_tau:.3f}s, "
            f"delay={self.decoder_delay_steps} steps"
        )

        return {
            "k_pos": self.decoder_gain_positive,
            "k_neg": self.decoder_gain_negative,
            "tau": self.decoder_tau,
            "delay_steps": self.decoder_delay_steps,
        }


def main():
    rospy.init_node("tiago_calibration", anonymous=True)

    # All Tiago arm joints (arm_1_joint to arm_7_joint -> indices 0-6)
    arm_joints = [0, 2, 3, 5]

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "calibration",
        "velocity_calibration.json",
    )

    for target_joint in arm_joints:
        rospy.loginfo(
            f"[CALIBRATION] Starting calibration for joint {target_joint} "
            f"({arm_joints.index(target_joint) + 1}/{len(arm_joints)})"
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

        calibration = TiagoCalibration(
            target_joint,
            use_manual_limits=use_manual_limits,
        )

        calibration.calibration(
            max_iterations=100,
            num_regions=1,
            beta=0.5,
            fit_every=3,
        )

        rospy.loginfo(
            f"[CALIBRATION] Joint {target_joint} done "
            f"({arm_joints.index(target_joint) + 1}/{len(arm_joints)})"
        )


if __name__ == "__main__":
    main()
