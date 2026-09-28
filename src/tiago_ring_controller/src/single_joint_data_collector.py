#!/usr/bin/env python3

import csv
import json
import os

import nest
import numpy as np
import rospy

from single_ring import SingleRingModel
from tiago_controller import TiagoPublisher, TiagoSubscriber
from tiago_ring_controller.artifacts import (
    load_json_legacy,
    save_json_legacy,
    save_npz_compressed_legacy,
    save_numpy_legacy,
)
from tiago_ring_controller.control.controller import (
    DecoderParameters,
    DriveControlCore,
)
from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE
from tiago_ring_controller.config import module_config_path
from tiago_ring_controller.math.control import (
    apply_delay,
    decode_velocity,
    exponential_filter,
)
from tiago_ring_controller.evaluation.metrics import collector_batch_statistics
from tiago_ring_controller.evaluation.serialization import (
    RING_BATCH_STAT_FIELDS,
    RING_TRIAL_SCALAR_FIELDS,
)


class SingleJointDataCollector:
    def __init__(self, target_joint, use_manual_limits=False):
        self.target_joint = target_joint

        self.publisher = TiagoPublisher()
        self.subscriber = TiagoSubscriber()
        self.publisher.wait_for_joint_state(timeout=5.0)

        self.ring_model = SingleRingModel()
        self.control_profile = COLLECTOR_PROFILE

        self.time_step = self.control_profile.time_step_ms
        self.decoder_gain_positive = 1e-4
        self.decoder_gain_negative = -1e-4
        self.decoder_tau = 0.3
        self.decoder_delay_steps = 0
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

        if not self._load_limits_from_config() or use_manual_limits:
            self.joint_min, self.joint_max = self._find_joint_limits_manually()
        else:
            rospy.loginfo(
                f"Joint {self.target_joint} limits: "
                f"[{self.joint_min:.4f}, {self.joint_max:.4f}] rad"
            )

        self._load_decoder_from_config()

    def _load_limits_from_config(self):
        try:
            config = load_json_legacy(self.config_path)
            key = str(self.target_joint)
            if key in config and "joint_min" in config[key] and "joint_max" in config[key]:
                self.joint_min = float(config[key]["joint_min"])
                self.joint_max = float(config[key]["joint_max"])
                return True
        except Exception:
            pass
        return False

    def _load_decoder_from_config(self):
        try:
            config = load_json_legacy(self.config_path)
            key = str(self.target_joint)
            if key not in config:
                return
            cfg = config[key]
            required = ["decoder_gain_positive", "decoder_gain_negative", "decoder_tau", "decoder_delay_steps"]
            if all(k in cfg for k in required):
                self.decoder_gain_positive = float(cfg["decoder_gain_positive"])
                self.decoder_gain_negative = float(cfg["decoder_gain_negative"])
                self.decoder_tau = float(cfg["decoder_tau"])
                self.decoder_delay_steps = int(cfg["decoder_delay_steps"])
            elif "raw_drive_velocity_gain" in cfg:
                gain = float(cfg["raw_drive_velocity_gain"])
                self.decoder_gain_positive = gain
                self.decoder_gain_negative = -gain
        except Exception as e:
            rospy.logwarn(f"[DECODER LOAD] {e}")

    def _output_dir(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "collected_data",
            f"joint_{self.target_joint}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _joint_to_ring_index(self, position, half_width):
        return self.control_profile.joint_to_ring_index(
            position,
            self.joint_min,
            self.joint_max,
            self.ring_model.population_size,
            requested_half_width=half_width,
        )

    def _get_spike_counts(self, recorders):
        spikes = self.ring_model._collect_spikes(recorders)
        return np.array([len(s) for s in spikes], dtype=float)

    def set_ring_goal(self, goal_position, half_width=5):
        idx = self._joint_to_ring_index(goal_position, half_width)
        self.goal_inject_idx = idx
        self.ring_model.r2._inject_bump(idx, half_width)

    def set_ring_state(self, half_width=5):
        pos = self.publisher.current_positions[self.target_joint]
        idx = self._joint_to_ring_index(pos, half_width)
        self.ring_state_inject_idx = idx
        self.ring_model.r1._inject_bump(idx, half_width)

    def get_ring_state(self):
        if self.r1_delta_spike_counts is None:
            return None
        if self.r1_delta_spike_counts.sum() == 0:
            return None
        return int(np.argmax(self.r1_delta_spike_counts))

    def _compute_centroid(self):
        if self.r1_delta_spike_counts is None:
            return np.nan
        total = self.r1_delta_spike_counts.sum()
        if total == 0:
            return np.nan
        indices = np.arange(len(self.r1_delta_spike_counts), dtype=float)
        return float(np.dot(indices, self.r1_delta_spike_counts) / total)

    def run_ring_simulation(self):
        def total_spikes(recorders):
            return sum(len(s) for s in self.ring_model._collect_spikes(recorders))

        left_before = total_spikes(self.ring_model.gain_modulation.left_gain_spike_recorders)
        right_before = total_spikes(self.ring_model.gain_modulation.right_gain_spike_recorders)
        r1_before = self._get_spike_counts(self.ring_model.r1.ring_attractor.ring_spike_recorders)

        nest.Simulate(self.time_step)

        left_after = total_spikes(self.ring_model.gain_modulation.left_gain_spike_recorders)
        right_after = total_spikes(self.ring_model.gain_modulation.right_gain_spike_recorders)
        r1_after = self._get_spike_counts(self.ring_model.r1.ring_attractor.ring_spike_recorders)

        self.left_count = int(left_after - left_before)
        self.right_count = int(right_after - right_before)
        self.r1_delta_spike_counts = np.maximum(r1_after - r1_before, 0.0)
        self.state_count = float(self.r1_delta_spike_counts.sum())

        return self.right_count - self.left_count

    def _decode_velocity(self, delayed_drive):
        return decode_velocity(
            delayed_drive,
            self.decoder_gain_positive,
            self.decoder_gain_negative,
        )

    def _filter_spikes(self, spike_history, dt_s, tau):
        return exponential_filter(spike_history, dt_s, tau)

    def _apply_delay(self, drive_history, delay_steps):
        return apply_delay(drive_history, delay_steps)

    def _collect_raster_data(self):
        def get_events(recorders):
            all_times, all_senders = [], []
            for rec in recorders:
                try:
                    ev = nest.GetStatus(rec, "events")[0]
                    all_times.extend(ev.get("times", []))
                    all_senders.extend(ev.get("senders", []))
                except Exception:
                    pass
            return np.array(all_times, dtype=float), np.array(all_senders, dtype=int)

        r1_times, r1_senders = get_events(self.ring_model.r1.ring_attractor.ring_spike_recorders)
        r2_times, r2_senders = get_events(self.ring_model.r2.ring_attractor.ring_spike_recorders)
        left_times, left_senders = get_events(self.ring_model.gain_modulation.left_gain_spike_recorders)
        right_times, right_senders = get_events(self.ring_model.gain_modulation.right_gain_spike_recorders)

        return {
            "r1_times": r1_times, "r1_senders": r1_senders,
            "r2_times": r2_times, "r2_senders": r2_senders,
            "left_times": left_times, "left_senders": left_senders,
            "right_times": right_times, "right_senders": right_senders,
        }

    def run_trial(self, goal_state):
        dt_s = self.time_step / 1000.0
        decoder_tau = self.decoder_tau
        alpha = np.exp(-dt_s / decoder_tau)

        self.publisher.prime_command_state(force=True)
        q_start = float(self.publisher.current_positions[self.target_joint])
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

        time_hist, pos_hist, vel_hist = [], [], []
        signed_spike_hist, left_hist, right_hist, r1_hist, bump_hist, centroid_hist = [], [], [], [], [], []
        filtered_hist, delayed_hist, vcmd_hist, err_hist = [], [], [], []

        sample_buffer = []

        for _ in range(self.lookahead):
            sig = self.run_ring_simulation()
            _, integrated_drive, dd = drive_core.advance(
                sig,
                self.decoder_delay_steps,
            )
            sample_buffer.append({
                "sig": float(sig), "left": int(self.left_count), "right": int(self.right_count),
                "r1": float(self.state_count), "bump": self.get_ring_state(),
                "centroid": self._compute_centroid(),
                "filtered": float(integrated_drive), "delayed": float(dd),
                "vcmd": float(self._decode_velocity(dd)),
            })

        self.publisher.publish_receding_trajectory(
            self.target_joint, [s["vcmd"] for s in sample_buffer], dt_s)

        for step in range(self.max_steps):
            rospy.sleep(dt_s)
            s = sample_buffer[0]
            cur_pos = float(self.publisher.current_positions[self.target_joint])
            cur_vel = float((self.publisher.current_velocities or [0.0] * 7)[self.target_joint])

            time_hist.append(float((step + 1) * dt_s))
            pos_hist.append(cur_pos)
            vel_hist.append(cur_vel)
            signed_spike_hist.append(s["sig"])
            left_hist.append(s["left"])
            right_hist.append(s["right"])
            r1_hist.append(s["r1"])
            bump_hist.append(s["bump"] if s["bump"] is not None else -1)
            centroid_hist.append(s["centroid"])
            filtered_hist.append(s["filtered"])
            delayed_hist.append(s["delayed"])
            vcmd_hist.append(s["vcmd"])
            err_hist.append(float(goal_state - cur_pos))

            rospy.loginfo(
                f"[{step:03d}] sig={s['sig']:.1f} L={s['left']} R={s['right']} "
                f"drv={s['filtered']:.2f} d_drv={s['delayed']:.2f} "
                f"vcmd={s['vcmd']:.6f} vmeas={cur_vel:.6f} "
                f"q={cur_pos:.4f} err={goal_state - cur_pos:.4f}"
            )

            if abs(s["delayed"]) < self.drive_threshold:
                consecutive_settled += 1
                if consecutive_settled >= self.n_settle:
                    stop_reason = "drive_settled"
                    break
            else:
                consecutive_settled = 0

            sig = self.run_ring_simulation()
            _, integrated_drive, dd = drive_core.advance(
                sig,
                self.decoder_delay_steps,
            )

            sample_buffer = sample_buffer[1:] + [{
                "sig": float(sig), "left": int(self.left_count), "right": int(self.right_count),
                "r1": float(self.state_count), "bump": self.get_ring_state(),
                "centroid": self._compute_centroid(),
                "filtered": float(integrated_drive), "delayed": float(dd),
                "vcmd": float(self._decode_velocity(dd)),
            }]
            self.publisher.publish_receding_trajectory(
                self.target_joint, [s["vcmd"] for s in sample_buffer], dt_s)

        self.publisher.stop_joint(self.target_joint, dt_s)
        self.publisher.wait_for_settled(self.target_joint)

        q_final = float(self.publisher.current_positions[self.target_joint])
        raw_err = goal_state - q_final

        filtered = self._filter_spikes(signed_spike_hist, dt_s, self.decoder_tau)
        delayed = self._apply_delay(filtered, self.decoder_delay_steps)
        S_pos = float(np.sum(np.maximum(delayed, 0)) * dt_s)
        S_neg = float(np.sum(np.maximum(-delayed, 0)) * dt_s)
        dq_pred = self.decoder_gain_positive * S_pos + self.decoder_gain_negative * S_neg

        raster = self._collect_raster_data()

        scalars = {
            "q_start": q_start, "q_goal": float(goal_state), "q_final": q_final,
            "dq_desired": float(dq_desired), "dq_actual": float(q_final - q_start),
            "decoder_predicted_dq_rad": float(dq_pred),
            "raw_position_error_rad": float(raw_err),
            "abs_position_error_rad": float(abs(raw_err)),
            "aligned_error_rad": float(raw_err * movement_direction),
            "S_raw": float(np.sum(signed_spike_hist) * dt_s),
            "S_filtered": float(np.sum(filtered_hist) * dt_s),
            "S_delayed": float(np.sum(delayed_hist) * dt_s),
            "S_pos": S_pos, "S_neg": S_neg,
            "S_vel": float(np.sum(vcmd_hist) * dt_s),
            "n_steps": len(time_hist),
            "stop_reason": stop_reason,
            "goal_ring_index": self.goal_inject_idx,
            "initial_ring_index": self.ring_state_inject_idx,
            "decoder_gain_positive": self.decoder_gain_positive,
            "decoder_gain_negative": self.decoder_gain_negative,
            "decoder_tau": self.decoder_tau,
            "decoder_delay_steps": self.decoder_delay_steps,
        }

        timeseries = {
            "time": np.array(time_hist),
            "joint_position": np.array(pos_hist),
            "joint_velocity": np.array(vel_hist),
            "signed_spike": np.array(signed_spike_hist),
            "left_gain_spikes": np.array(left_hist),
            "right_gain_spikes": np.array(right_hist),
            "r1_spike_count": np.array(r1_hist),
            "r1_bump_index": np.array(bump_hist, dtype=float),
            "r1_centroid": np.array(centroid_hist, dtype=float),
            "filtered_drive": np.array(filtered_hist),
            "delayed_drive": np.array(delayed_hist),
            "decoded_velocity": np.array(vcmd_hist),
            "position_error": np.array(err_hist),
        }

        return {"scalars": scalars, "timeseries": timeseries, "raster": raster}

    def _compute_batch_stats(self, trials):
        return collector_batch_statistics(trials)

    def reset_ring_model(self):
        try:
            nest.ResetKernel()
        except Exception as e:
            rospy.logwarn(f"[RING RESET] {e}")
        self.ring_model = SingleRingModel()
        self.goal_inject_idx = None
        self.ring_state_inject_idx = None
        self.r1_delta_spike_counts = None

    def _find_joint_limits_manually(self):
        velocity = 1.0
        timeout = 5.0
        self.publisher.publish_velocity(self.target_joint, -velocity, timeout)
        rospy.sleep(timeout)
        joint_min = float(self.publisher.current_positions[self.target_joint])
        self.publisher.publish_velocity(self.target_joint, velocity, timeout)
        rospy.sleep(timeout)
        joint_max = float(self.publisher.current_positions[self.target_joint])
        self.publisher.publish_velocity(self.target_joint, 0.0, timeout)
        rospy.loginfo(f"Joint {self.target_joint} limits: [{joint_min:.4f}, {joint_max:.4f}] rad")
        return joint_min, joint_max

    def run(self, num_iterations=100, num_batches=5):
        out = self._output_dir()
        trials_dir = os.path.join(out, "trials")
        batches_dir = os.path.join(out, "batches")
        os.makedirs(trials_dir, exist_ok=True)
        os.makedirs(batches_dir, exist_ok=True)
        batch_size = max(1, num_iterations // num_batches)
        total_regions = num_batches + 2
        region_edges = np.linspace(self.joint_min, self.joint_max, total_regions + 1)

        scalar_fields = RING_TRIAL_SCALAR_FIELDS

        batch_stat_fields = RING_BATCH_STAT_FIELDS

        batch_summaries = []
        cumulative_time_ms = 0.0
        current_batch_trials = []

        meta = {
            "joint_index": int(self.target_joint),
            "joint_min": float(self.joint_min),
            "joint_max": float(self.joint_max),
            "num_iterations": int(num_iterations),
            "num_batches": int(num_batches),
            "batch_size": int(batch_size),
            "total_goal_regions": int(total_regions),
            "used_goal_region_ids": [int(i) for i in range(1, total_regions - 1)],
            "skipped_goal_region_ids": [0, int(total_regions - 1)],
        }
        save_json_legacy(os.path.join(out, "session_meta.json"), meta)

        with open(os.path.join(out, "trials_summary.csv"), "w", newline="") as trials_f:
            writer = csv.DictWriter(trials_f, fieldnames=scalar_fields)
            writer.writeheader()

            for i in range(1, num_iterations + 1):
                batch_idx = min((i - 1) // batch_size + 1, num_batches)
                region_id = batch_idx
                goal_lo = float(region_edges[region_id])
                goal_hi = float(region_edges[region_id + 1])
                goal = np.random.uniform(goal_lo, goal_hi)
                cur = float(self.publisher.current_positions[self.target_joint])

                rospy.loginfo(
                    f"[{i}/{num_iterations} | batch {batch_idx}/{num_batches}] "
                    f"region={region_id}/{total_regions - 1} "
                    f"current={cur:.4f} goal={goal:.4f} rad"
                )

                self.publisher.reset_pose()
                self.publisher.prime_command_state(force=True)

                trial = self.run_trial(goal)

                row = {
                    "joint_index": int(self.target_joint),
                    "batch_idx": int(batch_idx),
                    "iteration_idx": int(i),
                    **trial["scalars"],
                }
                writer.writerow({k: row[k] for k in scalar_fields})
                trials_f.flush()

                save_npz_compressed_legacy(
                    os.path.join(trials_dir, f"trial_{i:04d}_timeseries.npz"),
                    **trial["timeseries"],
                )
                save_npz_compressed_legacy(
                    os.path.join(trials_dir, f"trial_{i:04d}_raster.npz"),
                    **trial["raster"],
                )

                current_batch_trials.append(trial)
                trial_duration_ms = (self.lookahead + trial["scalars"]["n_steps"]) * self.time_step
                trial["_time_offset_ms"] = cumulative_time_ms
                cumulative_time_ms += trial_duration_ms

                rospy.loginfo(
                    f"  abs_error={trial['scalars']['abs_position_error_rad']:.4f} rad  "
                    f"stop={trial['scalars']['stop_reason']}"
                )

                batch_done = (
                    i == batch_idx * batch_size
                    or (batch_idx == num_batches and i == num_iterations)
                )
                if batch_done:
                    stats = self._compute_batch_stats(current_batch_trials)
                    abs_list = np.array([t["scalars"]["abs_position_error_rad"] for t in current_batch_trials])
                    save_numpy_legacy(
                        os.path.join(
                            batches_dir,
                            f"batch_{batch_idx:02d}_abs_errors.npy",
                        ),
                        abs_list,
                    )

                    keys = ["r1", "r2", "left", "right"]
                    agg = {k + "_times": [] for k in keys}
                    agg.update({k + "_senders": [] for k in keys})
                    agg["trial_boundaries_ms"] = []
                    agg["goal_indices"] = []
                    agg["start_indices"] = []
                    for t in current_batch_trials:
                        off = t["_time_offset_ms"]
                        r = t["raster"]
                        for k in keys:
                            agg[k + "_times"].append(r[k + "_times"] + off)
                            agg[k + "_senders"].append(r[k + "_senders"])
                        agg["trial_boundaries_ms"].append(off)
                        agg["goal_indices"].append(t["scalars"]["goal_ring_index"])
                        agg["start_indices"].append(t["scalars"]["initial_ring_index"])
                    save_agg = {k: np.concatenate(v) for k, v in agg.items() if k not in ("trial_boundaries_ms", "goal_indices", "start_indices")}
                    save_agg["trial_boundaries_ms"] = np.array(agg["trial_boundaries_ms"], dtype=float)
                    save_agg["goal_indices"] = np.array(agg["goal_indices"], dtype=float)
                    save_agg["start_indices"] = np.array(agg["start_indices"], dtype=float)
                    save_npz_compressed_legacy(
                        os.path.join(
                            batches_dir,
                            f"batch_{batch_idx:02d}_raster.npz",
                        ),
                        **save_agg,
                    )

                    batch_summaries.append({
                        "joint_index": int(self.target_joint),
                        "batch_idx": int(batch_idx),
                        **stats,
                    })
                    current_batch_trials = []
                    rospy.loginfo(
                        f"[BATCH {batch_idx}] "
                        f"mean_abs_error={stats['mean_abs_error_rad']:.4f} "
                        f"std={stats['std_abs_error_rad']:.4f} rad"
                    )

                self.reset_ring_model()

        with open(os.path.join(out, "batches_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=batch_stat_fields)
            writer.writeheader()
            writer.writerows(batch_summaries)

        rospy.loginfo(f"[DONE] Data saved to {out}")
        return out


def main():
    rospy.init_node("single_joint_data_collector", anonymous=True)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config", "calibration", "velocity_calibration.json",
    )

    for target_joint in [5]:
        rospy.loginfo(f"=== Starting collection for joint {target_joint} ===")
        use_manual_limits = True
        try:
            config = load_json_legacy(config_path)
            key = str(target_joint)
            if key in config and "joint_min" in config[key] and "joint_max" in config[key]:
                use_manual_limits = False
        except Exception:
            pass

        collector = SingleJointDataCollector(target_joint, use_manual_limits)
        collector.run(num_iterations=50, num_batches=5)
        rospy.loginfo(f"=== Done joint {target_joint} ===")


if __name__ == "__main__":
    main()
