#!/usr/bin/env python3

import csv
import json
import os

import numpy as np
import rospy

from tiago_controller import TiagoPublisher, TiagoSubscriber
from tiago_ring_controller.artifacts import (
    load_json_legacy,
    save_json_legacy,
    save_npz_compressed_legacy,
    save_numpy_legacy,
)
from tiago_ring_controller.control.controller import PIDControllerCore
from tiago_ring_controller.config import module_config_path
from tiago_ring_controller.evaluation.metrics import collector_batch_statistics
from tiago_ring_controller.evaluation.serialization import (
    PID_BATCH_STAT_FIELDS,
    PID_TRIAL_SCALAR_FIELDS,
)


class PIDController:
    """Compatibility name for the extracted state-only PID controller."""

    def __init__(self, kp, ki, kd, windup_limit=2.0, output_limit=1.0):
        PIDControllerCore.__init__(
            self, kp, ki, kd, windup_limit, output_limit
        )

    def reset(self):
        return PIDControllerCore.reset(self)

    def step(self, error, dt):
        return PIDControllerCore.step(self, error, dt)


class SingleJointPIDDataCollector:
    def __init__(self, target_joint, use_manual_limits=False):
        self.target_joint = target_joint

        self.publisher = TiagoPublisher()
        self.subscriber = TiagoSubscriber()
        self.publisher.wait_for_joint_state(timeout=5.0)

        self.time_step = 50.0  # ms
        self.max_steps = 400
        self.n_settle = 10
        self.error_threshold = 0.01  # rad — settle when |error| < this

        # PID gains — tune per-joint via config if desired
        self.kp = 2.0
        self.ki = 0.1
        self.kd = 0.05
        self.output_limit = 0.5  # rad/s

        self.pid = PIDController(self.kp, self.ki, self.kd, output_limit=self.output_limit)

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

        self._load_pid_from_config()

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

    def _load_pid_from_config(self):
        try:
            config = load_json_legacy(self.config_path)
            key = str(self.target_joint)
            if key not in config:
                return
            cfg = config[key]
            if "pid_kp" in cfg:
                self.kp = float(cfg["pid_kp"])
            if "pid_ki" in cfg:
                self.ki = float(cfg["pid_ki"])
            if "pid_kd" in cfg:
                self.kd = float(cfg["pid_kd"])
            if "pid_output_limit" in cfg:
                self.output_limit = float(cfg["pid_output_limit"])
            self.pid = PIDController(self.kp, self.ki, self.kd, output_limit=self.output_limit)
        except Exception as e:
            rospy.logwarn(f"[PID LOAD] {e}")

    def _output_dir(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "collected_data",
            f"joint_{self.target_joint}",
            "pid",
        )
        os.makedirs(path, exist_ok=True)
        return path

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

    def run_trial(self, goal_state):
        dt_s = self.time_step / 1000.0

        self.publisher.prime_command_state(force=True)
        self.pid.reset()

        q_start = float(self.publisher.current_positions[self.target_joint])
        dq_desired = goal_state - q_start
        movement_direction = np.sign(dq_desired) if abs(dq_desired) > 1e-9 else 0.0

        consecutive_settled = 0
        stop_reason = "max_steps"

        time_hist, pos_hist, vel_hist, err_hist = [], [], [], []
        p_hist, i_hist, d_hist, vcmd_hist = [], [], [], []

        for step in range(self.max_steps):
            cur_pos = float(self.publisher.current_positions[self.target_joint])
            cur_vel = float((self.publisher.current_velocities or [0.0] * 7)[self.target_joint])
            error = goal_state - cur_pos

            vcmd, terms = self.pid.step(error, dt_s)

            time_hist.append(float((step + 1) * dt_s))
            pos_hist.append(cur_pos)
            vel_hist.append(cur_vel)
            err_hist.append(float(error))
            p_hist.append(terms["p"])
            i_hist.append(terms["i"])
            d_hist.append(terms["d"])
            vcmd_hist.append(vcmd)

            rospy.loginfo(
                f"[{step:03d}] err={error:.4f} P={terms['p']:.4f} "
                f"I={terms['i']:.4f} D={terms['d']:.4f} "
                f"vcmd={vcmd:.6f} vmeas={cur_vel:.6f} q={cur_pos:.4f}"
            )

            self.publisher.publish_receding_trajectory(
                self.target_joint, [vcmd] * 4, dt_s)

            if abs(error) < self.error_threshold:
                consecutive_settled += 1
                if consecutive_settled >= self.n_settle:
                    stop_reason = "error_settled"
                    break
            else:
                consecutive_settled = 0

            rospy.sleep(dt_s)

        self.publisher.stop_joint(self.target_joint, dt_s)
        self.publisher.wait_for_settled(self.target_joint)

        q_final = float(self.publisher.current_positions[self.target_joint])
        raw_err = goal_state - q_final
        dq_pred = float(np.sum(vcmd_hist) * dt_s)

        scalars = {
            "q_start": q_start,
            "q_goal": float(goal_state),
            "q_final": q_final,
            "dq_desired": float(dq_desired),
            "dq_actual": float(q_final - q_start),
            "predicted_dq_rad": dq_pred,
            "raw_position_error_rad": float(raw_err),
            "abs_position_error_rad": float(abs(raw_err)),
            "aligned_error_rad": float(raw_err * movement_direction),
            "S_vel": float(np.sum(vcmd_hist) * dt_s),
            "n_steps": len(time_hist),
            "stop_reason": stop_reason,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "output_limit": self.output_limit,
        }

        timeseries = {
            "time": np.array(time_hist),
            "joint_position": np.array(pos_hist),
            "joint_velocity": np.array(vel_hist),
            "position_error": np.array(err_hist),
            "p_term": np.array(p_hist),
            "i_term": np.array(i_hist),
            "d_term": np.array(d_hist),
            "decoded_velocity": np.array(vcmd_hist),
        }

        return {"scalars": scalars, "timeseries": timeseries}

    def _compute_batch_stats(self, trials):
        return collector_batch_statistics(
            trials,
            prediction_key="predicted_dq_rad",
            settled_stop_reason="error_settled",
        )

    def run(self, num_iterations=50, num_batches=5):
        out = self._output_dir()
        trials_dir = os.path.join(out, "trials")
        batches_dir = os.path.join(out, "batches")
        os.makedirs(trials_dir, exist_ok=True)
        os.makedirs(batches_dir, exist_ok=True)
        batch_size = max(1, num_iterations // num_batches)

        scalar_fields = PID_TRIAL_SCALAR_FIELDS

        batch_stat_fields = PID_BATCH_STAT_FIELDS

        batch_summaries = []
        current_batch_trials = []

        meta = {
            "joint_index": int(self.target_joint),
            "joint_min": float(self.joint_min),
            "joint_max": float(self.joint_max),
            "num_iterations": int(num_iterations),
            "num_batches": int(num_batches),
            "batch_size": int(batch_size),
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "output_limit": self.output_limit,
            "controller": "pid",
        }
        save_json_legacy(os.path.join(out, "session_meta.json"), meta)

        with open(os.path.join(out, "trials_summary.csv"), "w", newline="") as trials_f:
            writer = csv.DictWriter(trials_f, fieldnames=scalar_fields)
            writer.writeheader()

            for i in range(1, num_iterations + 1):
                batch_idx = min((i - 1) // batch_size + 1, num_batches)
                goal = np.random.uniform(self.joint_min, self.joint_max)
                cur = float(self.publisher.current_positions[self.target_joint])

                rospy.loginfo(
                    f"[PID {i}/{num_iterations} | batch {batch_idx}/{num_batches}] "
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

                current_batch_trials.append(trial)

                rospy.loginfo(
                    f"  abs_error={trial['scalars']['abs_position_error_rad']:.4f} rad  "
                    f"stop={trial['scalars']['stop_reason']}"
                )

                batch_done = (
                    (i == batch_idx * batch_size and current_batch_trials)
                    or (batch_idx == num_batches and i == num_iterations and current_batch_trials)
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

        with open(os.path.join(out, "batches_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=batch_stat_fields)
            writer.writeheader()
            writer.writerows(batch_summaries)

        rospy.loginfo(f"[DONE] PID data saved to {out}")
        return out


def main():
    rospy.init_node("single_joint_pid_data_collector", anonymous=True)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config", "calibration", "velocity_calibration.json",
    )

    for target_joint in range(7):
        rospy.loginfo(f"=== Starting PID collection for joint {target_joint} ===")
        use_manual_limits = True
        try:
            config = load_json_legacy(config_path)
            key = str(target_joint)
            if key in config and "joint_min" in config[key] and "joint_max" in config[key]:
                use_manual_limits = False
        except Exception:
            pass

        collector = SingleJointPIDDataCollector(target_joint, use_manual_limits)
        collector.run(num_iterations=50, num_batches=5)
        rospy.loginfo(f"=== Done joint {target_joint} ===")


if __name__ == "__main__":
    main()
