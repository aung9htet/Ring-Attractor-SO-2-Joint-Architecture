"""The comparison scripts without ROS or NEST: validate_velocity's goal plan, run_legacy_trials' wrapper.

docs/archive/velocity/bounded-ring.md.  The legacy collector itself is never imported
here; a stand-in module checks that the wrapper only redirects the output
directory and replaces the force-torque logger.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_legacy_trials  # noqa: E402
import validate_velocity  # noqa: E402
from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE  # noqa: E402


class GoalPlanTests(unittest.TestCase):
    def test_goals_on_both_sides_far_enough_and_same_for_every_system(self):
        plan = validate_velocity.goal_plan(0.5, 8, 13579)
        goals = [g for g, _ in plan]
        self.assertEqual(len(plan), 8)
        self.assertEqual(sum(g < 0.5 for g in goals), 4)
        self.assertTrue(all(abs(g - 0.5) >= 0.15 - 1e-9 and 0.05 <= g <= 0.95 for g in goals))
        self.assertEqual(plan, validate_velocity.goal_plan(0.5, 8, 13579))
        edge = validate_velocity.goal_plan(0.05, 6, 1)              # one side only: all goals above
        self.assertTrue(all(g > 0.05 for g, _ in edge))
        repeated = validate_velocity.goal_plan(0.9, 20, 5)          # few goals: repeats with new seeds
        self.assertEqual(len(repeated), 20)
        self.assertEqual(len(set(repeated)), 20)

    def test_systems(self):
        self.assertEqual(validate_velocity.SYSTEMS["B"], dict(calibration="transport", bounds="bounded"))
        self.assertEqual(validate_velocity.SYSTEMS["G"]["calibration"], "legacy")


def legacy_result(q_start=0.0, q_goal=0.6, q_final=0.5, steps=40):
    lo, hi = -1.4, 1.4
    positions = np.linspace(q_start, q_final, steps)
    centroid = np.array([COLLECTOR_PROFILE.joint_to_ring_index(q, lo, hi, 200, 5) for q in positions], dtype=float)
    return {
        "scalars": {"q_start": q_start, "q_goal": q_goal, "q_final": q_final, "n_steps": steps, "stop_reason": "drive_settled"},
        "timeseries": {"r1_centroid": centroid, "joint_position": positions},
    }


class LegacyWrapperTests(unittest.TestCase):
    def test_metrics_from_the_collector_timeseries(self):
        m = run_legacy_trials.legacy_metrics(legacy_result(), -1.4, 1.4)
        self.assertAlmostEqual(m["goal_error_rad"], 0.1)
        self.assertAlmostEqual(m["joint_moved"], 0.5)
        self.assertLess(m["state_error_abs_median"], 0.02)            # the bump sits on the joint: the mapping inverts
        self.assertAlmostEqual(m["bump_moved_rad"], 0.5 * 36 / 39, delta=0.03)
        self.assertFalse(m["wrong_direction"] or m["no_drive"])
        wrong = run_legacy_trials.legacy_metrics(legacy_result(q_final=-0.2), -1.4, 1.4)
        self.assertTrue(wrong["wrong_direction"])
        still = run_legacy_trials.legacy_metrics(legacy_result(q_final=0.01), -1.4, 1.4)
        self.assertTrue(still["no_drive"])

    def test_disturbances_wrap_the_publisher_and_leave_it_as_it_was(self):
        class Publisher:
            def __init__(self):
                self.commanded_positions = [0.0] * 7
                self.sent = []

            def publish_receding_trajectory(self, joint_idx, velocities, dt):
                self.sent.append(list(velocities))
                self.commanded_positions[joint_idx] += velocities[0] * dt

        def drive(**kwargs):
            publisher = Publisher()
            original = publisher.publish_receding_trajectory
            with run_legacy_trials.LegacyDisturbance(publisher, 5, 0.05, at_s=0.1, **kwargs) as disturbance:
                for _ in range(8):
                    publisher.publish_receding_trajectory(5, [0.2] * 4, 0.05)
            self.assertEqual(publisher.publish_receding_trajectory, original)      # the frozen publisher is itself again
            return publisher, disturbance

        plain, off = drive()
        self.assertFalse(off.active)
        self.assertAlmostEqual(plain.commanded_positions[5], 8 * 0.2 * 0.05)
        scaled, _ = drive(motor_gain_scale=0.7)
        self.assertEqual(scaled.sent[0], [0.2 * 0.7] * 4)
        self.assertAlmostEqual(scaled.commanded_positions[5], 0.7 * plain.commanded_positions[5])
        pushed, disturbance = drive(push_rad=0.2)
        self.assertTrue(disturbance.pushed)
        self.assertAlmostEqual(pushed.commanded_positions[5], plain.commanded_positions[5] + 0.2)
        held, disturbance = drive(hold_s=0.2)
        self.assertEqual((disturbance.dropped, len(held.sent)), (4, 4))
        self.assertAlmostEqual(held.commanded_positions[5], 4 * 0.2 * 0.05)

    def test_wrapper_redirects_output_and_stubs_the_logger(self):
        calls = []

        class Publisher:
            def reset_pose(self):
                calls.append("reset")

            def prime_command_state(self, force=False):
                calls.append("prime")

        class Collector:
            def __init__(self, target_joint, use_manual_limits=False):
                self.target_joint, self.joint_min, self.joint_max = target_joint, -1.4, 1.4
                self.subscriber = module.TiagoSubscriber()
                self.publisher = Publisher()

            def run_trial(self, goal):
                calls.append(("trial", goal, self._output_dir()))
                return legacy_result(q_goal=goal, q_final=goal - 0.05)

            def reset_ring_model(self):
                calls.append("rebuild")

            def _output_dir(self):
                raise AssertionError("the legacy output directory must not be used")

        module = types.ModuleType("single_joint_data_collector")
        module.SingleJointDataCollector = Collector
        module.TiagoSubscriber = lambda: (_ for _ in ()).throw(AssertionError("the force-torque logger must be stubbed"))
        rospy = types.ModuleType("rospy")
        rospy.init_node = lambda *a, **k: None
        saved = {name: sys.modules.get(name) for name in ("single_joint_data_collector", "rospy")}
        cwd = os.getcwd()                                         # main() moves to src/ for the legacy model's ./config
        sys.modules.update({"single_joint_data_collector": module, "rospy": rospy})
        try:
            with tempfile.TemporaryDirectory() as out:
                self.assertEqual(run_legacy_trials.main(["--joint", "5", "--goals", "0.6", "-0.4", "--out", out]), 0)
                self.assertEqual([c if isinstance(c, str) else c[0] for c in calls],
                                 ["reset", "prime", "trial", "rebuild", "reset", "prime", "trial", "rebuild"])
                self.assertTrue(calls[2][2].startswith(out))
                summary = json.loads((Path(out) / "legacy_trials_j5.json").read_text())
                self.assertEqual([round(t["goal_error_rad"], 6) for t in summary["trials"]], [0.05, 0.05])
                self.assertTrue((Path(out) / "legacy_j5_goal+0.600_r0.json").exists())
        finally:
            os.chdir(cwd)
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
