"""The upper-reference controllers without ROS or NEST: the tuned controller, its runner on the simulated
joint, and the paired statistics of the report (docs/results.md)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import results_report  # noqa: E402
import run_baselines  # noqa: E402
from tiago_ring_controller.control.controller import PIDControllerCore  # noqa: E402
from tiago_ring_controller.control.pid import AntiWindupPID  # noqa: E402
from tiago_ring_controller.graph.templates import ARM_MAX_VELOCITY, joint_limits  # noqa: E402


def integrate(step, goal=2.0, limit=0.5, dt=0.05, ticks=400):
    """A joint that moves at exactly the commanded velocity; returns the trace of its angle."""

    q, trace = 0.0, []
    for _ in range(ticks):
        q += float(np.clip(step(goal - q, q, dt), -limit, limit)) * dt
        trace.append(q)
    return np.array(trace)


class PidTests(unittest.TestCase):
    def test_a_saturated_long_move_does_not_wind_the_integral_up(self):
        proper = AntiWindupPID(2.0, 0.1, 0.05, 0.5)
        trace = integrate(lambda e, q, dt: proper.step(e, q, dt)[0])
        self.assertLess(trace.max() - 2.0, 0.01)                    # no overshoot plateau
        self.assertLess(abs(trace[-1] - 2.0), 0.005)
        # the PID of the draft with the same gains: the integral grows during the saturated part and parks the joint past the goal
        draft = PIDControllerCore(2.0, 0.1, 0.05, output_limit=0.5)
        wound = integrate(lambda e, q, dt: draft.step(e, dt)[0])
        self.assertGreater(wound.max() - 2.0, 0.05)
        self.assertGreater(abs(wound[-1] - 2.0), 0.02)               # and it is still there after 20 s

    def test_terms_and_limits(self):
        pid = AntiWindupPID(4.0, 1.0, 0.1, 1.0)
        output, terms = pid.step(1.0, 0.0, 0.05)                     # saturated: the integral stays where it was
        self.assertEqual((output, terms["saturated"], pid.integral), (1.0, 1.0, 0.0))
        output, terms = pid.step(0.1, 0.9, 0.05)                     # derivative on the measurement: the joint moved 0.9 rad in a tick
        self.assertAlmostEqual(terms["d"], -0.1 * 0.9 / 0.05)
        self.assertEqual(output, -1.0)
        output, _ = pid.step(0.01, 0.9, 0.05)                        # unsaturated: the integral advances
        self.assertAlmostEqual(pid.integral, 0.1 * 0.05 + 0.01 * 0.05)
        pid.reset()
        self.assertEqual((pid.integral, pid.previous), (0.0, None))
        with self.assertRaises(ValueError):
            AntiWindupPID(1.0, 0.0, 0.0, 0.0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.engine = run_baselines.make_engine("fake")

    def test_p_controller_reaches_the_goal_from_a_start_pose_within_the_velocity_limit(self):
        lo, hi = joint_limits(5)
        start, goal = lo + 0.2 * (hi - lo), lo + 0.9 * (hi - lo)
        row = run_baselines.run_trial(self.engine, "p", 5, goal, {5: start})
        self.assertEqual((row["system"], row["stop_reason"]), ("P", "settled"))
        self.assertAlmostEqual(row["start_rad"], start, places=6)
        self.assertLess(abs(row["goal_error_rad"]), 1e-3)
        self.assertTrue(row["reached_tolerance"])
        self.assertLessEqual(max(abs(v) for v in row["trace"]["v_cmd"]), ARM_MAX_VELOCITY[5] + 1e-9)
        self.assertAlmostEqual(max(abs(v) for v in row["trace"]["v_cmd"]), 0.6 * ARM_MAX_VELOCITY[5])   # its own limit: 60% of the joint's
        self.assertEqual(row["clamp_ticks"], 0)                      # so the joint's limit is never hit
        self.assertEqual(len(row["trace"]["q"]), row["n_steps"] + 2)

    def test_pid_corrects_a_wrong_motor_gain_a_push_and_a_hold(self):
        lo, hi = joint_limits(4)
        goal = lo + 0.8 * (hi - lo)
        for kwargs in ({"motor_gain_scale": 0.7}, {"motor_gain_scale": 1.3}, {"push_rad": 0.2}, {"hold_s": 2.0}):
            row = run_baselines.run_trial(self.engine, "p", 4, goal, {4: lo + 0.3 * (hi - lo)}, **kwargs)
            self.assertLess(abs(row["goal_error_rad"]), 2e-3, kwargs)

    def test_goal_plan_is_the_one_of_validate_velocity(self):
        import argparse

        import validate_velocity

        args = argparse.Namespace(joints=[5], starts=None, seeds=[13579], trials=8, repeats=2)
        trials, skipped = run_baselines.planned_trials(args, None)
        lo, hi = joint_limits(5)
        start = (validate_velocity.START_POSE[5] - lo) / (hi - lo)
        expected = [lo + f * (hi - lo) for f, _ in validate_velocity.goal_plan(start, 8, 13579) for _ in range(2)]
        self.assertEqual([round(t[2], 9) for t in trials], [round(g, 9) for g in expected])
        self.assertEqual(skipped, [])


class StatisticsTests(unittest.TestCase):
    def test_sign_test(self):
        self.assertAlmostEqual(results_report.sign_test(10, 0), 2.0 / 1024)
        self.assertAlmostEqual(results_report.sign_test(5, 5), 1.0)
        self.assertAlmostEqual(results_report.sign_test(1, 7), 2 * (1 + 8) / 256.0)
        self.assertTrue(np.isnan(results_report.sign_test(0, 0)))
        self.assertLess(results_report.sign_test(290, 10), 1e-60)                    # hundreds of trials: no overflow

    def test_bootstrap_interval_brackets_the_median(self):
        values = np.linspace(-0.1, 0.1, 41)
        low, high = results_report.median_interval(values)
        self.assertLess(low, np.median(np.abs(values)))
        self.assertGreater(high, np.median(np.abs(values)))
        self.assertEqual(results_report.median_interval(values), (low, high))        # seeded: the same every time

    def test_paired_table_reads_only_what_the_test_supports(self):
        def rows(errors, offset=0):
            return [{"repeat_key": (5, k), "error": e} for k, e in enumerate(errors, start=offset)]

        systems = {"F3": rows([0.01] * 12), "B3": rows([0.1] * 12), "P": rows([0.001] * 12), "N": rows([0.02, 0.005] * 3, offset=6)}
        table = "\n".join(results_report.paired_table(systems, results_report.NAMES))
        self.assertIn("| v3, bounded ring, 3x | 12 | -0.090 | 100% | 0% | under 0.001 | v4 is closer |", table)
        self.assertIn("| upper reference: tuned controller | 12 | +0.009 | 0% | 100% | under 0.001 | upper reference: tuned controller is closer |", table)
        self.assertIn("| upper reference: robot's own controller | 6 |", table)
        self.assertIn("| 50% | 50% | 1.000 | no difference shown |", table)


if __name__ == "__main__":
    unittest.main()
