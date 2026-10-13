"""Behaviour tests for kangaroo/calibrate.py (docs/kangaroo/DECISIONS.md D16-D19). Simulation running:
    cd /work/src/tiago_ring_controller/src && python3 /work/kangaroo/test_calibrate.py
"""
import sys
import unittest
from pathlib import Path

import rclpy
from rclpy.parameter import Parameter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_velocity import ArmVelocityPublisher, wait_ready  # noqa: E402
from calibrate import calibrate, reset_arm  # noqa: E402

PAL_HOME_LEFT = [0.239, 1.321, 1.57, 0.8]  # PAL 4-DoF home, read from the running sim (KB 4c)
J0_LIMITS = (-2.618, 2.618)
RIGHT_J1_MEASURED = (-1.1396, 1.4623)  # results/joint_limits/right_j1_20261013_125903 (D20)


class CalibrateOnKangaroo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('test_calibrate', parameter_overrides=[Parameter('use_sim_time', value=True)])
        cls.arm = ArmVelocityPublisher(cls.node, 'left')
        cls.right = ArmVelocityPublisher(cls.node, 'right')
        assert wait_ready(cls.node, [cls.arm, cls.right]), 'simulation or arm controller not running'

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_reset_returns_arm_to_home(self):
        reset_arm(self.node, self.arm, 'left')
        for ref, meas, home in zip(self.arm.reference_positions, self.arm.current_positions, PAL_HOME_LEFT):
            self.assertAlmostEqual(ref, home, delta=1e-3)
            self.assertAlmostEqual(meas, home, delta=0.02)

    def test_calibration_learns_correctly_signed_decoder(self):
        spec, trials = calibrate(self.node, self.arm, 'left', 0, seed=7, n_trials=6)
        # Positive drive must move the joint up, negative drive down: k_pos > 0 > k_neg.
        self.assertGreater(spec.gain_positive, 0.0)
        self.assertLess(spec.gain_negative, 0.0)
        self.assertEqual(spec.source, 'calibration_fit')
        self.assertIn(spec.delay_steps, range(5))
        self.assertTrue(0.10 <= spec.tau <= 1.20)
        self.assertEqual(len(trials), 6)
        for t in trials:
            self.assertTrue(J0_LIMITS[0] <= t['q_goal'] <= J0_LIMITS[1])

    def test_balanced_calibration_learns_both_signs_near_a_limit(self):
        # F11: home 1.321 sits 0.14 rad under right j1's measured max; unbalanced goals left k_pos undetermined.
        spec, trials = calibrate(self.node, self.right, 'right', 1, seed=7, n_trials=6,
                                 limits=RIGHT_J1_MEASURED, balanced=True)
        self.assertGreater(spec.gain_positive, 0.0)
        self.assertLess(spec.gain_negative, 0.0)
        self.assertEqual(sum(t['q_goal'] > 1.321 for t in trials), 3)
        for t in trials:
            self.assertTrue(RIGHT_J1_MEASURED[0] <= t['q_goal'] <= RIGHT_J1_MEASURED[1])

    def test_calibration_from_a_mid_range_start_pose(self):
        # Q13 (c): right j1 starts at the measured mid-range, 0.161 rad, so it has room both ways.
        start = [-0.239, 0.161, -1.57, 0.8]
        spec, trials = calibrate(self.node, self.right, 'right', 1, seed=7, n_trials=6,
                                 limits=RIGHT_J1_MEASURED, balanced=True, start=start)
        self.assertGreater(spec.gain_positive, 0.0)
        self.assertLess(spec.gain_negative, 0.0)
        for t in trials:
            self.assertAlmostEqual(t['q_start'], 0.161, delta=0.02)
        self.assertEqual(sum(t['q_goal'] > 0.161 for t in trials), 3)


if __name__ == '__main__':
    unittest.main()
