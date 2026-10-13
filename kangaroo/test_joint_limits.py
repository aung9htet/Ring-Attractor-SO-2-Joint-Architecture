"""Behaviour test for joint_limits.measure_limits (simulation running). Run from src/tiago_ring_controller/src:
    python3 /work/kangaroo/test_joint_limits.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rclpy
from rclpy.parameter import Parameter

from arm_velocity import ArmVelocityPublisher, wait_ready
from calibrate import HOME
from ring_reach import LIMITS
from joint_limits import measure_limits


class MeasureLimitsTest(unittest.TestCase):
    def test_right_j1_range_is_inside_urdf_and_stops_at_obstruction(self):
        rclpy.init()
        node = rclpy.create_node('test_joint_limits', parameter_overrides=[Parameter('use_sim_time', value=True)])
        arm = ArmVelocityPublisher(node, 'right')
        self.assertTrue(wait_ready(node, [arm]))
        lo, hi = measure_limits(node, arm, 'right', 1)
        rclpy.shutdown()
        self.assertLess(lo, HOME['right'][1])
        self.assertGreater(hi, HOME['right'][1])
        self.assertGreater(lo, LIMITS[1][0] - 0.02)
        self.assertLess(hi, LIMITS[1][1] + 0.02)
        self.assertLess(hi, 1.6)  # F9: stalls at 1.473-1.477 rad, URDF max 2.443


if __name__ == '__main__':
    unittest.main()
