"""Behaviour test for kangaroo_ring_controller/limits.py measure_limits (simulation running; any directory):
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_limits.py
"""
import sys
import unittest
from pathlib import Path

import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller.arm import HOME, connect  # noqa: E402
from kangaroo_ring_controller.limits import measure_limits  # noqa: E402
from kangaroo_ring_controller.ring import LIMITS  # noqa: E402


class MeasureLimitsTest(unittest.TestCase):
    def test_right_j1_range_is_inside_urdf_and_stops_at_obstruction(self):
        node, arms = connect('test_joint_limits', ['right'])
        lo, hi = measure_limits(arms['right'], 1)
        rclpy.shutdown()
        self.assertLess(lo, HOME['right'][1])
        self.assertGreater(hi, HOME['right'][1])
        self.assertGreater(lo, LIMITS[1][0] - 0.02)
        self.assertLess(hi, LIMITS[1][1] + 0.02)
        self.assertLess(hi, 1.6)  # F9: stalls at 1.473-1.477 rad, URDF max 2.443


if __name__ == '__main__':
    unittest.main()
