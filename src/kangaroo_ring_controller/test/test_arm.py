"""Behaviour test: each Kangaroo arm joint can be velocity-controlled individually (docs/kangaroo/DECISIONS.md D9).

Needs the simulation (with arm controllers) running. Inside the container, from /work:
    ros2 launch /opt/kangaroo_sim.launch.py &
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_arm.py
Expected motion comes from kinematics (dq = v * T), not from the code under test.
"""
import sys
import unittest
from pathlib import Path

import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller.arm import DT, HORIZON, N_JOINTS, SIDES, connect  # noqa: E402

SPEED = 0.2       # rad/s, well under the 1.95 rad/s joint limit
TICKS = 40        # 40 * 0.05 s = 2.0 s  ->  |dq| = 0.4 rad, inside every joint's range from home
STILL_TOL = 0.02  # rad allowed on joints that were not commanded


class ArmVelocityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node, cls.arms = connect('arm_velocity_test')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def positions(self):
        return {side: list(arm.current_positions) for side, arm in self.arms.items()}

    def toward_centre(self, side, idx, speed):
        """Velocity sign that moves the joint toward zero, keeping it inside its limits."""
        return -speed if self.positions()[side][idx] > 0 else speed

    def drive(self, commands):
        """Hold {(side, joint_idx): velocity} for TICKS control ticks, stop, settle; return dq per side."""
        before = self.positions()
        for _ in range(TICKS):
            for (side, idx), v in commands.items():
                self.arms[side].publish_receding_trajectory(idx, [v] * HORIZON)
            self.arms['left'].sleep(DT)
        for side, idx in commands:
            self.arms[side].stop_joint(idx)
        self.arms['left'].sleep(1.0)
        after = self.positions()
        return {side: [a - b for a, b in zip(after[side], before[side])] for side in SIDES}

    def assert_motion(self, dq, commands):
        for side in SIDES:
            for idx in range(N_JOINTS):
                v = commands.get((side, idx))
                if v is None:
                    self.assertLess(abs(dq[side][idx]), STILL_TOL, f'{side} joint {idx} moved but was not commanded')
                else:
                    expected = v * TICKS * DT
                    self.assertAlmostEqual(dq[side][idx], expected, delta=0.1 * abs(expected),
                                           msg=f'{side} joint {idx}')

    def test_each_joint_individually(self):
        for side in SIDES:
            for idx in range(N_JOINTS):
                with self.subTest(side=side, joint=idx):
                    commands = {(side, idx): self.toward_centre(side, idx, SPEED)}
                    self.assert_motion(self.drive(commands), commands)
                    back = {key: -v for key, v in commands.items()}
                    self.assert_motion(self.drive(back), back)

    def test_both_arms_at_once_independently(self):
        commands = {('left', 0): self.toward_centre('left', 0, SPEED),
                    ('right', 1): self.toward_centre('right', 1, SPEED / 2)}
        self.assert_motion(self.drive(commands), commands)
        back = {key: -v for key, v in commands.items()}
        self.assert_motion(self.drive(back), back)


if __name__ == '__main__':
    unittest.main()
