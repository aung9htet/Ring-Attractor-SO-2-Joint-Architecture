"""Behaviour tests for kangaroo_ring_controller/ring.py (docs/kangaroo/DECISIONS.md D11-D12).

Spike test: ROS-free, needs NEST. Robot tests need the simulation running:
    ros2 launch /opt/kangaroo_sim.launch.py &
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_ring.py
(runs from any directory: ring_core passes absolute config paths to SingleRingModel)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller.joint_setup import uncalibrated  # noqa: E402
from kangaroo_ring_controller.ring import RingDrive  # noqa: E402

LEFT_J0 = uncalibrated('left', 0)  # URDF range of arm_left_1_joint (-2.618, 2.618) rad, D11


class RingDriveSpikes(unittest.TestCase):
    def ticks(self, seed, n=20):
        ring = RingDrive(LEFT_J0, seed=seed)
        ring.start(q=0.239, goal=-0.8)
        return [ring.tick() for _ in range(n)]

    def test_same_seed_same_spikes(self):
        self.assertEqual(self.ticks(12345), self.ticks(12345))

    def test_seed_is_applied(self):
        self.assertNotEqual(self.ticks(12345), self.ticks(1))


class RingReachOnKangaroo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kangaroo_ring_controller.arm import connect  # ROS imports stay out of the NEST-only spike tests
        cls.node, cls.arms = connect('test_ring_reach')

    @classmethod
    def tearDownClass(cls):
        import rclpy
        rclpy.shutdown()

    def test_ring_drives_joint_toward_goal_and_nothing_else(self):
        from kangaroo_ring_controller.ring import run_trial
        left, right = self.arms['left'], self.arms['right']
        before = left.current_positions + right.current_positions
        goal = before[0] - 1.0
        out = run_trial(left, LEFT_J0, goal)
        after = left.current_positions + right.current_positions
        moved = out['q_final'] - out['q_start']

        # Moves toward goal (sign oracle) and visibly (> 0.01 rad).
        self.assertGreater(-moved, 0.01)
        # Plant follows decoder: Δq ≈ Σ v·dt (±10 %, floor 0.005 rad for small moves).
        expected = sum(out['decoded_velocity']) * 0.05
        self.assertAlmostEqual(moved, expected, delta=max(0.1 * abs(expected), 0.005))
        # Only the target joint moves, on either arm.
        for i in range(1, 8):
            self.assertLess(abs(after[i] - before[i]), 0.02, f'joint slot {i} moved')

    def test_a_prebuilt_ring_is_used(self):
        from kangaroo_ring_controller.ring import RingDrive, run_trial
        arm = self.arms['left']
        ring = RingDrive(LEFT_J0, seed=4)
        out = run_trial(arm, LEFT_J0, arm.current_positions[0] + 0.3, seed=4, ring=ring,
                        on_tick=lambda tick: tick['step'] == 2)
        self.assertEqual(out['goal_ring_index'], ring.goal_idx)  # run_trial started this ring, built no other

    def test_a_slow_delayed_decoder_still_reaches(self):
        # D28 left j3 (tau 0.93 s, delay 4), goal halfway to the max: moved 0.002 rad under TIAGo's settle rule (F14).
        from kangaroo_ring_controller import RESULTS
        from kangaroo_ring_controller.joint_setup import load
        from kangaroo_ring_controller.ring import run_trial
        setup = load(RESULTS / 'calibration/left_j3_seed2026_20261014_001609/calibration.json')
        arm = self.arms['left']
        arm.reset(setup.start)
        out = run_trial(arm, setup, 1.638, 100)
        self.assertGreater(out['q_final'] - out['q_start'], 0.1)


class RunTrialRefusesAMismatch(unittest.TestCase):
    """Checks that run before anything is published: no simulation needed (D35, Q9 S2)."""

    @classmethod
    def setUpClass(cls):
        import rclpy
        from kangaroo_ring_controller.arm import ArmVelocityPublisher
        rclpy.init()
        cls.node = rclpy.create_node('test_run_trial_checks')
        cls.left = ArmVelocityPublisher(cls.node, 'left')

    @classmethod
    def tearDownClass(cls):
        import rclpy
        rclpy.shutdown()

    def test_a_setup_for_the_other_arm(self):
        from kangaroo_ring_controller.joint_setup import uncalibrated
        from kangaroo_ring_controller.ring import run_trial
        with self.assertRaisesRegex(ValueError, 'right setup on the left arm'):
            run_trial(self.left, uncalibrated('right', 0), 0.5, seed=1)

    def test_a_ring_built_for_another_seed_or_setup(self):
        from dataclasses import replace
        from kangaroo_ring_controller.joint_setup import uncalibrated
        from kangaroo_ring_controller.ring import RingDrive, run_trial
        setup = uncalibrated('left', 0)
        ring = RingDrive(setup, seed=4)
        for other_setup, seed in ((setup, 5), (replace(setup, limits=(-1.0, 1.0)), 4)):
            with self.assertRaisesRegex(ValueError, 'ring built for'):
                run_trial(self.left, other_setup, 0.5, seed=seed, ring=ring)


if __name__ == '__main__':
    unittest.main()
