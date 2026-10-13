"""Behaviour tests for kangaroo/ring_reach.py (docs/kangaroo/DECISIONS.md D11-D12).

Spike test: ROS-free, needs NEST. Robot tests need the simulation running:
    ros2 launch /opt/kangaroo_sim.launch.py &
    cd /work/src/tiago_ring_controller/src && python3 /work/kangaroo/test_ring_reach.py
(run from the ring package dir: SingleRingModel loads ./config/... relative paths, single_ring.py:154-155)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_reach import RingDrive  # noqa: E402

LIMITS = (-2.618, 2.618)  # arm_<side>_1_joint URDF limits (Kangaroo-native range, D11)


class RingDriveSpikes(unittest.TestCase):
    def ticks(self, seed, n=20):
        ring = RingDrive(*LIMITS, seed=seed)
        ring.start(q=0.239, goal=-0.8)
        return [ring.tick() for _ in range(n)]

    def test_same_seed_same_spikes(self):
        self.assertEqual(self.ticks(12345), self.ticks(12345))

    def test_seed_is_applied(self):
        self.assertNotEqual(self.ticks(12345), self.ticks(1))


class RingReachOnKangaroo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rclpy
        from rclpy.parameter import Parameter
        from arm_velocity import ArmVelocityPublisher, wait_ready
        rclpy.init()
        cls.node = rclpy.create_node('test_ring_reach',
                                     parameter_overrides=[Parameter('use_sim_time', value=True)])
        cls.arms = {s: ArmVelocityPublisher(cls.node, s) for s in ('left', 'right')}
        assert wait_ready(cls.node, cls.arms.values()), 'simulation or arm controllers not running'

    @classmethod
    def tearDownClass(cls):
        import rclpy
        rclpy.shutdown()

    def test_ring_drives_joint_toward_goal_and_nothing_else(self):
        from ring_reach import run_trial
        left, right = self.arms['left'], self.arms['right']
        before = left.current_positions + right.current_positions
        goal = before[0] - 1.0
        out = run_trial(self.node, left, 0, goal)
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


if __name__ == '__main__':
    unittest.main()
