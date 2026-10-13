"""ROS 2 port of TIAGo's joint velocity path for one Kangaroo arm (docs/kangaroo/DECISIONS.md).

A velocity command for one joint becomes a short receding-horizon JointTrajectory sent to PAL's
arm_<side>_controller, as TiagoPublisher.publish_receding_trajectory did on TIAGo.

CLI (inside the container, simulation running):
    python3 kangaroo/arm_velocity.py <left|right> <joint 0-3> <velocity rad/s> <seconds>
"""
import argparse
import sys
import time
from pathlib import Path

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Reuse the ROS-free trajectory builder from the TIAGo package unchanged (DECISIONS.md D2).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/tiago_ring_controller/src'))
from tiago_ring_controller.control.trajectory import build_receding_trajectory  # noqa: E402

DT = 0.05     # s, TIAGo control tick (control/profiles.py)
HORIZON = 4   # points per published horizon, TIAGo lookahead
N_JOINTS = 4  # PAL 4-DoF arm: arm_<side>_1_joint .. arm_<side>_4_joint


def sleep_sim(node, seconds):
    """Spin `node` until `seconds` of simulation time have passed (DECISIONS.md D4)."""
    end = node.get_clock().now() + Duration(seconds=seconds)
    while node.get_clock().now() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def wait_ready(node, arms, timeout_s=30.0):
    """Spin until sim time runs, joint states arrive and every arm controller is listening."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > 0 and all(arm.ready() for arm in arms):
            return True
    return False


class ArmVelocityPublisher:
    """Velocity control of one joint at a time on one arm, mirroring TiagoPublisher (Mode A)."""

    def __init__(self, node, side):
        self.joint_names = [f'arm_{side}_{i}_joint' for i in range(1, N_JOINTS + 1)]
        self.current_positions = None
        self.reference_positions = None
        self.commanded_positions = None
        self._pub = node.create_publisher(JointTrajectory, f'/arm_{side}_controller/joint_trajectory', 1)
        node.create_subscription(JointState, '/joint_states', self._on_joint_states, 1)
        node.create_subscription(JointTrajectoryControllerState, f'/arm_{side}_controller/controller_state',
                                 self._on_controller_state, 1)

    def _on_joint_states(self, msg):
        by_name = dict(zip(msg.name, msg.position))
        if all(name in by_name for name in self.joint_names):
            self.current_positions = [by_name[name] for name in self.joint_names]

    def _on_controller_state(self, msg):
        """The JTC's held reference: under gravity the P-only loop sits ~0.01 rad off it (DECISIONS.md D13)."""
        by_name = dict(zip(msg.joint_names, msg.reference.positions))
        if all(name in by_name for name in self.joint_names):
            self.reference_positions = [by_name[name] for name in self.joint_names]

    def ready(self):
        return (self.current_positions is not None and self.reference_positions is not None
                and self._pub.get_subscription_count() > 0)

    def prime_command_state(self, force=False):
        """Start commanding from the controller's held reference, not the measured pose (D13)."""
        if self.reference_positions is None:
            return False
        if force or self.commanded_positions is None:
            self.commanded_positions = list(self.reference_positions)
        return True

    def publish_receding_trajectory(self, joint_idx, velocities, dt=DT):
        """Publish a horizon moving only `joint_idx`; the command state advances by one step."""
        if not self.prime_command_state():
            return
        trajectory = build_receding_trajectory(self.commanded_positions, joint_idx, velocities, dt)
        if trajectory is None:
            return
        msg = JointTrajectory(joint_names=self.joint_names)  # stamp 0: start on receipt (DECISIONS.md D5)
        for point in trajectory.points:
            msg.points.append(JointTrajectoryPoint(
                positions=list(point.positions),
                velocities=list(point.velocities),
                time_from_start=Duration(seconds=point.time_from_start_s).to_msg()))
        self._pub.publish(msg)
        self.commanded_positions = list(trajectory.next_commanded_positions)

    def stop_joint(self, joint_idx, dt=DT):
        """Three zero-velocity points, as TiagoPublisher.stop_joint."""
        self.publish_receding_trajectory(joint_idx, [0.0] * 3, dt)

    def move_to(self, positions, seconds=2.0):
        """One-point trajectory to `positions` over `seconds`; the command state becomes `positions`."""
        msg = JointTrajectory(joint_names=self.joint_names)
        msg.points.append(JointTrajectoryPoint(positions=list(positions), velocities=[0.0] * N_JOINTS,
                                               time_from_start=Duration(seconds=seconds).to_msg()))
        self._pub.publish(msg)
        self.commanded_positions = list(positions)


def main():
    parser = argparse.ArgumentParser(description='Move one Kangaroo arm joint at a constant velocity.')
    parser.add_argument('side', choices=['left', 'right'])
    parser.add_argument('joint', type=int, choices=range(N_JOINTS), help='joint index 0-3')
    parser.add_argument('velocity', type=float, help='rad/s')
    parser.add_argument('seconds', type=float)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('arm_velocity', parameter_overrides=[Parameter('use_sim_time', value=True)])
    arm = ArmVelocityPublisher(node, args.side)
    if not wait_ready(node, [arm]):
        sys.exit('simulation or arm controller not running')
    start = arm.current_positions[args.joint]
    for _ in range(round(args.seconds / DT)):
        arm.publish_receding_trajectory(args.joint, [args.velocity] * HORIZON)
        sleep_sim(node, DT)
    arm.stop_joint(args.joint)
    sleep_sim(node, 1.0)
    end = arm.current_positions[args.joint]
    print(f'{arm.joint_names[args.joint]}: {start:+.3f} -> {end:+.3f} rad '
          f'(moved {end - start:+.3f}, expected {args.velocity * args.seconds:+.3f})')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
