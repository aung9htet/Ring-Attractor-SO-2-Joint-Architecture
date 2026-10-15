"""Velocity control of one Kangaroo arm joint: ROS 2 port of TIAGo's joint velocity path (DECISIONS.md D1-D7, D13).

A velocity command for one joint becomes a short receding-horizon JointTrajectory sent to PAL's
arm_<side>_controller, as TiagoPublisher.publish_receding_trajectory did on TIAGo.
"""
import sys
import time

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .joint_setup import HOME  # noqa: F401  (re-exported; joint_setup also puts the TIAGo package on sys.path)
from tiago_ring_controller.control.trajectory import build_receding_trajectory  # reused unchanged (D2)

DT = 0.05     # s, TIAGo control tick (control/profiles.py)
HORIZON = 4   # points per published horizon, TIAGo lookahead
N_JOINTS = 4  # PAL 4-DoF arm: arm_<side>_1_joint .. arm_<side>_4_joint
SIDES = ('left', 'right')


def wait_ready(node, arms, timeout_s=30.0):
    """Spin until sim time runs, joint states arrive and every arm controller is listening."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > 0 and all(arm.ready() for arm in arms):
            return True
    return False


def add_arm_args(parser):
    """The `side` and `joint` arguments every command-line tool takes."""
    parser.add_argument('side', choices=SIDES)
    parser.add_argument('joint', type=int, choices=range(N_JOINTS), help=f'joint index 0-{N_JOINTS - 1}')


def connect(node_name, sides=SIDES):
    """rclpy.init, a sim-time node and one ArmVelocityPublisher per side; exits if the simulation is not up."""
    rclpy.init()
    node = rclpy.create_node(node_name, parameter_overrides=[Parameter('use_sim_time', value=True)])
    arms = {side: ArmVelocityPublisher(node, side) for side in sides}
    if not wait_ready(node, list(arms.values())):
        sys.exit('simulation or arm controller not running')
    return node, arms


class ArmVelocityPublisher:
    """Velocity control of one joint at a time on one arm, mirroring TiagoPublisher (Mode A).

    Owns its sim-time `node` and `side` (D36): sleep, hold and reset need neither passed again.
    """

    def __init__(self, node, side):
        self.node, self.side = node, side
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
        """The JTC's held reference: under gravity the P-only loop sits ~0.01 rad off it (D13)."""
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
        msg = JointTrajectory(joint_names=self.joint_names)  # stamp 0: start on receipt (D5)
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


    def sleep(self, seconds):
        """Spin the node until `seconds` of simulation time have passed (D4)."""
        end = self.node.get_clock().now() + Duration(seconds=seconds)
        while self.node.get_clock().now() < end:
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def hold(self, joint, velocity, seconds):
        """Command `velocity` rad/s on `joint` for `seconds` of sim time, one receding horizon per tick."""
        for _ in range(round(seconds / DT)):
            self.publish_receding_trajectory(joint, [velocity] * HORIZON)
            self.sleep(DT)

    def reset(self, pose=None, seconds=2.0, attempts=3):
        """Return the arm to `pose` (rad, 4 joints; default its side's PAL home), as reset_pose did on TIAGo (D18, D25).

        Checks the controller reference reached the pose and resends if not: in a fresh process right after a trial
        that ended on max_steps, a single reset was ignored (2 of 20 evaluation resets; cause unknown). Retries fixed
        4 of 4.
        """
        pose = pose or HOME[self.side]
        for _ in range(attempts):
            self.move_to(pose, seconds)
            self.sleep(seconds + 1.0)
            if max(abs(r - h) for r, h in zip(self.reference_positions, pose)) < 1e-3:
                return
        raise RuntimeError(f'{self.side} arm did not reach {pose} after {attempts} attempts')
