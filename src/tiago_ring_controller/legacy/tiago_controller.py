#!/usr/bin/env python3

import rospy
import csv
import os
import math
import actionlib
from datetime import datetime

from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

from tiago_ring_controller.control.trajectory import (
    build_receding_trajectory,
    tracking_drift,
)
from tiago_ring_controller.ros.logging import (
    SENSOR_LOG_SCHEMA,
    format_sensor_row,
    log_filename,
    package_root_for_module,
    timestamp_token,
)
from tiago_ring_controller.ros.transport import (
    ARM_COMMAND_TOPIC,
    JOINT_NAMES,
    JOINT_STATES_TOPIC,
    PLAY_MOTION_ACTION,
    RESET_MOTION_NAME,
    WRIST_FT_TOPIC,
    gravity_compensation_topic,
    joint_state_vectors,
    subscriber_joint_positions,
)


class TiagoPublisher:
    def __init__(self):
        self.joint_names = list(JOINT_NAMES)

        self.current_positions = None
        self.current_velocities = None

        # Internal commanded state for Mode A receding-horizon streaming.
        # This avoids rebuilding every trajectory from delayed/noisy measured state.
        self.commanded_positions = None
        self.commanded_velocities = None
        self.command_state_initialized = False

        self.pub = rospy.Publisher(
            ARM_COMMAND_TOPIC,
            JointTrajectory,
            queue_size=1
        )

        # Per-joint effort publishers. Real hardware only; often ignored in Gazebo.
        self.torque_pubs = {
            name: rospy.Publisher(
                gravity_compensation_topic(name),
                Float64,
                queue_size=1
            )
            for name in self.joint_names
        }

        self.play_motion_client = actionlib.SimpleActionClient(
            PLAY_MOTION_ACTION,
            PlayMotionAction
        )

        self.joint_state_sub = rospy.Subscriber(
            JOINT_STATES_TOPIC,
            JointState,
            self.joint_state_cb,
            queue_size=1
        )

    def joint_state_cb(self, msg):
        vectors = joint_state_vectors(
            msg.name,
            msg.position,
            msg.velocity,
            self.joint_names,
        )
        if vectors is None:
            return
        positions, velocities = vectors
        self.current_positions = list(positions)
        self.current_velocities = list(velocities)

    def wait_for_joint_state(self, timeout=5.0):
        deadline = rospy.Time.now() + rospy.Duration(timeout)

        while not rospy.is_shutdown() and self.current_positions is None:
            if rospy.Time.now() > deadline:
                rospy.logwarn("Timed out waiting for /joint_states")
                return False
            rospy.sleep(0.05)

        return self.current_positions is not None

    def prime_command_state(self, force=False):
        """
        Initialize internal commanded state from measured joint state.

        Call this after reset_pose() and before starting a Mode A stream.
        """
        if not self.wait_for_joint_state():
            return False

        if force or not self.command_state_initialized:
            self.commanded_positions = list(self.current_positions)

            if self.current_velocities is None:
                self.commanded_velocities = [0.0] * len(self.joint_names)
            else:
                self.commanded_velocities = list(self.current_velocities)

            self.command_state_initialized = True
            rospy.loginfo("Mode A command state primed from current joint state")

        return True

    def reset_command_state(self):
        self.commanded_positions = None
        self.commanded_velocities = None
        self.command_state_initialized = False

    def publish_receding_trajectory(
        self,
        joint_idx,
        velocities,
        dt,
        rebase_threshold=None
    ):
        """
        Publish a receding-horizon trajectory for one joint.

        Mode A behavior:
        - Uses internal commanded state, not latest measured joint state.
        - Publishes a horizon of future waypoints.
        - Advances internal commanded state by ONE timestep, because the caller
          sleeps dt before publishing the next horizon.

        Args:
            joint_idx: target joint index, 0-6.
            velocities: list of velocity commands in rad/s.
            dt: timestep between waypoints in seconds.
            rebase_threshold: optional position drift threshold in rad.
        """
        if not self.prime_command_state():
            rospy.logwarn("publish_receding_trajectory: could not prime command state")
            return

        if len(velocities) == 0:
            rospy.logwarn("publish_receding_trajectory: empty velocity buffer")
            return

        if rebase_threshold is not None and self.current_positions is not None:
            drift = tracking_drift(
                self.current_positions,
                self.commanded_positions,
                joint_idx,
            )

            if drift > rebase_threshold:
                rospy.logwarn(
                    f"Rebasing command state for joint {joint_idx}: "
                    f"tracking drift {drift:.4f} > {rebase_threshold:.4f}"
                )
                self.prime_command_state(force=True)

        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.joint_names

        trajectory = build_receding_trajectory(
            self.commanded_positions,
            joint_idx,
            velocities,
            dt,
        )
        # The explicit empty-buffer guard above makes this unreachable.
        if trajectory is None:
            return

        for point_data in trajectory.points:
            point = JointTrajectoryPoint()
            point.positions = list(point_data.positions)
            point.velocities = list(point_data.velocities)
            point.time_from_start = rospy.Duration.from_sec(
                point_data.time_from_start_s
            )

            msg.points.append(point)

        self.pub.publish(msg)

        # Critical Mode A update:
        # Advance commanded state by ONE consumed step, not by the whole horizon.
        first_vel = velocities[0]
        self.commanded_positions[joint_idx] += first_vel * dt
        self.commanded_velocities = [0.0] * len(self.joint_names)
        self.commanded_velocities[joint_idx] = first_vel

    def stop_joint(self, joint_idx, dt=0.05):
        """
        Send a short zero-velocity horizon for the joint.
        """
        self.publish_receding_trajectory(joint_idx, [0.0, 0.0, 0.0], dt)

    def wait_for_settled(
        self,
        joint_idx,
        vel_threshold=0.01,
        timeout=8.0,
        poll_dt=0.05
    ):
        """
        Block until measured joint velocity is below threshold.
        """
        deadline = rospy.Time.now() + rospy.Duration(timeout)

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                rospy.logwarn(f"wait_for_settled: timeout for joint {joint_idx}")
                break

            if (
                self.current_velocities is not None
                and abs(self.current_velocities[joint_idx]) < vel_threshold
            ):
                break

            rospy.sleep(poll_dt)

    def reset_pose(self, timeout=10.0):
        """
        Return the robot to the tiago_experiment_start_1 home pose.

        After this, commanded trajectory state is invalid and must be re-primed.
        """
        if not self.play_motion_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logwarn("play_motion server not available — cannot reset pose")
            return

        goal = PlayMotionGoal()
        goal.motion_name = RESET_MOTION_NAME
        goal.skip_planning = False

        self.play_motion_client.send_goal(goal)
        self.play_motion_client.wait_for_result(rospy.Duration(timeout))

        rospy.loginfo("reset_pose: tiago_experiment_start_1 completed")

        self.reset_command_state()

    def publish_position(self, target_joint_index, magnitude_deg, duration=2.0):
        """
        Legacy helper: move target joint by magnitude_deg degrees.
        Not used for Mode A calibration.
        """
        if self.current_positions is None:
            rospy.logwarn("No joint state yet.")
            return

        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = list(self.current_positions)
        point.velocities = [0.0] * len(self.joint_names)
        point.positions[target_joint_index] += math.radians(magnitude_deg)
        point.time_from_start = rospy.Duration(duration)

        msg.points = [point]
        self.pub.publish(msg)

    def publish_torque(self, target_joint_index, torque_nm):
        """
        Legacy helper: direct torque command.
        Real hardware only; usually not effective in default Gazebo.
        """
        name = self.joint_names[target_joint_index]
        self.torque_pubs[name].publish(Float64(data=torque_nm))

    def publish_velocity(
        self,
        target_joint_index,
        velocity_rad_s,
        total_time=1.0,
        ramp_time=0.2,
        sample_dt=0.05,
        blocking=True
    ):
        """
        Legacy helper for manual limit discovery.

        Keep this if TiagoCalibration.find_joint_limits_manually() still uses it.
        Do not use this for Mode A calibration streaming.
        """
        if self.current_positions is None:
            rospy.logwarn("No joint state yet.")
            return

        if total_time <= 2.0 * ramp_time:
            raise ValueError("total_time must be > 2 * ramp_time")

        q0 = list(self.current_positions)

        v_sign = 1.0 if velocity_rad_s >= 0.0 else -1.0
        v = abs(velocity_rad_s)
        a = v / ramp_time
        cruise_time = total_time - 2.0 * ramp_time

        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.joint_names

        t = sample_dt

        while t <= total_time + 1e-9:
            if t < ramp_time:
                s = 0.5 * a * t * t
                sd = a * t
            elif t < ramp_time + cruise_time:
                s = 0.5 * v * ramp_time + v * (t - ramp_time)
                sd = v
            else:
                td = t - (ramp_time + cruise_time)
                s = (
                    0.5 * v * ramp_time
                    + v * cruise_time
                    + v * td
                    - 0.5 * a * td * td
                )
                sd = max(v - a * td, 0.0)

            point = JointTrajectoryPoint()
            point.positions = list(q0)
            point.velocities = [0.0] * len(self.joint_names)

            point.positions[target_joint_index] = q0[target_joint_index] + v_sign * s
            point.velocities[target_joint_index] = v_sign * sd
            point.time_from_start = rospy.Duration.from_sec(t)

            msg.points.append(point)

            t += sample_dt

        self.pub.publish(msg)

        if blocking:
            rospy.sleep(total_time + 0.1)


class TiagoSubscriber:
    def __init__(self):
        self.joint_names = list(JOINT_NAMES)

        self.ft_sub = rospy.Subscriber(
            WRIST_FT_TOPIC,
            WrenchStamped,
            self.ft_callback,
            queue_size=1
        )

        self.js_sub = rospy.Subscriber(
            JOINT_STATES_TOPIC,
            JointState,
            self.joint_state_callback,
            queue_size=1
        )

        log_dir = os.path.join(
            package_root_for_module(__file__),
            'experiment_results',
            SENSOR_LOG_SCHEMA.directory_name,
        )
        os.makedirs(log_dir, exist_ok=True)

        timestamp = timestamp_token(datetime.now())
        log_path = os.path.join(
            log_dir,
            log_filename(SENSOR_LOG_SCHEMA, timestamp),
        )

        self.log_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        self.csv_writer.writerow(SENSOR_LOG_SCHEMA.columns)

        self.log_file.flush()

        self.current_joint_positions = [0.0] * len(self.joint_names)
        self.start_time = rospy.get_time()

        rospy.loginfo("TiagoSubscriber initialized")
        rospy.loginfo(f"Sensor data logging to: {log_path}")

    def joint_state_callback(self, msg):
        positions = subscriber_joint_positions(
            msg.name,
            msg.position,
            self.joint_names,
        )
        if positions is None:
            return
        self.current_joint_positions = list(positions)

    def ft_callback(self, msg):
        elapsed = rospy.get_time() - self.start_time

        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z

        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z

        joints = self.current_joint_positions

        self.csv_writer.writerow(
            format_sensor_row(
                elapsed,
                (fx, fy, fz),
                (tx, ty, tz),
                joints,
            )
        )

        self.log_file.flush()

    def shutdown(self):
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()

        rospy.loginfo("TiagoSubscriber stopped. Data logged.")
