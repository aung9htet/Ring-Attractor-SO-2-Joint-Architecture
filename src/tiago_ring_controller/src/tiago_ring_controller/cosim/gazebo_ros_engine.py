"""Gazebo/ROS engine: owns the robot's time through pause/step/unpause.

The engine talks to ROS only through a :class:`RosTransport`.  The real
:class:`RospyTransport` imports ``rospy`` lazily in its constructor, so this
module imports cleanly without ROS and the engine is unit-tested against a
fake transport.  All waits are wall-clock waits with timeouts; nothing here
calls ``rospy.sleep`` or ``rospy.Rate``.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..control.trajectory import CommandState, TrajectoryPointData, build_stop_trajectory
from ..ros.transport import JOINT_NAMES, RESET_MOTION_NAME, joint_state_vectors
from .datapack import DataPack
from .fakes import command_horizons, command_limits
from .engine import Engine, EngineError, StepTimeoutError
from .stepping import GazeboStepper, StepResult, make_stepper


JointSnapshot = Tuple[Tuple[float, ...], Tuple[float, ...], float]


def finite_difference_velocities(previous: JointSnapshot, current: JointSnapshot) -> Optional[List[float]]:
    """Joint velocities from the positions of two snapshots; ``None`` when the stamp did not advance."""

    elapsed = current[2] - previous[2]
    if elapsed <= 0.0:
        return None
    return [(after - before) / elapsed for before, after in zip(previous[0], current[0])]


class RosTransport(ABC):
    """Minimal ROS/Gazebo surface used by :class:`GazeboRosEngine`."""

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def latest_joint_state(self) -> Optional[JointSnapshot]:
        """``(positions, velocities, stamp_s)`` copied under a lock, or ``None``."""

    @abstractmethod
    def sim_time_s(self) -> float: ...

    @abstractmethod
    def wait_until_sim_time(self, target_s: float, timeout_s: float) -> bool: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def unpause(self) -> None: ...

    @abstractmethod
    def publish_trajectory(
        self, joint_names: Sequence[str], points: Sequence[TrajectoryPointData], stamp_s: float
    ) -> None: ...

    @abstractmethod
    def play_motion(self, motion_name: str, timeout_s: float) -> str:
        """Run a ``play_motion`` goal and return the terminal state name."""

    def step_world(self, iterations: int) -> float:
        raise EngineError("PluginStepper requires the cosim step plugin (plan B phase 5)")

    def model_pose(self, name: str) -> Optional[Tuple[float, float, float]]:
        """``(x, y, yaw)`` of a simulated model in the world, or ``None`` when the transport cannot tell."""

        return None

    def set_model_pose(self, name: str, x: float, y: float, yaw: float) -> bool:
        """Put a simulated model at ``(x, y, yaw)`` with zero velocity; False when unsupported."""

        return False

    def get_param(self, name: str) -> Any:
        return None

    def wait_for_joint_state(self, timeout_s: float, poll_s: float = 0.02) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.latest_joint_state() is None:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)
        return True


class RospyTransport(RosTransport):
    """The real thing.  Imports rospy and the message types on construction."""

    def __init__(
        self,
        node_name: str = "cosim_gazebo_engine",
        joint_names: Sequence[str] = JOINT_NAMES,
        anonymous: bool = True,
        init_node: bool = True,
    ) -> None:
        import rospy  # noqa: WPS433 - deliberate lazy import
        import actionlib
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Empty
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
        from actionlib_msgs.msg import GoalStatus
        from ..ros.transport import ARM_COMMAND_TOPIC, JOINT_STATES_TOPIC, PLAY_MOTION_ACTION

        self._rospy = rospy
        self._actionlib = actionlib
        self._Clock = Clock
        self._JointState = JointState
        self._Empty = Empty
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._PlayMotionAction = PlayMotionAction
        self._PlayMotionGoal = PlayMotionGoal
        self._GoalStatus = GoalStatus
        self._topics = (ARM_COMMAND_TOPIC, JOINT_STATES_TOPIC, PLAY_MOTION_ACTION)
        self.node_name = node_name
        self.anonymous = anonymous
        self.init_node = init_node
        self.joint_names = tuple(joint_names)
        self._lock = threading.Lock()
        self._clock_cv = threading.Condition()
        self._sim_time_s = 0.0
        self._snapshot: Optional[JointSnapshot] = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        rospy = self._rospy
        if self.init_node:
            rospy.init_node(self.node_name, anonymous=self.anonymous)
        arm_topic, joint_topic, action_name = self._topics
        self._pub = rospy.Publisher(arm_topic, self._JointTrajectory, queue_size=1)
        self._joint_sub = rospy.Subscriber(joint_topic, self._JointState, self._joint_cb, queue_size=1)
        self._clock_sub = rospy.Subscriber("/clock", self._Clock, self._clock_cb, queue_size=1)
        rospy.wait_for_service("/gazebo/pause_physics", timeout=10.0)
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=10.0)
        self._pause = rospy.ServiceProxy("/gazebo/pause_physics", self._Empty)
        self._unpause = rospy.ServiceProxy("/gazebo/unpause_physics", self._Empty)
        self._client = self._actionlib.SimpleActionClient(action_name, self._PlayMotionAction)
        self._initialized = True

    def shutdown(self) -> None:
        if not self._initialized:
            return
        self._joint_sub.unregister()
        self._clock_sub.unregister()
        self._initialized = False

    def _joint_cb(self, msg: Any) -> None:
        vectors = joint_state_vectors(msg.name, msg.position, msg.velocity, self.joint_names)
        if vectors is None:
            return
        positions, velocities = vectors
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else self._sim_time_s
        with self._lock:
            self._snapshot = (tuple(positions), tuple(velocities), float(stamp))

    def _clock_cb(self, msg: Any) -> None:
        with self._clock_cv:
            self._sim_time_s = msg.clock.to_sec()
            self._clock_cv.notify_all()

    def latest_joint_state(self) -> Optional[JointSnapshot]:
        with self._lock:
            return self._snapshot

    def sim_time_s(self) -> float:
        with self._clock_cv:
            return self._sim_time_s

    def wait_until_sim_time(self, target_s: float, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._clock_cv:
            while self._sim_time_s < target_s:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._clock_cv.wait(min(remaining, 0.05))
            return True

    def pause(self) -> None:
        self._pause()

    def unpause(self) -> None:
        self._unpause()

    def publish_trajectory(
        self, joint_names: Sequence[str], points: Sequence[TrajectoryPointData], stamp_s: float
    ) -> None:
        rospy = self._rospy
        msg = self._JointTrajectory()
        msg.header.stamp = rospy.Time.from_sec(stamp_s)
        msg.joint_names = list(joint_names)
        for point_data in points:
            point = self._JointTrajectoryPoint()
            point.positions = list(point_data.positions)
            point.velocities = list(point_data.velocities)
            point.time_from_start = rospy.Duration.from_sec(point_data.time_from_start_s)
            msg.points.append(point)
        self._pub.publish(msg)

    def play_motion(self, motion_name: str, timeout_s: float) -> str:
        rospy = self._rospy
        if not self._client.wait_for_server(rospy.Duration(5.0)):
            return "NO_SERVER"
        goal = self._PlayMotionGoal()
        goal.motion_name = motion_name
        goal.skip_planning = False
        self._client.send_goal(goal)
        # wait_for_result blocks on ROS time; that is fine here because reset
        # runs unpaused by design (see GazeboRosEngine.reset).
        finished = self._client.wait_for_result(rospy.Duration(timeout_s))
        if not finished:
            return "TIMEOUT"
        state = self._client.get_state()
        return self._GoalStatus.to_string(state) if hasattr(self._GoalStatus, "to_string") else str(state)

    def get_param(self, name: str) -> Any:
        return self._rospy.get_param(name, None)

    def _model_services(self):
        if getattr(self, "_get_model", None) is None:
            from gazebo_msgs.srv import GetModelState, SetModelState

            self._get_model = self._rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            self._set_model = self._rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        return self._get_model, self._set_model

    def model_pose(self, name: str) -> Optional[Tuple[float, float, float]]:
        get_model, _ = self._model_services()
        try:
            state = get_model(model_name=name, relative_entity_name="world")
        except Exception:  # noqa: BLE001 - no such service or model: the caller treats it as unknown
            return None
        if not state.success:
            return None
        q = state.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (float(state.pose.position.x), float(state.pose.position.y), float(yaw))

    def set_model_pose(self, name: str, x: float, y: float, yaw: float) -> bool:
        from gazebo_msgs.msg import ModelState

        get_model, set_model = self._model_services()
        try:
            current = get_model(model_name=name, relative_entity_name="world")
            if not current.success:
                return False
            state = ModelState()
            state.model_name = name
            state.reference_frame = "world"
            state.pose.position.x, state.pose.position.y = float(x), float(y)
            state.pose.position.z = current.pose.position.z
            state.pose.orientation.z, state.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            return bool(set_model(model_state=state).success)
        except Exception:  # noqa: BLE001
            return False


class GazeboRosEngine(Engine):
    """Composes the existing command-state contract with lock-stepped physics."""

    inputs = frozenset({"arm_velocity_cmd"})
    outputs = frozenset({"joint_state", "sim_clock"})

    def __init__(
        self,
        transport: RosTransport,
        stepper: Optional[GazeboStepper] = None,
        stepper_name: str = "clock_wait",
        max_step_size_s: float = 0.001,
        step_timeout_s: float = 5.0,
        joint_names: Sequence[str] = JOINT_NAMES,
        reset_motion: str = RESET_MOTION_NAME,
        reset_timeout_s: float = 20.0,
        reset_retries: int = 2,
        base_model: str = "tiago",
        base_pose: Optional[Sequence[float]] = (0.0, 0.0, 0.0),
        joint_state_timeout_s: float = 10.0,
        settle_vel_threshold: float = 0.01,
        settle_timeout_s: float = 8.0,
        settle_window_s: float = 0.1,
        name: str = "robot",
    ) -> None:
        super().__init__(name)
        self.transport = transport
        self.stepper = stepper or make_stepper(stepper_name, transport, max_step_size_s, step_timeout_s)
        self.max_step_size_s = float(max_step_size_s)
        self.joint_names = tuple(joint_names)
        self.reset_motion = reset_motion
        self.reset_timeout_s = float(reset_timeout_s)
        self.reset_retries = int(reset_retries)
        self.reset_failures = 0
        self._reset_positions: Optional[List[float]] = None
        #: the robot's mobile base is not anchored: an arm that pushes on the table shoves it.  A rebuild
        #: reset puts it back at ``base_pose`` (x, y, yaw; ``None`` leaves it alone); ``base_log`` holds
        #: where it was found, where the trial started and where it ended.
        self.base_model = base_model
        self.base_pose = None if base_pose is None else tuple(float(v) for v in base_pose)
        self.base_log: Dict[str, Any] = {}
        self.joint_state_timeout_s = float(joint_state_timeout_s)
        self.settle_vel_threshold = float(settle_vel_threshold)
        self.settle_timeout_s = float(settle_timeout_s)
        self.settle_window_s = float(settle_window_s)
        self.command_state = CommandState()
        self.pending: Optional[DataPack] = None
        self.last_step: Optional[StepResult] = None
        self.step_log: List[StepResult] = []
        self.published: List[Dict[str, Any]] = []
        #: velocities from position differences between joint-state stamps.  The velocity field
        #: of /joint_states is not used for decisions: Gazebo reports about -1.45 rad/s for a
        #: resting arm_6_joint (docs/archive/velocity/calibration.md).
        self.velocities_fd: List[float] = [0.0] * len(self.joint_names)
        self._fd_reference: Optional[JointSnapshot] = None
        self.clamped_joints: List[int] = []
        self.start_pose: Dict[int, float] = {}
        self.start_speed = 0.4      # rad/s of the move to a start pose
        self.paused = False
        self.rebuild_count = 0

    # -- lifecycle --------------------------------------------------------
    def _do_initialize(self) -> None:
        self.transport.initialize()
        if not self.transport.wait_for_joint_state(self.joint_state_timeout_s):
            raise EngineError("no /joint_states within %.1f s" % self.joint_state_timeout_s)

    def _snapshot(self) -> JointSnapshot:
        snapshot = self.transport.latest_joint_state()
        if snapshot is None:
            raise EngineError("no joint state available")
        return snapshot

    def _update_velocity_estimate(self) -> Optional[List[float]]:
        """Refresh ``velocities_fd`` from the newest snapshot; ``None`` while no newer joint state arrived."""

        current = self._snapshot()
        if self._fd_reference is None:
            self._fd_reference = current
            return None
        velocities = finite_difference_velocities(self._fd_reference, current)
        if velocities is None:
            return None
        self.velocities_fd = velocities
        self._fd_reference = current
        return velocities

    def wait_until_still(self, joint_indices: Optional[Sequence[int]] = None) -> bool:
        """Simulation unpaused: wait until the joints stop moving, judged from positions.

        Each check compares two joint states ``settle_window_s`` of simulated
        time apart.  Returns ``False`` after ``settle_timeout_s`` of wall time.
        """

        joints = range(len(self.joint_names)) if joint_indices is None else list(joint_indices)
        deadline = time.monotonic() + self.settle_timeout_s
        self._fd_reference = self._snapshot()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            advanced = self.transport.wait_until_sim_time(
                self.transport.sim_time_s() + self.settle_window_s, min(remaining, 1.0)
            )
            velocities = self._update_velocity_estimate()
            if velocities is not None and all(abs(velocities[j]) < self.settle_vel_threshold for j in joints):
                return True
            if not advanced:
                time.sleep(0.01)     # clock not running: do not spin

    def wait_for_settled(self, joint_index: int) -> bool:
        """One joint of :meth:`wait_until_still` (simulation unpaused)."""

        return self.wait_until_still([joint_index])

    def set_start_pose(self, positions: Mapping[int, float]) -> None:
        """Joint angles (by index) a rebuild reset moves to after the home motion."""

        self.start_pose = {int(joint): float(value) for joint, value in positions.items()}

    def move_to(self, positions: Mapping[int, float]) -> bool:
        """Simulation unpaused: one smooth trajectory to ``positions``, then wait until still."""

        current = list(self._snapshot()[0])
        target = list(current)
        for joint, value in positions.items():
            target[int(joint)] = float(value)
        distance = max(abs(a - b) for a, b in zip(target, current))
        duration = max(2.0, distance / self.start_speed)
        point = TrajectoryPointData(positions=tuple(target), velocities=(0.0,) * len(target), time_from_start_s=duration)
        self.transport.publish_trajectory(self.joint_names, [point], self.transport.sim_time_s())
        self.transport.wait_until_sim_time(self.transport.sim_time_s() + duration + 0.2, duration * 3 + 10.0)
        return self.wait_until_still()

    def _do_reset(self) -> None:
        if self.reset_mode == "rebuild":
            self.transport.unpause()
            self.paused = False
            state = self.transport.play_motion(self.reset_motion, self.reset_timeout_s)
            for _ in range(self.reset_retries):
                if state == "SUCCEEDED":
                    break
                # The motion is planned from the current pose and can fail from where a trial left the
                # arm: go straight back to where the last reset ended, then ask again.
                if self._reset_positions is not None:
                    self.move_to(dict(enumerate(self._reset_positions)))
                else:
                    self.wait_until_still()
                self.reset_failures += 1
                state = self.transport.play_motion(self.reset_motion, self.reset_timeout_s)
            if state != "SUCCEEDED":
                raise EngineError("play_motion %r ended in state %s" % (self.reset_motion, state))
            # The home motion leaves residual velocity; wait for all joints.
            self.wait_until_still()
            self._reset_positions = list(self._snapshot()[0])
            self.base_log = {"found": self.transport.model_pose(self.base_model)}
            if self.base_pose is not None and self.transport.set_model_pose(self.base_model, *self.base_pose):
                self.wait_until_still()
            if self.start_pose:
                self.move_to(self.start_pose)
            self.base_log["start"] = self.transport.model_pose(self.base_model)
            self.rebuild_count += 1
        # continue: the arm stays where the previous trial's stop left it
        # (physics already paused by stop_and_settle); just re-prime.
        snapshot = self._snapshot()
        self.velocities_fd = [0.0] * len(self.joint_names)
        self._fd_reference = snapshot
        self.clamped_joints = []
        self.command_state.reset()
        self.command_state.prime(snapshot[0], list(self.velocities_fd), force=True)
        if not self.paused:
            self.transport.pause()
            self.paused = True
        self.stepper.reset()
        self.pending = None
        self.last_step = None
        self.step_log = []
        self.published = []

    # -- datapacks --------------------------------------------------------
    def _do_get_datapacks(self) -> Dict[str, DataPack]:
        positions, velocities, stamp_s = self._snapshot()
        sim_time_ms = self.transport.sim_time_s() * 1000.0
        step = self.last_step
        return {
            "joint_state": DataPack(
                "joint_state",
                self.t_ms,
                {
                    "positions": list(positions),
                    "velocities": list(velocities),
                    "velocities_fd": list(self.velocities_fd),
                    "commanded_positions": list(self.command_state.commanded_positions or []),
                    "clamped_joints": list(self.clamped_joints),
                    "stamp_ms": stamp_s * 1000.0,
                    "sim_time_ms": sim_time_ms,
                },
            ),
            "sim_clock": DataPack(
                "sim_clock",
                self.t_ms,
                {
                    "sim_time_ms": sim_time_ms,
                    "stepper": self.stepper.name,
                    "sim_time_before_ms": None if step is None else step.sim_time_before_s * 1000.0,
                    "sim_time_after_ms": None if step is None else step.sim_time_after_s * 1000.0,
                    "overshoot_ms": 0.0 if step is None else step.overshoot_s * 1000.0,
                    "step_wall_s": 0.0 if step is None else step.wall_s,
                },
            ),
        }

    def _do_set_datapacks(self, packs: Dict[str, DataPack]) -> None:
        self.pending = packs.get("arm_velocity_cmd")

    def _publish_horizon(self, joint_index: int, velocities: Sequence[float], dt_s: float,
                         horizons: Optional[Dict[int, List[float]]] = None,
                         limits: Optional[Dict[int, Tuple[float, float]]] = None) -> None:
        horizons = horizons or {int(joint_index): list(velocities)}
        trajectory = self.command_state.build_multi(horizons, dt_s, limits)
        if trajectory is None:
            raise EngineError("empty velocity horizon")
        self.clamped_joints = list(trajectory.clamped_joints)
        stamp_s = self.transport.sim_time_s()
        self.transport.publish_trajectory(self.joint_names, trajectory.points, stamp_s)
        self.published.append(
            {"t_ms": self.t_ms, "joint_index": int(joint_index), "joint_indices": sorted(horizons),
             "stamp_s": stamp_s, "velocities": [float(v) for v in velocities],
             "points": [list(point.positions) for point in trajectory.points],
             "clamped_joints": list(trajectory.clamped_joints)}
        )

    def _do_advance(self, dt_ms: float) -> None:
        if not self.paused:
            raise EngineError("advance() requires paused physics; call reset() first")
        dt_s = dt_ms / 1000.0
        if self.pending is not None:
            cmd = self.pending
            self.pending = None
            self._publish_horizon(
                int(cmd["joint_index"]), cmd["velocities"], dt_s, command_horizons(cmd), command_limits(cmd)
            )
        else:
            self.clamped_joints = []
        iterations = dt_s / self.max_step_size_s
        if abs(iterations - round(iterations)) > 1e-9:
            raise EngineError("dt %.6f s is not a whole number of physics iterations" % dt_s)
        self.last_step = self.stepper.step(int(round(iterations)))
        self.step_log.append(self.last_step)
        self._update_velocity_estimate()

    def stop_and_settle(self, joint_index: int, dt_s: float = 0.05) -> bool:
        """Publish the legacy stop horizon, then step until the joint's position stops changing."""

        stop = build_stop_trajectory(
            list(self.command_state.commanded_positions or self._snapshot()[0]), joint_index, dt_s
        )
        self.transport.publish_trajectory(self.joint_names, stop.points, self.transport.sim_time_s())
        self.command_state.commanded_positions[:] = list(stop.next_commanded_positions)
        iterations = int(round(dt_s / self.max_step_size_s))
        elapsed = 0.0
        while elapsed < self.settle_timeout_s:
            try:
                self.stepper.step(iterations)
            except StepTimeoutError:
                return False
            elapsed += dt_s
            velocities = self._update_velocity_estimate()
            if velocities is not None and abs(velocities[joint_index]) < self.settle_vel_threshold:
                return True
        return False

    def _do_finish_trial(self) -> None:
        if self.published:
            for joint in self.published[-1].get("joint_indices") or [self.published[-1]["joint_index"]]:
                self.stop_and_settle(int(joint))
        self.base_log["end"] = self.transport.model_pose(self.base_model)

    def base_shift(self) -> Dict[str, float]:
        """How far the base moved: ``trial_m`` during the last trial, ``found_m`` how far from
        ``base_pose`` the reset found it (NaN when the transport cannot tell)."""

        def distance(a, b):
            return float("nan") if a is None or b is None else float(math.hypot(a[0] - b[0], a[1] - b[1]))

        log = self.base_log
        return {"trial_m": distance(log.get("start"), log.get("end")), "found_m": distance(log.get("found"), self.base_pose)}

    def _do_shutdown(self) -> None:
        try:
            self.transport.unpause()
            self.paused = False
        finally:
            self.transport.shutdown()


__all__ = ["GazeboRosEngine", "JointSnapshot", "RosTransport", "RospyTransport", "finite_difference_velocities"]
