"""Declarative ROS contracts and pure message-data helpers.

This module deliberately imports no ROS packages, so contract and trajectory
tests can run without a ROS master or generated message modules.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


JOINT_NAMES: Tuple[str, ...] = (
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_joint",
)

ARM_COMMAND_TOPIC = "/arm_controller/command"
JOINT_STATES_TOPIC = "/joint_states"
WRIST_FT_TOPIC = "/wrist_ft"
PLAY_MOTION_ACTION = "play_motion"
RESET_MOTION_NAME = "tiago_experiment_start_1"


def gravity_compensation_topic(joint_name: str) -> str:
    """Return the existing per-joint effort command topic."""

    return "/gravity_compensation/%s/command" % joint_name


GRAVITY_COMPENSATION_TOPICS: Tuple[str, ...] = tuple(
    gravity_compensation_topic(name) for name in JOINT_NAMES
)


@dataclass(frozen=True)
class TopicContract:
    name: str
    message_type: str
    direction: str
    queue_size: Optional[int] = None
    buffer_size: Optional[int] = None


@dataclass(frozen=True)
class ActionContract:
    name: str
    action_type: str
    goal_type: str
    server_wait_s: float
    default_result_timeout_s: float
    motion_name: str
    skip_planning: bool


@dataclass(frozen=True)
class NodeContract:
    script: str
    requested_name: Optional[str]
    anonymous: Optional[bool]


ARM_COMMAND_CONTRACT = TopicContract(
    ARM_COMMAND_TOPIC,
    "trajectory_msgs/JointTrajectory",
    "publisher",
    queue_size=1,
)
JOINT_STATES_CONTRACT = TopicContract(
    JOINT_STATES_TOPIC,
    "sensor_msgs/JointState",
    "subscriber",
    queue_size=1,
)
WRIST_FT_SUBSCRIBER_CONTRACT = TopicContract(
    WRIST_FT_TOPIC,
    "geometry_msgs/WrenchStamped",
    "subscriber",
    queue_size=1,
)
TORQUE_RECORDER_FT_CONTRACT = TopicContract(
    WRIST_FT_TOPIC,
    "geometry_msgs/WrenchStamped",
    "subscriber",
    queue_size=None,
)
GRAVITY_COMPENSATION_CONTRACTS: Tuple[TopicContract, ...] = tuple(
    TopicContract(
        topic,
        "std_msgs/Float64",
        "publisher",
        queue_size=1,
    )
    for topic in GRAVITY_COMPENSATION_TOPICS
)
PLAY_MOTION_CONTRACT = ActionContract(
    name=PLAY_MOTION_ACTION,
    action_type="play_motion_msgs/PlayMotionAction",
    goal_type="play_motion_msgs/PlayMotionGoal",
    server_wait_s=5.0,
    default_result_timeout_s=10.0,
    motion_name=RESET_MOTION_NAME,
    skip_planning=False,
)

NODE_CONTRACTS: Dict[str, NodeContract] = {
    "record_experiments.py": NodeContract(
        "record_experiments.py", "torque_recorder", True
    ),
    "analysis_single_joint.py": NodeContract(
        "analysis_single_joint.py", "tiago_decoder_analysis", True
    ),
    "calibrate_single_joint.py": NodeContract(
        "calibrate_single_joint.py", "tiago_calibration", True
    ),
    "single_joint_data_collector.py": NodeContract(
        "single_joint_data_collector.py", "single_joint_data_collector", True
    ),
    "single_joint_pid_data_collector.py": NodeContract(
        "single_joint_pid_data_collector.py",
        "single_joint_pid_data_collector",
        True,
    ),
    "experiment_camera.py": NodeContract(
        "experiment_camera.py", "experiment_camera_stream", True
    ),
    "demo_graphs.py": NodeContract("demo_graphs.py", "demo_graphs", True),
    "tiago_controller.py": NodeContract("tiago_controller.py", None, None),
}


def joint_state_vectors(
    names: Iterable[str],
    positions: Iterable[float],
    velocities: Optional[Iterable[float]],
    joint_names: Sequence[str] = JOINT_NAMES,
) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """Mirror ``TiagoPublisher.joint_state_cb`` without ROS messages.

    Missing required positions return ``None``.  A missing, empty, or partial
    velocity vector is zero-filled by joint name, as in the existing callback.
    """

    name_list = list(names)
    position_map = dict(zip(name_list, positions))
    velocity_list = [] if velocities is None else list(velocities)
    velocity_map = dict(zip(name_list, velocity_list)) if velocity_list else {}
    try:
        ordered_positions = tuple(position_map[name] for name in joint_names)
    except KeyError:
        return None
    ordered_velocities = tuple(velocity_map.get(name, 0.0) for name in joint_names)
    return ordered_positions, ordered_velocities


def subscriber_joint_positions(
    names: Iterable[str],
    positions: Iterable[float],
    joint_names: Sequence[str] = JOINT_NAMES,
) -> Optional[Tuple[float, ...]]:
    """Mirror the sensor logger's all-or-nothing joint position update."""

    position_map = dict(zip(names, positions))
    ordered = tuple(position_map.get(name) for name in joint_names)
    if any(position is None for position in ordered):
        return None
    return ordered  # type: ignore[return-value]


__all__ = [
    "ARM_COMMAND_CONTRACT",
    "ARM_COMMAND_TOPIC",
    "ActionContract",
    "GRAVITY_COMPENSATION_CONTRACTS",
    "GRAVITY_COMPENSATION_TOPICS",
    "JOINT_NAMES",
    "JOINT_STATES_CONTRACT",
    "JOINT_STATES_TOPIC",
    "NODE_CONTRACTS",
    "NodeContract",
    "PLAY_MOTION_ACTION",
    "PLAY_MOTION_CONTRACT",
    "RESET_MOTION_NAME",
    "TORQUE_RECORDER_FT_CONTRACT",
    "TopicContract",
    "WRIST_FT_SUBSCRIBER_CONTRACT",
    "WRIST_FT_TOPIC",
    "gravity_compensation_topic",
    "joint_state_vectors",
    "subscriber_joint_positions",
]
