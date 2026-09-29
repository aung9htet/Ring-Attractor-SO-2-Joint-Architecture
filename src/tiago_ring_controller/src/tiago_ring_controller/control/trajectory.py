"""Pure numerical construction for TIAGo receding trajectories."""

from dataclasses import dataclass
from typing import Mapping, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TrajectoryPointData:
    """ROS-independent content of one ``JointTrajectoryPoint``."""

    positions: Tuple[float, ...]
    velocities: Tuple[float, ...]
    time_from_start_s: float


@dataclass(frozen=True)
class RecedingTrajectoryData:
    """A horizon and the one-step command state consumed by its caller."""

    points: Tuple[TrajectoryPointData, ...]
    next_commanded_positions: Tuple[float, ...]
    next_commanded_velocities: Tuple[float, ...]


def build_receding_trajectory(
    base_positions: Sequence[float],
    joint_idx: int,
    velocities: Iterable[float],
    dt_s: float,
) -> Optional[RecedingTrajectoryData]:
    """Build the current Mode-A horizon without importing ROS.

    Target position is integrated across the complete horizon, while the
    returned command state advances by exactly the first velocity step.  An
    empty velocity iterable returns ``None``, matching the legacy publisher's
    early return without publishing or changing command state.
    """

    velocity_list = list(velocities)
    if len(velocity_list) == 0:
        return None

    base = list(base_positions)
    position = base[joint_idx]
    points: List[TrajectoryPointData] = []

    for i, velocity in enumerate(velocity_list):
        position += velocity * dt_s
        point_positions = list(base)
        point_velocities = [0.0] * len(base)
        point_positions[joint_idx] = position
        point_velocities[joint_idx] = velocity
        points.append(
            TrajectoryPointData(
                positions=tuple(point_positions),
                velocities=tuple(point_velocities),
                time_from_start_s=dt_s * (i + 1),
            )
        )

    first_velocity = velocity_list[0]
    next_positions = list(base)
    next_positions[joint_idx] += first_velocity * dt_s
    next_velocities = [0.0] * len(base)
    next_velocities[joint_idx] = first_velocity

    return RecedingTrajectoryData(
        points=tuple(points),
        next_commanded_positions=tuple(next_positions),
        next_commanded_velocities=tuple(next_velocities),
    )


def build_multi_joint_trajectory(
    base_positions: Sequence[float],
    horizons: Mapping[int, Iterable[float]],
    dt_s: float,
) -> Optional[RecedingTrajectoryData]:
    """One receding horizon for several joints at once (plan phase 5).

    Each commanded joint integrates its own velocity list; joints without a
    horizon hold their commanded position.  Horizons of different lengths are
    padded with zero velocity.  For a single joint the result equals
    :func:`build_receding_trajectory` point for point.
    """

    lists = {int(joint): list(velocities) for joint, velocities in horizons.items()}
    lists = {joint: values for joint, values in lists.items() if values}
    if not lists:
        return None
    base = list(base_positions)
    length = max(len(values) for values in lists.values())
    positions = list(base)
    points: List[TrajectoryPointData] = []
    for i in range(length):
        point_velocities = [0.0] * len(base)
        for joint, values in lists.items():
            velocity = values[i] if i < len(values) else 0.0
            positions[joint] += velocity * dt_s
            point_velocities[joint] = velocity
        points.append(
            TrajectoryPointData(
                positions=tuple(positions),
                velocities=tuple(point_velocities),
                time_from_start_s=dt_s * (i + 1),
            )
        )
    next_positions = list(base)
    next_velocities = [0.0] * len(base)
    for joint, values in lists.items():
        next_positions[joint] += values[0] * dt_s
        next_velocities[joint] = values[0]
    return RecedingTrajectoryData(
        points=tuple(points),
        next_commanded_positions=tuple(next_positions),
        next_commanded_velocities=tuple(next_velocities),
    )


def build_stop_trajectory(
    base_positions: Sequence[float],
    joint_idx: int,
    dt_s: float = 0.05,
) -> RecedingTrajectoryData:
    """Build the existing three-point zero-velocity stop horizon."""

    result = build_receding_trajectory(
        base_positions=base_positions,
        joint_idx=joint_idx,
        velocities=[0.0, 0.0, 0.0],
        dt_s=dt_s,
    )
    # The fixed non-empty velocity list makes this unreachable; keeping the
    # assertion local avoids changing the public Optional contract above.
    assert result is not None
    return result


def tracking_drift(
    measured_positions: Sequence[float],
    commanded_positions: Sequence[float],
    joint_idx: int,
) -> float:
    """Return the absolute drift used by optional command-state rebasing."""

    return abs(measured_positions[joint_idx] - commanded_positions[joint_idx])


class CommandState:
    """ROS-independent owner of the Mode-A one-step command state."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.commanded_positions: Optional[List[float]] = None
        self.commanded_velocities: Optional[List[float]] = None
        self.initialized = False

    def prime(
        self,
        measured_positions: Sequence[float],
        measured_velocities: Optional[Sequence[float]] = None,
        force: bool = False,
    ) -> bool:
        """Prime from measured state, preserving the facade's force behavior."""

        if force or not self.initialized:
            self.commanded_positions = list(measured_positions)
            if measured_velocities is None:
                self.commanded_velocities = [0.0] * len(measured_positions)
            else:
                self.commanded_velocities = list(measured_velocities)
            self.initialized = True
        return True

    def maybe_rebase(
        self,
        measured_positions: Sequence[float],
        measured_velocities: Optional[Sequence[float]],
        joint_idx: int,
        threshold: Optional[float],
    ) -> bool:
        """Re-prime when the existing strict ``drift > threshold`` test fires."""

        if threshold is None or self.commanded_positions is None:
            return False
        if tracking_drift(
            measured_positions,
            self.commanded_positions,
            joint_idx,
        ) > threshold:
            self.prime(measured_positions, measured_velocities, force=True)
            return True
        return False

    def build(
        self,
        joint_idx: int,
        velocities: Iterable[float],
        dt_s: float,
    ) -> Optional[RecedingTrajectoryData]:
        """Build a horizon and consume exactly one command step."""

        if not self.initialized or self.commanded_positions is None:
            raise RuntimeError("Command state has not been primed")

        result = build_receding_trajectory(
            self.commanded_positions,
            joint_idx,
            velocities,
            dt_s,
        )
        if result is None:
            return None

        # Match the compatibility facade: callers may retain a reference to
        # the commanded-position list, so consuming one horizon mutates that
        # list in place.  Commanded velocities are replaced by a fresh list in
        # the legacy implementation.
        self.commanded_positions[:] = result.next_commanded_positions
        self.commanded_velocities = list(result.next_commanded_velocities)
        return result

    def build_multi(
        self,
        horizons: Mapping[int, Iterable[float]],
        dt_s: float,
    ) -> Optional[RecedingTrajectoryData]:
        """Several joints per horizon; consumes one step of each commanded joint."""

        if not self.initialized or self.commanded_positions is None:
            raise RuntimeError("Command state has not been primed")
        result = build_multi_joint_trajectory(self.commanded_positions, horizons, dt_s)
        if result is None:
            return None
        self.commanded_positions[:] = result.next_commanded_positions
        self.commanded_velocities = list(result.next_commanded_velocities)
        return result


__all__ = [
    "CommandState",
    "build_multi_joint_trajectory",
    "RecedingTrajectoryData",
    "TrajectoryPointData",
    "build_receding_trajectory",
    "build_stop_trajectory",
    "tracking_drift",
]
