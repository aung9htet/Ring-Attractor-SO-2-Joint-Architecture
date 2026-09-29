"""Joint (robot binding) and Goal (constant or scheduled angle source)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from .base import Block, Port, signal_in, signal_out
from .params import GOAL_SCHEMA, JOINT_SCHEMA


class Joint(Block):
    """One arm joint of the robot engine: measured angle out, velocity horizon in."""

    type_name: ClassVar[str] = "Joint"
    schema = JOINT_SCHEMA
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_out("angle", "measured joint angle in rad"),
        signal_out("velocity_measured", "measured joint velocity in rad/s"),
        signal_in("velocity", "receding-horizon velocities to command", required=False),
    )

    @property
    def index(self) -> int:
        return int(self.params["index"])

    def limits(self) -> Tuple[float, float]:
        return float(self.params["joint_min"]), float(self.params["joint_max"])

    def read(self, joint_state: Mapping[str, Any]) -> Dict[str, float]:
        """Extract this joint's angle and velocity from a ``joint_state`` datapack."""

        positions = joint_state["positions"]
        velocities = joint_state.get("velocities") or [0.0] * len(positions)
        return {"angle": float(positions[self.index]), "velocity_measured": float(velocities[self.index])}


class Goal(Block):
    """A target angle: constant, or a schedule ``[(t_ms, angle), ...]`` set at run time."""

    type_name: ClassVar[str] = "Goal"
    schema = GOAL_SCHEMA
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_out("angle", "goal angle in rad"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.schedule: Optional[Sequence[Tuple[float, float]]] = None
        self.override: Optional[float] = None

    def set(self, angle_rad: float) -> None:
        self.override = float(angle_rad)

    def set_schedule(self, schedule: Sequence[Tuple[float, float]]) -> None:
        self.schedule = sorted((float(t), float(a)) for t, a in schedule)

    def value(self, t_ms: float = 0.0) -> float:
        if self.schedule:
            angle = self.schedule[0][1]
            for at, value in self.schedule:
                if t_ms + 1e-9 >= at:
                    angle = value
            return angle
        if self.override is not None:
            return self.override
        return float(self.params["angle_rad"])

    def step(self, t_ms: float = 0.0) -> Dict[str, float]:
        return {"angle": self.value(t_ms)}


__all__ = ["Goal", "Joint"]
