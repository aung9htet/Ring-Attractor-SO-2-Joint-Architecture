"""Deterministic fake engines so the loop is testable without simulators.

* :class:`FakeRobotEngine` — seven-joint kinematic model.  The commanded
  state is the unchanged :class:`CommandState` (first horizon point per
  tick); the measured joint follows the commanded one through a first-order
  lag integrated in sub-steps.
* :class:`FakeNestEngine` — either a scripted sequence of ``(left, right)``
  counts (for exact loop-order tests) or a stub in which the state bump
  drifts towards the goal bump and the gain counts are proportional to the
  remaining distance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..control.trajectory import CommandState, build_stop_trajectory
from ..ros.transport import JOINT_NAMES
from .datapack import DataPack
from .engine import Engine, EngineError


def ring_readout(delta: np.ndarray) -> Tuple[float, Optional[int], float]:
    """Return ``(total, argmax_or_None, centroid_or_nan)`` like the collector."""

    delta = np.asarray(delta, dtype=float)
    total = float(delta.sum())
    if total <= 0.0:
        return total, None, float("nan")
    indices = np.arange(len(delta), dtype=float)
    return total, int(np.argmax(delta)), float(np.dot(indices, delta) / total)


class FakeRobotEngine(Engine):
    inputs = frozenset({"arm_velocity_cmd"})
    outputs = frozenset({"joint_state", "sim_clock"})

    def __init__(
        self,
        name: str = "robot",
        home: Optional[Sequence[float]] = None,
        tau_s: float = 0.08,
        substeps: int = 50,
        settle_vel_threshold: float = 0.01,
        settle_timeout_s: float = 8.0,
    ) -> None:
        super().__init__(name)
        self.joint_names = tuple(JOINT_NAMES)
        self.home = [0.0] * len(self.joint_names) if home is None else [float(v) for v in home]
        if len(self.home) != len(self.joint_names):
            raise ValueError("home pose must have %d joints" % len(self.joint_names))
        self.tau_s = float(tau_s)
        self.substeps = int(substeps)
        self.settle_vel_threshold = float(settle_vel_threshold)
        self.settle_timeout_s = float(settle_timeout_s)
        self.command_state = CommandState()
        self.positions: List[float] = list(self.home)
        self.velocities: List[float] = [0.0] * len(self.home)
        self.sim_time_s = 0.0
        self.pending: Optional[DataPack] = None
        self.published: List[Dict[str, Any]] = []
        self.reset_count = 0

    def _do_reset(self) -> None:
        self.positions = list(self.home)
        self.velocities = [0.0] * len(self.home)
        self.sim_time_s = 0.0
        self.pending = None
        self.published = []
        self.command_state.reset()
        self.command_state.prime(self.positions, self.velocities, force=True)
        self.reset_count += 1

    def _do_get_datapacks(self) -> Dict[str, DataPack]:
        return {
            "joint_state": DataPack(
                "joint_state",
                self.t_ms,
                {
                    "positions": list(self.positions),
                    "velocities": list(self.velocities),
                    "commanded_positions": list(self.command_state.commanded_positions or []),
                    "sim_time_ms": self.sim_time_s * 1000.0,
                },
            ),
            "sim_clock": DataPack(
                "sim_clock",
                self.t_ms,
                {"sim_time_ms": self.sim_time_s * 1000.0, "overshoot_ms": 0.0},
            ),
        }

    def _do_set_datapacks(self, packs: Dict[str, DataPack]) -> None:
        self.pending = packs.get("arm_velocity_cmd")

    def _track(self, target: Sequence[float], dt_s: float) -> None:
        h = dt_s / self.substeps
        gain = 1.0 - math.exp(-h / self.tau_s)
        before = list(self.positions)
        for _ in range(self.substeps):
            for j in range(len(self.positions)):
                self.positions[j] += (target[j] - self.positions[j]) * gain
        self.velocities = [(a - b) / dt_s for a, b in zip(self.positions, before)]
        self.sim_time_s += dt_s

    def _do_advance(self, dt_ms: float) -> None:
        dt_s = dt_ms / 1000.0
        if self.pending is not None:
            cmd = self.pending
            self.pending = None
            velocities = [float(v) for v in cmd["velocities"]]
            trajectory = self.command_state.build(int(cmd["joint_index"]), velocities, dt_s)
            if trajectory is None:
                raise EngineError("empty velocity horizon")
            self.published.append(
                {
                    "t_ms": self.t_ms,
                    "joint_index": int(cmd["joint_index"]),
                    "velocities": velocities,
                    "points": [list(point.positions) for point in trajectory.points],
                }
            )
            target = list(trajectory.points[0].positions)
        else:
            target = list(self.command_state.commanded_positions or self.positions)
        self._track(target, dt_s)

    def stop_and_settle(self, joint_index: int, dt_s: float = 0.05) -> None:
        """Zero-velocity horizon, then integrate until the joint is still."""

        stop = build_stop_trajectory(
            list(self.command_state.commanded_positions or self.positions), joint_index, dt_s
        )
        target = list(stop.points[0].positions)
        self.command_state.commanded_positions[:] = list(stop.next_commanded_positions)
        elapsed = 0.0
        while elapsed < self.settle_timeout_s:
            self._track(target, dt_s)
            elapsed += dt_s
            if abs(self.velocities[joint_index]) < self.settle_vel_threshold:
                break

    def _do_finish_trial(self) -> None:
        joint = 0
        if self.published:
            joint = self.published[-1]["joint_index"]
        self.stop_and_settle(joint)


class FakeNestEngine(Engine):
    inputs = frozenset({"state_bump", "goal_bump"})
    outputs = frozenset({"ring_counts"})

    def __init__(
        self,
        name: str = "nest",
        population_size: int = 100,
        script: Optional[Sequence[Tuple[int, int]]] = None,
        count_gain: float = 2.0,
        drift_per_step: float = 1.0,
        bump_half_width: int = 5,
    ) -> None:
        super().__init__(name)
        self.population_size = int(population_size)
        self.script = None if script is None else [(int(l), int(r)) for l, r in script]
        self.count_gain = float(count_gain)
        self.drift_per_step = float(drift_per_step)
        self.bump_half_width = int(bump_half_width)
        self.state_index: Optional[float] = None
        self.goal_index: Optional[float] = None
        self.pending: Dict[str, DataPack] = {}
        self.applied_bumps: List[Dict[str, Any]] = []
        self.last: Optional[DataPack] = None
        self.reset_count = 0

    def _do_reset(self) -> None:
        self.state_index = None
        self.goal_index = None
        self.pending = {}
        self.applied_bumps = []
        self.last = None
        self.reset_count += 1

    def _do_get_datapacks(self) -> Dict[str, DataPack]:
        return {} if self.last is None else {"ring_counts": self.last}

    def _do_set_datapacks(self, packs: Dict[str, DataPack]) -> None:
        self.pending.update(packs)

    def _apply_pending(self) -> None:
        for key in ("goal_bump", "state_bump"):
            pack = self.pending.pop(key, None)
            if pack is None:
                continue
            self.applied_bumps.append(
                {"name": key, "center_index": int(pack["center_index"]),
                 "applied_at_step": self.step_index, "t_ms": pack.t_ms}
            )
            if key == "goal_bump":
                self.goal_index = float(pack["center_index"])
            else:
                self.state_index = float(pack["center_index"])

    def _counts(self) -> Tuple[int, int]:
        step = self.step_index
        if self.script is not None:
            if step < len(self.script):
                return self.script[step]
            return (0, 0)
        if self.state_index is None or self.goal_index is None:
            return (0, 0)
        diff = self.goal_index - self.state_index
        magnitude = int(round(self.count_gain * abs(diff)))
        return (magnitude, 0) if diff < 0 else (0, magnitude)

    def _do_advance(self, dt_ms: float) -> None:
        self._apply_pending()
        left, right = self._counts()
        delta = np.zeros(self.population_size)
        if self.state_index is not None:
            center = int(round(self.state_index)) % self.population_size
            for offset in range(-self.bump_half_width, self.bump_half_width + 1):
                delta[(center + offset) % self.population_size] = 1.0
            delta[center] = 2.0
            if self.goal_index is not None:
                diff = self.goal_index - self.state_index
                self.state_index += max(-self.drift_per_step, min(self.drift_per_step, diff))
        total, bump, centroid = ring_readout(delta)
        t_after = self.t_ms + dt_ms
        self.last = DataPack(
            "ring_counts",
            t_after,
            {
                "left": int(left),
                "right": int(right),
                "r1_delta": delta.tolist(),
                "r1_spike_count": total,
                "r1_bump_index": bump,
                "r1_centroid": centroid,
                "t_nest_ms": t_after,
                "nest_step": self.step_index + 1,
                "hidden_ms": 0.0,
                "step_mode": "fake",
            },
        )


__all__ = ["FakeNestEngine", "FakeRobotEngine", "ring_readout"]
