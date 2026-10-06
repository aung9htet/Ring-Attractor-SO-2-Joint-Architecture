"""Gazebo stepping strategies behind one ``step(n_iterations)`` interface.

* :class:`ClockWaitStepper` (phase 1, pure Python): unpause, wait on the
  wall clock until ``/clock`` reaches the target, pause.  Bounded but not
  exact: the overshoot (one physics iteration at RTF≈1) is measured and
  returned so it can be logged with every tick.  Targets follow an absolute
  schedule, so the overshoot of one step is not added to the next.
* :class:`PluginStepper` (phase 2/5, exact): a Gazebo world plugin exposes a
  ``/cosim/step`` service that calls ``world->Step(n)`` while physics stays
  paused.  This class is the hook; the transport raises until the plugin
  exists.

Nothing here calls ``rospy.sleep`` or ``rospy.Rate``: with ``/use_sim_time``
those block on ``/clock`` and would hang while Gazebo is paused.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .engine import StepTimeoutError


@dataclass(frozen=True)
class StepResult:
    iterations: int
    sim_time_before_s: float
    sim_time_after_s: float
    target_s: float
    wall_s: float

    @property
    def advanced_s(self) -> float:
        return self.sim_time_after_s - self.sim_time_before_s

    @property
    def overshoot_s(self) -> float:
        return self.sim_time_after_s - self.target_s


class GazeboStepper(ABC):
    name = "stepper"

    def reset(self) -> None:
        """Called when a trial starts: the clock may have run freely since the last step."""

        return None

    @abstractmethod
    def step(self, iterations: int) -> StepResult:
        raise NotImplementedError


class ClockWaitStepper(GazeboStepper):
    name = "clock_wait"

    def __init__(
        self,
        transport: Any,
        max_step_size_s: float = 0.001,
        timeout_s: float = 5.0,
        clock_tolerance_s: float = 1e-9,
    ) -> None:
        self.transport = transport
        self.max_step_size_s = float(max_step_size_s)
        self.timeout_s = float(timeout_s)
        self.clock_tolerance_s = float(clock_tolerance_s)
        #: target of the previous step, while consecutive steps form one schedule
        self._schedule_s: Optional[float] = None

    def reset(self) -> None:
        self._schedule_s = None

    def step(self, iterations: int) -> StepResult:
        iterations = int(iterations)
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        started = time.monotonic()
        before = float(self.transport.sim_time_s())
        span = iterations * self.max_step_size_s
        target = before + span
        if self._schedule_s is not None:
            # Continue from the previous target, not from where the clock stopped, so N steps
            # advance N spans plus one overshoot.  Re-anchor when the clock ran away from it.
            scheduled = self._schedule_s + span
            if scheduled - before >= 0.5 * span:
                target = scheduled
        self.transport.unpause()
        try:
            reached = self.transport.wait_until_sim_time(
                target - self.clock_tolerance_s, self.timeout_s
            )
        finally:
            self.transport.pause()
        after = float(self.transport.sim_time_s())
        if not reached:
            raise StepTimeoutError(
                "Gazebo clock reached %.6f s, target %.6f s, within %.1f s wall"
                % (after, target, self.timeout_s)
            )
        self._schedule_s = target
        return StepResult(iterations, before, after, target, time.monotonic() - started)


class PluginStepper(GazeboStepper):
    name = "plugin"

    def __init__(self, transport: Any, max_step_size_s: float = 0.001) -> None:
        self.transport = transport
        self.max_step_size_s = float(max_step_size_s)

    def step(self, iterations: int) -> StepResult:
        iterations = int(iterations)
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        started = time.monotonic()
        before = float(self.transport.sim_time_s())
        target = before + iterations * self.max_step_size_s
        after = float(self.transport.step_world(iterations))
        return StepResult(iterations, before, after, target, time.monotonic() - started)


def make_stepper(name: str, transport: Any, max_step_size_s: float, timeout_s: float) -> GazeboStepper:
    if name == "clock_wait":
        return ClockWaitStepper(transport, max_step_size_s, timeout_s)
    if name == "plugin":
        return PluginStepper(transport, max_step_size_s)
    raise ValueError("unknown stepper %r" % name)


__all__ = ["ClockWaitStepper", "GazeboStepper", "PluginStepper", "StepResult", "make_stepper"]
