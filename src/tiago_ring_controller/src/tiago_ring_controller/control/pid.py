"""A tuned position controller used as an upper reference next to the paper's PID baseline (docs/results.md).

Measured joint angle and goal in, joint velocity out, once per tick.  It is the
PID form with the things a tuned controller needs on a rate-limited actuator:
the integral is not advanced while the output is saturated, the derivative
acts on the measurement, and the output limit is a fraction of the joint's
velocity limit.  The tuning grid (``scripts/run_baselines.py tune``) chose
``ki = kd = 0``: in effect a proportional controller.  The baseline of the
comparison is the PID of the draft (``control.controller.PIDControllerCore``
in ``legacy/single_joint_pid_data_collector.py``), not this.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

#: settings of the reference controller: gains (rad/s per rad, rad/s per rad s, rad/s per rad/s) and the output limit as a
#: fraction of the joint's velocity limit (a joint commanded faster than it can follow overshoots, because the
#: streamed position runs ahead of it).  One set for all joints, chosen by ``scripts/run_baselines.py tune``
#: (docs/results.md, "Upper reference").
DEFAULT_GAINS = {"kp": 8.0, "ki": 0.0, "kd": 0.0, "speed_fraction": 0.6}


class AntiWindupPID:
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float) -> None:
        if output_limit <= 0.0:
            raise ValueError("output_limit must be positive")
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.output_limit = float(output_limit)
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.previous: Optional[float] = None

    def step(self, error: float, measurement: float, dt: float) -> Tuple[float, Dict[str, float]]:
        """The velocity command for this tick and its three terms."""

        rate = 0.0 if self.previous is None else (float(measurement) - self.previous) / dt
        self.previous = float(measurement)
        p, d = self.kp * float(error), -self.kd * rate
        candidate = self.integral + float(error) * dt
        raw = p + self.ki * candidate + d
        # integrate only when it does not push a saturated output further into saturation
        if abs(raw) <= self.output_limit or raw * float(error) < 0.0:
            self.integral = candidate
        else:
            raw = p + self.ki * self.integral + d
        output = max(-self.output_limit, min(self.output_limit, raw))
        return float(output), {"p": float(p), "i": float(self.ki * self.integral), "d": float(d), "saturated": float(abs(raw) > self.output_limit)}


__all__ = ["AntiWindupPID", "DEFAULT_GAINS"]
