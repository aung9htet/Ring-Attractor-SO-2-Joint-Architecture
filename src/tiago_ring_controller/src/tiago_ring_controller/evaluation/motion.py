"""Motion metrics of one trial from the joint trace: the same code scores every system.

Used by ``scripts/validate_velocity.py`` (graph systems) and
``scripts/run_legacy_trials.py`` (the frozen ring and PID collectors), so the
comparison in ``docs/results.md`` does not depend on who ran the trial.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .metrics import settling_time

#: a joint that moved less than this (rad) produced no drive
NO_DRIVE_RAD = 0.03
#: the settling definition of the paper: |error| under ``SETTLE_TOLERANCE`` for ``SETTLE_SAMPLES`` samples
SETTLE_TOLERANCE = 0.05
SETTLE_SAMPLES = 5


def motion_metrics(positions: Sequence[float], goal_rad: float, dt_s: float,
                   tolerance: float = SETTLE_TOLERANCE, samples: int = SETTLE_SAMPLES) -> Dict[str, Any]:
    """Scalars of one move.

    ``positions``: the joint angle at every tick, first the start, last the
    final angle.  ``time_to_90_s`` is the first time 90% of the travelled
    distance is covered; ``settling_time_s`` the first time the error stays
    under ``tolerance`` for ``samples`` samples (NaN when it never does, with
    ``reached_tolerance`` False); ``overshoot_rad`` the largest excursion past
    the goal in the direction of travel.
    """

    q = np.asarray(positions, dtype=float)
    if q.size < 2:
        raise ValueError("a joint trace needs at least two samples")
    moved = float(q[-1] - q[0])
    direction = float(np.sign(goal_rad - q[0]))
    reached = np.nonzero(np.abs(q - q[0]) >= 0.9 * abs(moved))[0] if abs(moved) >= NO_DRIVE_RAD else []
    error = float(goal_rad) - q
    times = np.arange(q.size) * float(dt_s)
    within = np.abs(error) < tolerance
    settled = any(np.all(within[k:k + samples]) for k in range(max(q.size - samples + 1, 0)))
    return {
        "goal_error_rad": float(error[-1]),
        "joint_displacement": moved,
        "time_to_90_s": float(reached[0] * dt_s) if len(reached) else float("nan"),
        "peak_speed": float(np.max(np.abs(np.diff(q))) / dt_s),
        "overshoot_rad": float(max(0.0, np.max(direction * (q - goal_rad)))) if direction else 0.0,
        "settling_time_s": settling_time(error, times, tolerance, samples) if settled else float("nan"),
        "reached_tolerance": bool(settled),
        "no_drive": bool(abs(moved) < NO_DRIVE_RAD),
        "wrong_direction": bool(abs(moved) >= NO_DRIVE_RAD and np.sign(moved) != direction),
    }


__all__ = ["NO_DRIVE_RAD", "SETTLE_SAMPLES", "SETTLE_TOLERANCE", "motion_metrics"]
