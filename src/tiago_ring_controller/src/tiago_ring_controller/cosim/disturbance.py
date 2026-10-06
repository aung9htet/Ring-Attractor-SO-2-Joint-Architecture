"""Disturbances on the robot side of a trial, for robustness tests (docs/v4-tuning.md).

The robot is driven by streamed positions, so a disturbance is something the
ring is not told about:

* ``push``: the commanded position of one joint jumps by ``push_rad`` at
  ``at_s`` (the joint goes there; nothing in the network caused it);
* ``hold``: for ``hold_s`` seconds from ``at_s`` the velocity commands are
  dropped (the joint stays where it is while the network keeps commanding).

Works on every robot engine with a ``command_state`` (the fake robot and the
Gazebo engine): it wraps the engine's ``_do_set_datapacks``.
"""

from __future__ import annotations

from typing import Any, Dict


class Disturbance:
    def __init__(self, engine: Any, joint: int, push_rad: float = 0.0, hold_s: float = 0.0, at_s: float = 1.0) -> None:
        self.engine, self.joint = engine, int(joint)
        self.push_rad, self.hold_s, self.at_s = float(push_rad), float(hold_s), float(at_s)
        self.pushed = False
        self.dropped = 0
        self._original = engine._do_set_datapacks
        engine._do_set_datapacks = self._set_datapacks

    @property
    def active(self) -> bool:
        return bool(self.push_rad or self.hold_s)

    def _set_datapacks(self, packs: Dict[str, Any]) -> None:
        t_s = float(self.engine.t_ms) / 1000.0 + 1e-9         # tick times are sums of floats
        if t_s < self.at_s:
            self.pushed = False                      # a new trial
        elif self.push_rad and not self.pushed:
            commanded = self.engine.command_state.commanded_positions
            if commanded:
                commanded[self.joint] += self.push_rad
                self.pushed = True
        if self.hold_s and self.at_s <= t_s < self.at_s + self.hold_s and "arm_velocity_cmd" in packs:
            packs = {name: pack for name, pack in packs.items() if name != "arm_velocity_cmd"}
            self.dropped += 1
        self._original(packs)

    def remove(self) -> None:
        self.engine._do_set_datapacks = self._original


__all__ = ["Disturbance"]
