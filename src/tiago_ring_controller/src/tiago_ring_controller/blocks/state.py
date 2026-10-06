"""RingState: the joint angle the state ring holds (population-vector readout)."""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar, Deque, Dict, Optional, Tuple

import numpy as np

from ..control.mapping import RingMapper
from .base import Block, Port, signal_in, signal_out
from .params import RING_STATE_SCHEMA


class RingState(Block):
    """Ring counts → population-vector average → ring index → joint angle.

    The counts of the last ``window_ticks`` NEST steps are summed, their
    circular population-vector average gives a fractional ring index, and the
    inverse of the encoder mapping turns it into a joint angle, clamped to the
    joint limits (a bump just past an arc end reads as that limit; ``index``
    and ``in_arc`` keep the raw position).  ``strength``
    is the length of the population vector (1 for a single tight bump, near 0
    for activity spread around the ring); ``in_arc`` says whether the index
    lies on the joint's arc (``arc_margin`` indices of slack).  Use the same
    mapping parameters as the encoder that places the joint on the ring.
    """

    type_name: ClassVar[str] = "RingState"
    schema = RING_STATE_SCHEMA
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("counts", "per-neuron spike counts of the state ring per tick (Ring.counts)"),
        signal_out("angle", "joint angle the ring holds, within the joint limits, rad (NaN while the ring is silent)"),
        signal_out("index", "fractional ring index of the population vector"),
        signal_out("strength", "population-vector length, 0-1"),
        signal_out("in_arc", "True when the index lies on the joint's arc"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.reset()

    def reset(self) -> None:
        self.window: Deque[np.ndarray] = deque(maxlen=int(self.params["window_ticks"]))
        self._last_step: Optional[int] = None

    def mapper(self, ring_size: int) -> RingMapper:
        p = self.params
        return RingMapper(p["mapping"], p["joint_min"], p["joint_max"], ring_size, p["half_width"], p["arc_fraction"])

    def read(self, counts: Any) -> Dict[str, Any]:
        """Readout of one count vector (no window)."""

        values = np.asarray(counts, dtype=float)
        total = float(values.sum())
        if values.size == 0 or total <= 0.0:
            return {"angle": float("nan"), "index": float("nan"), "strength": 0.0, "in_arc": False}
        size = values.size
        vector = np.sum(values * np.exp(2j * np.pi * np.arange(size) / size)) / total
        index = float((np.angle(vector) % (2.0 * np.pi)) * size / (2.0 * np.pi))
        mapper = self.mapper(size)
        low, high = mapper.arc
        margin = float(self.params["arc_margin"])
        # Express the index on the side of the seam nearest the arc, so a bump just past an
        # arc end that wraps (e.g. 199 for an arc starting at 0) reads as just outside it.
        middle = (low + high) / 2.0
        unwrapped = middle + ((index - middle + size / 2.0) % size - size / 2.0)
        angle = float(mapper.angle(unwrapped))
        if self.params["mapping"] != "circular":
            angle = min(max(angle, float(self.params["joint_min"])), float(self.params["joint_max"]))
        return {
            "angle": angle, "index": float(unwrapped), "strength": float(abs(vector)),
            "in_arc": bool(low - margin <= unwrapped <= high + margin),
        }

    def step(self, counts: Any, nest_step: Optional[int] = None) -> Dict[str, Any]:
        if counts is not None and (nest_step is None or nest_step != self._last_step):
            self.window.append(np.asarray(counts, dtype=float))
            self._last_step = nest_step
        if not self.window:
            return {"angle": float("nan"), "index": float("nan"), "strength": 0.0, "in_arc": False}
        return self.read(np.sum(self.window, axis=0))


__all__ = ["RingState"]
