"""Probe: records any signal per tick."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

from .base import Block, Port, signal_in
from .params import PROBE_SCHEMA


class Probe(Block):
    """Records the value of one signal every tick (kept in memory, written by the runner)."""

    type_name: ClassVar[str] = "Probe"
    schema = PROBE_SCHEMA
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("signal", "any signal port"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.samples: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.samples = []

    def record(self, t_ms: float, value: Any) -> None:
        self.samples.append({"t_ms": float(t_ms), "value": value})
        keep = int(self.params["keep"])
        if keep and len(self.samples) > keep:
            del self.samples[: len(self.samples) - keep]


__all__ = ["Probe"]
