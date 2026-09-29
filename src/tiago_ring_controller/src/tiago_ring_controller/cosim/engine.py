"""Engine protocol: anything that owns simulated time and advances in steps.

The base class implements the clock bookkeeping and the lifecycle guards so
that concrete engines (NEST, Gazebo/ROS, and the fakes used by the unit
suite) only implement the ``_do_*`` hooks.  Importing this module has no
side effects and does not import NEST or ROS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping

from .datapack import DataPack


class EngineError(RuntimeError):
    """Base class for engine failures."""


class EngineStateError(EngineError):
    """A lifecycle or clock invariant was violated."""


class StepTimeoutError(EngineError):
    """A simulator did not reach its target time within the wall-clock budget."""


@dataclass
class EngineClock:
    """Simulated time owned by one engine, in milliseconds."""

    t_ms: float = 0.0
    step_index: int = 0

    def reset(self) -> None:
        self.t_ms = 0.0
        self.step_index = 0

    def advance(self, dt_ms: float) -> None:
        self.t_ms += float(dt_ms)
        self.step_index += 1


class Engine(ABC):
    """Template for a time-owning simulator participant.

    Lifecycle: ``initialize()`` once, then per trial ``reset()``, any number
    of ``get_datapacks()`` / ``set_datapacks()`` / ``advance()`` calls,
    ``finish_trial()``, and finally ``shutdown()``.
    """

    #: DataPack names this engine accepts through :meth:`set_datapacks`.
    inputs: FrozenSet[str] = frozenset()
    #: DataPack names this engine may return from :meth:`get_datapacks`.
    outputs: FrozenSet[str] = frozenset()

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("engine name must be non-empty")
        self.name = str(name)
        self.clock = EngineClock()
        self._initialized = False
        self._in_trial = False

    # -- public lifecycle -------------------------------------------------
    @property
    def t_ms(self) -> float:
        return self.clock.t_ms

    @property
    def step_index(self) -> int:
        return self.clock.step_index

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        self._do_initialize()
        self._initialized = True

    def reset(self) -> None:
        """Start a new trial: rebuild or re-home, and zero the clock."""

        self._require_initialized("reset")
        self.clock.reset()
        self._do_reset()
        self._in_trial = True

    def get_datapacks(self) -> Dict[str, DataPack]:
        self._require_initialized("get_datapacks")
        packs = self._do_get_datapacks()
        unknown = set(packs) - set(self.outputs)
        if unknown:
            raise EngineStateError(
                "engine %r produced undeclared datapacks %s" % (self.name, sorted(unknown))
            )
        return dict(packs)

    def set_datapacks(self, packs: Mapping[str, DataPack]) -> None:
        self._require_initialized("set_datapacks")
        unknown = set(packs) - set(self.inputs)
        if unknown:
            raise EngineError(
                "engine %r does not accept datapacks %s" % (self.name, sorted(unknown))
            )
        for key, pack in packs.items():
            if not isinstance(pack, DataPack):
                raise EngineError("input %r is not a DataPack" % key)
        if packs:
            self._do_set_datapacks(dict(packs))

    def advance(self, dt_ms: float) -> None:
        self._require_initialized("advance")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        self._do_advance(float(dt_ms))
        self.clock.advance(dt_ms)

    def finish_trial(self) -> None:
        """Hook run once after the loop's stop condition fires."""

        if self._in_trial:
            self._do_finish_trial()
            self._in_trial = False

    def shutdown(self) -> None:
        if not self._initialized:
            return
        try:
            self.finish_trial()
        finally:
            self._do_shutdown()
            self._initialized = False

    # -- hooks ------------------------------------------------------------
    def _do_initialize(self) -> None:
        return None

    def _do_reset(self) -> None:
        return None

    @abstractmethod
    def _do_get_datapacks(self) -> Dict[str, DataPack]:
        raise NotImplementedError

    def _do_set_datapacks(self, packs: Dict[str, DataPack]) -> None:
        return None

    @abstractmethod
    def _do_advance(self, dt_ms: float) -> None:
        raise NotImplementedError

    def _do_finish_trial(self) -> None:
        return None

    def _do_shutdown(self) -> None:
        return None

    # -- helpers ----------------------------------------------------------
    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise EngineStateError(
                "engine %r: %s() called before initialize()" % (self.name, operation)
            )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "%s(name=%r, t_ms=%r)" % (type(self).__name__, self.name, self.t_ms)


__all__ = [
    "Engine",
    "EngineClock",
    "EngineError",
    "EngineStateError",
    "StepTimeoutError",
]
