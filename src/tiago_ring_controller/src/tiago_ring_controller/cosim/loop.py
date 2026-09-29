"""Fixed-time-increment co-simulation loop with NRP-style one-step delay.

One tick at loop time ``t`` performs, in this fixed order:

1. ``cache = {engine.get_datapacks()}``      state of every engine at ``t``
2. ``outputs = tf(cache)`` for every TF      pure, deterministic order
3. ``engine.set_datapacks(outputs)``         inputs applied for ``[t, t+dt)``
4. ``engine.advance(dt)``                    every engine advances exactly ``dt``
5. ``t += dt``

Data produced at ``t`` is therefore consumed at ``t + dt``: a documented,
fixed one-step latency.

The optional *lead* reproduces the legacy receding horizon: before the first
main tick, the lead engine (NEST) is advanced ``nest_lead_steps`` times while
the other engines hold at ``t = 0``.  TFs run at each lead sub-step so that
stateful TFs observe every NEST sample in order, but only the lead engine
receives inputs during the lead phase.  Afterwards the lead engine's clock is
always ``nest_lead_steps * dt`` ahead of the loop clock, and the loop checks
that invariant after every tick.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .datapack import DataPack, datapacks_from_dict, datapacks_to_dict
from .engine import Engine, EngineError, EngineStateError
from .tf import TickContext, TransceiverFunction


PHASE_LEAD = "lead"
PHASE_MAIN = "main"


@dataclass(frozen=True)
class TickRecord:
    """Everything the loop saw and emitted in one tick."""

    tick: int
    phase: str
    t_ms: float
    inputs: Dict[str, DataPack]
    outputs: Dict[str, DataPack]
    advanced: Tuple[str, ...]
    wall_s: float = 0.0

    def to_dict(self, include_timing: bool = True) -> Dict[str, Any]:
        value = {
            "tick": self.tick,
            "phase": self.phase,
            "t_ms": self.t_ms,
            "inputs": datapacks_to_dict(self.inputs),
            "outputs": datapacks_to_dict(self.outputs),
            "advanced": list(self.advanced),
        }
        if include_timing:
            value["wall_s"] = self.wall_s
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TickRecord":
        return cls(
            tick=int(value["tick"]),
            phase=str(value["phase"]),
            t_ms=float(value["t_ms"]),
            inputs=datapacks_from_dict(value.get("inputs", {})),
            outputs=datapacks_from_dict(value.get("outputs", {})),
            advanced=tuple(str(name) for name in value.get("advanced", ())),
            wall_s=float(value.get("wall_s", 0.0)),
        )


@dataclass
class TrialRecord:
    """A reset followed by lead sub-steps and main ticks until a stop."""

    dt_ms: float
    nest_lead_steps: int
    engines: Tuple[str, ...]
    transceivers: Tuple[str, ...]
    ticks: List[TickRecord] = field(default_factory=list)
    #: engine state right after the last tick's advance (before finish_trial)
    end_state: Dict[str, DataPack] = field(default_factory=dict)
    #: engine state after finish_trial (robot stopped and settled)
    final_state: Dict[str, DataPack] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def lead_ticks(self) -> List[TickRecord]:
        return [tick for tick in self.ticks if tick.phase == PHASE_LEAD]

    @property
    def main_ticks(self) -> List[TickRecord]:
        return [tick for tick in self.ticks if tick.phase == PHASE_MAIN]

    @property
    def n_steps(self) -> int:
        return len(self.main_ticks)

    def to_dict(self, include_timing: bool = True) -> Dict[str, Any]:
        return {
            "schema": "tiago_ring_controller.cosim.TrialRecord/1",
            "dt_ms": self.dt_ms,
            "nest_lead_steps": self.nest_lead_steps,
            "engines": list(self.engines),
            "transceivers": list(self.transceivers),
            "ticks": [tick.to_dict(include_timing) for tick in self.ticks],
            "end_state": datapacks_to_dict(self.end_state),
            "final_state": datapacks_to_dict(self.final_state),
            "stop_reason": self.stop_reason,
            "meta": {k: v for k, v in self.meta.items() if include_timing or k != "timing"},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialRecord":
        return cls(
            dt_ms=float(value["dt_ms"]),
            nest_lead_steps=int(value["nest_lead_steps"]),
            engines=tuple(str(name) for name in value["engines"]),
            transceivers=tuple(str(name) for name in value["transceivers"]),
            ticks=[TickRecord.from_dict(item) for item in value.get("ticks", [])],
            end_state=datapacks_from_dict(value.get("end_state", {})),
            final_state=datapacks_from_dict(value.get("final_state", {})),
            stop_reason=value.get("stop_reason"),
            meta=dict(value.get("meta", {})),
        )

    def to_json(self, indent: Optional[int] = None, include_timing: bool = True) -> str:
        return json.dumps(self.to_dict(include_timing), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TrialRecord":
        return cls.from_dict(json.loads(text))


#: Returns a stop reason, or ``None`` to keep running.
StopCondition = Callable[[TickRecord], Optional[str]]


class LoopObserver:
    """Per-tick hook for monitors, cameras and loggers.

    Observers only *read* records; they run after the engines have advanced
    and cannot change what the loop does, so they never affect determinism.
    """

    def on_reset(self, loop: "FTILoop") -> None:
        return None

    def on_tick(self, record: TickRecord) -> None:
        return None

    def on_trial_end(self, record: TrialRecord) -> None:
        return None


class MaxSteps:
    """Stop after ``max_steps`` main ticks, like the legacy ``for`` loop."""

    reason = "max_steps"

    def __init__(self, max_steps: int) -> None:
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)

    def __call__(self, record: TickRecord) -> Optional[str]:
        if record.tick + 1 >= self.max_steps:
            return self.reason
        return None


class SettledFlag:
    """Stop when an output datapack reports ``settled`` (legacy drive settle)."""

    def __init__(
        self,
        pack_name: str = "arm_velocity_cmd",
        key: str = "settled",
        reason: str = "drive_settled",
    ) -> None:
        self.pack_name = pack_name
        self.key = key
        self.reason = reason

    def __call__(self, record: TickRecord) -> Optional[str]:
        pack = record.outputs.get(self.pack_name)
        if pack is not None and bool(pack.get(self.key, False)):
            return self.reason
        return None


class AnyOf:
    """First matching condition wins; order encodes legacy precedence."""

    def __init__(self, *conditions: StopCondition) -> None:
        self.conditions = tuple(conditions)

    def __call__(self, record: TickRecord) -> Optional[str]:
        for condition in self.conditions:
            reason = condition(record)
            if reason is not None:
                return reason
        return None


class FTILoop:
    """Fixed-time-increment scheduler owning time for all engines."""

    def __init__(
        self,
        engines: Sequence[Engine],
        tfs: Sequence[TransceiverFunction],
        dt_ms: float,
        nest_lead_steps: int = 0,
        lead_engine: Optional[str] = None,
        stop_condition: Optional[StopCondition] = None,
        clock_tolerance_ms: float = 1e-6,
        observers: Optional[Sequence[LoopObserver]] = None,
    ) -> None:
        self.engines: List[Engine] = list(engines)
        self.tfs: List[TransceiverFunction] = list(tfs)
        self.observers: List[LoopObserver] = list(observers or [])
        if float(dt_ms) <= 0:
            raise ValueError("dt_ms must be positive")
        self.dt_ms = float(dt_ms)
        if int(nest_lead_steps) < 0:
            raise ValueError("nest_lead_steps must be >= 0")
        self.nest_lead_steps = int(nest_lead_steps)
        self.stop_condition = stop_condition
        self.clock_tolerance_ms = float(clock_tolerance_ms)

        names = [engine.name for engine in self.engines]
        if len(set(names)) != len(names):
            raise ValueError("engine names must be unique: %r" % names)
        if not self.engines:
            raise ValueError("at least one engine is required")

        seen: Dict[str, str] = {}
        for engine in self.engines:
            for output in engine.outputs:
                if output in seen:
                    raise ValueError(
                        "datapack %r is produced by both %r and %r"
                        % (output, seen[output], engine.name)
                    )
                seen[output] = engine.name

        if self.nest_lead_steps > 0:
            if lead_engine is None:
                lead_engine = "nest" if "nest" in names else None
            if lead_engine is None or lead_engine not in names:
                raise ValueError(
                    "nest_lead_steps > 0 requires a lead_engine among %r" % names
                )
        self.lead_engine = lead_engine

        self.t_ms = 0.0
        self.tick_index = 0
        self.records: List[TickRecord] = []
        self.reset_wall_s: Dict[str, float] = {}
        self.reset_mode = "rebuild"
        self._initialized = False

    # -- lifecycle --------------------------------------------------------
    def add_observer(self, observer: LoopObserver) -> None:
        self.observers.append(observer)

    def initialize(self) -> None:
        for engine in self.engines:
            engine.initialize()
        self._initialized = True

    def shutdown(self) -> None:
        errors = []
        for engine in self.engines:
            try:
                engine.shutdown()
            except Exception as exc:  # pragma: no cover - best-effort teardown
                errors.append((engine.name, exc))
        self._initialized = False
        if errors:
            raise EngineError("shutdown errors: %r" % errors)

    def reset(self, mode: str = "rebuild", notify_observers: bool = True) -> None:
        if not self._initialized:
            self.initialize()
        self.reset_mode = mode
        self.reset_wall_s = {}
        for engine in self.engines:
            started = time.monotonic()
            engine.reset(mode)
            self.reset_wall_s[engine.name] = time.monotonic() - started
        for tf in self.tfs:
            tf.reset()
        self.t_ms = 0.0
        self.tick_index = 0
        self.records = []
        if notify_observers:
            for observer in self.observers:
                observer.on_reset(self)

    # -- one tick ---------------------------------------------------------
    def _collect(self) -> Dict[str, DataPack]:
        cache: Dict[str, DataPack] = {}
        for engine in self.engines:
            for key, pack in engine.get_datapacks().items():
                if key in cache:
                    raise EngineStateError("duplicate datapack %r in tick cache" % key)
                cache[key] = pack
        return cache

    def _transceive(
        self, cache: Mapping[str, DataPack], ctx: TickContext
    ) -> Dict[str, DataPack]:
        outputs: Dict[str, DataPack] = {}
        for tf in self.tfs:
            # Engine datapacks plus what earlier transceivers produced this tick, so a
            # signal chain (sensor -> decoder -> command) runs within one tick.
            view = dict(cache, **outputs) if outputs else cache
            produced = tf(view, ctx)
            for key, pack in produced.items():
                if key not in tf.outputs:
                    raise EngineError(
                        "transceiver %r emitted undeclared datapack %r" % (tf.name, key)
                    )
                if key in outputs:
                    raise EngineError(
                        "datapack %r emitted by more than one transceiver" % key
                    )
                if not isinstance(pack, DataPack):
                    raise EngineError(
                        "transceiver %r emitted a non-DataPack for %r" % (tf.name, key)
                    )
                outputs[key] = pack
        return outputs

    @staticmethod
    def _distribute(outputs: Mapping[str, DataPack], engines: Sequence[Engine]) -> None:
        for engine in engines:
            subset = {key: pack for key, pack in outputs.items() if key in engine.inputs}
            if subset:
                engine.set_datapacks(subset)

    def _lead_offset_ms(self, engine: Engine) -> float:
        if self.lead_engine is not None and engine.name == self.lead_engine:
            return self.nest_lead_steps * self.dt_ms
        return 0.0

    def _check_clocks(self) -> None:
        for engine in self.engines:
            expected = self.t_ms + self._lead_offset_ms(engine)
            if abs(engine.t_ms - expected) > self.clock_tolerance_ms:
                raise EngineStateError(
                    "engine %r clock %.6f ms, loop expects %.6f ms"
                    % (engine.name, engine.t_ms, expected)
                )

    def _step(self, ctx: TickContext, targets: Sequence[Engine]) -> TickRecord:
        started = time.monotonic()
        cache = self._collect()
        outputs = self._transceive(cache, ctx)
        self._distribute(outputs, targets)
        for engine in targets:
            engine.advance(self.dt_ms)
        record = TickRecord(
            tick=ctx.tick,
            phase=ctx.phase,
            t_ms=ctx.t_ms,
            inputs=cache,
            outputs=outputs,
            advanced=tuple(engine.name for engine in targets),
            wall_s=time.monotonic() - started,
        )
        self.records.append(record)
        for observer in self.observers:
            observer.on_tick(record)
        return record

    def lead_step(self, lead_index: int) -> TickRecord:
        """Advance only the lead engine; the loop clock does not move."""

        if self.lead_engine is None:
            raise EngineStateError("lead_step() called without a lead engine")
        ctx = TickContext(
            t_ms=self.t_ms, tick=-(self.nest_lead_steps - lead_index),
            phase=PHASE_LEAD, lead_index=lead_index,
        )
        targets = [engine for engine in self.engines if engine.name == self.lead_engine]
        return self._step(ctx, targets)

    def tick(self) -> TickRecord:
        """Advance every engine by exactly ``dt`` and move the loop clock."""

        ctx = TickContext(t_ms=self.t_ms, tick=self.tick_index, phase=PHASE_MAIN)
        record = self._step(ctx, self.engines)
        self.t_ms += self.dt_ms
        self.tick_index += 1
        self._check_clocks()
        return record

    # -- trials -----------------------------------------------------------
    def run_trial(
        self,
        reset: bool = True,
        max_ticks: Optional[int] = None,
        meta: Optional[Mapping[str, Any]] = None,
        reset_mode: str = "rebuild",
    ) -> TrialRecord:
        if self.stop_condition is None and max_ticks is None:
            raise ValueError("run_trial needs a stop_condition or max_ticks")
        if reset:
            self.reset(reset_mode)
        elif not self._initialized:
            self.initialize()

        lead_started = time.monotonic()
        for lead_index in range(self.nest_lead_steps):
            self.lead_step(lead_index)
        self._check_clocks()
        lead_wall_s = time.monotonic() - lead_started

        stop_reason: Optional[str] = None
        while True:
            record = self.tick()
            if self.stop_condition is not None:
                stop_reason = self.stop_condition(record)
            if stop_reason is None and max_ticks is not None and self.tick_index >= max_ticks:
                stop_reason = "max_ticks"
            if stop_reason is not None:
                break

        end_state = self._collect()
        for engine in self.engines:
            engine.finish_trial()
        final_state = self._collect()

        trial = TrialRecord(
            dt_ms=self.dt_ms,
            nest_lead_steps=self.nest_lead_steps,
            engines=tuple(engine.name for engine in self.engines),
            transceivers=tuple(tf.name for tf in self.tfs),
            ticks=list(self.records),
            end_state=end_state,
            final_state=final_state,
            stop_reason=stop_reason,
            meta=dict(meta or {}),
        )
        trial.meta.setdefault("timing", {})
        trial.meta["timing"].update(
            {"reset_wall_s": dict(self.reset_wall_s), "lead_wall_s": lead_wall_s}
        )
        trial.meta["reset_mode"] = self.reset_mode
        for observer in self.observers:
            observer.on_trial_end(trial)
        return trial


__all__ = [
    "AnyOf",
    "FTILoop",
    "LoopObserver",
    "MaxSteps",
    "PHASE_LEAD",
    "PHASE_MAIN",
    "SettledFlag",
    "StopCondition",
    "TickRecord",
    "TrialRecord",
]
