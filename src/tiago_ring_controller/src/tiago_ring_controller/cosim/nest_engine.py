"""NEST engine wrapping the *unchanged* single-ring model.

The model is built through the existing facades (``SingleRingModel`` and its
components) exactly as the legacy scripts do; nothing about neurons, synapses,
recorders or stimulus devices changes here.  What this engine changes is
only the harness around the model:

* stepping is ``Prepare()`` once, ``Run(dt)`` per tick, ``Cleanup()`` at the
  end of the trial (``nest_step_mode="run"``), or ``Simulate(dt)`` per tick
  for byte-exact legacy behaviour (``nest_step_mode="simulate"``);
* readout is a per-tick ``n_events`` delta on NodeCollections of the existing
  recorders (three calls instead of ~600), with a per-recorder fallback;
* stimulus application goes through a :class:`StimulusPort`.  The default
  :class:`LegacyInjectStimulusPort` calls the model's own ``_inject_bump``
  (which internally runs ``Simulate(50)``) and therefore only supports bumps
  *before* the first step of a trial.  Closing the loop with per-tick
  proprioception needs build-time generators (plan B.4) and is a follow-up:
  a port that supports mid-trial bumps plugs in here without touching the
  loop or the TFs.

The NEST module is injected as ``backend``; this module never imports it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..nest.kernel import recorder_events
from .datapack import DataPack
from .engine import Engine, EngineError, EngineStateError
from .fakes import ring_readout


class StimulusNotSupportedError(EngineError):
    """The stimulus port cannot apply a bump at this point of the trial."""


@dataclass
class RingModelPorts:
    """The parts of a built model the engine touches; nothing else."""

    population_size: int
    r1_recorders: Sequence[Any]
    r2_recorders: Sequence[Any]
    left_recorders: Sequence[Any]
    right_recorders: Sequence[Any]
    inject_state: Callable[[int, int], None]
    inject_goal: Callable[[int, int], None]
    model: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_single_ring_model(cls, model: Any) -> "RingModelPorts":
        """Adapt ``single_ring.SingleRingModel`` (or a duck-typed stand-in)."""

        return cls(
            population_size=int(model.population_size),
            r1_recorders=list(model.r1.ring_attractor.ring_spike_recorders),
            r2_recorders=list(model.r2.ring_attractor.ring_spike_recorders),
            left_recorders=list(model.gain_modulation.left_gain_spike_recorders),
            right_recorders=list(model.gain_modulation.right_gain_spike_recorders),
            inject_state=model.r1._inject_bump,
            inject_goal=model.r2._inject_bump,
            model=model,
        )


class StimulusPort(ABC):
    """How bumps reach the network.  Swappable to close the loop later."""

    #: True when the port can apply bumps between ``Run`` calls.
    supports_mid_trial = False

    def reset(self) -> None:
        return None

    @abstractmethod
    def apply(self, engine: "NestEngine", name: str, bump: DataPack) -> None:
        raise NotImplementedError


class LegacyInjectStimulusPort(StimulusPort):
    """Legacy ``inject_stimulus`` (creates a generator and runs ``Simulate``).

    Legal only before the engine has stepped, i.e. at the start of a trial.
    The hidden simulated time it consumes is accounted in ``hidden_ms``.
    """

    supports_mid_trial = False

    def __init__(self, duration_ms: float = 50.0) -> None:
        self.duration_ms = float(duration_ms)
        self.hidden_ms = 0.0
        self.applied: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.hidden_ms = 0.0
        self.applied = []

    def apply(self, engine: "NestEngine", name: str, bump: DataPack) -> None:
        if engine.step_index > 0 or engine.prepared:
            raise StimulusNotSupportedError(
                "%s at NEST step %d: the legacy stimulus port injects through "
                "inject_stimulus (Simulate inside) and only supports bumps before "
                "the first step of a trial. Continuous proprioception needs "
                "build-time stimulus generators (plan B.4)." % (name, engine.step_index)
            )
        center = int(bump["center_index"])
        half_width = int(bump["half_width"])
        if name == "goal_bump":
            engine.ports.inject_goal(center, half_width)
        elif name == "state_bump":
            engine.ports.inject_state(center, half_width)
        else:
            raise EngineError("unknown bump %r" % name)
        self.hidden_ms += float(bump.get("duration_ms", self.duration_ms))
        self.applied.append(
            {"name": name, "center_index": center, "half_width": half_width,
             "nest_step": engine.step_index}
        )


def _global_id(backend: Any, node: Any) -> int:
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            value = getter("global_id")
            if isinstance(value, (list, tuple)):
                value = value[0]
            return int(value)
        except (KeyError, TypeError, AttributeError, ValueError):
            pass
    return int(backend.GetStatus(node, "global_id")[0])


class SpikeCountReader:
    """Cumulative ``n_events`` for a list of recorders, cost independent of time."""

    def __init__(self, backend: Any, recorders: Sequence[Any]) -> None:
        self.backend = backend
        self.recorders = list(recorders)
        self.collection = None
        self.order: Optional[np.ndarray] = None
        factory = getattr(backend, "NodeCollection", None)
        if callable(factory) and self.recorders:
            try:
                ids = [_global_id(backend, recorder) for recorder in self.recorders]
                sorted_ids = sorted(ids)
                self.collection = factory(sorted_ids)
                position = {gid: idx for idx, gid in enumerate(sorted_ids)}
                self.order = np.array([position[gid] for gid in ids], dtype=int)
                self.read()
            except Exception:
                self.collection = None
                self.order = None

    @property
    def mode(self) -> str:
        return "node_collection" if self.collection is not None else "events"

    def read(self) -> np.ndarray:
        if not self.recorders:
            return np.zeros(0, dtype=float)
        if self.collection is not None:
            values = self.collection.get("n_events")
            if np.isscalar(values):
                values = [values]
            counts = np.asarray(values, dtype=float)
            if counts.shape != (len(self.recorders),):
                raise EngineError("unexpected n_events shape %r" % (counts.shape,))
            return counts[self.order]
        return np.array(
            [len(recorder_events(self.backend, r).get("times", [])) for r in self.recorders],
            dtype=float,
        )


class NestEngine(Engine):
    inputs = frozenset({"state_bump", "goal_bump"})
    outputs = frozenset({"ring_counts"})

    def __init__(
        self,
        backend: Any,
        model_factory: Callable[[], RingModelPorts],
        step_mode: str = "run",
        stimulus_port: Optional[StimulusPort] = None,
        name: str = "nest",
        population_size: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        self._declared_population_size = None if population_size is None else int(population_size)
        if step_mode not in ("run", "simulate"):
            raise ValueError("step_mode must be 'run' or 'simulate'")
        self.backend = backend
        self.model_factory = model_factory
        self.step_mode = step_mode
        self.stimulus_port = stimulus_port or LegacyInjectStimulusPort()
        self.ports: Optional[RingModelPorts] = None
        self.prepared = False
        self.pending: Dict[str, DataPack] = {}
        self.last: Optional[DataPack] = None
        self._readers: Dict[str, SpikeCountReader] = {}
        self._prev: Dict[str, np.ndarray] = {}
        #: simulated time the kernel has run since the last rebuild, including
        #: the hidden time inside stimulus injection
        self.kernel_time_ms = 0.0
        #: kernel time at which the current trial started (raster window)
        self.trial_start_ms = 0.0
        self.rebuild_count = 0

    # -- lifecycle --------------------------------------------------------
    def _do_reset(self) -> None:
        self._cleanup()
        if self.reset_mode == "rebuild" or self.ports is None:
            self.ports = self.model_factory()
            self._readers = {
                "left": SpikeCountReader(self.backend, self.ports.left_recorders),
                "right": SpikeCountReader(self.backend, self.ports.right_recorders),
                "r1": SpikeCountReader(self.backend, self.ports.r1_recorders),
            }
            self.kernel_time_ms = 0.0
            self.rebuild_count += 1
        # continue: the network keeps its state; the new goal / state bumps
        # are injected on top of it at the first advance of the trial.
        self.trial_start_ms = self.kernel_time_ms
        self.stimulus_port.reset()
        self.pending = {}
        self.last = None
        self._snapshot()

    def _snapshot(self) -> None:
        self._prev = {key: reader.read() for key, reader in self._readers.items()}

    def _cleanup(self) -> None:
        if self.prepared:
            self.backend.Cleanup()
            self.prepared = False

    def _do_finish_trial(self) -> None:
        self._cleanup()

    def _do_shutdown(self) -> None:
        self._cleanup()

    @property
    def population_size(self) -> Optional[int]:
        if self.ports is not None:
            return self.ports.population_size
        return self._declared_population_size

    @property
    def hidden_ms(self) -> float:
        return float(getattr(self.stimulus_port, "hidden_ms", 0.0))

    @property
    def readout_mode(self) -> str:
        return self._readers["r1"].mode if self._readers else "unknown"

    # -- datapacks --------------------------------------------------------
    def _do_get_datapacks(self) -> Dict[str, DataPack]:
        return {} if self.last is None else {"ring_counts": self.last}

    def _do_set_datapacks(self, packs: Dict[str, DataPack]) -> None:
        self.pending.update(packs)

    def _apply_pending(self) -> None:
        if not self.pending:
            return
        if self.ports is None:
            raise EngineStateError("NEST engine has not been reset")
        # Legacy order: goal (r2) first, then state (r1).
        hidden_before = self.hidden_ms
        for key in ("goal_bump", "state_bump"):
            bump = self.pending.pop(key, None)
            if bump is not None:
                self.stimulus_port.apply(self, key, bump)
        # Stimulus injection may have advanced hidden simulated time; do not
        # count those spikes in the next tick's delta (legacy before/after).
        self.kernel_time_ms += self.hidden_ms - hidden_before
        self._snapshot()

    def _do_advance(self, dt_ms: float) -> None:
        if self.ports is None:
            raise EngineStateError("NEST engine has not been reset")
        self._apply_pending()
        if self.step_mode == "run":
            if not self.prepared:
                self.backend.Prepare()
                self.prepared = True
            self.backend.Run(dt_ms)
        else:
            self.backend.Simulate(dt_ms)
        self.kernel_time_ms += dt_ms

        current = {key: reader.read() for key, reader in self._readers.items()}
        left = int(np.sum(current["left"] - self._prev["left"]))
        right = int(np.sum(current["right"] - self._prev["right"]))
        r1_delta = np.maximum(current["r1"] - self._prev["r1"], 0.0)
        self._prev = current
        total, bump, centroid = ring_readout(r1_delta)
        t_after = self.t_ms + dt_ms
        self.last = DataPack(
            "ring_counts",
            t_after,
            {
                "left": left,
                "right": right,
                "r1_delta": r1_delta.tolist(),
                "r1_spike_count": total,
                "r1_bump_index": bump,
                "r1_centroid": centroid,
                "t_nest_ms": t_after,
                "nest_step": self.step_index + 1,
                "hidden_ms": self.hidden_ms,
                "step_mode": self.step_mode,
                "readout_mode": self.readout_mode,
            },
        )

    # -- trial-end readout ------------------------------------------------
    def raster(self, since_ms: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Event rasters in the collector's ``_collect_raster_data`` schema.

        Recorders accumulate across ``continue`` trials, so by default only
        events from the current trial (``trial_start_ms`` onwards) are
        returned; after a rebuild that is everything.
        """

        if self.ports is None:
            raise EngineStateError("NEST engine has not been reset")
        start = self.trial_start_ms if since_ms is None else float(since_ms)

        def gather(recorders: Sequence[Any]):
            times: List[float] = []
            senders: List[int] = []
            for recorder in recorders:
                try:
                    events = recorder_events(self.backend, recorder)
                except Exception:
                    continue
                for t, sender in zip(events.get("times", []), events.get("senders", [])):
                    if float(t) > start or start <= 0.0:
                        times.append(float(t))
                        senders.append(int(sender))
            return np.array(times, dtype=float), np.array(senders, dtype=int)

        r1_times, r1_senders = gather(self.ports.r1_recorders)
        r2_times, r2_senders = gather(self.ports.r2_recorders)
        left_times, left_senders = gather(self.ports.left_recorders)
        right_times, right_senders = gather(self.ports.right_recorders)
        return {
            "r1_times": r1_times, "r1_senders": r1_senders,
            "r2_times": r2_times, "r2_senders": r2_senders,
            "left_times": left_times, "left_senders": left_senders,
            "right_times": right_times, "right_senders": right_senders,
        }


__all__ = [
    "LegacyInjectStimulusPort",
    "NestEngine",
    "RingModelPorts",
    "SpikeCountReader",
    "StimulusNotSupportedError",
    "StimulusPort",
]
