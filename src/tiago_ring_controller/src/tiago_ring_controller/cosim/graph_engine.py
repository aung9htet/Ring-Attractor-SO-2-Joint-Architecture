"""NEST engine that holds a block ``Graph`` (any number of neural blocks).

Inputs are the encoders' angle signals (``"<encoder id>.angle"`` datapacks with
an ``angle`` field); outputs are one datapack per neural block with signal
outputs, named by the block id (``Ring``: per-neuron deltas, total, bump index,
centroid; ``Gain``: left/right totals; ``Homeostasis``: warm/cold/left/right;
``SignedProduct``/``OutputRing``: per-cell deltas).  Bumps are generator rate
changes followed by a kernel recalibration (see ``GeneratorStimulusPort``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from ..blocks.base import BuildContext, PortKind
from ..nest.kernel import recorder_events

if TYPE_CHECKING:  # pragma: no cover - the graph package imports this module
    from ..graph.graph import BuiltGraph, Graph
from .datapack import DataPack
from .engine import Engine, EngineStateError
from .fakes import ring_readout
from .nest_engine import SpikeCountReader


class GraphNestEngine(Engine):
    def __init__(
        self,
        backend: Any,
        graph: "Graph",
        step_mode: Optional[str] = None,
        name: str = "nest",
        config_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        graph.validate()
        self.backend = backend
        self.graph = graph
        self.step_mode = step_mode or graph.simulation.step_mode
        if self.step_mode not in ("run", "simulate"):
            raise ValueError("step_mode must be 'run' or 'simulate'")
        self.config_dir = config_dir
        self.encoder_ids = [block.id for block in graph.neural_blocks() if hasattr(block, "drive")]
        #: encoder id -> (source block id, source port): the datapack and field carrying its angle
        self.encoder_sources: Dict[str, Any] = {}
        for encoder_id in self.encoder_ids:
            for edge in graph.edges_into(graph.blocks[encoder_id], "angle"):
                self.encoder_sources[encoder_id] = (edge.source.block.id, edge.source.port.name)
        self.inputs = frozenset(source for source, _ in self.encoder_sources.values())
        self.outputs = frozenset(
            block.id for block in graph.neural_blocks()
            if any(port.kind is PortKind.SIGNAL and not port.is_input for port in block.ports)
        )
        self.built: Optional["BuiltGraph"] = None
        self.prepared = False
        self.pending: Dict[str, DataPack] = {}
        self.last: Dict[str, DataPack] = {}
        self._readers: Dict[str, Dict[str, SpikeCountReader]] = {}
        self._prev: Dict[str, Dict[str, np.ndarray]] = {}
        self._ring_of_encoder: Dict[str, str] = {}
        self.kernel_time_ms = 0.0
        self.trial_start_ms = 0.0
        self.rebuild_count = 0
        self.recalibrations = 0
        self.applied: List[Dict[str, Any]] = []

    # -- lifecycle --------------------------------------------------------
    def _do_reset(self) -> None:
        self._cleanup()
        if self.reset_mode == "rebuild" or self.built is None:
            ctx = BuildContext(
                self.backend, config_dir=self.config_dir, rng_seed=self.graph.simulation.rng_seed,
                local_num_threads=self.graph.simulation.local_num_threads, dt_ms=self.graph.simulation.dt_ms,
            )
            self.built = self.graph.build(ctx)
            self._readers = {
                block_id: {port: SpikeCountReader(self.backend, recorders) for port, recorders in sources.items()}
                for block_id, sources in self.built.counts_sources().items()
            }
            self._ring_of_encoder = {
                block.id: getattr(block, "ring_id", None) for block in self.built.encoders()
            }
            self.kernel_time_ms = 0.0
            self.rebuild_count += 1
        else:
            for block in self.graph.blocks.values():
                block.reset()
        self.trial_start_ms = self.kernel_time_ms
        for encoder in self.built.encoders():
            encoder.clear()
        self.pending = {}
        self.last = {}
        self.applied = []
        self._snapshot()

    def _snapshot(self) -> None:
        self._prev = {
            block_id: {port: reader.read() for port, reader in readers.items()}
            for block_id, readers in self._readers.items()
        }

    def _cleanup(self) -> None:
        if self.prepared:
            self.backend.Cleanup()
            self.prepared = False

    def recalibrate(self) -> None:
        if self.prepared:
            self.backend.Cleanup()
            self.backend.Prepare()
            self.recalibrations += 1

    def _do_finish_trial(self) -> None:
        self._cleanup()

    def _do_shutdown(self) -> None:
        self._cleanup()

    @property
    def hidden_ms(self) -> float:
        return 0.0

    @property
    def readout_mode(self) -> str:
        for readers in self._readers.values():
            for reader in readers.values():
                return reader.mode
        return "unknown"

    @property
    def population_size(self) -> Optional[int]:
        for block in self.graph.neural_blocks():
            if block.type_name == "Ring":
                return int(block.params["population_size"])
        return None

    # -- datapacks --------------------------------------------------------
    def _do_get_datapacks(self) -> Dict[str, DataPack]:
        return dict(self.last)

    def _do_set_datapacks(self, packs: Dict[str, DataPack]) -> None:
        self.pending.update(packs)

    def _ring_centroid(self, encoder_id: str) -> Optional[float]:
        ring_id = self._ring_of_encoder.get(encoder_id)
        pack = self.last.get(ring_id) if ring_id else None
        return None if pack is None else pack.get("centroid")

    def _apply_pending(self, dt_ms: float) -> None:
        if not self.pending:
            return
        if self.built is None:
            raise EngineStateError("graph engine has not been reset")
        changed = False
        for encoder in self.built.encoders():
            source = self.encoder_sources.get(encoder.id)
            pack = None if source is None else self.pending.get(source[0])
            if pack is None:
                continue
            angle = float(pack[source[1]])
            if encoder.drive(angle, self.kernel_time_ms, dt_ms, self._ring_centroid(encoder.id)):
                changed = True
                self.applied.append({
                    "encoder": encoder.id, "angle": angle, "index": getattr(encoder, "last_index", None),
                    "nest_step": self.step_index, "t_nest_ms": self.kernel_time_ms,
                })
        self.pending = {}
        if changed:
            self.recalibrate()

    def _do_advance(self, dt_ms: float) -> None:
        if self.built is None:
            raise EngineStateError("graph engine has not been reset")
        self._apply_pending(dt_ms)
        if self.step_mode == "run":
            if not self.prepared:
                self.backend.Prepare()
                self.prepared = True
            self.backend.Run(dt_ms)
        else:
            self.backend.Simulate(dt_ms)
        self.kernel_time_ms += dt_ms
        expired = [encoder.expire(self.kernel_time_ms) for encoder in self.built.encoders()]
        if any(expired):
            self.recalibrate()

        t_after = self.t_ms + dt_ms
        step = self.step_index + 1
        packs: Dict[str, DataPack] = {}
        for block_id, readers in self._readers.items():
            block = self.graph.blocks[block_id]
            data: Dict[str, Any] = {"nest_step": step, "t_nest_ms": t_after}
            for port, reader in readers.items():
                current = reader.read()
                delta = np.maximum(current - self._prev[block_id][port], 0.0)
                self._prev[block_id][port] = current
                if block.type_name == "Ring":
                    total, bump, centroid = ring_readout(delta)
                    data.update({"counts": delta.tolist(), "total": total, "bump_index": bump, "centroid": centroid})
                elif block.type_name == "Homeostasis":
                    data["counts"] = {label: int(delta[i]) for i, label in enumerate(block.population.labels)}
                elif port in ("left_counts", "right_counts"):
                    data[port] = int(np.sum(delta))
                else:
                    data[port] = delta.tolist()
            packs[block_id] = DataPack(block_id, t_after, data)
        self.last = packs

    # -- trial-end readout ------------------------------------------------
    def raster(self, since_ms: Optional[float] = None) -> Dict[str, np.ndarray]:
        """``{"<ring id>_times", "<ring id>_senders", ...}`` for every ring (current trial)."""

        if self.built is None:
            raise EngineStateError("graph engine has not been reset")
        start = self.trial_start_ms if since_ms is None else float(since_ms)
        result: Dict[str, np.ndarray] = {}
        for ring in self.built.rings():
            times: List[float] = []
            senders: List[int] = []
            for recorder in ring.population.recorder_list():
                try:
                    events = recorder_events(self.backend, recorder)
                except Exception:
                    continue
                for t, sender in zip(events.get("times", []), events.get("senders", [])):
                    if float(t) > start or start <= 0.0:
                        times.append(float(t))
                        senders.append(int(sender))
            result[ring.id + "_times"] = np.array(times, dtype=float)
            result[ring.id + "_senders"] = np.array(senders, dtype=int)
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.name, "step_mode": self.step_mode, "encoders": list(self.encoder_ids),
            "outputs": sorted(self.outputs), "rebuilds": self.rebuild_count, "recalibrations": self.recalibrations,
            "build": None if self.built is None else self.built.describe(),
        }


__all__ = ["GraphNestEngine"]
