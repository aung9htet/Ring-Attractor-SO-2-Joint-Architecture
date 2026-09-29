"""The ``Graph``: blocks, edges, validation and the NEST build order.

This is the primary interface (plan 5b).  A graph file serialises it (phase 4)
and the editor edits the file; nothing is editor-only.

Build order (``Graph.build``): a neural block builds once every build-bound
spike input is connected to a built source; blocks become ready in insertion
order; late-bound spike edges (feedback, encoder generators) are wired after
all blocks exist, in edge order.  Unresolvable dependencies are a cycle among
build-bound edges and are reported as such.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..blocks.base import Block, BlockError, BuildContext, Composite, Connection, PortKind, PortRef

SCHEMA_ID = "ring-blocks/1"
STEP_MODES = ("run", "simulate")
RESET_MODES = ("rebuild", "continue")
EDGE_PARAM_NAMES = ("weight", "margin")


class GraphError(ValueError):
    """Validation problems, all of them at once."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("graph is invalid:\n  " + "\n  ".join(self.problems))


@dataclass
class Simulation:
    dt_ms: float = 50.0
    nest_lead_steps: int = 4
    max_steps: int = 400
    rng_seed: Optional[int] = 13579
    local_num_threads: int = 1
    step_mode: str = "run"
    reset_mode: str = "rebuild"

    def validate(self) -> List[str]:
        problems = []
        if self.dt_ms <= 0:
            problems.append("simulation.dt_ms must be positive")
        if self.nest_lead_steps < 0:
            problems.append("simulation.nest_lead_steps must be >= 0")
        if self.max_steps <= 0:
            problems.append("simulation.max_steps must be positive")
        if self.step_mode not in STEP_MODES:
            problems.append("simulation.step_mode must be one of %r" % (STEP_MODES,))
        if self.reset_mode not in RESET_MODES:
            problems.append("simulation.reset_mode must be one of %r" % (RESET_MODES,))
        if self.local_num_threads < 1:
            problems.append("simulation.local_num_threads must be >= 1")
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return OrderedDict(
            dt_ms=self.dt_ms, nest_lead_steps=self.nest_lead_steps, max_steps=self.max_steps,
            rng_seed=self.rng_seed, local_num_threads=self.local_num_threads,
            step_mode=self.step_mode, reset_mode=self.reset_mode,
        )


@dataclass
class Edge:
    source: PortRef
    target: PortRef
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "%s -> %s" % (self.source.key, self.target.key)

    @property
    def kind(self) -> PortKind:
        return self.source.port.kind


@dataclass
class BuiltGraph:
    """What ``Graph.build`` returns: the built blocks and their readouts."""

    graph: "Graph"
    ctx: BuildContext
    order: List[str]
    build_seconds: float

    @property
    def blocks(self) -> "OrderedDict[str, Block]":
        return self.graph.blocks

    def block(self, id: str) -> Block:
        return self.graph.blocks[id]

    def counts_sources(self) -> Dict[str, Dict[str, Any]]:
        """block id -> {signal port -> recorder list} for every neural block."""

        result: Dict[str, Dict[str, Any]] = OrderedDict()
        for block in self.graph.neural_blocks():
            sources = block.counts_sources()
            if sources:
                result[block.id] = sources
        return result

    def encoders(self) -> List[Block]:
        return [block for block in self.graph.neural_blocks() if hasattr(block, "drive")]

    def rings(self) -> List[Block]:
        return [block for block in self.graph.neural_blocks() if block.type_name == "Ring"]

    def describe(self) -> Dict[str, Any]:
        return {
            "order": list(self.order), "build_seconds": round(self.build_seconds, 4),
            "log": list(self.ctx.build_log), "graph": self.graph.describe(),
        }


class Graph:
    """Blocks + edges; validate, build; serialisation in ``graph.schema`` (phase 4)."""

    def __init__(self, name: str = "", simulation: Optional[Simulation] = None, **simulation_kwargs: Any) -> None:
        self.name = name
        self.simulation = simulation or Simulation(**simulation_kwargs)
        self.blocks: "OrderedDict[str, Block]" = OrderedDict()
        self.composites: "OrderedDict[str, Composite]" = OrderedDict()
        self._port_maps: Dict[str, Dict[str, PortRef]] = {}
        self.edges: List[Edge] = []
        self.robot: Dict[str, Any] = {"engine": "fake", "stepper": "clock_wait"}

    # -- construction -----------------------------------------------------
    def add(self, block: Block) -> Block:
        if block.id in self.blocks or block.id in self.composites:
            raise GraphError(["duplicate block id %r" % block.id])
        if isinstance(block, Composite):
            sub_blocks, edges, port_map = block.expand()
            self.composites[block.id] = block
            self._port_maps[block.id] = port_map
            for sub in sub_blocks:
                self.add(sub)
            for source, target, params in edges:
                self.connect(source, target, **params)
            return block
        self.blocks[block.id] = block
        return block

    def resolve(self, ref: PortRef) -> PortRef:
        """Map a composite's port to the sub-block port implementing it."""

        seen = 0
        while isinstance(ref.block, Composite):
            port_map = self._port_maps.get(ref.block.id)
            if port_map is None:
                raise GraphError(["composite %r was not added to the graph" % ref.block.id])
            if ref.port.name not in port_map:
                raise GraphError(["composite %r has no mapping for port %r" % (ref.block.id, ref.port.name)])
            ref = port_map[ref.port.name]
            seen += 1
            if seen > 8:
                raise GraphError(["composite port mapping loops at %r" % ref.key])
        return ref

    def connect(self, source: PortRef, target: PortRef, **params: Any) -> Edge:
        if not isinstance(source, PortRef) or not isinstance(target, PortRef):
            raise GraphError(["connect() takes PortRefs, e.g. graph.connect(r1.spikes, f1.ring)"])
        edge = Edge(self.resolve(source), self.resolve(target), dict(params))
        self.edges.append(edge)
        return edge

    def block_ids(self) -> List[str]:
        return list(self.blocks)

    def neural_blocks(self) -> List[Block]:
        return [block for block in self.blocks.values() if block.neural]

    def signal_blocks(self) -> List[Block]:
        return [block for block in self.blocks.values() if not block.neural]

    def edges_into(self, block: Block, port_name: Optional[str] = None) -> List[Edge]:
        return [
            edge for edge in self.edges
            if edge.target.block is block and (port_name is None or edge.target.port.name == port_name)
        ]

    def edges_from(self, block: Block, port_name: Optional[str] = None) -> List[Edge]:
        return [
            edge for edge in self.edges
            if edge.source.block is block and (port_name is None or edge.source.port.name == port_name)
        ]

    # -- validation -------------------------------------------------------
    def problems(self) -> List[str]:
        problems: List[str] = list(self.simulation.validate())
        for edge in self.edges:
            for ref, role in ((edge.source, "source"), (edge.target, "target")):
                if ref.block.id not in self.blocks or self.blocks[ref.block.id] is not ref.block:
                    problems.append("edge %s: %s block %r is not in the graph" % (edge.key, role, ref.block.id))
            if edge.source.port.is_input:
                problems.append("edge %s: source %r is an input port" % (edge.key, edge.source.port.name))
            if not edge.target.port.is_input:
                problems.append("edge %s: target %r is an output port" % (edge.key, edge.target.port.name))
            if edge.source.port.kind is not edge.target.port.kind:
                problems.append(
                    "edge %s: kinds differ (%s -> %s)" % (edge.key, edge.source.port.kind.value, edge.target.port.kind.value)
                )
            unknown = sorted(set(edge.params) - set(EDGE_PARAM_NAMES))
            if unknown:
                problems.append("edge %s: unknown edge parameters %s" % (edge.key, unknown))
        for block in self.blocks.values():
            for port in block.input_ports():
                incoming = self.edges_into(block, port.name)
                if port.required and not incoming:
                    problems.append("%s.%s: required input is not connected" % (block.id, port.name))
                if len(incoming) > 1 and not port.multi:
                    problems.append(
                        "%s.%s: %d producers, at most one allowed" % (block.id, port.name, len(incoming))
                    )
        problems.extend(self._signal_cycles())
        problems.extend(self._build_order_problems())
        return problems

    def validate(self) -> "Graph":
        problems = self.problems()
        if problems:
            raise GraphError(problems)
        return self

    def _signal_cycles(self) -> List[str]:
        successors: Dict[str, List[str]] = {id: [] for id in self.blocks}
        for edge in self.edges:
            if edge.kind is PortKind.SIGNAL and edge.source.block.id in successors:
                successors[edge.source.block.id].append(edge.target.block.id)
        state: Dict[str, int] = {}
        problems: List[str] = []
        seen_cycles = set()

        def visit(node: str, path: List[str]) -> None:
            state[node] = 1
            for nxt in successors.get(node, ()):
                if state.get(nxt) == 1:
                    cycle = path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt]
                    key = frozenset(cycle)
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        problems.append("signal cycle within a tick: %s" % " -> ".join(cycle))
                elif state.get(nxt) is None:
                    visit(nxt, path + [nxt])
            state[node] = 2

        for id in self.blocks:
            if state.get(id) is None:
                visit(id, [id])
        return problems

    def build_order(self) -> Tuple[List[str], List[str]]:
        """Neural blocks in build order, plus the ones that never became ready."""

        pending = [block for block in self.neural_blocks()]
        built: List[str] = []
        done = set()
        while pending:
            progressed = False
            for block in list(pending):
                needs = [
                    edge.source.block.id for edge in self.edges_into(block)
                    if edge.kind is PortKind.SPIKES and not edge.target.port.late
                ]
                if all(source in done for source in needs):
                    built.append(block.id)
                    done.add(block.id)
                    pending.remove(block)
                    progressed = True
            if not progressed:
                break
        return built, [block.id for block in pending]

    def _build_order_problems(self) -> List[str]:
        _, stuck = self.build_order()
        if not stuck:
            return []
        return ["cycle among build-bound spike edges involving %s (mark an input as late-bound)" % stuck]

    # -- build ------------------------------------------------------------
    def build(self, ctx: BuildContext, configure_kernel: bool = True) -> BuiltGraph:
        """Build every neural block in NEST; signal blocks are reset and configured."""

        import time

        from ..nest.kernel import configure_kernel as _configure_kernel

        self.validate()
        started = time.perf_counter()
        if configure_kernel:
            _configure_kernel(
                ctx.backend, reset_kernel=True, verbosity="M_ERROR",
                local_num_threads=self.simulation.local_num_threads, rng_seed=self.simulation.rng_seed,
            )
        order, _ = self.build_order()
        for block_id in order:
            block = self.blocks[block_id]
            inputs: Dict[str, List[Connection]] = {}
            for edge in self.edges_into(block):
                if edge.kind is PortKind.SPIKES and not edge.target.port.late:
                    inputs.setdefault(edge.target.port.name, []).append(
                        Connection(edge.source.block, edge.source.port, block, edge.target.port, dict(edge.params))
                    )
            try:
                block.build(ctx, inputs)
            except BlockError:
                raise
            except Exception as exc:  # noqa: BLE001 - re-raise with the block id
                raise BlockError("%s %r: build failed: %s" % (block.type_name, block.id, exc))
        for edge in self.edges:
            if edge.kind is PortKind.SPIKES and edge.target.port.late:
                connection = Connection(edge.source.block, edge.source.port, edge.target.block, edge.target.port, dict(edge.params))
                edge.target.block.connect_late(ctx, connection)
        for block in self.blocks.values():
            configure = getattr(block, "configure", None)
            if callable(configure):
                configure(self.simulation.dt_ms)
            block.reset()
        return BuiltGraph(self, ctx, order, time.perf_counter() - started)

    # -- description --------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return OrderedDict(
            schema=SCHEMA_ID, name=self.name, simulation=self.simulation.to_dict(),
            blocks=[block.describe() for block in self.blocks.values()],
            composites=[block.describe() for block in self.composites.values()],
            edges=[OrderedDict(**{"from": e.source.key, "to": e.target.key}, **({"params": e.params} if e.params else {})) for e in self.edges],
            robot=dict(self.robot),
        )


__all__ = ["BuiltGraph", "Edge", "Graph", "GraphError", "SCHEMA_ID", "Simulation"]
