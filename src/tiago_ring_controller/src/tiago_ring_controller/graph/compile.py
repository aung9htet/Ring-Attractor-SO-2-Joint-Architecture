"""Compile a ``Graph`` into engines, transceiver functions and an ``FTILoop``.

Datapack conventions of a compiled graph:

* every block with signal outputs produces one datapack named by its id whose
  fields are its output ports (neural blocks through ``GraphNestEngine``,
  signal blocks through the transceivers below);
* an edge ``a.x -> b.y`` therefore means "read datapack ``a``, field ``x``";
* the robot engine keeps its ``joint_state`` / ``arm_velocity_cmd`` contract:
  ``Joint`` blocks read the first and the one fed by a ``Decoder`` writes the
  second (the ``MotorTF`` schema, so records, stop conditions, the dashboard
  and the monitor work unchanged);
* when the graph has the single-joint motif (Decoder ← Gain ← Ring), a
  ``LegacyViewTF`` also emits ``ring_counts`` for the existing observers.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..blocks.base import Block, PortKind
from ..blocks.decoder import Decoder, ProfileDecoder
from ..blocks.joint import Goal, Joint
from ..blocks.probe import Probe
from ..cosim.config import CosimConfig
from ..cosim.datapack import DataPack
from ..cosim.engine import Engine
from ..cosim.fakes import FakeRobotEngine
from ..cosim.graph_engine import GraphNestEngine
from ..cosim.loop import AnyOf, FTILoop, LoopObserver, MaxSteps, SettledFlag
from ..cosim.tf import GoalTF, TickContext, TransceiverFunction
from .graph import Graph, GraphError

ROBOT_ENGINES = ("fake", "gazebo")


# -- structure discovery --------------------------------------------------------
@dataclass
class Primary:
    """The single-joint motif of a graph, when present (for legacy views and records)."""

    decoder: Optional[Decoder] = None
    gain: Optional[Block] = None
    state_ring: Optional[Block] = None
    goal_ring: Optional[Block] = None
    joint: Optional[Joint] = None
    goal: Optional[Goal] = None
    state_encoder: Optional[Block] = None
    goal_encoder: Optional[Block] = None

    @property
    def complete(self) -> bool:
        return all(x is not None for x in (self.decoder, self.gain, self.state_ring, self.joint))


def source_of(graph: Graph, block: Block, port: str) -> Optional[Block]:
    edges = graph.edges_into(block, port)
    return edges[0].source.block if edges else None


def find_primary(graph: Graph) -> Primary:
    primary = Primary()
    joints = [b for b in graph.blocks.values() if isinstance(b, Joint)]
    goals = [b for b in graph.blocks.values() if isinstance(b, Goal)]
    primary.joint = joints[0] if joints else None
    primary.goal = goals[0] if goals else None
    for joint in joints:
        decoder = source_of(graph, joint, "velocity")
        if isinstance(decoder, Decoder):
            primary.joint, primary.decoder = joint, decoder
            break
    if primary.decoder is None:
        decoders = [b for b in graph.blocks.values() if isinstance(b, Decoder)]
        primary.decoder = decoders[0] if decoders else None
    if primary.decoder is not None:
        gain = source_of(graph, primary.decoder, "left_counts")
        if gain is not None and gain.type_name == "Gain":
            primary.gain = gain
            ring = source_of(graph, gain, "ring")
            if ring is not None and ring.type_name == "Ring":
                primary.state_ring = ring
            left = source_of(graph, gain, "left_in")
            comparator = left if left is not None and left.type_name == "Homeostasis" else None
            if comparator is not None:
                target = source_of(graph, comparator, "target_features")
                goal_ring = source_of(graph, target, "ring") if target is not None and target.type_name == "FourierReadout" else None
                if goal_ring is not None and goal_ring.type_name == "Ring":
                    primary.goal_ring = goal_ring
    for ring, attr in ((primary.state_ring, "state_encoder"), (primary.goal_ring, "goal_encoder")):
        if ring is None:
            continue
        for edge in graph.edges_into(ring, "stim"):
            if edge.source.block.type_name == "Encoder":
                setattr(primary, attr, edge.source.block)
    return primary


# -- transceivers ---------------------------------------------------------------
class BlockTF(TransceiverFunction):
    """Base for signal-block transceivers: reads its input edges as (datapack, field)."""

    def __init__(self, graph: Graph, block: Block) -> None:
        self.graph = graph
        self.block = block
        self.name = block.id
        self.sources: Dict[str, Any] = OrderedDict()
        for port in block.input_ports():
            if port.kind is not PortKind.SIGNAL:
                continue
            edges = graph.edges_into(block, port.name)
            if edges:
                self.sources[port.name] = (edges[0].source.block.id, edges[0].source.port.name)
        self.inputs = frozenset(source for source, _ in self.sources.values())
        self.outputs = frozenset({block.id})

    def reset(self) -> None:
        self.block.reset()

    def read(self, inputs: Mapping[str, DataPack], port: str, default: Any = None) -> Any:
        source = self.sources.get(port)
        if source is None:
            return default
        pack = inputs.get(source[0])
        if pack is None:
            return default
        return pack.get(source[1], default)

    def emit(self, ctx: TickContext, **fields: Any) -> Dict[str, DataPack]:
        return {self.block.id: DataPack(self.block.id, ctx.t_ms, fields)}


class GoalBlockTF(BlockTF, GoalTF):
    """``Goal`` → ``{"angle": ...}``; ``set_goal`` is what the runner and the dashboard call."""

    def __init__(self, graph: Graph, block: Goal, encoder: Optional[Block] = None) -> None:
        BlockTF.__init__(self, graph, block)
        self.encoder = encoder
        self.goal_rad = block.value()
        self.last_index = None
        self.mode = "once"

    def set_goal(self, goal_rad: float) -> None:
        self.block.set(goal_rad)
        self.goal_rad = float(goal_rad)

    def reset(self) -> None:
        self.block.reset()
        self.last_index = None

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        angle = self.block.value(ctx.t_ms)
        self.goal_rad = angle
        if self.encoder is not None and getattr(self.encoder, "ring_size", 0):
            self.last_index = self.encoder.ring_index(angle)
        return self.emit(ctx, angle=angle)


class JointSensorTF(BlockTF):
    """``joint_state`` → ``{"angle", "velocity_measured"}`` of one joint."""

    def __init__(self, graph: Graph, block: Joint) -> None:
        BlockTF.__init__(self, graph, block)
        self.inputs = frozenset({"joint_state"})
        self.last_index = None
        self.encoder = None
        for edge in graph.edges_from(block, "angle"):
            if edge.target.block.type_name == "Encoder":
                self.encoder = edge.target.block

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        state = inputs.get("joint_state")
        if state is None:
            return {}
        values = self.block.read(state)
        if self.last_index is None and self.encoder is not None and getattr(self.encoder, "ring_size", 0):
            self.last_index = self.encoder.ring_index(values["angle"])
        return self.emit(ctx, **values)


class DecoderTF(BlockTF):
    def __init__(self, graph: Graph, block: Decoder) -> None:
        BlockTF.__init__(self, graph, block)
        self._last_step: Optional[int] = None

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        left = self.read(inputs, "left_counts")
        right = self.read(inputs, "right_counts")
        if left is None or right is None:
            return {}
        source_pack = inputs.get(self.sources["left_counts"][0])
        nest_step = None if source_pack is None else source_pack.get("nest_step")
        if nest_step is not None and self._last_step is not None and nest_step <= self._last_step:
            return {}  # sample already consumed (engine not advanced this tick)
        self._last_step = nest_step
        result = self.block.step(
            float(left), float(right), is_lead=ctx.is_lead, centroid=self.read(inputs, "centroid"), nest_step=nest_step,
        )
        return self.emit(ctx, **result)

    def reset(self) -> None:
        self.block.reset()
        self._last_step = None


class JointCommandTF(BlockTF):
    """``Decoder.velocity -> Joint.velocity`` → ``arm_velocity_cmd`` (the MotorTF schema)."""

    def __init__(self, graph: Graph, block: Joint, decoder: Decoder, source_id: str) -> None:
        BlockTF.__init__(self, graph, block)
        self.name = block.id + ".command"
        self.decoder = decoder
        self.source_id = source_id
        self.inputs = frozenset({source_id})
        self.outputs = frozenset({"arm_velocity_cmd"})
        self.dt_s = graph.simulation.dt_ms / 1000.0
        self.nest_lead_steps = graph.simulation.nest_lead_steps
        self._ring_fields: Optional[LegacyViewTF] = None

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        pack = inputs.get(self.source_id)
        if pack is None:
            return {}
        consumed = dict(pack["consumed"])
        # The collector record and the dashboard expect the ring fields on the consumed sample.
        view = self._ring_fields.last if self._ring_fields is not None else None
        consumed.setdefault("r1_spike_count", 0.0 if view is None else view.get("r1_spike_count", 0.0))
        consumed.setdefault("r1_bump_index", None if view is None else view.get("r1_bump_index"))
        consumed.setdefault("r1_centroid", float("nan") if view is None else view.get("r1_centroid", float("nan")))
        return {
            "arm_velocity_cmd": DataPack(
                "arm_velocity_cmd", ctx.t_ms,
                {
                    "joint_index": self.block.index, "dt_s": self.dt_s,
                    "velocities": list(pack["velocity"]), "horizon_len": len(pack["velocity"]),
                    "nest_lead_steps": self.nest_lead_steps, "consumed": consumed,
                    "consecutive_settled": int(pack.get("consecutive_settled", 0)),
                    "settled": bool(pack["settled"]), "phase": ctx.phase,
                },
            )
        }


class ProfileDecoderTF(BlockTF):
    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        profile = self.read(inputs, "profile")
        if profile is None:
            return {}
        return self.emit(ctx, **self.block.step(profile))


class ProbeTF(BlockTF):
    def __init__(self, graph: Graph, block: Probe) -> None:
        BlockTF.__init__(self, graph, block)
        self.outputs = frozenset()

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        value = self.read(inputs, "signal")
        if value is not None:
            self.block.record(ctx.t_ms, value)
        return {}


class LegacyViewTF(TransceiverFunction):
    """``ring_counts`` (the NestEngine schema) from the primary ring and gain datapacks."""

    name = "legacy_view"
    outputs = frozenset({"ring_counts"})

    def __init__(self, primary: Primary, step_mode: str) -> None:
        self.primary = primary
        self.step_mode = step_mode
        self.inputs = frozenset({primary.state_ring.id, primary.gain.id})
        self.last: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self.last = None

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        ring = inputs.get(self.primary.state_ring.id)
        gain = inputs.get(self.primary.gain.id)
        if ring is None or gain is None:
            return {}
        data = {
            "left": int(gain["left_counts"]), "right": int(gain["right_counts"]),
            "r1_delta": list(ring["counts"]), "r1_spike_count": float(ring["total"]),
            "r1_bump_index": ring["bump_index"], "r1_centroid": ring["centroid"],
            "t_nest_ms": ring["t_nest_ms"], "nest_step": ring["nest_step"], "hidden_ms": 0.0,
            "step_mode": self.step_mode, "readout_mode": "node_collection",
        }
        self.last = data
        return {"ring_counts": DataPack("ring_counts", ring.t_ms, data)}


# -- compilation ------------------------------------------------------------------
@dataclass
class CompiledGraph:
    graph: Graph
    loop: FTILoop
    config: CosimConfig
    nest_engine: Optional[GraphNestEngine]
    robot_engine: Engine
    tfs: List[TransceiverFunction]
    primary: Primary
    meta: Dict[str, Any] = field(default_factory=dict)


def graph_cosim_config(graph: Graph, primary: Optional[Primary] = None) -> CosimConfig:
    """A ``CosimConfig`` describing the graph run, for records, writers and the dashboard."""

    primary = primary or find_primary(graph)
    sim = graph.simulation
    values: Dict[str, Any] = dict(
        dt_ms=sim.dt_ms, nest_lead_steps=sim.nest_lead_steps, max_steps=sim.max_steps,
        rng_seed=sim.rng_seed, local_num_threads=sim.local_num_threads, nest_step_mode=sim.step_mode,
        reset_mode=sim.reset_mode, stepper=graph.robot.get("stepper", "clock_wait"), nest_model="vectorised",
    )
    encoder = primary.state_encoder or primary.goal_encoder
    if encoder is not None and encoder.params["mapping"] != "circular":
        values["profile"] = encoder.params["mapping"]
        values["stimulus_half_width"] = encoder.params["half_width"]
        values["proprioception_mode"] = encoder.params["mode"] if encoder.params["mode"] != "corrective" else "continuous"
    if primary.joint is not None:
        values["joint_index"] = primary.joint.index
        values["joint_min"], values["joint_max"] = primary.joint.limits()
    if primary.decoder is not None:
        p = primary.decoder.params
        values.update(
            decoder_gain_positive=p["gain_positive"], decoder_gain_negative=p["gain_negative"],
            decoder_tau_s=p["tau_s"], decoder_delay_steps=p["delay_steps"],
            drive_threshold=p["drive_threshold"], n_settle=p["n_settle"],
        )
    return CosimConfig(**values)


def _signal_order(graph: Graph) -> List[Block]:
    """Signal blocks in dependency order over signal edges (sensors and goals first)."""

    blocks = graph.signal_blocks()
    remaining = list(blocks)
    ordered: List[Block] = []
    done = set()
    while remaining:
        progressed = False
        for block in list(remaining):
            needs = [
                edge.source.block.id for edge in graph.edges_into(block)
                if edge.kind is PortKind.SIGNAL and not edge.source.block.neural and edge.source.block is not block
            ]
            if all(source in done for source in needs):
                ordered.append(block)
                done.add(block.id)
                remaining.remove(block)
                progressed = True
        if not progressed:  # validated graphs have no signal cycles
            raise GraphError(["signal cycle among %s" % [b.id for b in remaining]])
    return ordered


def make_robot_engine(graph: Graph, config: CosimConfig, engines: str, transport: Any = None) -> Engine:
    kind = graph.robot.get("engine", "fake") if engines == "full" else "fake"
    if kind == "fake":
        return FakeRobotEngine("robot")
    if kind == "gazebo":
        from ..cosim.runner import make_gazebo_engine

        return make_gazebo_engine(config, transport)
    raise GraphError(["robot.engine must be one of %r, got %r" % (ROBOT_ENGINES, kind)])


def compile_graph(
    graph: Graph,
    engines: str = "fake",
    backend: Any = None,
    observers: Optional[Sequence[LoopObserver]] = None,
    transport: Any = None,
    config_dir: Optional[str] = None,
) -> CompiledGraph:
    """``engines``: ``"fake"`` (fake NEST backend + fake robot), ``"nest"`` (real NEST + fake robot), ``"full"``."""

    if engines not in ("fake", "nest", "full"):
        raise GraphError(["engines must be 'fake', 'nest' or 'full'"])
    graph.validate()
    primary = find_primary(graph)
    config = graph_cosim_config(graph, primary)
    if backend is None:
        if engines == "fake":
            from ..cosim.fake_backend import FakeNestBackend

            backend = FakeNestBackend()
        else:
            import nest as backend  # noqa: WPS433 - lazy by design

    nest_engine = GraphNestEngine(backend, graph, config_dir=config_dir) if graph.neural_blocks() else None
    robot_engine = make_robot_engine(graph, config, engines, transport)
    engine_list: List[Engine] = [engine for engine in (nest_engine, robot_engine) if engine is not None]

    tfs: List[TransceiverFunction] = []
    decoder_tf_ids: Dict[str, str] = {}
    for block in _signal_order(graph):
        if isinstance(block, Goal):
            encoder = None
            for edge in graph.edges_from(block, "angle"):
                if edge.target.block.type_name == "Encoder":
                    encoder = edge.target.block
            tfs.append(GoalBlockTF(graph, block, encoder))
        elif isinstance(block, Joint):
            tfs.append(JointSensorTF(graph, block))
            decoder = source_of(graph, block, "velocity")
            if isinstance(decoder, Decoder):
                tfs.append(JointCommandTF(graph, block, decoder, decoder.id))
        elif isinstance(block, Decoder):
            tfs.append(DecoderTF(graph, block))
        elif isinstance(block, ProfileDecoder):
            tfs.append(ProfileDecoderTF(graph, block))
        elif isinstance(block, Probe):
            tfs.append(ProbeTF(graph, block))
    legacy_view = None
    if primary.complete and nest_engine is not None:
        legacy_view = LegacyViewTF(primary, graph.simulation.step_mode)
        for tf in tfs:
            if isinstance(tf, JointCommandTF):
                tf._ring_fields = legacy_view
        tfs.insert(0, legacy_view)
    # Sensors and goals must run before the blocks reading them; JointCommandTF after its decoder.
    tfs.sort(key=lambda tf: 0 if isinstance(tf, LegacyViewTF) else 1 if isinstance(tf, (JointSensorTF, GoalBlockTF)) else 3 if isinstance(tf, JointCommandTF) else 2)

    lead_steps = graph.simulation.nest_lead_steps if nest_engine is not None else 0
    loop = FTILoop(
        engine_list, tfs, dt_ms=graph.simulation.dt_ms, nest_lead_steps=lead_steps,
        lead_engine="nest" if lead_steps > 0 else None,
        stop_condition=AnyOf(SettledFlag(), MaxSteps(graph.simulation.max_steps)),
        observers=observers,
    )
    return CompiledGraph(
        graph=graph, loop=loop, config=config, nest_engine=nest_engine, robot_engine=robot_engine,
        tfs=tfs, primary=primary, meta={"engines": engines, "graph": graph.describe()},
    )


__all__ = [
    "BlockTF", "CompiledGraph", "DecoderTF", "GoalBlockTF", "JointCommandTF", "JointSensorTF",
    "LegacyViewTF", "Primary", "ProbeTF", "ProfileDecoderTF", "compile_graph", "find_primary",
    "graph_cosim_config", "make_robot_engine",
]
