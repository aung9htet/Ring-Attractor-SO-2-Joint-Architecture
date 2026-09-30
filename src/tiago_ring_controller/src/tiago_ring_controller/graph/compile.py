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
* when the graph has the single-joint motif (Decoder ← Gain ← Ring), the NEST
  engine also emits the ``ring_counts`` datapack and the goal / joint
  transceivers emit ``goal_bump`` / ``state_bump``, so the dashboard, the
  monitor and the collector record work unchanged.
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

    def __init__(self, graph: Graph, block: Goal, encoder: Optional[Block] = None, emit_bump: bool = False) -> None:
        BlockTF.__init__(self, graph, block)
        self.encoder = encoder
        self.goal_rad = block.value()
        self.last_index = None
        self.mode = "once"
        self.emit_bump = emit_bump and encoder is not None
        self.outputs = frozenset({block.id, "goal_bump"}) if self.emit_bump else frozenset({block.id})

    def set_goal(self, goal_rad: float) -> None:
        self.block.set(goal_rad)
        self.goal_rad = float(goal_rad)

    def reset(self) -> None:
        self.block.reset()
        self.last_index = None

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        angle = self.block.value(ctx.t_ms)
        self.goal_rad = angle
        packs = self.emit(ctx, angle=angle)
        if self.encoder is not None and getattr(self.encoder, "ring_size", 0):
            self.last_index = self.encoder.ring_index(angle)
        if self.emit_bump and self.last_index is not None:
            # The dashboard and the monitor draw the goal marker from this pack (legacy GoalTF schema).
            packs["goal_bump"] = DataPack("goal_bump", ctx.t_ms, {
                "center_index": int(self.last_index), "half_width": self.encoder.effective_half_width(),
                "goal_rad": float(angle), "source": "goal", "mode": self.encoder.params["mode"],
            })
        return packs


class JointSensorTF(BlockTF):
    """``joint_state`` → ``{"angle", "velocity_measured"}`` of one joint."""

    def __init__(self, graph: Graph, block: Joint, emit_bump: bool = False) -> None:
        BlockTF.__init__(self, graph, block)
        self.inputs = frozenset({"joint_state"})
        self.last_index = None
        self.encoder = None
        for edge in graph.edges_from(block, "angle"):
            if edge.target.block.type_name == "Encoder":
                self.encoder = edge.target.block
        self.emit_bump = emit_bump and self.encoder is not None
        self.outputs = frozenset({block.id, "state_bump"}) if self.emit_bump else frozenset({block.id})

    def reset(self) -> None:
        self.block.reset()
        self.last_index = None

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        state = inputs.get("joint_state")
        if state is None:
            return {}
        values = self.block.read(state)
        packs = self.emit(ctx, **values)
        if self.last_index is None and self.encoder is not None and getattr(self.encoder, "ring_size", 0):
            self.last_index = self.encoder.ring_index(values["angle"])
            if not self.emit_bump:
                return packs
            # First measurement of the trial: the observers' initial-index marker (legacy ProprioceptionTF schema).
            packs["state_bump"] = DataPack("state_bump", ctx.t_ms, {
                "center_index": int(self.last_index), "half_width": self.encoder.effective_half_width(),
                "joint_position": float(values["angle"]), "source": "proprioception", "mode": self.encoder.params["mode"],
            })
        return packs


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
    """``Decoder.velocity -> Joint.velocity`` → ``"<joint>.command"`` (one joint's horizon)."""

    def __init__(self, graph: Graph, block: Joint, decoder: Decoder, source_id: str) -> None:
        BlockTF.__init__(self, graph, block)
        self.name = block.id + ".command"
        self.decoder = decoder
        self.source_id = source_id
        self.inputs = frozenset({source_id, "ring_counts"})
        self.outputs = frozenset({self.name})

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        pack = inputs.get(self.source_id)
        if pack is None:
            return {}
        consumed = dict(pack["consumed"])
        # The collector record and the dashboard expect the ring fields on the consumed sample;
        # the NEST engine emits ring_counts when the graph has the single-joint motif.
        view = inputs.get("ring_counts")
        consumed.setdefault("r1_spike_count", 0.0 if view is None else view.get("r1_spike_count", 0.0))
        consumed.setdefault("r1_bump_index", None if view is None else view.get("r1_bump_index"))
        consumed.setdefault("r1_centroid", float("nan") if view is None else view.get("r1_centroid", float("nan")))
        return {
            self.name: DataPack(
                self.name, ctx.t_ms,
                {
                    "joint_index": self.block.index, "velocities": list(pack["velocity"]),
                    "consumed": consumed, "consecutive_settled": int(pack.get("consecutive_settled", 0)),
                    "settled": bool(pack["settled"]),
                },
            )
        }


class ArmCommandTF(TransceiverFunction):
    """Merge every ``"<joint>.command"`` into one ``arm_velocity_cmd`` (the MotorTF schema plus ``commands``).

    The legacy fields describe the primary joint (records, dashboard, monitor);
    ``settled`` is True only when every commanded joint has settled; the robot
    engines build one trajectory from ``commands`` (all joints per tick).
    """

    name = "arm_command"
    outputs = frozenset({"arm_velocity_cmd"})

    def __init__(self, command_tfs: Sequence[JointCommandTF], primary_joint: Optional[Joint], graph: Graph) -> None:
        self.command_tfs = list(command_tfs)
        self.primary_id = None
        for tf in self.command_tfs:
            if primary_joint is not None and tf.block is primary_joint:
                self.primary_id = tf.name
        if self.primary_id is None and self.command_tfs:
            self.primary_id = self.command_tfs[0].name
        self.inputs = frozenset(tf.name for tf in self.command_tfs)
        self.dt_s = graph.simulation.dt_ms / 1000.0
        self.nest_lead_steps = graph.simulation.nest_lead_steps

    def __call__(self, inputs: Mapping[str, DataPack], ctx: TickContext) -> Dict[str, DataPack]:
        packs = [inputs[tf.name] for tf in self.command_tfs if tf.name in inputs]
        if not packs:
            return {}
        primary = inputs.get(self.primary_id) or packs[0]
        commands = [
            {"joint_index": int(p["joint_index"]), "velocities": list(p["velocities"]), "settled": bool(p["settled"])}
            for p in packs
        ]
        return {
            "arm_velocity_cmd": DataPack(
                "arm_velocity_cmd", ctx.t_ms,
                {
                    "joint_index": int(primary["joint_index"]), "dt_s": self.dt_s,
                    "velocities": list(primary["velocities"]), "horizon_len": len(primary["velocities"]),
                    "nest_lead_steps": self.nest_lead_steps, "consumed": dict(primary["consumed"]),
                    "consecutive_settled": int(primary.get("consecutive_settled", 0)),
                    "settled": all(c["settled"] for c in commands), "phase": ctx.phase,
                    "commands": commands,
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
    """``engines="full"`` always drives the real robot (Gazebo through ROS); ``fake`` / ``nest`` use the fake robot.

    ``graph.robot["engine"]`` documents the intended target and must be a known
    kind; the stepper comes from ``graph.robot["stepper"]`` through the config.
    """

    kind = graph.robot.get("engine", "gazebo")
    if kind not in ROBOT_ENGINES:
        raise GraphError(["robot.engine must be one of %r, got %r" % (ROBOT_ENGINES, kind)])
    if engines != "full":
        return FakeRobotEngine("robot")
    from ..cosim.runner import make_gazebo_engine

    return make_gazebo_engine(config, transport)


def resolve_joint_limits(graph: Graph, transport: Any) -> Dict[str, Any]:
    """Apply URDF limits to ``Joint`` blocks with ``limits_source="urdf"`` (and their encoders).

    Encoders whose ``joint_min``/``joint_max`` equal the joint's previous limits
    (the templates set them from the same source) follow.  Returns what changed.
    """

    from ..cosim.limits import joint_limits_from_transport

    changed: Dict[str, Any] = {}
    for joint in [b for b in graph.blocks.values() if isinstance(b, Joint)]:
        if joint.params["limits_source"] != "urdf":
            continue
        limits = joint_limits_from_transport(transport, joint.index)
        if limits is None:
            raise GraphError(["Joint %r: no URDF limits for arm joint %d on the parameter server" % (joint.id, joint.index)])
        old = joint.limits()
        joint.params["joint_min"], joint.params["joint_max"] = float(limits[0]), float(limits[1])
        followers = []
        for block in graph.blocks.values():
            params = block.params
            if block is not joint and "joint_min" in params and (params["joint_min"], params["joint_max"]) == old:
                params["joint_min"], params["joint_max"] = float(limits[0]), float(limits[1])
                followers.append(block.id)
        changed[joint.id] = {"from": old, "to": limits, "followers": followers}
    return changed


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
            tfs.append(GoalBlockTF(graph, block, encoder, emit_bump=encoder is not None and encoder is primary.goal_encoder))
        elif isinstance(block, Joint):
            tfs.append(JointSensorTF(graph, block, emit_bump=block is primary.joint))
            decoder = source_of(graph, block, "velocity")
            if isinstance(decoder, Decoder):
                tfs.append(JointCommandTF(graph, block, decoder, decoder.id))
        elif isinstance(block, Decoder):
            tfs.append(DecoderTF(graph, block))
        elif isinstance(block, ProfileDecoder):
            tfs.append(ProfileDecoderTF(graph, block))
        elif isinstance(block, Probe):
            tfs.append(ProbeTF(graph, block))
    if primary.complete and nest_engine is not None:
        nest_engine.set_legacy_view(
            primary.state_ring.id, primary.gain.id, None if primary.goal_ring is None else primary.goal_ring.id
        )
    command_tfs = [tf for tf in tfs if isinstance(tf, JointCommandTF)]
    if command_tfs:
        tfs.append(ArmCommandTF(command_tfs, primary.joint, graph))
    # Sensors and goals must run before the blocks reading them; commands after their decoders; the merge last.
    tfs.sort(key=lambda tf: 1 if isinstance(tf, (JointSensorTF, GoalBlockTF))
             else 3 if isinstance(tf, JointCommandTF) else 4 if isinstance(tf, ArmCommandTF) else 2)

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
    "ArmCommandTF", "BlockTF", "CompiledGraph", "DecoderTF", "GoalBlockTF", "JointCommandTF", "JointSensorTF",
    "Primary", "ProbeTF", "ProfileDecoderTF", "compile_graph", "find_primary",
    "graph_cosim_config", "make_robot_engine", "resolve_joint_limits",
]
