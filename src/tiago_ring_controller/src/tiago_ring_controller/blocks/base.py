"""Block, Port and BuildContext: the primitives every block type shares.

A block has a type name, a :class:`ParamSchema` (values validated on
construction), typed ports and fixed internal topology.  ``SPIKES`` ports carry
NEST node collections (wired at build time), ``SIGNAL`` ports carry floats or
small vectors once per tick.

Spike inputs come in two bindings.  A *build-bound* input (``Port.late`` False)
must be connected to a built source before the block itself builds, because
the block's structure depends on it (a readout needs its ring).  A *late-bound*
input (``Port.late`` True) is connected after both blocks exist; the target
block owns the connection pattern (``Ring.stim`` accepts encoder generators and
the gain's shifted feedback).  Spike cycles therefore build in a well-defined
order: feedback edges are late-bound.

Blocks never import NEST; the backend arrives in the :class:`BuildContext`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import load_neuron_parameters, source_config_path
from .params import ParamError, ParamSchema


class BlockError(ValueError):
    """A block could not be constructed, connected or built; names the block."""


class PortKind(Enum):
    SPIKES = "spikes"
    SIGNAL = "signal"


@dataclass(frozen=True)
class Port:
    name: str
    kind: PortKind
    direction: str  # "in" | "out"
    doc: str = ""
    required: bool = True   # inputs only: must be connected before build
    multi: bool = False     # inputs only: several producers allowed
    late: bool = False      # spike inputs only: connected after both blocks are built

    def __post_init__(self) -> None:
        if self.direction not in ("in", "out"):
            raise ValueError("port %s: direction must be 'in' or 'out'" % self.name)
        if self.late and (self.kind is not PortKind.SPIKES or self.direction != "in"):
            raise ValueError("port %s: only spike inputs can be late-bound" % self.name)

    @property
    def is_input(self) -> bool:
        return self.direction == "in"

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind.value, "direction": self.direction,
            "doc": self.doc, "required": self.required, "multi": self.multi, "late": self.late,
        }


def spikes_in(name: str, doc: str = "", required: bool = True, multi: bool = False, late: bool = False) -> Port:
    return Port(name, PortKind.SPIKES, "in", doc, required, multi, late)


def spikes_out(name: str, doc: str = "") -> Port:
    return Port(name, PortKind.SPIKES, "out", doc)


def signal_in(name: str, doc: str = "", required: bool = True) -> Port:
    return Port(name, PortKind.SIGNAL, "in", doc, required)


def signal_out(name: str, doc: str = "") -> Port:
    return Port(name, PortKind.SIGNAL, "out", doc)


@dataclass(frozen=True)
class PortRef:
    """``block.port`` as an object, for ``Graph.connect``."""

    block: "Block"
    port: Port

    @property
    def key(self) -> str:
        return "%s.%s" % (self.block.id, self.port.name)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return "PortRef(%s)" % self.key


@dataclass
class Connection:
    """One resolved edge into an input port."""

    source: "Block"
    source_port: Port
    target: "Block"
    target_port: Port
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "%s.%s -> %s.%s" % (self.source.id, self.source_port.name, self.target.id, self.target_port.name)


class BuildContext:
    """What a block needs to build: the backend, config paths, seed, threads."""

    def __init__(
        self,
        backend: Any,
        config_dir: Optional[str] = None,
        rng_seed: Optional[int] = None,
        local_num_threads: int = 1,
        dt_ms: float = 50.0,
    ) -> None:
        self.backend = backend
        self.config_dir = config_dir or source_config_path()
        self.rng_seed = rng_seed
        self.local_num_threads = int(local_num_threads)
        self.dt_ms = float(dt_ms)
        self._neuron_cache: Dict[str, Dict[str, Any]] = {}
        self.build_log: List[str] = []

    def artifact(self, *parts: str) -> str:
        return os.path.join(self.config_dir, *parts)

    def neuron_parameters(self, population: str, override: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The ``neuron_params.json`` set for ``population``, or the block's own."""

        if override:
            return dict(override)
        if population not in self._neuron_cache:
            self._neuron_cache[population] = load_neuron_parameters(
                self.artifact("model_params", "neuron_params.json"), population
            )
        return dict(self._neuron_cache[population])


class Block:
    """Base class; subclasses declare ``type_name``, ``schema`` and ``ports``."""

    type_name: ClassVar[str] = "Block"
    version: ClassVar[int] = 1
    schema: ClassVar[ParamSchema] = ParamSchema("Block", [])
    ports: ClassVar[Tuple[Port, ...]] = ()
    #: True for blocks that create NEST structure (they live in the NEST engine)
    neural: ClassVar[bool] = False

    def __init__(self, id: str, **params: Any) -> None:
        if not id or not isinstance(id, str) or "." in id or " " in id:
            raise BlockError("block id %r must be a non-empty string without dots or spaces" % (id,))
        self.id = id
        try:
            self.params: Dict[str, Any] = self.schema.resolve(params, owner="%s %r" % (self.type_name, id))
        except ParamError as exc:
            raise BlockError(str(exc))
        self.built = False
        #: spike output port name -> node collection / population after build
        self.outputs: Dict[str, Any] = {}
        self.ui: Dict[str, Any] = {}

    # -- ports --------------------------------------------------------------
    @classmethod
    def port(cls, name: str) -> Port:
        for port in cls.ports:
            if port.name == name:
                return port
        raise BlockError("%s has no port %r (ports: %s)" % (cls.type_name, name, [p.name for p in cls.ports]))

    @classmethod
    def input_ports(cls) -> List[Port]:
        return [p for p in cls.ports if p.is_input]

    @classmethod
    def output_ports(cls) -> List[Port]:
        return [p for p in cls.ports if not p.is_input]

    def __getattr__(self, name: str) -> PortRef:
        # Only called when normal lookup fails: resolve declared ports.
        ports = type(self).__dict__.get("ports") or type(self).ports
        for port in ports:
            if port.name == name:
                return PortRef(self, port)
        raise AttributeError("%s %r has no attribute or port %r" % (type(self).type_name, self.__dict__.get("id"), name))

    def ref(self, name: str) -> PortRef:
        return PortRef(self, self.port(name))

    # -- lifecycle ----------------------------------------------------------
    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        """Create the fixed topology.  ``inputs`` holds the build-bound spike edges."""

        self.built = True

    def connect_late(self, ctx: BuildContext, connection: Connection) -> None:
        """Wire a late-bound spike input; the target owns the pattern."""

        raise BlockError("%s %r: port %r accepts no late connections" % (self.type_name, self.id, connection.target_port.name))

    def reset(self) -> None:
        """Per-trial state (signal blocks); NEST structure is untouched."""

        return None

    def counts_sources(self) -> Dict[str, Any]:
        """Signal outputs that are spike-count readouts: port name -> population."""

        return {}

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type_name, "version": self.version, "params": dict(self.params)}

    @classmethod
    def describe_type(cls) -> Dict[str, Any]:
        return {
            "type": cls.type_name, "version": cls.version, "neural": cls.neural,
            "params": cls.schema.describe(), "ports": [p.describe() for p in cls.ports],
            "doc": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return "%s(%r)" % (self.type_name, self.id)

    # -- helpers for subclasses --------------------------------------------
    def _single_input(self, inputs: Mapping[str, Sequence[Connection]], port: str) -> Connection:
        connections = list(inputs.get(port, ()))
        if len(connections) != 1:
            raise BlockError("%s %r: port %r needs exactly one source, got %d" % (self.type_name, self.id, port, len(connections)))
        return connections[0]

    def _source_population(self, connection: Connection) -> Any:
        value = connection.source.outputs.get(connection.source_port.name)
        if value is None:
            raise BlockError(
                "%s %r: source %s.%s is not built" % (self.type_name, self.id, connection.source.id, connection.source_port.name)
            )
        return value

    def error(self, message: str) -> BlockError:
        return BlockError("%s %r: %s" % (self.type_name, self.id, message))


class Composite(Block):
    """A block that expands into primitives with internal edges.

    ``expand()`` returns ``(blocks, edges, port_map)``: the sub-blocks (ids are
    prefixed with the composite id), the internal edges as ``(PortRef, PortRef,
    params)`` and a map from the composite's port names to the sub-block port
    that implements them.
    """

    composite: ClassVar[bool] = True

    def expand(self) -> Tuple[List[Block], List[Tuple[PortRef, PortRef, Dict[str, Any]]], Dict[str, PortRef]]:
        raise NotImplementedError

    def sub_id(self, name: str) -> str:
        return "%s__%s" % (self.id, name)


def iter_ports(blocks: Iterable[Block]) -> Iterable[PortRef]:
    for block in blocks:
        for port in block.ports:
            yield PortRef(block, port)


__all__ = [
    "Block", "BlockError", "BuildContext", "Composite", "Connection", "Port", "PortKind", "PortRef",
    "iter_ports", "signal_in", "signal_out", "spikes_in", "spikes_out",
]
