"""Block layer: declared parameters, typed ports, primitives and composites.

``REGISTRY`` maps a type name to its class; ``block_from_type`` builds one
from a graph-file entry.  Importing this package needs neither NEST nor ROS.
"""

from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Type

from .base import (
    Block,
    BlockError,
    BuildContext,
    Composite,
    Connection,
    Port,
    PortKind,
    PortRef,
    signal_in,
    signal_out,
    spikes_in,
    spikes_out,
)
from .comparator import Homeostasis
from .composites import JointTriple, Transport
from .decoder import Decoder, ProfileDecoder
from .encoder import Encoder, FeatureEncoder
from .gain import Gain
from .joint import Goal, Joint
from .multi_ring import OutputRing, SignedProduct
from .params import PRIMITIVE_SCHEMAS, ParamError, ParamSchema, ParamSpec, schema_for
from .probe import Probe
from .readout import FourierReadout
from .ring import Ring
from .taskgain import TaskGain

REGISTRY: "OrderedDict[str, Type[Block]]" = OrderedDict(
    (cls.type_name, cls)
    for cls in (
        Ring, FourierReadout, Homeostasis, Gain, Encoder, FeatureEncoder, Decoder, Joint, Goal, Probe,
        SignedProduct, OutputRing, ProfileDecoder, TaskGain, Transport, JointTriple,
    )
)


def block_from_type(type_name: str, id: str, params: Optional[Mapping[str, Any]] = None) -> Block:
    try:
        cls = REGISTRY[type_name]
    except KeyError:
        raise BlockError("unknown block type %r for %r; known: %s" % (type_name, id, list(REGISTRY)))
    return cls(id, **dict(params or {}))


def describe_types() -> Dict[str, Dict[str, Any]]:
    """Palette description for the editor: params and ports of every type."""

    return OrderedDict((name, cls.describe_type()) for name, cls in REGISTRY.items())


__all__ = [
    "Block", "BlockError", "BuildContext", "Composite", "Connection", "Decoder", "Encoder", "FeatureEncoder",
    "FourierReadout", "Gain", "Goal", "Homeostasis", "Joint", "JointTriple", "OutputRing", "PRIMITIVE_SCHEMAS",
    "ParamError", "ParamSchema", "ParamSpec", "Port", "PortKind", "PortRef", "Probe", "ProfileDecoder", "REGISTRY",
    "Ring", "SignedProduct", "TaskGain", "Transport", "block_from_type", "describe_types", "schema_for",
    "signal_in", "signal_out", "spikes_in", "spikes_out",
]
