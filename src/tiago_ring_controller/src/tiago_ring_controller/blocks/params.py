"""Declared block parameters: ``ParamSpec`` and ``ParamSchema``.

Every editable block parameter is declared once with a type, default, bounds,
unit and doc.  Values live in the graph file; :meth:`ParamSchema.resolve`
validates them on load (and the editor uses :meth:`ParamSchema.describe` for
its panels).  All problems of one block are reported together.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PARAM_TYPES = ("int", "float", "bool", "str", "dict", "list")


class ParamError(ValueError):
    """One or more block parameters are invalid; the message lists them all."""

    def __init__(self, owner: str, problems: Sequence[str]) -> None:
        self.owner = owner
        self.problems = list(problems)
        super().__init__("%s: %s" % (owner, "; ".join(self.problems)))


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    unit: str = ""
    doc: str = ""

    def __post_init__(self) -> None:
        if self.type not in PARAM_TYPES:
            raise ValueError("%s: unknown parameter type %r" % (self.name, self.type))
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
        # The default must satisfy the spec itself.
        self.coerce(self.default)

    def coerce(self, value: Any) -> Any:
        """Return the validated value (ints accepted as floats), else raise."""

        kind = self.type
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError("expected bool, got %r" % (value,))
            result: Any = value
        elif kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                if isinstance(value, float) and float(value).is_integer():
                    value = int(value)
                else:
                    raise ValueError("expected int, got %r" % (value,))
            result = int(value)
        elif kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("expected float, got %r" % (value,))
            result = float(value)
        elif kind == "str":
            if not isinstance(value, str):
                raise ValueError("expected str, got %r" % (value,))
            result = value
        elif kind == "dict":
            if not isinstance(value, Mapping):
                raise ValueError("expected an object, got %r" % (value,))
            result = dict(value)
        else:  # list
            if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__iter__"):
                raise ValueError("expected a list, got %r" % (value,))
            result = list(value)
        if self.minimum is not None and kind in ("int", "float") and result < self.minimum:
            raise ValueError("%r is below the minimum %r" % (result, self.minimum))
        if self.maximum is not None and kind in ("int", "float") and result > self.maximum:
            raise ValueError("%r is above the maximum %r" % (result, self.maximum))
        if self.choices is not None and result not in self.choices:
            raise ValueError("%r is not one of %r" % (result, self.choices))
        return result

    def describe(self) -> Dict[str, Any]:
        item: Dict[str, Any] = {"name": self.name, "type": self.type, "default": self.default}
        for key in ("minimum", "maximum", "unit", "doc"):
            value = getattr(self, key)
            if value not in (None, ""):
                item[key] = value
        if self.choices is not None:
            item["choices"] = list(self.choices)
        return item


class ParamSchema:
    """An ordered set of :class:`ParamSpec` for one block type."""

    def __init__(self, block_type: str, specs: Sequence[ParamSpec]) -> None:
        self.block_type = block_type
        self.specs: "OrderedDict[str, ParamSpec]" = OrderedDict()
        for spec in specs:
            if spec.name in self.specs:
                raise ValueError("%s: duplicate parameter %r" % (block_type, spec.name))
            self.specs[spec.name] = spec

    @property
    def names(self) -> List[str]:
        return list(self.specs)

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def __getitem__(self, name: str) -> ParamSpec:
        return self.specs[name]

    def defaults(self) -> Dict[str, Any]:
        return OrderedDict((name, spec.default) for name, spec in self.specs.items())

    def resolve(self, values: Optional[Mapping[str, Any]] = None, owner: Optional[str] = None) -> Dict[str, Any]:
        """Validate ``values``; unknown names rejected, missing ones defaulted."""

        owner = owner or self.block_type
        given = dict(values or {})
        problems: List[str] = []
        resolved: Dict[str, Any] = OrderedDict()
        for name in given:
            if name not in self.specs:
                problems.append("unknown parameter %r" % name)
        for name, spec in self.specs.items():
            if name in given:
                try:
                    resolved[name] = spec.coerce(given[name])
                except ValueError as exc:
                    problems.append("%s: %s" % (name, exc))
            else:
                resolved[name] = spec.default
        if problems:
            raise ParamError(owner, problems)
        return resolved

    def describe(self) -> List[Dict[str, Any]]:
        return [spec.describe() for spec in self.specs.values()]

    def dumps(self, values: Optional[Mapping[str, Any]] = None, owner: Optional[str] = None) -> str:
        return json.dumps(self.resolve(values, owner), indent=2, sort_keys=False)

    def loads(self, text: str, owner: Optional[str] = None) -> Dict[str, Any]:
        return self.resolve(json.loads(text), owner)


def _neuron_param_spec(population: str) -> ParamSpec:
    return ParamSpec(
        "neuron_params", "dict", {},
        doc="iaf_psc_alpha parameters for the %s population; empty means the config file's %r set"
        % (population, population),
    )


RING_SCHEMA = ParamSchema("Ring", [
    ParamSpec("population_size", "int", 200, minimum=3, doc="ring neurons N"),
    ParamSpec("variant", "str", "legacy", choices=("legacy", "builder"), doc="distance profile"),
    ParamSpec("max_distance", "float", 50.0, minimum=0.0, doc="distance assigned to the far side"),
    ParamSpec("excitation_std_dev", "float", 10.0, minimum=0.0, doc="Mexican-hat excitation width"),
    ParamSpec("inhibition_std_dev", "float", 5.0, minimum=0.0, doc="Mexican-hat inhibition width"),
    _neuron_param_spec("ring"),
])

FOURIER_READOUT_SCHEMA = ParamSchema("FourierReadout", [
    ParamSpec("num_fourier_k", "int", 20, minimum=1, doc="harmonics K; 2K feature neurons"),
    ParamSpec("weight_scale", "float", 200.0, doc="multiplies the (N, 2K) mask"),
    ParamSpec("dc_baseline", "float", 200.0, unit="pA", doc="dc_generator amplitude into the features"),
    ParamSpec("mask_source", "str", "analytic", choices=("analytic", "artifact"),
              doc="analytic sine masks or the N_<N>_fourier_weights.npy artifact"),
])

HOMEOSTASIS_SCHEMA = ParamSchema("Homeostasis", [
    ParamSpec("warm_exc_weight", "float", 0.0, doc="cold -> right"),
    ParamSpec("warm_inh_weight", "float", 0.0, doc="cold -> left"),
    ParamSpec("cold_exc_weight", "float", 0.0, doc="warm -> left"),
    ParamSpec("cold_inh_weight", "float", 0.0, doc="warm -> right"),
    ParamSpec("decision_lateral_weight", "float", 0.0, doc="left <-> right"),
    ParamSpec("weight_scale", "float", 5.0e6, doc="multiplies the fitted feature weights"),
    ParamSpec("weights_artifact", "str", "", doc="N_<N>_homeostasis_weights.npy; empty = per-N default"),
    _neuron_param_spec("homeostasis"),
])

GAIN_SCHEMA = ParamSchema("Gain", [
    ParamSpec("left_homeostasis_gain_weight", "float", 0.0, doc="decision left -> left population"),
    ParamSpec("right_homeostasis_gain_weight", "float", 0.0, doc="decision right -> right population"),
    ParamSpec("ring_to_gain_weight", "float", 0.0, doc="ring i -> left/right i"),
    ParamSpec("gain_to_ring_weight", "float", 0.0, doc="feedback weight (shifted +-1)"),
    ParamSpec("cross_inhibition_weight", "float", 0.0, doc="decision left -> right population and vice versa"),
    ParamSpec("margin", "int", 5, minimum=0, doc="ring neurons excluded from feedback at each end"),
    _neuron_param_spec("gain"),
])

ENCODER_SCHEMA = ParamSchema("Encoder", [
    ParamSpec("half_width", "int", 5, minimum=0, doc="bump half width in ring indices"),
    ParamSpec("rate_hz", "float", 200.0, minimum=0.0, unit="Hz", doc="generator rate inside the window"),
    ParamSpec("weight", "float", 4.5e3, doc="generator -> ring synapse weight"),
    ParamSpec("duration_ticks", "int", 1, minimum=1, doc="ticks a bump stays on"),
    ParamSpec("mode", "str", "once", choices=("once", "continuous", "corrective"), doc="when to stimulate"),
    ParamSpec("dead_band", "int", 0, minimum=0, doc="corrective mode: minimum centroid error in indices"),
    ParamSpec("mapping", "str", "collector", choices=("collector", "analysis", "calibration"),
              doc="angle -> ring index mapping (legacy control profile)"),
])

DECODER_SCHEMA = ParamSchema("Decoder", [
    ParamSpec("source", "str", "gain_counts", choices=("gain_counts", "centroid_velocity", "decision_counts")),
    ParamSpec("spike_scale", "float", 1.0, doc="multiplies right - left before the filter"),
    ParamSpec("tau_s", "float", 0.3, minimum=0.0, unit="s"),
    ParamSpec("delay_steps", "int", 0, minimum=0),
    ParamSpec("gain_positive", "float", 1.0e-4),
    ParamSpec("gain_negative", "float", -1.0e-4),
    ParamSpec("drive_threshold", "float", 5.0, minimum=0.0),
    ParamSpec("n_settle", "int", 10, minimum=1),
    ParamSpec("horizon", "int", 4, minimum=1, doc="trajectory points published per tick"),
])

JOINT_SCHEMA = ParamSchema("Joint", [
    ParamSpec("index", "int", 5, minimum=0, maximum=6, doc="arm joint index"),
    ParamSpec("joint_min", "float", -1.0, unit="rad", doc="lower limit (URDF or calibration)"),
    ParamSpec("joint_max", "float", 1.0, unit="rad", doc="upper limit"),
])

GOAL_SCHEMA = ParamSchema("Goal", [
    ParamSpec("angle_rad", "float", 0.0, unit="rad"),
])

PRIMITIVE_SCHEMAS: Dict[str, ParamSchema] = OrderedDict(
    (schema.block_type, schema)
    for schema in (
        RING_SCHEMA, FOURIER_READOUT_SCHEMA, HOMEOSTASIS_SCHEMA, GAIN_SCHEMA,
        ENCODER_SCHEMA, DECODER_SCHEMA, JOINT_SCHEMA, GOAL_SCHEMA,
    )
)


def schema_for(block_type: str) -> ParamSchema:
    try:
        return PRIMITIVE_SCHEMAS[block_type]
    except KeyError:
        raise KeyError("unknown block type %r; known: %s" % (block_type, list(PRIMITIVE_SCHEMAS)))


__all__ = [
    "DECODER_SCHEMA", "ENCODER_SCHEMA", "FOURIER_READOUT_SCHEMA", "GAIN_SCHEMA", "GOAL_SCHEMA",
    "HOMEOSTASIS_SCHEMA", "JOINT_SCHEMA", "PARAM_TYPES", "PRIMITIVE_SCHEMAS", "RING_SCHEMA",
    "ParamError", "ParamSchema", "ParamSpec", "schema_for",
]
