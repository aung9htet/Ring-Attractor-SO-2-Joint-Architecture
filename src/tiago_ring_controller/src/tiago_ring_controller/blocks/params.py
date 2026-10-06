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
    ParamSpec("wall", "list", [], doc="bounded ring: [low, high] ring indices; neurons outside get their bias current "
              "scaled by outside_current_scale so no bump can rest there (empty = no wall)"),
    ParamSpec("outside_current_scale", "float", 1.0, minimum=0.0, maximum=1.0,
              doc="bias current factor outside the wall"),
    ParamSpec("wall_taper", "int", 0, minimum=0, doc="indices over which the factor changes linearly at each wall end"),
])

FOURIER_READOUT_SCHEMA = ParamSchema("FourierReadout", [
    ParamSpec("num_fourier_k", "int", 20, minimum=1, doc="harmonics K; 2K feature neurons"),
    ParamSpec("weight_scale", "float", 200.0, doc="multiplies the (N, 2K) mask"),
    ParamSpec("dc_baseline", "float", 200.0, unit="pA", doc="dc_generator amplitude into the features"),
    ParamSpec("mask_source", "str", "analytic", choices=("analytic", "artifact"),
              doc="analytic push-pull sine masks (identical to the checked-in artifacts) or a .npy file"),
    ParamSpec("weights_artifact", "str", "", doc="artifact path; empty = ring_decoding_weights/N_<N>_fourier_weights.npy"),
])

# Defaults below equal config/model_params/homeostasis_params.json and gain_modulation_params.json
# (checked by test_blocks_fake), so a graph file needs no parameter artifact for them.
HOMEOSTASIS_SCHEMA = ParamSchema("Homeostasis", [
    ParamSpec("warm_exc_weight", "float", 20000.0, doc="cold -> right"),
    ParamSpec("warm_inh_weight", "float", -20000.0, doc="cold -> left"),
    ParamSpec("cold_exc_weight", "float", 20000.0, doc="warm -> left"),
    ParamSpec("cold_inh_weight", "float", -20000.0, doc="warm -> right"),
    ParamSpec("decision_lateral_weight", "float", -20000.0, doc="left <-> right"),
    ParamSpec("weight_scale", "float", 5.0e6, doc="multiplies the fitted feature weights"),
    ParamSpec("ramp_current", "float", 0.0, unit="pA",
              doc="added to the warm and cold neurons' bias current: keeps both ramps above the zero-rate floor across "
                  "the joint arc (the decision reads their difference); 0 = legacy"),
    ParamSpec("weights_artifact", "str", "", doc="N_<N>_homeostasis_weights.npy; empty = homeostasis/ per feature count"),
    ParamSpec("metadata_artifact", "str", "", doc="matching metadata JSON (feature order); empty = per feature count"),
    _neuron_param_spec("homeostasis"),
])

GAIN_SCHEMA = ParamSchema("Gain", [
    ParamSpec("left_homeostasis_gain_weight", "float", 10000.0, doc="decision left -> left population"),
    ParamSpec("right_homeostasis_gain_weight", "float", 10000.0, doc="decision right -> right population"),
    ParamSpec("ring_to_gain_weight", "float", 450000.0, doc="ring i -> left/right i"),
    ParamSpec("gain_to_ring_weight", "float", -0.6, doc="feedback weight (shifted +-1); an edge may override it"),
    ParamSpec("cross_inhibition_weight", "float", -30000.0, doc="decision left -> right population and vice versa"),
    ParamSpec("margin", "int", 5, minimum=0, doc="ring neurons excluded from feedback at each end"),
    _neuron_param_spec("gain"),
    ParamSpec("limit_arc", "list", [], doc="bounded ring: [low, high] ring indices of the joint limits; the ring's drive "
              "of the right (upward) population fades to 0 at high, of the left population at low (empty = none)"),
    ParamSpec("limit_taper", "float", 10.0, minimum=0.0, doc="indices over which the drive fades at a limit"),
])

ENCODER_SCHEMA = ParamSchema("Encoder", [
    ParamSpec("half_width", "int", 5, minimum=0, doc="bump half width in ring indices"),
    ParamSpec("rate_hz", "float", 200.0, minimum=0.0, unit="Hz", doc="generator rate inside the window"),
    ParamSpec("weight", "float", 4.5e3, doc="generator -> ring synapse weight"),
    ParamSpec("duration_ticks", "int", 1, minimum=1, doc="ticks a bump stays on"),
    ParamSpec("mode", "str", "once", choices=("once", "continuous", "corrective"), doc="when to stimulate"),
    ParamSpec("dead_band", "int", 0, minimum=0, doc="corrective mode: minimum |centroid - index| in ring indices"),
    ParamSpec("mapping", "str", "collector", choices=("collector", "analysis", "calibration", "circular", "bounded"),
              doc="angle -> ring index: a legacy control profile over [joint_min, joint_max], the full circle, "
                  "or 'bounded': the joint range on arc_fraction of the ring, centred opposite the seam"),
    ParamSpec("joint_min", "float", -1.0, unit="rad", doc="lower joint limit of the mapping"),
    ParamSpec("joint_max", "float", 1.0, unit="rad", doc="upper joint limit of the mapping"),
    ParamSpec("arc_fraction", "float", 0.70, minimum=0.05, maximum=0.95,
              doc="bounded mapping: fraction of the ring the joint range covers"),
])

FEATURE_ENCODER_SCHEMA = ParamSchema("FeatureEncoder", [
    ParamSpec("num_fourier_k", "int", 20, minimum=1, doc="harmonics K; 2K push-pull generators"),
    ParamSpec("rate_scale", "float", 100.0, minimum=0.0, unit="Hz", doc="rate for a feature value of 1"),
    ParamSpec("weight", "float", 1.0, doc="generator -> target synapse weight scale (multiplied by fitted weights)"),
    ParamSpec("joint_min", "float", -1.0, unit="rad"),
    ParamSpec("joint_max", "float", 1.0, unit="rad"),
    ParamSpec("mapping", "str", "collector", choices=("collector", "analysis", "calibration", "circular", "bounded")),
    ParamSpec("population_size", "int", 200, minimum=3, doc="ring size the angle is mapped onto"),
    ParamSpec("arc_fraction", "float", 0.70, minimum=0.05, maximum=0.95, doc="bounded mapping: fraction of the ring"),
])

DECODER_SCHEMA = ParamSchema("Decoder", [
    ParamSpec("source", "str", "gain_counts", choices=("gain_counts", "centroid_velocity", "decision_counts")),
    ParamSpec("spike_scale", "float", 1.0, doc="multiplies right - left before the filter"),
    ParamSpec("dead_zone", "float", 0.0, minimum=0.0, unit="spikes/tick",
              doc="signed counts below this give no drive, larger ones are reduced by it (the bump "
                  "does not move below a minimum gain rate); applied before the filter"),
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
    ParamSpec("joint_min", "float", -1.0, unit="rad", doc="lower limit"),
    ParamSpec("joint_max", "float", 1.0, unit="rad", doc="upper limit"),
    ParamSpec("limits_source", "str", "params", choices=("params", "urdf"),
              doc="'urdf': read the limits from /robot_description when a robot transport is available "
                  "(encoders mapped with the old limits follow)"),
    ParamSpec("max_velocity", "float", 1.0, minimum=0.0, unit="rad/s",
              doc="commanded velocities are clipped to this magnitude; commanded positions never pass the limits"),
])

GOAL_SCHEMA = ParamSchema("Goal", [
    ParamSpec("angle_rad", "float", 0.0, unit="rad"),
])

PROBE_SCHEMA = ParamSchema("Probe", [
    ParamSpec("keep", "int", 0, minimum=0, doc="samples kept (0 = all)"),
])

SIGNED_PRODUCT_SCHEMA = ParamSchema("SignedProduct", [
    ParamSpec("grid_size", "int", 32, minimum=1, doc="g; 16 g^2 conjunction cells"),
    ParamSpec("dc_baseline", "float", 120.0, unit="pA", doc="calibrated signed_product_dc_baseline"),
    ParamSpec("input_weight", "float", 700.0, doc="calibrated signed_product_input_weight"),
    ParamSpec("epsilon", "float", 1.0e-9, minimum=0.0),
])

OUTPUT_RING_SCHEMA = ParamSchema("OutputRing", [
    ParamSpec("size", "int", 100, minimum=1),
    ParamSpec("dc_baseline", "float", 200.0, unit="pA"),
    ParamSpec("weight_scale", "float", 1000.0, doc="signed_product_output_weight_scale"),
    ParamSpec("weights_artifact", "str", "", doc=".npz; empty = ring_decoding_weights/N_<N>_J_2_multi_ring_sawtooth_weights.npz"),
    ParamSpec("weights_key", "str", "W_signed_lift", doc="array in the .npz, shape (n_features, size)"),
])

RING_STATE_SCHEMA = ParamSchema("RingState", [
    ParamSpec("mapping", "str", "collector", choices=("collector", "analysis", "calibration", "circular", "bounded"),
              doc="the state encoder's mapping, inverted to read the joint angle"),
    ParamSpec("joint_min", "float", -1.0, unit="rad"),
    ParamSpec("joint_max", "float", 1.0, unit="rad"),
    ParamSpec("half_width", "int", 5, minimum=0, doc="the state encoder's half width (legacy mappings use it)"),
    ParamSpec("arc_fraction", "float", 0.70, minimum=0.05, maximum=0.95, doc="bounded mapping: fraction of the ring"),
    ParamSpec("window_ticks", "int", 3, minimum=1, doc="ticks of counts summed before the readout"),
    ParamSpec("arc_margin", "float", 5.0, minimum=0.0, doc="ring indices of slack for in_arc"),
])

PROFILE_DECODER_SCHEMA = ParamSchema("ProfileDecoder", [
    ParamSpec("method", "str", "sawtooth", choices=("sawtooth", "scalar_ramp", "centroid")),
])

TASK_GAIN_SCHEMA = ParamSchema("TaskGain", [
    ParamSpec("weights_artifact", "str", "", doc="fitted bilinear weights (research; not implemented)"),
])

TRANSPORT_SCHEMA = ParamSchema("Transport", [
    ParamSpec("homeostasis", "dict", {}, doc="Homeostasis parameter overrides"),
    ParamSpec("gain", "dict", {}, doc="Gain parameter overrides"),
])

JOINT_TRIPLE_SCHEMA = ParamSchema("JointTriple", [
    ParamSpec("population_size", "int", 200, minimum=3),
    ParamSpec("num_fourier_k", "int", 20, minimum=1),
    ParamSpec("half_width", "int", 5, minimum=0),
    ParamSpec("mapping", "str", "collector", choices=("collector", "analysis", "calibration", "bounded")),
    ParamSpec("joint_min", "float", -1.0, unit="rad"),
    ParamSpec("joint_max", "float", 1.0, unit="rad"),
    ParamSpec("arc_fraction", "float", 0.70, minimum=0.05, maximum=0.95, doc="bounded mapping: fraction of the ring"),
    ParamSpec("ring", "dict", {}, doc="Ring parameter overrides of the three rings (the bounded ring's wall)"),
    ParamSpec("homeostasis", "dict", {}, doc="Homeostasis parameter overrides of both comparators"),
    ParamSpec("gain", "dict", {}, doc="Gain parameter overrides of both gain pairs (cross inhibition is 0 unless given)"),
    ParamSpec("state_mode", "str", "corrective", choices=("once", "continuous", "corrective"),
              doc="how the state ring follows the measured angle (continuous drive biases the body comparator)"),
    ParamSpec("state_rate_hz", "float", 200.0, minimum=0.0, unit="Hz"),
    ParamSpec("state_dead_band", "int", 1, minimum=0, doc="state_mode corrective: indices the state bump may be off"),
    ParamSpec("body_gain", "dict", {}, doc="Gain parameter overrides of the body pair only (on top of gain)"),
    ParamSpec("goal_mode", "str", "corrective", choices=("once", "continuous", "corrective"),
              doc="how the goal ring is held on the goal angle"),
    ParamSpec("goal_dead_band", "int", 2, minimum=0, doc="goal_mode corrective: indices the goal bump may be off"),
    ParamSpec("goal_feedback_weight", "float", -0.6, doc="goal gain pair -> predictive ring (shifted +-1)"),
    ParamSpec("body_feedback_weight", "float", -3.0, doc="body gain pair -> predictive ring (how hard the body pulls)"),
    ParamSpec("goal_reference", "dict", {"exc_weight": 2.0e4, "inh_weight": 6.0e4, "window": 10},
              doc="goal ring -> goal gain pair (Gain.reference edge parameters); empty = no sharp comparison"),
    ParamSpec("body_reference", "dict", {"exc_weight": 2.0e4, "inh_weight": 1.5e5, "window": 10},
              doc="state ring -> body gain pair (Gain.reference edge parameters); empty = no sharp comparison"),
    ParamSpec("window_ticks", "int", 3, minimum=1, doc="ticks summed by the two RingState readouts"),
])

PRIMITIVE_SCHEMAS: Dict[str, ParamSchema] = OrderedDict(
    (schema.block_type, schema)
    for schema in (
        RING_SCHEMA, FOURIER_READOUT_SCHEMA, HOMEOSTASIS_SCHEMA, GAIN_SCHEMA,
        ENCODER_SCHEMA, FEATURE_ENCODER_SCHEMA, DECODER_SCHEMA, JOINT_SCHEMA, GOAL_SCHEMA, PROBE_SCHEMA,
        SIGNED_PRODUCT_SCHEMA, OUTPUT_RING_SCHEMA, PROFILE_DECODER_SCHEMA, RING_STATE_SCHEMA, TASK_GAIN_SCHEMA,
        TRANSPORT_SCHEMA, JOINT_TRIPLE_SCHEMA,
    )
)


def schema_for(block_type: str) -> ParamSchema:
    try:
        return PRIMITIVE_SCHEMAS[block_type]
    except KeyError:
        raise KeyError("unknown block type %r; known: %s" % (block_type, list(PRIMITIVE_SCHEMAS)))


__all__ = [
    "DECODER_SCHEMA", "ENCODER_SCHEMA", "FEATURE_ENCODER_SCHEMA", "FOURIER_READOUT_SCHEMA", "GAIN_SCHEMA",
    "GOAL_SCHEMA", "HOMEOSTASIS_SCHEMA", "JOINT_SCHEMA", "JOINT_TRIPLE_SCHEMA", "OUTPUT_RING_SCHEMA",
    "PARAM_TYPES", "PRIMITIVE_SCHEMAS", "PROBE_SCHEMA", "PROFILE_DECODER_SCHEMA", "RING_SCHEMA", "RING_STATE_SCHEMA",
    "SIGNED_PRODUCT_SCHEMA", "TASK_GAIN_SCHEMA", "TRANSPORT_SCHEMA",
    "ParamError", "ParamSchema", "ParamSpec", "schema_for",
]
