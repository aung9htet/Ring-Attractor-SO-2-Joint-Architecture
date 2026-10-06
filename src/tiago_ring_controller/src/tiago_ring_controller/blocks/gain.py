"""Gain block: opponent left/right populations gated by the decision pair."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..contracts import GainSpec
from ..nest.populations import (
    BuildError,
    build_gain_populations,
    connect_gain_feedback_populations,
    connect_gain_inputs,
    connect_gain_reference,
)
from .base import Block, BuildContext, Connection, Port, signal_out, spikes_in, spikes_out
from .params import GAIN_SCHEMA


#: defaults of a ``reference`` edge: excitation from the side the bump should move to, inhibition from the other
REFERENCE_EXC_WEIGHT = 2.0e4
REFERENCE_INH_WEIGHT = 6.0e4
REFERENCE_WINDOW = 10


class Gain(Block):
    """Left/right gain populations: ring AND decision; feedback shifts the ring's bump."""

    type_name: ClassVar[str] = "Gain"
    schema = GAIN_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("ring", "the ring whose bump gates the populations (one_to_one)"),
        spikes_in("left_in", "decision neuron for left"),
        spikes_in("right_in", "decision neuron for right"),
        spikes_in("reference", "a ring the gated bump is steered toward, index by index (edge params exc_weight, "
                  "inh_weight, window; late-bound)", required=False, multi=True, late=True),
        spikes_out("feedback", "shifted ±1 feedback into a ring's stim port (late-bound)"),
        signal_out("left_counts", "left population spikes per tick"),
        signal_out("right_counts", "right population spikes per tick"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.populations = None
        self.feedback_targets: Dict[str, int] = {}
        self.references: Dict[str, int] = {}

    def spec(self) -> GainSpec:
        p = self.params
        return GainSpec(
            left_homeostasis_gain_weight=p["left_homeostasis_gain_weight"],
            right_homeostasis_gain_weight=p["right_homeostasis_gain_weight"],
            ring_to_gain_weight=p["ring_to_gain_weight"], gain_to_ring_weight=p["gain_to_ring_weight"],
            cross_inhibition_weight=p["cross_inhibition_weight"],
        )

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        self.feedback_targets = {}
        self.references = {}
        ring = self._source_population(self._single_input(inputs, "ring"))
        left = self._source_population(self._single_input(inputs, "left_in"))
        right = self._source_population(self._single_input(inputs, "right_in"))
        left_drive, right_drive = self.drive_profiles(ring.size)
        try:
            self.populations = build_gain_populations(
                ctx.backend, self.id, ring.size, ctx.neuron_parameters("gain", self.params["neuron_params"])
            )
            connect_gain_inputs(ctx.backend, self.populations, ring, left, right, self.spec(), left_drive, right_drive)
        except BuildError as exc:
            raise self.error(str(exc))
        self.outputs = {"feedback": self}
        ctx.build_log.append("%s: gain populations on %s" % (self.id, ring.name))
        self.built = True

    def drive_profiles(self, ring_size: int):
        """``(left, right)`` per-neuron factors on the ring's drive, or ``(None, None)``.

        A bounded ring's limits are ``limit_arc``: the right population (which
        moves the bump up) fades to 0 at the upper limit, the left one at the
        lower limit, both over ``limit_taper`` indices.  Spikes and bump motion
        stop together, so the decoder stays valid at the limits.
        """

        arc = self.params["limit_arc"]
        if not arc:
            return None, None
        if len(arc) != 2 or not 0 <= arc[0] < arc[1] < ring_size:
            raise self.error("limit_arc must be [low, high] ring indices with low < high, got %r" % (arc,))
        taper = max(float(self.params["limit_taper"]), 1e-9)
        indices = np.arange(ring_size, dtype=float)
        right = np.clip((float(arc[1]) - indices) / taper, 0.0, 1.0)
        left = np.clip((indices - float(arc[0])) / taper, 0.0, 1.0)
        return left, right

    def connect_stim(self, ctx: BuildContext, ring: Block, connection: Connection) -> None:
        """Called by ``Ring.connect_late`` for ``gain.feedback -> ring.stim``."""

        weight = float(connection.params.get("weight", self.params["gain_to_ring_weight"]))
        margin = int(connection.params.get("margin", self.params["margin"]))
        if ring.population.size != self.populations.left.size:
            raise self.error("feedback target %r has %d neurons, gain has %d" % (ring.id, ring.population.size, self.populations.left.size))
        per_side = connect_gain_feedback_populations(ctx.backend, self.populations, ring.population, weight, margin)
        self.feedback_targets[ring.id] = per_side
        ctx.build_log.append("%s: feedback -> %s (%d per side, weight %g)" % (self.id, ring.id, per_side, weight))

    def connect_late(self, ctx: BuildContext, connection: Connection) -> None:
        """``ring.spikes -> gain.reference``: the sharp comparison (docs/architecture.md)."""

        if connection.target_port.name != "reference":
            raise self.error("port %r is not late-bound" % connection.target_port.name)
        params = connection.params
        exc = float(params.get("exc_weight", REFERENCE_EXC_WEIGHT))
        inh = float(params.get("inh_weight", REFERENCE_INH_WEIGHT))
        window = int(params.get("window", REFERENCE_WINDOW))
        try:
            count = connect_gain_reference(ctx.backend, self.populations, self._source_population(connection), exc, inh, window)
        except BuildError as exc_info:
            raise self.error(str(exc_info))
        self.references[connection.source.id] = count
        ctx.build_log.append("%s: reference <- %s (window %d, exc %g, inh %g)" % (self.id, connection.source.id, window, exc, inh))

    def counts_sources(self) -> Dict[str, Any]:
        return {
            "left_counts": self.populations.left.recorder_list(),
            "right_counts": self.populations.right.recorder_list(),
        }


__all__ = ["Gain", "REFERENCE_EXC_WEIGHT", "REFERENCE_INH_WEIGHT", "REFERENCE_WINDOW"]
