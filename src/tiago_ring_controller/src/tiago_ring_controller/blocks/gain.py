"""Gain block: opponent left/right populations gated by the decision pair."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Sequence, Tuple

from ..contracts import GainSpec
from ..nest.populations import (
    BuildError,
    build_gain_populations,
    connect_gain_feedback_populations,
    connect_gain_inputs,
)
from .base import Block, BuildContext, Connection, Port, signal_out, spikes_in, spikes_out
from .params import GAIN_SCHEMA


class Gain(Block):
    """Left/right gain populations: ring AND decision; feedback shifts the ring's bump."""

    type_name: ClassVar[str] = "Gain"
    schema = GAIN_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("ring", "the ring whose bump gates the populations (one_to_one)"),
        spikes_in("left_in", "decision neuron for left"),
        spikes_in("right_in", "decision neuron for right"),
        spikes_out("feedback", "shifted ±1 feedback into a ring's stim port (late-bound)"),
        signal_out("left_counts", "left population spikes per tick"),
        signal_out("right_counts", "right population spikes per tick"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.populations = None
        self.feedback_targets: Dict[str, int] = {}

    def spec(self) -> GainSpec:
        p = self.params
        return GainSpec(
            left_homeostasis_gain_weight=p["left_homeostasis_gain_weight"],
            right_homeostasis_gain_weight=p["right_homeostasis_gain_weight"],
            ring_to_gain_weight=p["ring_to_gain_weight"], gain_to_ring_weight=p["gain_to_ring_weight"],
            cross_inhibition_weight=p["cross_inhibition_weight"],
        )

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        ring = self._source_population(self._single_input(inputs, "ring"))
        left = self._source_population(self._single_input(inputs, "left_in"))
        right = self._source_population(self._single_input(inputs, "right_in"))
        try:
            self.populations = build_gain_populations(
                ctx.backend, self.id, ring.size, ctx.neuron_parameters("gain", self.params["neuron_params"])
            )
            connect_gain_inputs(ctx.backend, self.populations, ring, left, right, self.spec())
        except BuildError as exc:
            raise self.error(str(exc))
        self.outputs = {"feedback": self}
        ctx.build_log.append("%s: gain populations on %s" % (self.id, ring.name))
        self.built = True

    def connect_stim(self, ctx: BuildContext, ring: Block, connection: Connection) -> None:
        """Called by ``Ring.connect_late`` for ``gain.feedback -> ring.stim``."""

        weight = float(connection.params.get("weight", self.params["gain_to_ring_weight"]))
        margin = int(connection.params.get("margin", self.params["margin"]))
        if ring.population.size != self.populations.left.size:
            raise self.error("feedback target %r has %d neurons, gain has %d" % (ring.id, ring.population.size, self.populations.left.size))
        per_side = connect_gain_feedback_populations(ctx.backend, self.populations, ring.population, weight, margin)
        self.feedback_targets[ring.id] = per_side
        ctx.build_log.append("%s: feedback -> %s (%d per side, weight %g)" % (self.id, ring.id, per_side, weight))

    def counts_sources(self) -> Dict[str, Any]:
        return {
            "left_counts": self.populations.left.recorder_list(),
            "right_counts": self.populations.right.recorder_list(),
        }


__all__ = ["Gain"]
