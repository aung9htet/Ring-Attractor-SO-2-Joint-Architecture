"""Homeostasis block: warm/cold/left/right decision circuit with fitted feature weights."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..artifacts import load_json_legacy, load_numpy_legacy
from ..contracts import HomeostasisSpec
from ..nest.populations import BuildError, build_decision_population, connect_features_to_decision
from .base import Block, BuildContext, Connection, Port, signal_out, spikes_in, spikes_out
from .params import HOMEOSTASIS_SCHEMA


class Homeostasis(Block):
    """Compares state and target features: warm/cold, then a left/right decision pair."""

    type_name: ClassVar[str] = "Homeostasis"
    schema = HOMEOSTASIS_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("state_features", "features of the ring being moved (→ warm)"),
        spikes_in("target_features", "features of the reference (→ cold)"),
        spikes_out("left", "decision neuron: move left"),
        spikes_out("right", "decision neuron: move right"),
        signal_out("counts", "warm, cold, left, right spike-count deltas per tick"),
        signal_out("left_counts", "left decision spikes per tick (motor-signal experiment)"),
        signal_out("right_counts", "right decision spikes per tick"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.population = None
        self.feature_names = []

    def spec(self) -> HomeostasisSpec:
        p = self.params
        return HomeostasisSpec(
            config_dir="", tie_epsilon=1e-9, warm_cold_weight_scale=p["weight_scale"], warm_cold_bias_scale=0.0,
            warm_exc_weight=p["warm_exc_weight"], warm_inh_weight=p["warm_inh_weight"],
            cold_exc_weight=p["cold_exc_weight"], cold_inh_weight=p["cold_inh_weight"],
            decision_lateral_weight=p["decision_lateral_weight"], left_label=0, right_label=1,
        )

    def fitted_weights(self, ctx: BuildContext, n_features: int, ring_size: int):
        p = self.params
        weights_path = p["weights_artifact"] or ctx.artifact("homeostasis", "N_%d_homeostasis_weights.npy" % ring_size)
        metadata_path = p["metadata_artifact"] or os.path.join(
            os.path.dirname(weights_path), "N_%d_homeostasis_metadata.json" % ring_size
        )
        try:
            matrix = np.asarray(load_numpy_legacy(weights_path), dtype=float)
            metadata = load_json_legacy(metadata_path)
        except (OSError, ValueError) as exc:
            raise self.error("cannot load fitted weights: %s" % exc)
        names = list(metadata["feature_names"])
        if len(names) != n_features or matrix.shape != (2 * n_features, 2):
            raise self.error(
                "fitted weights %s (%r, %d features) do not match %d input features"
                % (weights_path, matrix.shape, len(names), n_features)
            )
        return matrix[:n_features, 0].copy(), matrix[n_features:, 1].copy(), names

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        state = self._single_input(inputs, "state_features")
        target = self._single_input(inputs, "target_features")
        state_pop = self._source_population(state)
        target_pop = self._source_population(target)
        if state_pop.size != target_pop.size:
            raise self.error("state features (%d) and target features (%d) differ in size" % (state_pop.size, target_pop.size))
        # The fitted artifact is keyed by the ring size the readouts were trained on.
        ring_size = getattr(state.source, "source_ring_size", None)
        if not ring_size:
            raise self.error(
                "%s %r does not expose the ring size behind its features" % (state.source.type_name, state.source.id)
            )
        warm, cold, names = self.fitted_weights(ctx, state_pop.size, ring_size)
        try:
            self.population = build_decision_population(
                ctx.backend, self.id, ctx.neuron_parameters("homeostasis", self.params["neuron_params"]), self.spec()
            )
            connect_features_to_decision(ctx.backend, self.population, state_pop, target_pop, warm, cold, self.params["weight_scale"])
        except BuildError as exc:
            raise self.error(str(exc))
        self.feature_names = names
        self.outputs = {"left": self.population.node("left"), "right": self.population.node("right")}
        ctx.build_log.append("%s: comparator on %s vs %s (N=%d artifact)" % (self.id, state_pop.name, target_pop.name, ring_size))
        self.built = True

    def counts_sources(self) -> Dict[str, Any]:
        return {"counts": self.population.recorder_list()}


__all__ = ["Homeostasis"]
