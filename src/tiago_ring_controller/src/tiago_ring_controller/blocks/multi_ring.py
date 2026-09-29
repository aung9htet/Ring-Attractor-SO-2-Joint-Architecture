"""Forward-kinematics primitives: SignedProduct (two rings → conjunction features) and OutputRing."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..artifacts import load_numpy_legacy
from ..features import signed_product_feature_order
from ..nest.multi_ring import build_signed_product_layer
from ..nest.populations import BuildError, build_output_ring_population
from .base import Block, BuildContext, Connection, Port, signal_out, spikes_in, spikes_out
from .params import OUTPUT_RING_SCHEMA, SIGNED_PRODUCT_SCHEMA


class SignedProduct(Block):
    """16 conjunction populations on a g×g grid over two rings (positive/negative per term)."""

    type_name: ClassVar[str] = "SignedProduct"
    schema = SIGNED_PRODUCT_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("ring_a", "first joint ring (q1)"),
        spikes_in("ring_b", "second joint ring (q2)"),
        spikes_out("features", "16·g² conjunction cells in the fixed feature order"),
        signal_out("counts", "spike-count deltas of every cell per tick, feature order"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.layer = None
        self.ring_size = 0
        self.feature_order: List[str] = []
        self._recorders: List[Any] = []

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        a = self._source_population(self._single_input(inputs, "ring_a"))
        b = self._source_population(self._single_input(inputs, "ring_b"))
        if a.size != b.size:
            raise self.error("rings differ in size (%d vs %d)" % (a.size, b.size))
        p = self.params
        if p["grid_size"] > a.size:
            raise self.error("grid_size %d exceeds the ring size %d" % (p["grid_size"], a.size))
        try:
            self.layer = build_signed_product_layer(
                ctx.backend, a.neuron_list(), b.neuron_list(), a.size, p["grid_size"],
                p["dc_baseline"], p["input_weight"], epsilon=p["epsilon"],
            )
        except (BuildError, ValueError) as exc:
            raise self.error(str(exc))
        self.ring_size = a.size
        self.feature_order = list(self.layer.feature_order)
        self._recorders = []
        for name in self.feature_order:
            recorders = self.layer.populations[name]["recs"]
            self._recorders.extend(recorders[i] for i in range(len(recorders)))
        self.outputs = {"features": self.layer.flat_nodes}
        ctx.build_log.append("%s: %d conjunction cells over %s x %s" % (self.id, len(self._recorders), a.name, b.name))
        self.built = True

    @property
    def n_features(self) -> int:
        return len(self._recorders)

    def counts_sources(self) -> Dict[str, Any]:
        return {"counts": list(self._recorders)}


class OutputRing(Block):
    """One population driven by a ridge-fitted (features × size) matrix plus a DC baseline."""

    type_name: ClassVar[str] = "OutputRing"
    schema = OUTPUT_RING_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("features", "SignedProduct features (or any feature population)"),
        spikes_out("spikes", "the output ring neurons"),
        signal_out("profile", "per-neuron spike-count deltas per tick"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.population = None

    def weights(self, ctx: BuildContext, n_features: int, ring_size_hint: int) -> np.ndarray:
        p = self.params
        path = p["weights_artifact"] or ctx.artifact(
            "ring_decoding_weights", "N_%d_J_2_multi_ring_sawtooth_weights.npz" % ring_size_hint
        )
        try:
            data = load_numpy_legacy(path, allow_pickle=True)
            matrix = np.asarray(data[p["weights_key"]], dtype=float)
        except (OSError, ValueError, KeyError) as exc:
            raise self.error("cannot load %s[%s]: %s" % (path, p["weights_key"], exc))
        if matrix.shape != (n_features, int(p["size"])):
            raise self.error("%s[%s] has shape %r, expected %r" % (path, p["weights_key"], matrix.shape, (n_features, int(p["size"]))))
        return matrix

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        connection = self._single_input(inputs, "features")
        source_nodes = self._source_population(connection)
        n_features = len(source_nodes)
        # The artifact is keyed by the joint-ring size the feature layer was trained on.
        hint = int(getattr(connection.source, "ring_size", 0) or 100)
        matrix = self.weights(ctx, n_features, hint)
        p = self.params
        try:
            self.population = build_output_ring_population(
                ctx.backend, self.id, source_nodes, matrix, p["size"], p["dc_baseline"], p["weight_scale"]
            )
        except BuildError as exc:
            raise self.error(str(exc))
        self.outputs = {"spikes": self.population}
        ctx.build_log.append("%s: output ring %d from %d features (%s)" % (self.id, p["size"], n_features, p["weights_key"]))
        self.built = True

    def counts_sources(self) -> Dict[str, Any]:
        return {"profile": self.population.recorder_list()}


__all__ = ["OutputRing", "SignedProduct"]
