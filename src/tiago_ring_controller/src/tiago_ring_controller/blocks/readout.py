"""FourierReadout block: 2K push-pull sine features of one ring."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence, Tuple

import numpy as np

from ..artifacts import load_numpy_legacy
from ..nest.populations import BuildError, build_readout_population
from ..training.analytic import build_fourier_weights
from .base import Block, BuildContext, Connection, Port, spikes_in, spikes_out
from .params import FOURIER_READOUT_SCHEMA


class FourierReadout(Block):
    """2K feature neurons (sin_pos/sin_neg per harmonic) with a DC baseline, driven by a ring."""

    type_name: ClassVar[str] = "FourierReadout"
    schema = FOURIER_READOUT_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("ring", "the ring to read out"),
        spikes_out("features", "2K feature neurons, sin_pos_k1, sin_neg_k1, ..."),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.population = None
        self._ring_size = 0

    def weights(self, ctx: BuildContext, population_size: int) -> np.ndarray:
        p = self.params
        harmonics = int(p["num_fourier_k"])
        if p["mask_source"] == "analytic":
            return np.asarray(build_fourier_weights(population_size, harmonics), dtype=float)
        path = p["weights_artifact"] or ctx.artifact("ring_decoding_weights", "N_%d_fourier_weights.npy" % population_size)
        try:
            matrix = np.asarray(load_numpy_legacy(path), dtype=float)
        except (OSError, ValueError) as exc:
            raise self.error("cannot load Fourier weights %s: %s" % (path, exc))
        if matrix.shape != (population_size, 2 * harmonics):
            raise self.error("Fourier weights %s have shape %r, expected %r" % (path, matrix.shape, (population_size, 2 * harmonics)))
        return matrix

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        ring = self._source_population(self._single_input(inputs, "ring"))
        p = self.params
        try:
            self.population = build_readout_population(
                ctx.backend, self.id, ring, self.weights(ctx, ring.size), p["num_fourier_k"],
                p["weight_scale"], p["dc_baseline"],
            )
        except BuildError as exc:
            raise self.error(str(exc))
        self._ring_size = ring.size
        self.outputs = {"features": self.population}
        ctx.build_log.append("%s: readout K=%d on %s" % (self.id, p["num_fourier_k"], ring.name))
        self.built = True

    @property
    def feature_names(self):
        return list(self.population.feature_names) if self.population is not None else []

    @property
    def source_ring_size(self) -> int:
        """Ring size the features were built on (keys the fitted comparator artifact)."""

        return int(self._ring_size)


__all__ = ["FourierReadout"]
