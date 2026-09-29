"""Encoders: angle → generator rates on a ring (Encoder) or on 2K features (FeatureEncoder)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..control.profiles import get_legacy_control_profile
from ..math.circular import preferred_angles
from ..nest.populations import (
    BuildError,
    Population,
    StimulusGenerators,
    build_stimulus_generators,
    set_stimulus_rates,
)
from .base import Block, BuildContext, Connection, Port, signal_in, spikes_out
from .params import ENCODER_SCHEMA, FEATURE_ENCODER_SCHEMA


class Encoder(Block):
    """Angle → bump: N Poisson generators one_to_one on the target ring, rate set per tick.

    Modes: ``once`` (first angle of a trial only, legacy parity), ``continuous``
    (every tick), ``corrective`` (only when the ring's centroid is further than
    ``dead_band`` from the mapped index).  A window stays on for
    ``duration_ticks`` ticks.  The block owns no NEST structure until it is
    connected to a ring (late-bound ``stim``), because the generator count is
    the ring size.
    """

    type_name: ClassVar[str] = "Encoder"
    schema = ENCODER_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("angle", "angle in rad (Joint.angle or Goal.angle)"),
        spikes_out("stim", "generator drive into one ring's stim port"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.profile = get_legacy_control_profile(self.params["mapping"])
        self.stimulus: Optional[StimulusGenerators] = None
        self.ring_id: Optional[str] = None
        self.ring_size = 0
        self.reset()

    # -- structure ----------------------------------------------------------
    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        self.outputs = {"stim": self}
        self.built = True

    def connect_stim(self, ctx: BuildContext, ring: Block, connection: Connection) -> None:
        if self.stimulus is not None:
            raise self.error("already drives ring %r; one encoder drives one ring" % self.ring_id)
        try:
            self.stimulus = build_stimulus_generators(ctx.backend, self.id, ring.population, self.params["weight"])
        except BuildError as exc:
            raise self.error(str(exc))
        self.ring_id = ring.id
        self.ring_size = ring.population.size
        self._backend = ctx.backend
        ctx.build_log.append("%s: %d generators -> %s (%s)" % (self.id, self.ring_size, ring.id, self.params["mode"]))

    # -- per-tick behaviour ---------------------------------------------------
    def reset(self) -> None:
        self._emitted = False
        self._expires_at: Optional[float] = None
        self.last_index: Optional[int] = None
        self.last_angle: Optional[float] = None
        self.active = False

    def ring_index(self, angle_rad: float) -> int:
        return self.profile.joint_to_ring_index(
            float(angle_rad), self.params["joint_min"], self.params["joint_max"], self.ring_size,
            requested_half_width=self.params["half_width"],
        )

    def effective_half_width(self) -> int:
        return self.profile.effective_half_width(self.params["half_width"])

    def rates_for(self, angle_rad: float) -> np.ndarray:
        return self.rates_for_index(self.ring_index(angle_rad))

    def rates_for_index(self, index: int) -> np.ndarray:
        self.last_index = int(index)
        return self.stimulus.bump_rates(int(index), self.effective_half_width(), self.params["rate_hz"])

    def drive_index(self, index: int, t_ms: float, dt_ms: float, rate_hz: Optional[float] = None) -> bool:
        """Stimulate a ring index directly (tests and fitting tools); ignores the mode."""

        if self.stimulus is None:
            raise self.error("is not connected to a ring")
        rates = self.stimulus.bump_rates(int(index), self.effective_half_width(),
                                         self.params["rate_hz"] if rate_hz is None else float(rate_hz))
        set_stimulus_rates(self._backend, self.stimulus, rates)
        self.last_index = int(index)
        self._emitted = True
        self.active = True
        self._expires_at = float(t_ms) + self.params["duration_ticks"] * float(dt_ms)
        return True

    def drive(self, angle_rad: float, t_ms: float, dt_ms: float, centroid: Optional[float] = None) -> bool:
        """Apply the mode; return True when generator rates changed."""

        if self.stimulus is None:
            raise self.error("is not connected to a ring")
        mode = self.params["mode"]
        self.last_angle = float(angle_rad)
        if mode == "once" and self._emitted:
            return False
        if mode == "corrective":
            index = self.ring_index(angle_rad)
            if centroid is not None and np.isfinite(centroid):
                error = abs(float(centroid) - index)
                error = min(error, self.ring_size - error)
                if error <= self.params["dead_band"]:
                    return False
        rates = self.rates_for(angle_rad)
        set_stimulus_rates(self._backend, self.stimulus, rates)
        self._emitted = True
        self.active = True
        self._expires_at = float(t_ms) + self.params["duration_ticks"] * float(dt_ms)
        return True

    def expire(self, t_ms: float) -> bool:
        """Switch the window off once its duration has elapsed; True when changed."""

        if self.active and self._expires_at is not None and float(t_ms) + 1e-9 >= self._expires_at:
            set_stimulus_rates(self._backend, self.stimulus, np.zeros(self.ring_size))
            self.active = False
            self._expires_at = None
            return True
        return False

    def clear(self) -> bool:
        if self.stimulus is not None and self.active:
            set_stimulus_rates(self._backend, self.stimulus, np.zeros(self.ring_size))
            self.active = False
            self._expires_at = None
            return True
        return False


class FeatureEncoder(Block):
    """Angle → 2K push-pull feature rates on generators, so a comparator can take a reference with no ring.

    Feature k: ``sin_pos = max(sin(kθ), 0)``, ``sin_neg = max(-sin(kθ), 0)`` where
    θ is the preferred angle of the ring index the angle maps to; rates are
    ``rate_scale`` times those values.  The generators are a population whose
    ``features`` port plugs into ``Homeostasis``; the fitted comparator weights
    are those of the ring size in ``population_size``.
    """

    type_name: ClassVar[str] = "FeatureEncoder"
    schema = FEATURE_ENCODER_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("angle", "reference angle in rad"),
        spikes_out("features", "2K Poisson generators, sin_pos_k1, sin_neg_k1, ..."),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.profile = get_legacy_control_profile(self.params["mapping"])
        self.population: Optional[Population] = None
        self.reset()

    @property
    def source_ring_size(self) -> int:
        return int(self.params["population_size"])

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        n = 2 * int(self.params["num_fourier_k"])
        generators = ctx.backend.Create("poisson_generator", n, params={"rate": 0.0})
        self.population = Population(name=self.id, neurons=generators, recorders=None, size=n, model="poisson_generator")
        self._backend = ctx.backend
        self.outputs = {"features": self.population}
        ctx.build_log.append("%s: %d feature generators" % (self.id, n))
        self.built = True

    def reset(self) -> None:
        self.last_angle: Optional[float] = None
        self.rates = None

    def rates_for(self, angle_rad: float) -> np.ndarray:
        size = self.source_ring_size
        index = self.profile.joint_to_ring_index(
            float(angle_rad), self.params["joint_min"], self.params["joint_max"], size, requested_half_width=0
        )
        theta = preferred_angles(size)[int(index) % size]
        harmonics = int(self.params["num_fourier_k"])
        rates = np.zeros(2 * harmonics)
        for k in range(1, harmonics + 1):
            value = np.sin(k * theta)
            rates[2 * (k - 1)] = max(value, 0.0)
            rates[2 * (k - 1) + 1] = max(-value, 0.0)
        return self.params["rate_scale"] * rates

    def drive(self, angle_rad: float, t_ms: float, dt_ms: float, centroid: Optional[float] = None) -> bool:
        rates = self.rates_for(angle_rad)
        if self.rates is not None and np.array_equal(rates, self.rates):
            return False
        self._backend.SetStatus(self.population.neurons, [{"rate": float(r)} for r in rates])
        self.rates = rates
        self.last_angle = float(angle_rad)
        return True

    def expire(self, t_ms: float) -> bool:
        return False

    def clear(self) -> bool:
        if self.rates is not None and np.any(self.rates != 0.0):
            self._backend.SetStatus(self.population.neurons, [{"rate": 0.0} for _ in range(self.population.size)])
            self.rates = np.zeros(self.population.size)
            return True
        return False


__all__ = ["Encoder", "FeatureEncoder"]
