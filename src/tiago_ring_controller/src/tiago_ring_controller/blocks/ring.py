"""Ring block: N neurons, N recorders, Mexican-hat recurrence, late-bound stim."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Sequence, Tuple

from ..nest.populations import BuildError, build_ring_population, wall_profile
from .base import Block, BuildContext, Connection, Port, signal_out, spikes_in, spikes_out
from .params import RING_SCHEMA


class Ring(Block):
    """Ring attractor (fixed N² recurrence); ``stim`` accepts encoders and gain feedback."""

    type_name: ClassVar[str] = "Ring"
    schema = RING_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("stim", "stimulus into the ring: Encoder generators or Gain feedback", required=False, multi=True, late=True),
        spikes_out("spikes", "the N ring neurons"),
        signal_out("counts", "per-neuron spike-count deltas per tick, with total, bump index and centroid"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.population = None

    @property
    def population_size(self) -> int:
        return int(self.params["population_size"])

    def build(self, ctx: BuildContext, inputs: Mapping[str, Sequence[Connection]]) -> None:
        p = self.params
        try:
            self.population = build_ring_population(
                ctx.backend, self.id, p["population_size"],
                ctx.neuron_parameters("ring", p["neuron_params"]),
                variant=p["variant"], max_distance=p["max_distance"],
                excitation_std_dev=p["excitation_std_dev"], inhibition_std_dev=p["inhibition_std_dev"],
                current_profile=self.current_profile(),
            )
        except BuildError as exc:
            raise self.error(str(exc))
        self.outputs = {"spikes": self.population}
        ctx.build_log.append("%s: ring N=%d (%s)" % (self.id, self.population.size, p["variant"]))
        self.built = True

    def current_profile(self):
        """Bias-current factor per neuron for a bounded ring's wall, or ``None``."""

        p = self.params
        if not p["wall"] or p["outside_current_scale"] == 1.0:
            return None
        if len(p["wall"]) != 2 or not 0 <= p["wall"][0] < p["wall"][1] < p["population_size"]:
            raise self.error("wall must be [low, high] ring indices with low < high, got %r" % (p["wall"],))
        return wall_profile(p["population_size"], p["wall"][0], p["wall"][1], p["outside_current_scale"], p["wall_taper"])

    def connect_late(self, ctx: BuildContext, connection: Connection) -> None:
        if connection.target_port.name != "stim":
            raise self.error("port %r is not late-bound" % connection.target_port.name)
        provider = getattr(connection.source, "connect_stim", None)
        if provider is None:
            raise self.error(
                "%s %r cannot drive a ring's stim port" % (connection.source.type_name, connection.source.id)
            )
        provider(ctx, self, connection)

    def counts_sources(self) -> Dict[str, Any]:
        return {"counts": self.population.recorder_list()}


__all__ = ["Ring"]
