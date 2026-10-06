"""Joint angle <-> ring index on an arc of the ring (the bounded ring).

The legacy profiles map a joint's whole range onto almost the whole ring
(indices ``half_width`` to ``N - 1 - half_width``), so the joint limits sit next
to each other across the ring's seam.  :class:`ArcMapping` maps the range onto
an arc ``[first, first + span]`` instead; with :meth:`ArcMapping.bounded` the arc
is centred on index ``N / 2`` and the out-of-range sector straddles the seam,
as in the bounded ring attractor of Ferrari et al. (arXiv 2604.14021), where the
centre of the out-of-range sector lies opposite the middle of the range.

The mapping rounds to the nearest index and its inverse returns the angle of an
index (fractional indices allowed), so ``index(angle(i)) == i`` on the arc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ArcMapping:
    joint_min: float
    joint_max: float
    ring_size: int
    #: ring index of ``joint_min``
    first: float
    #: ring indices covered by the joint range (``joint_max`` maps to ``first + span``)
    span: float

    def __post_init__(self) -> None:
        if self.joint_max <= self.joint_min:
            raise ValueError("joint_max must be above joint_min")
        if self.span <= 0 or self.first < 0 or self.first + self.span > self.ring_size - 1:
            raise ValueError("arc [%g, %g] does not fit a ring of %d" % (self.first, self.first + self.span, self.ring_size))

    @classmethod
    def bounded(cls, joint_min: float, joint_max: float, ring_size: int, arc_fraction: float) -> "ArcMapping":
        """The joint range on ``arc_fraction`` of the ring, centred on index ``ring_size / 2``."""

        if not 0.0 < arc_fraction < 1.0:
            raise ValueError("arc_fraction must be in (0, 1)")
        span = float(round(arc_fraction * ring_size))
        return cls(float(joint_min), float(joint_max), int(ring_size), (ring_size - span) / 2.0, span)

    @property
    def arc(self) -> Tuple[float, float]:
        return self.first, self.first + self.span

    @property
    def radians_per_index(self) -> float:
        return (self.joint_max - self.joint_min) / self.span

    def index(self, angle: float) -> int:
        """Nearest ring index of a joint angle; angles outside the range land on the arc ends."""

        phase = min(max((float(angle) - self.joint_min) / (self.joint_max - self.joint_min), 0.0), 1.0)
        return int(round(self.first + phase * self.span))

    def angle(self, index: float) -> float:
        """Joint angle of a (fractional) ring index, not clamped to the range."""

        return self.joint_min + (float(index) - self.first) * self.radians_per_index

    def contains(self, index: float, margin: float = 0.0) -> bool:
        return self.first - margin <= float(index) <= self.first + self.span + margin

    def taper(self, length: float, end: str) -> np.ndarray:
        """Per-neuron factor that is 1 inside the arc and falls linearly to 0 at one arc end.

        ``end="upper"``: 1 up to ``last - length``, 0 from the arc's last index
        on.  ``end="lower"`` mirrors it at the first index.  On the other side of
        the arc the factor stays 1, so motion back into the arc is never faded.
        Used to fade the gain population that moves the bump toward that end.
        """

        indices = np.arange(self.ring_size, dtype=float)
        low, high = self.arc
        if end == "upper":
            return np.clip((high - indices) / max(length, 1e-9), 0.0, 1.0)
        if end == "lower":
            return np.clip((indices - low) / max(length, 1e-9), 0.0, 1.0)
        raise ValueError("end must be 'upper' or 'lower'")


MAPPINGS = ("collector", "analysis", "calibration", "circular", "bounded")


class RingMapper:
    """One interface over the three kinds of angle <-> index mapping the blocks use.

    ``mapping`` is a legacy control profile name, ``"circular"`` (the angle is the
    ring angle) or ``"bounded"`` (:class:`ArcMapping.bounded`).  ``index`` gives
    the integer index an encoder stimulates; ``angle`` is its inverse, used to
    read a joint angle back from the ring.
    """

    def __init__(self, mapping: str, joint_min: float, joint_max: float, ring_size: int,
                 half_width: int = 5, arc_fraction: float = 0.70) -> None:
        if mapping not in MAPPINGS:
            raise ValueError("mapping must be one of %r, got %r" % (MAPPINGS, mapping))
        if int(ring_size) <= 0:
            raise ValueError("ring size is unknown")
        self.mapping = mapping
        self.joint_min, self.joint_max = float(joint_min), float(joint_max)
        self.ring_size = int(ring_size)
        self.half_width = int(half_width)
        self.arc_mapping = ArcMapping.bounded(joint_min, joint_max, ring_size, arc_fraction) if mapping == "bounded" else None
        if mapping in ("circular", "bounded"):
            self.profile = None
        else:
            from .profiles import get_legacy_control_profile

            self.profile = get_legacy_control_profile(mapping)

    def index(self, angle: float) -> int:
        if self.arc_mapping is not None:
            return self.arc_mapping.index(angle)
        if self.profile is None:
            from ..math.circular import angle_to_neuron_index

            return angle_to_neuron_index(float(angle), self.ring_size)
        return self.profile.joint_to_ring_index(float(angle), self.joint_min, self.joint_max, self.ring_size,
                                                requested_half_width=self.half_width)

    def angle(self, index: float) -> float:
        if self.arc_mapping is not None:
            return self.arc_mapping.angle(index)
        if self.profile is None:
            return 2.0 * np.pi * float(index) / self.ring_size
        return self.profile.ring_index_to_joint(index, self.joint_min, self.joint_max, self.ring_size,
                                                requested_half_width=self.half_width)

    @property
    def radians_per_index(self) -> float:
        if self.arc_mapping is not None:
            return self.arc_mapping.radians_per_index
        if self.profile is None:
            return 2.0 * np.pi / self.ring_size
        return self.profile.radians_per_index(self.joint_min, self.joint_max, self.ring_size,
                                              requested_half_width=self.half_width)

    @property
    def effective_half_width(self) -> int:
        if self.profile is None:
            return self.half_width
        return self.profile.effective_half_width(self.half_width)

    @property
    def arc(self) -> Tuple[float, float]:
        """Ring indices of ``joint_min`` and ``joint_max`` (the whole ring for ``circular``)."""

        if self.arc_mapping is not None:
            return self.arc_mapping.arc
        if self.profile is None:
            return 0.0, float(self.ring_size)
        first, span = self.profile.index_range(self.ring_size, self.half_width)
        return first, first + span


__all__ = ["ArcMapping", "MAPPINGS", "RingMapper"]
