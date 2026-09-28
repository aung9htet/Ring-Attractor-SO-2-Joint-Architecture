"""Legacy single-joint control profiles.

The calibration, analysis, and data-collection scripts look similar but do
not currently preprocess ring activity in the same way.  This module names
those differences without attempting to reconcile them.  In particular,
calibration uses a population-size spike scale and a rounded 10 percent edge
margin, analysis doubles the requested bump half-width, and collection uses
the requested half-width unchanged.

The helpers are ROS- and NEST-independent.  They intentionally mirror the
existing arithmetic, including truncating ``int`` conversion in the full
range mapper and Python's rounded calibration mapper.
"""

from dataclasses import dataclass
from typing import Dict, Optional


FULL_RANGE_TRUNCATED = "full_range_truncated"
CALIBRATION_MARGIN_ROUNDED = "calibration_margin_rounded"


@dataclass(frozen=True)
class LegacyControlProfile:
    """Immutable description of an existing controller workflow."""

    name: str
    mapping_mode: str
    stimulus_half_width_multiplier: int
    spike_scale_reference_population: Optional[float]
    ring_edge_margin_fraction: Optional[float] = None
    time_step_ms: float = 50.0
    decoder_gain_positive: float = 1e-4
    decoder_gain_negative: float = -1e-4
    decoder_tau_s: float = 0.3
    decoder_delay_steps: int = 0
    lookahead: int = 4
    drive_threshold: float = 5.0
    n_settle: int = 10
    max_steps: int = 400

    def effective_half_width(self, requested_half_width: int = 5) -> int:
        """Return the half-width actually passed to the ring injection."""

        return int(requested_half_width) * self.stimulus_half_width_multiplier

    def spike_scale(self, population_size: int) -> float:
        """Return the workflow's multiplier for ``right - left`` counts."""

        if self.spike_scale_reference_population is None:
            return 1.0
        return self.spike_scale_reference_population / float(population_size)

    def signed_drive(
        self,
        right_count: float,
        left_count: float,
        population_size: int,
    ) -> float:
        """Return the current signed gain signal for this workflow."""

        return (right_count - left_count) * self.spike_scale(population_size)

    def joint_to_ring_index(
        self,
        joint_position: float,
        joint_min: float,
        joint_max: float,
        population_size: int,
        requested_half_width: int = 5,
    ) -> int:
        """Map a joint position exactly as the named legacy workflow does.

        ``requested_half_width`` is the value supplied by the caller.  The
        analysis profile applies its historical factor of two before mapping.
        """

        half_width = self.effective_half_width(requested_half_width)
        if self.mapping_mode == CALIBRATION_MARGIN_ROUNDED:
            return _calibration_joint_to_ring_index(
                joint_position=joint_position,
                joint_min=joint_min,
                joint_max=joint_max,
                population_size=population_size,
                stimulus_half_width=half_width,
                edge_margin_fraction=float(self.ring_edge_margin_fraction),
            )
        if self.mapping_mode == FULL_RANGE_TRUNCATED:
            return _full_range_joint_to_ring_index(
                joint_position=joint_position,
                joint_min=joint_min,
                joint_max=joint_max,
                population_size=population_size,
                stimulus_half_width=half_width,
            )
        raise ValueError("Unknown legacy mapping mode: %s" % self.mapping_mode)


def _full_range_joint_to_ring_index(
    joint_position: float,
    joint_min: float,
    joint_max: float,
    population_size: int,
    stimulus_half_width: int,
) -> int:
    """Mirror the collector/analysis truncated full-range mapping."""

    inject = (
        int(
            (joint_position - joint_min)
            / (joint_max - joint_min)
            * (population_size - stimulus_half_width * 2)
        )
        + stimulus_half_width
    )
    return max(
        stimulus_half_width,
        min(inject, population_size - 1 - stimulus_half_width),
    )


def _calibration_joint_to_ring_index(
    joint_position: float,
    joint_min: float,
    joint_max: float,
    population_size: int,
    stimulus_half_width: int,
    edge_margin_fraction: float,
) -> int:
    """Mirror calibration's clipped, rounded, edge-margin mapping."""

    n = int(population_size)
    denom = joint_max - joint_min
    if denom <= 0:
        return max(
            stimulus_half_width,
            min(n // 2, n - 1 - stimulus_half_width),
        )

    edge_margin = int(round(edge_margin_fraction * n))
    lower = max(stimulus_half_width, edge_margin)
    upper = min(n - 1 - stimulus_half_width, n - edge_margin)

    if upper < lower:
        lower = stimulus_half_width
        upper = n - 1 - stimulus_half_width

    phase = (joint_position - joint_min) / denom
    phase = float(min(max(phase, 0.0), 1.0))
    inject = int(round(lower + phase * (upper - lower)))
    return max(lower, min(inject, upper))


CALIBRATION_PROFILE = LegacyControlProfile(
    name="calibration",
    mapping_mode=CALIBRATION_MARGIN_ROUNDED,
    stimulus_half_width_multiplier=1,
    spike_scale_reference_population=100.0,
    ring_edge_margin_fraction=0.10,
)

ANALYSIS_PROFILE = LegacyControlProfile(
    name="analysis",
    mapping_mode=FULL_RANGE_TRUNCATED,
    stimulus_half_width_multiplier=2,
    spike_scale_reference_population=None,
)

COLLECTOR_PROFILE = LegacyControlProfile(
    name="collector",
    mapping_mode=FULL_RANGE_TRUNCATED,
    stimulus_half_width_multiplier=1,
    spike_scale_reference_population=None,
)

# Compatibility spelling for callers that use the workflow noun rather than
# the historical script/class adjective.  Both names refer to one immutable
# profile whose canonical name remains ``collector``.
COLLECTION_PROFILE = COLLECTOR_PROFILE

LEGACY_CONTROL_PROFILES: Dict[str, LegacyControlProfile] = {
    profile.name: profile
    for profile in (CALIBRATION_PROFILE, ANALYSIS_PROFILE, COLLECTOR_PROFILE)
}


def get_legacy_control_profile(name: str) -> LegacyControlProfile:
    """Return a named profile without creating a mutable copy."""

    return LEGACY_CONTROL_PROFILES[name]


__all__ = [
    "ANALYSIS_PROFILE",
    "CALIBRATION_MARGIN_ROUNDED",
    "CALIBRATION_PROFILE",
    "COLLECTOR_PROFILE",
    "COLLECTION_PROFILE",
    "FULL_RANGE_TRUNCATED",
    "LEGACY_CONTROL_PROFILES",
    "LegacyControlProfile",
    "get_legacy_control_profile",
]
