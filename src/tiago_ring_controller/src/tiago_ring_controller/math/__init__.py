"""Pure scientific mathematics used by the legacy NEST/ROS workflows."""

from .circular import (
    angle_to_neuron_index,
    angle_to_ring_index,
    circular_signed_difference,
    decode_population_angle,
    index_to_phase,
    preferred_angles,
    ring_index_to_angle,
)
from .control import apply_delay, decode_velocity, decoder_features, exponential_filter
from .fourier import build_push_pull_sine_weights, fourier_feature_names
from .ring import (
    builder_ring_distances,
    builder_ring_weight,
    legacy_ring_distances,
    legacy_ring_weight,
)

__all__ = [
    "angle_to_neuron_index",
    "angle_to_ring_index",
    "apply_delay",
    "builder_ring_distances",
    "builder_ring_weight",
    "build_push_pull_sine_weights",
    "circular_signed_difference",
    "decode_population_angle",
    "decode_velocity",
    "decoder_features",
    "exponential_filter",
    "fourier_feature_names",
    "index_to_phase",
    "legacy_ring_distances",
    "legacy_ring_weight",
    "preferred_angles",
    "ring_index_to_angle",
]
