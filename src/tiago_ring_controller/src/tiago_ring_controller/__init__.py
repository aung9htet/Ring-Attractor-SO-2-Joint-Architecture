"""Internal, side-effect-free helpers for :mod:`tiago_ring_controller`.

The historical modules in ``src/*.py`` remain the public compatibility
surface.  This package contains implementation building blocks that can be
adopted by those modules incrementally.  Importing it does not import ROS or
NEST and does not read configuration files.
"""

from .contracts import (
    ArtifactArrayDescriptor,
    ArtifactDescriptor,
    DecoderSpec,
    GainSpec,
    HomeostasisSpec,
    JointCalibration,
    MultiRingSpec,
    NeuronSpec,
    PIDSpec,
    RingSpec,
)

__all__ = [
    "ArtifactArrayDescriptor",
    "ArtifactDescriptor",
    "DecoderSpec",
    "GainSpec",
    "HomeostasisSpec",
    "JointCalibration",
    "MultiRingSpec",
    "NeuronSpec",
    "PIDSpec",
    "RingSpec",
]

__version__ = "0.0.1"
