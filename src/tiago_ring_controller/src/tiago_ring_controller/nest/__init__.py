"""Backend-injected NEST topology builders with no import-time NEST effects."""

from .comparator import DecisionCircuit, build_decision_circuit, connect_ring_features
from .gain import (
    GainNetwork,
    build_gain_network,
    connect_gain_feedback,
    connect_gain_network,
    create_gain_network,
)
from .kernel import configure_kernel, recorder_events, spike_counts
from .readout import FourierReadout, build_fourier_readout
from .ring import (
    RingNetwork,
    build_builder_ring,
    build_legacy_ring,
    build_ring_network,
    inject_stimulus,
)

__all__ = [
    "DecisionCircuit",
    "FourierReadout",
    "GainNetwork",
    "RingNetwork",
    "build_builder_ring",
    "build_decision_circuit",
    "build_fourier_readout",
    "build_gain_network",
    "build_legacy_ring",
    "build_ring_network",
    "configure_kernel",
    "connect_gain_feedback",
    "connect_gain_network",
    "connect_ring_features",
    "create_gain_network",
    "inject_stimulus",
    "recorder_events",
    "spike_counts",
]
