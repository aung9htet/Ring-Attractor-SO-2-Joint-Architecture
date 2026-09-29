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
from .populations import (
    BuildError,
    Population,
    build_decision_population,
    build_gain_populations,
    build_readout_population,
    build_ring_population,
    build_stimulus_generators,
    clear_stimulus,
    connect_features_to_decision,
    connect_gain_feedback_populations,
    connect_gain_populations,
    set_bump,
)
from .readout import FourierReadout, build_fourier_readout
from .ring import (
    RingNetwork,
    build_builder_ring,
    build_legacy_ring,
    build_ring_network,
    inject_stimulus,
)
from .single_ring import SingleRingNetwork, build_single_ring_network, load_single_ring_artifacts

__all__ = [
    "BuildError",
    "Population",
    "SingleRingNetwork",
    "build_decision_population",
    "build_gain_populations",
    "build_readout_population",
    "build_ring_population",
    "build_single_ring_network",
    "build_stimulus_generators",
    "clear_stimulus",
    "connect_features_to_decision",
    "connect_gain_feedback_populations",
    "connect_gain_populations",
    "load_single_ring_artifacts",
    "set_bump",
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
