"""The two-ring single-joint model built from the vectorised populations.

Same science as the legacy ``SingleRingModel`` (two rings, two Fourier
readouts, the fitted warm/cold/left/right comparator, opponent gain
populations, shifted feedback) with three engineering changes measured and
documented in ``docs/blocks/equivalence.md``:

* one ``Connect`` per projection (build time well under a second);
* build-time Poisson generators, one per ring neuron, so a bump is a rate
  change and never a hidden ``Simulate``;
* every parameter comes from the same ``config/`` files as before, validated
  before anything is created.

The NEST module is injected as ``backend``; this module never imports it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..artifacts import load_json_legacy, load_numpy_legacy
from ..config import (
    load_gain_spec,
    load_homeostasis_spec,
    load_neuron_parameters,
    load_ring_spec,
    source_config_path,
    source_root,
)
from ..contracts import GainSpec, HomeostasisSpec
from .kernel import configure_kernel
from .populations import (
    STIMULUS_RATE_HZ,
    STIMULUS_WEIGHT,
    BuildError,
    DecisionPopulation,
    GainPopulations,
    ReadoutPopulation,
    RingPopulation,
    StimulusGenerators,
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


@dataclass
class SingleRingArtifacts:
    """Everything read from ``config/`` for one build, resolved and validated."""

    population_size: int
    num_fourier_k: int
    readout_weight_scale: float
    output_dc_baseline: float
    neuron_parameters: Dict[str, Dict[str, Any]]
    homeostasis: HomeostasisSpec
    gain: GainSpec
    fourier_weights: np.ndarray
    warm_weights: np.ndarray
    cold_weights: np.ndarray
    feature_names: List[str]
    paths: Dict[str, str] = field(default_factory=dict)


def load_single_ring_artifacts(
    ring_params_file: Optional[str] = None,
    weights_dir: Optional[str] = None,
    neuron_params_file: Optional[str] = None,
    homeostasis_params_file: Optional[str] = None,
    gain_params_file: Optional[str] = None,
    homeostasis_dir: Optional[str] = None,
) -> SingleRingArtifacts:
    ring_params_file = ring_params_file or source_config_path("model_params", "ring_params.json")
    weights_dir = weights_dir or source_config_path("ring_decoding_weights")
    neuron_params_file = neuron_params_file or os.path.join(
        os.path.dirname(ring_params_file), "neuron_params.json"
    )
    homeostasis_params_file = homeostasis_params_file or source_config_path(
        "model_params", "homeostasis_params.json"
    )
    gain_params_file = gain_params_file or source_config_path("model_params", "gain_modulation_params.json")

    ring = load_ring_spec(ring_params_file)
    homeostasis = load_homeostasis_spec(homeostasis_params_file)
    gain = load_gain_spec(gain_params_file)
    neuron_parameters = {
        name: load_neuron_parameters(neuron_params_file, name)
        for name in ("ring", "homeostasis", "gain")
    }
    size = ring.population_size
    if homeostasis_dir is None:
        # The legacy facade resolves config_dir relative to the flat src folder.
        homeostasis_dir = os.path.join(source_root(), homeostasis.config_dir)

    fourier_path = os.path.join(weights_dir, "N_%d_fourier_weights.npy" % size)
    fourier = np.asarray(load_numpy_legacy(fourier_path), dtype=float)
    expected = (size, 2 * ring.num_fourier_k)
    if fourier.shape != expected:
        raise BuildError(
            "Fourier weights %s: shape %r, expected %r" % (fourier_path, fourier.shape, expected)
        )

    weights_path = os.path.join(homeostasis_dir, "N_%d_homeostasis_weights.npy" % size)
    metadata_path = os.path.join(homeostasis_dir, "N_%d_homeostasis_metadata.json" % size)
    matrix = np.asarray(load_numpy_legacy(weights_path), dtype=float)
    metadata = load_json_legacy(metadata_path)
    feature_names = list(metadata["feature_names"])
    n_features = len(feature_names)
    if matrix.shape != (2 * n_features, 2):
        raise BuildError(
            "homeostasis weights %s: shape %r, expected %r"
            % (weights_path, matrix.shape, (2 * n_features, 2))
        )
    if n_features != 2 * ring.num_fourier_k:
        raise BuildError(
            "homeostasis metadata lists %d features but the readout has %d"
            % (n_features, 2 * ring.num_fourier_k)
        )
    # Legacy order: warm takes rows [0, F) of column 0, cold rows [F, 2F) of column 1.
    return SingleRingArtifacts(
        population_size=size,
        num_fourier_k=ring.num_fourier_k,
        readout_weight_scale=ring.readout_weight_scale,
        output_dc_baseline=ring.output_dc_baseline,
        neuron_parameters=neuron_parameters,
        homeostasis=homeostasis,
        gain=gain,
        fourier_weights=fourier,
        warm_weights=matrix[:n_features, 0].copy(),
        cold_weights=matrix[n_features:, 1].copy(),
        feature_names=feature_names,
        paths={
            "ring_params": ring_params_file, "neuron_params": neuron_params_file,
            "homeostasis_params": homeostasis_params_file, "gain_params": gain_params_file,
            "fourier_weights": fourier_path, "homeostasis_weights": weights_path,
            "homeostasis_metadata": metadata_path,
        },
    )


@dataclass
class SingleRingNetwork:
    population_size: int
    r1: RingPopulation
    r2: RingPopulation
    f1: ReadoutPopulation
    f2: ReadoutPopulation
    decision: DecisionPopulation
    gain: GainPopulations
    stimulus: Dict[str, StimulusGenerators]
    artifacts: SingleRingArtifacts
    feedback_synapses_per_side: int
    build_seconds: float
    seed: Optional[int]
    local_num_threads: int
    backend: Any = field(repr=False, default=None)

    # -- stimulus (rate changes only) --------------------------------------
    def set_bump(self, ring: str, center_index: int, half_width: int, rate_hz: float = STIMULUS_RATE_HZ) -> np.ndarray:
        return set_bump(self.backend, self.stimulus[ring], center_index, half_width, rate_hz)

    def clear_bumps(self, ring: Optional[str] = None) -> None:
        for name, generators in self.stimulus.items():
            if ring is None or name == ring:
                clear_stimulus(self.backend, generators)

    # -- recorder lists in the legacy facade's shape ----------------------
    @property
    def r1_recorders(self) -> List[Any]:
        return self.r1.recorder_list()

    @property
    def r2_recorders(self) -> List[Any]:
        return self.r2.recorder_list()

    @property
    def left_recorders(self) -> List[Any]:
        return self.gain.left.recorder_list()

    @property
    def right_recorders(self) -> List[Any]:
        return self.gain.right.recorder_list()

    @property
    def decision_recorders(self) -> Dict[str, Any]:
        return {label + "_spike": self.decision.recorder_for(label) for label in self.decision.labels}

    def describe(self) -> Dict[str, Any]:
        return {
            "model": "vectorised_single_ring",
            "population_size": self.population_size,
            "num_fourier_k": self.artifacts.num_fourier_k,
            "feedback_synapses_per_side": self.feedback_synapses_per_side,
            "build_seconds": round(self.build_seconds, 4),
            "seed": self.seed,
            "local_num_threads": self.local_num_threads,
            "artifacts": dict(self.artifacts.paths),
        }


def build_single_ring_network(
    backend: Any,
    artifacts: Optional[SingleRingArtifacts] = None,
    seed: Optional[int] = None,
    local_num_threads: int = 1,
    reset_kernel: bool = True,
    variant: str = "legacy",
    feedback: bool = True,
    stimulus_weight: float = STIMULUS_WEIGHT,
    **artifact_paths: Any,
) -> SingleRingNetwork:
    """Build the model; ``artifact_paths`` go to :func:`load_single_ring_artifacts`."""

    if artifacts is None:
        artifacts = load_single_ring_artifacts(**artifact_paths)
    elif artifact_paths:
        raise TypeError("pass either artifacts or artifact paths, not both")
    started = time.perf_counter()
    configure_kernel(
        backend, reset_kernel=reset_kernel, verbosity="M_ERROR",
        local_num_threads=int(local_num_threads), rng_seed=seed,
    )
    size = artifacts.population_size
    neuron = artifacts.neuron_parameters
    r1 = build_ring_population(backend, "r1", size, neuron["ring"], variant=variant)
    r2 = build_ring_population(backend, "r2", size, neuron["ring"], variant=variant)
    f1 = build_readout_population(
        backend, "f1", r1, artifacts.fourier_weights, artifacts.num_fourier_k,
        artifacts.readout_weight_scale, artifacts.output_dc_baseline,
    )
    f2 = build_readout_population(
        backend, "f2", r2, artifacts.fourier_weights, artifacts.num_fourier_k,
        artifacts.readout_weight_scale, artifacts.output_dc_baseline,
    )
    decision = build_decision_population(backend, "cmp", neuron["homeostasis"], artifacts.homeostasis)
    connect_features_to_decision(
        backend, decision, f1, f2, artifacts.warm_weights, artifacts.cold_weights,
        artifacts.homeostasis.warm_cold_weight_scale,
    )
    gain = build_gain_populations(backend, "gain", size, neuron["gain"])
    connect_gain_populations(backend, gain, r1, decision, artifacts.gain)
    per_side = 0
    if feedback:
        per_side = connect_gain_feedback_populations(backend, gain, r1, artifacts.gain.gain_to_ring_weight)
    stimulus = {
        "r1": build_stimulus_generators(backend, "stim_r1", r1, stimulus_weight),
        "r2": build_stimulus_generators(backend, "stim_r2", r2, stimulus_weight),
    }
    return SingleRingNetwork(
        population_size=size, r1=r1, r2=r2, f1=f1, f2=f2, decision=decision, gain=gain,
        stimulus=stimulus, artifacts=artifacts, feedback_synapses_per_side=per_side,
        build_seconds=time.perf_counter() - started, seed=seed,
        local_num_threads=int(local_num_threads), backend=backend,
    )


__all__ = [
    "SingleRingArtifacts",
    "SingleRingNetwork",
    "build_single_ring_network",
    "load_single_ring_artifacts",
]
