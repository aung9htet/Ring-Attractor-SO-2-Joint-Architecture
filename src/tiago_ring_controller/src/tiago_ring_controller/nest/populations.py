"""Vectorised population builders: one ``Connect`` per projection.

The per-synapse builders in :mod:`.ring`, :mod:`.readout`, :mod:`.comparator`
and :mod:`.gain` stay untouched for the frozen legacy scripts.  The functions
here build the *same connection sets* (pairs and weights, proven by the
fake-NEST equivalence tests) with NEST node collections, ``one_to_one`` and
``all_to_all`` rules and weight matrices computed in NumPy, which turns a
20-30 s build of the N=200 model into a fraction of a second.

Conventions:

* every population is a :class:`Population`: a node collection of neurons and a
  node collection of one ``spike_recorder`` per neuron (so per-neuron
  ``n_events`` deltas, the raster and the bump index stay available);
* weight matrices are handed to callers as ``(pre, post)``; NEST wants
  ``(n_target, n_source)`` for ``all_to_all`` and the transpose is done here,
  in :func:`connect_matrix`, and nowhere else;
* build errors are :class:`BuildError` and name the population.

The NEST module is injected as ``backend``; this module never imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..contracts import GainSpec, HomeostasisSpec
from ..math.ring import ring_weight_matrix, stimulus_target_indices

STIMULUS_RATE_HZ = 200.0
STIMULUS_WEIGHT = 4.5e3
NEURON_MODEL = "iaf_psc_alpha"
FEEDBACK_MARGIN = 5


class BuildError(RuntimeError):
    """A population could not be built; the message names it."""


@dataclass
class Population:
    """Neurons plus one spike recorder per neuron, both as node collections."""

    name: str
    neurons: Any
    recorders: Any
    size: int
    model: str = NEURON_MODEL

    def neuron(self, index: int) -> Any:
        return self.neurons[int(index)]

    def recorder(self, index: int) -> Any:
        return self.recorders[int(index)]

    def neuron_list(self) -> List[Any]:
        return [self.neurons[i] for i in range(self.size)]

    def recorder_list(self) -> List[Any]:
        return [self.recorders[i] for i in range(self.size)]


@dataclass
class RingPopulation(Population):
    variant: str = "legacy"
    weight_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class ReadoutPopulation(Population):
    dc_generator: Any = None
    feature_names: List[str] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def feature_neuron(self, name: str) -> Any:
        return self.neurons[self.feature_names.index(name)]


@dataclass
class DecisionPopulation(Population):
    """warm, cold, left, right in that order."""

    labels = ("warm", "cold", "left", "right")

    def node(self, label: str) -> Any:
        return self.neurons[self.labels.index(label)]

    def recorder_for(self, label: str) -> Any:
        return self.recorders[self.labels.index(label)]


@dataclass
class GainPopulations:
    left: Population
    right: Population


@dataclass
class StimulusGenerators:
    """One ``poisson_generator`` per ring neuron, rate 0 until a bump is set."""

    name: str
    generators: Any
    size: int
    weight: float
    rates: np.ndarray

    def bump_rates(self, center_index: int, half_width: int, rate_hz: float) -> np.ndarray:
        rates = np.zeros(self.size, dtype=float)
        rates[stimulus_target_indices(self.size, int(center_index), int(half_width))] = float(rate_hz)
        return rates


# -- validation -------------------------------------------------------------
def validate_neuron_parameters(
    backend: Any, model: str, parameters: Mapping[str, Any], owner: str
) -> Dict[str, Any]:
    """Check a neuron parameter set against the backend's model defaults.

    Backends without ``GetDefaults`` (the fakes) only get the type check.
    """

    problems = []
    values = dict(parameters)
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append("%s=%r is not a number" % (key, value))
    defaults = None
    getter = getattr(backend, "GetDefaults", None)
    if callable(getter):
        try:
            defaults = getter(model)
        except Exception as exc:  # pragma: no cover - backend specific
            raise BuildError("%s: unknown neuron model %r (%s)" % (owner, model, exc))
    if defaults is not None:
        unknown = sorted(key for key in values if key not in defaults)
        if unknown:
            problems.append("unknown %s parameters %s" % (model, unknown))
    if problems:
        raise BuildError("%s: invalid neuron parameters: %s" % (owner, "; ".join(problems)))
    return values


# -- primitives -------------------------------------------------------------
def create_population(
    backend: Any,
    name: str,
    size: int,
    neuron_parameters: Optional[Mapping[str, Any]] = None,
    model: str = NEURON_MODEL,
) -> Population:
    size = int(size)
    if size < 1:
        raise BuildError("%s: population size must be positive, got %d" % (name, size))
    if neuron_parameters:
        params = validate_neuron_parameters(backend, model, neuron_parameters, name)
        neurons = backend.Create(model, size, params=params)
    else:
        neurons = backend.Create(model, size)
    recorders = backend.Create("spike_recorder", size)
    backend.Connect(neurons, recorders, "one_to_one")
    return Population(name=name, neurons=neurons, recorders=recorders, size=size, model=model)


def connect_matrix(backend: Any, pre: Any, post: Any, weights_pre_by_post: np.ndarray) -> None:
    """``all_to_all`` with a ``(pre, post)`` weight matrix (transposed for NEST)."""

    matrix = np.asarray(weights_pre_by_post, dtype=float)
    if matrix.ndim != 2:
        raise BuildError("weight matrix must be 2-D, got shape %r" % (matrix.shape,))
    backend.Connect(
        pre,
        post,
        {"rule": "all_to_all"},
        {"weight": np.ascontiguousarray(matrix.T)},
    )


def wall_profile(population_size: int, low: float, high: float, outside_scale: float, taper: int = 0) -> np.ndarray:
    """Per-neuron bias-current factor: 1 on ``[low, high]``, ``outside_scale`` around the seam.

    With ``taper`` the factor changes linearly over that many indices outside
    each end.  Used to make the out-of-range sector of a bounded ring unable to
    hold a bump.
    """

    indices = np.arange(int(population_size), dtype=float)
    distance = np.maximum(low - indices, 0.0) + np.maximum(indices - high, 0.0)
    if taper > 0:
        fraction = np.clip(distance / float(taper), 0.0, 1.0)
    else:
        fraction = (distance > 0).astype(float)
    return 1.0 - fraction * (1.0 - float(outside_scale))


def build_ring_population(
    backend: Any,
    name: str,
    population_size: int,
    neuron_parameters: Mapping[str, Any],
    variant: str = "legacy",
    max_distance: float = 50,
    excitation_std_dev: float = 10,
    inhibition_std_dev: float = 5,
    current_profile: Optional[Sequence[float]] = None,
) -> RingPopulation:
    """``current_profile`` multiplies each neuron's bias current ``I_e`` (bounded ring wall)."""

    try:
        matrix = ring_weight_matrix(
            population_size, variant, max_distance, excitation_std_dev, inhibition_std_dev
        )
    except ValueError as exc:
        raise BuildError("%s: %s" % (name, exc))
    base = create_population(backend, name, population_size, neuron_parameters)
    if current_profile is not None:
        factors = np.asarray(current_profile, dtype=float)
        if factors.shape != (int(population_size),):
            raise BuildError("%s: current profile has shape %r, expected (%d,)" % (name, factors.shape, int(population_size)))
        bias = float(dict(neuron_parameters).get("I_e", 0.0))
        backend.SetStatus(base.neurons, [{"I_e": bias * float(f)} for f in factors])
    connect_matrix(backend, base.neurons, base.neurons, matrix)
    return RingPopulation(
        name=base.name, neurons=base.neurons, recorders=base.recorders, size=base.size,
        model=base.model, variant=variant, weight_matrix=matrix,
    )


def readout_feature_names(num_fourier_k: int) -> List[str]:
    names = []
    for k in range(int(num_fourier_k)):
        names.append("sin_pos_k%d" % (k + 1))
        names.append("sin_neg_k%d" % (k + 1))
    return names


def build_readout_population(
    backend: Any,
    name: str,
    ring: Population,
    weights: np.ndarray,
    num_fourier_k: int,
    output_weight_scale: float,
    output_dc_baseline: float,
) -> ReadoutPopulation:
    matrix = np.asarray(weights, dtype=float)
    expected = (ring.size, 2 * int(num_fourier_k))
    if matrix.shape != expected:
        raise BuildError(
            "%s: Fourier weights shape mismatch: got %r, expected %r" % (name, matrix.shape, expected)
        )
    # The legacy readout creates its neurons with NEST's model defaults.
    base = create_population(backend, name, 2 * int(num_fourier_k))
    dc = backend.Create("dc_generator", params={"amplitude": float(output_dc_baseline)})
    backend.Connect(dc, base.neurons)
    connect_matrix(backend, ring.neurons, base.neurons, float(output_weight_scale) * matrix)
    return ReadoutPopulation(
        name=base.name, neurons=base.neurons, recorders=base.recorders, size=base.size,
        model=base.model, dc_generator=dc, feature_names=readout_feature_names(num_fourier_k),
        weights=matrix,
    )


def build_decision_population(
    backend: Any,
    name: str,
    neuron_parameters: Mapping[str, Any],
    spec: HomeostasisSpec,
) -> DecisionPopulation:
    base = create_population(backend, name, 4, neuron_parameters)
    pop = DecisionPopulation(
        name=base.name, neurons=base.neurons, recorders=base.recorders, size=4, model=base.model
    )
    # Six fixed lateral synapses, same pairs and weights as build_decision_circuit.
    for source, target, weight in (
        ("cold", "right", spec.warm_exc_weight),
        ("cold", "left", spec.warm_inh_weight),
        ("warm", "left", spec.cold_exc_weight),
        ("warm", "right", spec.cold_inh_weight),
        ("left", "right", spec.decision_lateral_weight),
        ("right", "left", spec.decision_lateral_weight),
    ):
        backend.Connect(pop.node(source), pop.node(target), syn_spec={"weight": float(weight)})
    return pop


def connect_features_to_decision(
    backend: Any,
    decision: DecisionPopulation,
    state_features: Population,
    target_features: Population,
    warm_weights: Sequence[float],
    cold_weights: Sequence[float],
    weight_scale: float,
) -> None:
    """State features → warm, target features → cold, one matrix each."""

    warm = np.asarray(warm_weights, dtype=float)
    cold = np.asarray(cold_weights, dtype=float)
    if warm.shape != (state_features.size,) or cold.shape != (target_features.size,):
        raise BuildError(
            "%s: feature weight vectors %r / %r do not match the readouts (%d / %d)"
            % (decision.name, warm.shape, cold.shape, state_features.size, target_features.size)
        )
    connect_matrix(backend, state_features.neurons, decision.node("warm"), (float(weight_scale) * warm)[:, None])
    connect_matrix(backend, target_features.neurons, decision.node("cold"), (float(weight_scale) * cold)[:, None])


def build_gain_populations(
    backend: Any,
    name: str,
    population_size: int,
    neuron_parameters: Mapping[str, Any],
) -> GainPopulations:
    left = create_population(backend, name + ".left", population_size, neuron_parameters)
    right = create_population(backend, name + ".right", population_size, neuron_parameters)
    return GainPopulations(left=left, right=right)


def connect_gain_populations(
    backend: Any,
    gain: GainPopulations,
    ring: Population,
    decision: DecisionPopulation,
    spec: GainSpec,
) -> None:
    connect_gain_inputs(backend, gain, ring, decision.node("left"), decision.node("right"), spec)


def connect_gain_inputs(
    backend: Any,
    gain: GainPopulations,
    ring: Population,
    left_decision: Any,
    right_decision: Any,
    spec: GainSpec,
    left_drive: Optional[Sequence[float]] = None,
    right_drive: Optional[Sequence[float]] = None,
) -> None:
    """Decision nodes gate the populations, the ring drives them one_to_one.

    ``left_drive`` / ``right_drive`` scale the ring's drive per neuron (the
    bounded ring fades the population that moves the bump toward a joint limit);
    ``None`` keeps the single weight of the legacy build.
    """

    if ring.size != gain.left.size or ring.size != gain.right.size:
        raise BuildError(
            "%s: ring size %d does not match gain populations (%d / %d)"
            % (gain.left.name, ring.size, gain.left.size, gain.right.size)
        )
    backend.Connect(left_decision, gain.left.neurons, syn_spec={"weight": float(spec.left_homeostasis_gain_weight)})
    backend.Connect(right_decision, gain.right.neurons, syn_spec={"weight": float(spec.right_homeostasis_gain_weight)})
    backend.Connect(left_decision, gain.right.neurons, syn_spec={"weight": float(spec.cross_inhibition_weight)})
    backend.Connect(right_decision, gain.left.neurons, syn_spec={"weight": float(spec.cross_inhibition_weight)})
    for population, drive in ((gain.left, left_drive), (gain.right, right_drive)):
        if drive is None:
            backend.Connect(ring.neurons, population.neurons, "one_to_one", {"weight": float(spec.ring_to_gain_weight)})
            continue
        factors = np.asarray(drive, dtype=float)
        if factors.shape != (ring.size,):
            raise BuildError("%s: drive profile has shape %r, expected (%d,)" % (population.name, factors.shape, ring.size))
        backend.Connect(ring.neurons, population.neurons, "one_to_one",
                        {"weight": float(spec.ring_to_gain_weight) * factors})


def build_output_ring_population(
    backend: Any,
    name: str,
    source_nodes: Any,
    weights_source_by_target: np.ndarray,
    size: int,
    dc_baseline: float,
    weight_scale: float,
) -> ReadoutPopulation:
    """One output ring: DC baseline plus an all_to_all fitted matrix from ``source_nodes``.

    Same synapse set as ``nest.multi_ring.build_output_rings`` for one name.
    """

    matrix = np.asarray(weights_source_by_target, dtype=float)
    n_source = len(source_nodes)
    if matrix.shape != (n_source, int(size)):
        raise BuildError(
            "%s: output weights shape %r, expected (%d, %d)" % (name, matrix.shape, n_source, int(size))
        )
    base = create_population(backend, name, int(size))
    dc = backend.Create("dc_generator", params={"amplitude": float(dc_baseline)})
    backend.Connect(dc, base.neurons)
    connect_matrix(backend, source_nodes, base.neurons, float(weight_scale) * matrix)
    return ReadoutPopulation(
        name=base.name, neurons=base.neurons, recorders=base.recorders, size=base.size,
        model=base.model, dc_generator=dc, feature_names=[], weights=matrix,
    )


def connect_gain_feedback_populations(
    backend: Any,
    gain: GainPopulations,
    ring: Population,
    weight: float,
    margin: int = FEEDBACK_MARGIN,
) -> int:
    """Shifted ±1 feedback excluding ``margin`` neurons at each end.

    Returns the number of synapses per side (0 when the ring is too small,
    exactly like the legacy loop ``range(5, N - 5)``).
    """

    size = ring.size
    margin = int(margin)
    count = size - 2 * margin
    if count <= 0:
        return 0
    backend.Connect(
        gain.left.neurons[margin:size - margin],
        ring.neurons[margin + 1:size - margin + 1],
        "one_to_one",
        {"weight": float(weight)},
    )
    backend.Connect(
        gain.right.neurons[margin:size - margin],
        ring.neurons[margin - 1:size - margin - 1],
        "one_to_one",
        {"weight": float(weight)},
    )
    return count


def connect_gain_reference(
    backend: Any,
    gain: GainPopulations,
    reference: Population,
    exc_weight: float,
    inh_weight: float,
    window: int,
) -> int:
    """A reference ring steers the gain populations directly (the sharp comparison of v4).

    Gain neuron ``i`` of the right population (which moves the bump up) is
    excited by reference neurons ``i+1 .. i+window`` and inhibited by
    ``i-window .. i-1``; the left population is the mirror image.  With the
    bump at ``i`` gating both populations, the side whose direction leads to
    the reference bump wins, down to an offset of one index.  No wrap around
    the seam.  One ``one_to_one`` call per shift; returns the synapses per
    population.
    """

    size = reference.size
    if size != gain.left.size or size != gain.right.size:
        raise BuildError(
            "%s: reference ring %s has %d neurons, gain populations %d / %d"
            % (gain.left.name, reference.name, size, gain.left.size, gain.right.size)
        )
    window = int(window)
    if not 1 <= window < size:
        raise BuildError("%s: reference window must be in 1..%d, got %d" % (gain.left.name, size - 1, window))
    exc, inh = float(exc_weight), -abs(float(inh_weight))
    count = 0
    for shift in range(1, window + 1):
        above, below = reference.neurons[shift:size], reference.neurons[0:size - shift]
        low, high = slice(0, size - shift), slice(shift, size)
        for source, population, target, weight in (
            (above, gain.right, low, exc), (below, gain.right, high, inh),
            (below, gain.left, high, exc), (above, gain.left, low, inh),
        ):
            if weight != 0.0:
                backend.Connect(source, population.neurons[target], "one_to_one", {"weight": weight})
        count += 2 * (size - shift)
    return count


# -- build-time stimulus generators (plan B.4) ------------------------------
def build_stimulus_generators(
    backend: Any, name: str, ring: Population, weight: float = STIMULUS_WEIGHT
) -> StimulusGenerators:
    generators = backend.Create("poisson_generator", ring.size, params={"rate": 0.0})
    backend.Connect(generators, ring.neurons, "one_to_one", {"weight": float(weight)})
    return StimulusGenerators(
        name=name, generators=generators, size=ring.size, weight=float(weight),
        rates=np.zeros(ring.size, dtype=float),
    )


def set_stimulus_rates(backend: Any, stimulus: StimulusGenerators, rates: Sequence[float]) -> None:
    """Set every generator's rate (a rate change is legal between ``Run`` calls)."""

    values = np.asarray(rates, dtype=float)
    if values.shape != (stimulus.size,):
        raise BuildError("%s: rates shape %r != (%d,)" % (stimulus.name, values.shape, stimulus.size))
    if np.any(values < 0.0):
        raise BuildError("%s: negative stimulus rate" % stimulus.name)
    backend.SetStatus(stimulus.generators, [{"rate": float(value)} for value in values])
    stimulus.rates = values


def set_bump(
    backend: Any,
    stimulus: StimulusGenerators,
    center_index: int,
    half_width: int,
    rate_hz: float = STIMULUS_RATE_HZ,
) -> np.ndarray:
    rates = stimulus.bump_rates(center_index, half_width, rate_hz)
    set_stimulus_rates(backend, stimulus, rates)
    return rates


def clear_stimulus(backend: Any, stimulus: StimulusGenerators) -> None:
    if np.any(stimulus.rates != 0.0):
        set_stimulus_rates(backend, stimulus, np.zeros(stimulus.size, dtype=float))


__all__ = [
    "BuildError",
    "DecisionPopulation",
    "FEEDBACK_MARGIN",
    "GainPopulations",
    "Population",
    "ReadoutPopulation",
    "RingPopulation",
    "STIMULUS_RATE_HZ",
    "STIMULUS_WEIGHT",
    "StimulusGenerators",
    "build_decision_population",
    "build_gain_populations",
    "build_output_ring_population",
    "build_readout_population",
    "build_ring_population",
    "build_stimulus_generators",
    "clear_stimulus",
    "connect_features_to_decision",
    "connect_gain_feedback_populations",
    "connect_gain_inputs",
    "connect_gain_populations",
    "connect_gain_reference",
    "connect_matrix",
    "create_population",
    "readout_feature_names",
    "set_bump",
    "set_stimulus_rates",
    "validate_neuron_parameters",
    "wall_profile",
]
