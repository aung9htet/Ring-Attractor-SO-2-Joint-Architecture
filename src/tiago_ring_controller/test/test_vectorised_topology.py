"""Fake-NEST equivalence: vectorised builders create the legacy connection sets.

Every vectorised builder (``nest.populations``) is compared with the per-synapse
builder the legacy facades call, as a *set* of ``(pre, post, weight)`` rows after
relabelling node ids to ``(population, index)``.  Creation order differs by
design; the synapses do not.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support_fake_nest import (  # noqa: E402
    RecordingNest,
    connection_rows,
    label_nodes,
    relabel_rows,
)
from tiago_ring_controller.config import (  # noqa: E402
    load_gain_spec,
    load_homeostasis_spec,
    load_neuron_parameters,
)
from tiago_ring_controller.contracts import GainSpec, HomeostasisSpec  # noqa: E402
from tiago_ring_controller.math.ring import (  # noqa: E402
    builder_ring_distances,
    builder_ring_weight,
    legacy_ring_distances,
    legacy_ring_weight,
    ring_weight_matrix,
)
from tiago_ring_controller.nest.comparator import (  # noqa: E402
    build_decision_circuit,
    connect_ring_feature_vectors,
)
from tiago_ring_controller.nest.gain import build_gain_network, connect_gain_feedback  # noqa: E402
from tiago_ring_controller.nest.populations import (  # noqa: E402
    BuildError,
    build_decision_population,
    build_gain_populations,
    build_readout_population,
    build_ring_population,
    build_stimulus_generators,
    clear_stimulus,
    connect_features_to_decision,
    connect_gain_feedback_populations,
    connect_gain_populations,
    create_population,
    set_bump,
    validate_neuron_parameters,
)
from tiago_ring_controller.nest.readout import build_fourier_readout  # noqa: E402
from tiago_ring_controller.nest.ring import build_ring_network  # noqa: E402
from tiago_ring_controller.nest.single_ring import (  # noqa: E402
    build_single_ring_network,
    load_single_ring_artifacts,
)


MODEL_PARAMS = SRC / "config/model_params"
RING_PARAMS = {"V_th": 0.8, "I_e": 450000.0}
HOMEOSTASIS_SPEC = HomeostasisSpec(
    config_dir="", tie_epsilon=1e-9, warm_cold_weight_scale=3.0, warm_cold_bias_scale=0.0,
    warm_exc_weight=11.0, warm_inh_weight=-12.0, cold_exc_weight=13.0, cold_inh_weight=-14.0,
    decision_lateral_weight=-15.0, left_label=0, right_label=1,
)
GAIN_SPEC = GainSpec(
    left_homeostasis_gain_weight=1.5, right_homeostasis_gain_weight=2.5,
    ring_to_gain_weight=3.5, gain_to_ring_weight=-0.6, cross_inhibition_weight=-4.5,
)


def legacy_ring_rows(size, variant):
    backend = RecordingNest()
    network = build_ring_network(backend, size, RING_PARAMS, variant=variant)
    labels = {}
    label_nodes(labels, "ring", network.neurons)
    label_nodes(labels, "rec", network.spike_recorders)
    return relabel_rows(connection_rows(backend), labels), backend


def vectorised_ring_rows(size, variant):
    backend = RecordingNest()
    ring = build_ring_population(backend, "r", size, RING_PARAMS, variant=variant)
    labels = {}
    label_nodes(labels, "ring", ring.neurons)
    label_nodes(labels, "rec", ring.recorders)
    return relabel_rows(connection_rows(backend), labels), backend, ring


class RingWeightMatrixTests(unittest.TestCase):
    def test_matrix_rows_follow_the_legacy_distance_profile(self):
        for size in (4, 5, 8, 9):
            for variant in ("legacy", "builder"):
                with self.subTest(size=size, variant=variant):
                    matrix = ring_weight_matrix(size, variant)
                    if variant == "legacy":
                        profile = [legacy_ring_weight(d) for d in legacy_ring_distances(size)]
                    else:
                        profile = [builder_ring_weight(d, size) for d in builder_ring_distances(size)]
                    for pre in range(size):
                        for shift in range(size):
                            self.assertEqual(matrix[pre, (pre + shift) % size], profile[shift])
                    self.assertTrue(np.all(np.diag(matrix) == 3.0))
        with self.assertRaises(ValueError):
            ring_weight_matrix(4, "nope")


class RingEquivalenceTests(unittest.TestCase):
    def test_same_synapses_as_the_per_synapse_builder_for_both_variants(self):
        for size in (4, 8, 9):
            for variant in ("legacy", "builder"):
                with self.subTest(size=size, variant=variant):
                    legacy, legacy_backend = legacy_ring_rows(size, variant)
                    fast, fast_backend, ring = vectorised_ring_rows(size, variant)
                    self.assertEqual(fast, legacy)
                    self.assertEqual(len(legacy_backend.connect_calls), size + size * size)
                    self.assertEqual(len(fast_backend.connect_calls), 2)
                    self.assertEqual(ring.weight_matrix.shape, (size, size))

    def test_create_calls_and_kernel_configuration(self):
        backend = RecordingNest()
        ring = build_ring_population(backend, "r", 6, RING_PARAMS)
        self.assertEqual(
            [(c["model"], c["n"], c["params"]) for c in backend.create_calls],
            [("iaf_psc_alpha", 6, RING_PARAMS), ("spike_recorder", 6, None)],
        )
        self.assertEqual(backend.kernel_calls, [])  # the assembly configures the kernel, not the builder
        self.assertEqual(ring.neuron(2).ids, (3,))
        self.assertEqual(ring.recorder(2).ids, (9,))
        self.assertEqual(backend.connect_calls[0]["conn_spec"], "one_to_one")
        weight = backend.connect_calls[1]["syn_spec"]["weight"]
        self.assertEqual(weight.shape, (6, 6))
        np.testing.assert_array_equal(weight, ring.weight_matrix.T)  # NEST wants (target, source)

    def test_build_errors_name_the_population(self):
        backend = RecordingNest()
        with self.assertRaisesRegex(BuildError, "^r: population size must be positive"):
            create_population(backend, "r", 0, RING_PARAMS)
        with self.assertRaisesRegex(BuildError, "^r: invalid neuron parameters: V_th='x' is not a number"):
            create_population(backend, "r", 3, {"V_th": "x"})
        self.assertEqual(backend.create_calls, [])

        class WithDefaults(RecordingNest):
            def GetDefaults(self, model):
                return {"V_th": -55.0, "I_e": 0.0}

        with self.assertRaisesRegex(BuildError, "unknown iaf_psc_alpha parameters \\['bogus'\\]"):
            validate_neuron_parameters(WithDefaults(), "iaf_psc_alpha", {"bogus": 1.0}, "r")
        self.assertEqual(validate_neuron_parameters(WithDefaults(), "iaf_psc_alpha", {"V_th": 0.8}, "r"), {"V_th": 0.8})


class ReadoutEquivalenceTests(unittest.TestCase):
    def test_same_synapses_dc_and_recorders(self):
        size, harmonics = 6, 2
        weights = np.round(np.random.default_rng(1).standard_normal((size, 2 * harmonics)), 3)
        legacy_backend = RecordingNest()
        ring = create_population(legacy_backend, "ring", size, RING_PARAMS)
        readout = build_fourier_readout(legacy_backend, ring.neuron_list(), weights, harmonics, 200.0, 250.0)
        labels = {}
        label_nodes(labels, "ring", ring.neurons)
        label_nodes(labels, "rrec", ring.recorders)
        label_nodes(labels, "f", readout.all_neurons)
        label_nodes(labels, "frec", readout.all_recorders)
        label_nodes(labels, "dc", readout.dc_generator)
        legacy = relabel_rows(connection_rows(legacy_backend), labels)

        fast_backend = RecordingNest()
        ring = create_population(fast_backend, "ring", size, RING_PARAMS)
        pop = build_readout_population(fast_backend, "f", ring, weights, harmonics, 200.0, 250.0)
        labels = {}
        label_nodes(labels, "ring", ring.neurons)
        label_nodes(labels, "rrec", ring.recorders)
        label_nodes(labels, "f", pop.neurons)
        label_nodes(labels, "frec", pop.recorders)
        label_nodes(labels, "dc", pop.dc_generator)
        self.assertEqual(relabel_rows(connection_rows(fast_backend), labels), legacy)
        self.assertEqual(pop.feature_names, ["sin_pos_k1", "sin_neg_k1", "sin_pos_k2", "sin_neg_k2"])
        self.assertEqual(pop.feature_neuron("sin_neg_k2").ids, (pop.neurons.ids[3],))
        # Readout neurons use NEST's defaults, like the legacy facade.
        self.assertEqual(fast_backend.create_calls[2], {"model": "iaf_psc_alpha", "n": 4, "params": None, "ids": (13, 14, 15, 16)})
        with self.assertRaisesRegex(BuildError, "^f: Fourier weights shape mismatch"):
            build_readout_population(fast_backend, "f", ring, weights[:, :2], harmonics, 1.0, 1.0)


class DecisionAndGainEquivalenceTests(unittest.TestCase):
    def _feature_weights(self, n_features):
        return np.round(np.random.default_rng(2).standard_normal((2 * n_features, 2)), 3)

    def test_decision_circuit_and_feature_projections(self):
        harmonics = 3
        features = 2 * harmonics
        matrix = self._feature_weights(features)
        names = ["sin_pos_k%d" % (k + 1) if i % 2 == 0 else "sin_neg_k%d" % (k + 1) for k in range(harmonics) for i in range(2)]

        legacy_backend = RecordingNest()
        f1 = create_population(legacy_backend, "f1", features)
        f2 = create_population(legacy_backend, "f2", features)
        circuit = build_decision_circuit(legacy_backend, RING_PARAMS, HOMEOSTASIS_SPEC)
        connect_ring_feature_vectors(
            legacy_backend, circuit,
            {name: f1.neuron(i) for i, name in enumerate(names)},
            {name: f2.neuron(i) for i, name in enumerate(names)},
            names, matrix[:, 0], matrix[:, 1], HOMEOSTASIS_SPEC.warm_cold_weight_scale,
        )
        labels = {}
        for pop in (f1, f2):
            label_nodes(labels, pop.name, pop.neurons)
            label_nodes(labels, pop.name + "rec", pop.recorders)
        for label, node in circuit.neurons.items():
            labels[node.ids[0]] = ("dec", label)
        for label, node in circuit.spike_recorders.items():
            labels[node.ids[0]] = ("drec", label.replace("_spike", ""))
        legacy = relabel_rows(connection_rows(legacy_backend), labels)

        fast_backend = RecordingNest()
        f1 = create_population(fast_backend, "f1", features)
        f2 = create_population(fast_backend, "f2", features)
        decision = build_decision_population(fast_backend, "cmp", RING_PARAMS, HOMEOSTASIS_SPEC)
        connect_features_to_decision(
            fast_backend, decision, f1, f2, matrix[:features, 0], matrix[features:, 1],
            HOMEOSTASIS_SPEC.warm_cold_weight_scale,
        )
        labels = {}
        for pop in (f1, f2):
            label_nodes(labels, pop.name, pop.neurons)
            label_nodes(labels, pop.name + "rec", pop.recorders)
        for index, label in enumerate(decision.labels):
            labels[decision.neurons.ids[index]] = ("dec", label)
            labels[decision.recorders.ids[index]] = ("drec", label)
        self.assertEqual(relabel_rows(connection_rows(fast_backend), labels), legacy)
        self.assertEqual(len(fast_backend.connect_calls), 1 + 1 + (1 + 6) + 2)  # recorders, laterals, two feature matrices
        with self.assertRaisesRegex(BuildError, "^cmp: feature weight vectors"):
            connect_features_to_decision(fast_backend, decision, f1, f2, matrix[:, 0], matrix[:, 1], 1.0)

    def test_gain_populations_and_shifted_feedback(self):
        for size, per_side in ((12, 2), (10, 0), (16, 6)):
            with self.subTest(size=size):
                legacy_backend = RecordingNest()
                ring = create_population(legacy_backend, "ring", size, RING_PARAMS)
                circuit = build_decision_circuit(legacy_backend, RING_PARAMS, HOMEOSTASIS_SPEC)
                network = build_gain_network(legacy_backend, ring.neuron_list(), circuit, RING_PARAMS, GAIN_SPEC)
                connect_gain_feedback(legacy_backend, network, ring.neuron_list(), GAIN_SPEC.gain_to_ring_weight)
                labels = {}
                label_nodes(labels, "ring", ring.neurons)
                label_nodes(labels, "rrec", ring.recorders)
                for label, node in circuit.neurons.items():
                    labels[node.ids[0]] = ("dec", label)
                for label, node in circuit.spike_recorders.items():
                    labels[node.ids[0]] = ("drec", label.replace("_spike", ""))
                label_nodes(labels, "left", network.left_neurons)
                label_nodes(labels, "right", network.right_neurons)
                label_nodes(labels, "lrec", network.left_spike_recorders)
                label_nodes(labels, "rrec2", network.right_spike_recorders)
                legacy = relabel_rows(connection_rows(legacy_backend), labels)

                fast_backend = RecordingNest()
                ring = create_population(fast_backend, "ring", size, RING_PARAMS)
                decision = build_decision_population(fast_backend, "cmp", RING_PARAMS, HOMEOSTASIS_SPEC)
                gain = build_gain_populations(fast_backend, "gain", size, RING_PARAMS)
                connect_gain_populations(fast_backend, gain, ring, decision, GAIN_SPEC)
                count = connect_gain_feedback_populations(fast_backend, gain, ring, GAIN_SPEC.gain_to_ring_weight)
                self.assertEqual(count, per_side)
                labels = {}
                label_nodes(labels, "ring", ring.neurons)
                label_nodes(labels, "rrec", ring.recorders)
                for index, label in enumerate(decision.labels):
                    labels[decision.neurons.ids[index]] = ("dec", label)
                    labels[decision.recorders.ids[index]] = ("drec", label)
                label_nodes(labels, "left", gain.left.neurons)
                label_nodes(labels, "right", gain.right.neurons)
                label_nodes(labels, "lrec", gain.left.recorders)
                label_nodes(labels, "rrec2", gain.right.recorders)
                self.assertEqual(relabel_rows(connection_rows(fast_backend), labels), legacy)
        with self.assertRaisesRegex(BuildError, "ring size 12 does not match"):
            other = create_population(fast_backend, "other", 12, RING_PARAMS)
            connect_gain_populations(fast_backend, gain, other, decision, GAIN_SPEC)


class StimulusGeneratorTests(unittest.TestCase):
    def test_generators_are_one_to_one_and_bumps_wrap_around(self):
        backend = RecordingNest()
        ring = create_population(backend, "ring", 12, RING_PARAMS)
        stimulus = build_stimulus_generators(backend, "stim", ring)
        self.assertEqual(backend.create_calls[-1]["model"], "poisson_generator")
        self.assertEqual(backend.create_calls[-1]["params"], {"rate": 0.0})
        call = backend.connect_calls[-1]
        self.assertEqual((call["conn_spec"], call["syn_spec"]), ("one_to_one", {"weight": 4500.0}))
        self.assertEqual(len(call["source"]), 12)
        rates = set_bump(backend, stimulus, 1, 2, 200.0)
        self.assertEqual(rates.tolist(), [200.0, 200.0, 200.0, 200.0, 0, 0, 0, 0, 0, 0, 0, 200.0])
        ids, value = backend.status_calls[-1]
        self.assertEqual(ids, stimulus.generators.ids)
        self.assertEqual([entry["rate"] for entry in value], rates.tolist())
        clear_stimulus(backend, stimulus)
        self.assertEqual([entry["rate"] for entry in backend.status_calls[-1][1]], [0.0] * 12)
        calls = len(backend.status_calls)
        clear_stimulus(backend, stimulus)  # already off: no call
        self.assertEqual(len(backend.status_calls), calls)
        with self.assertRaisesRegex(BuildError, "negative stimulus rate"):
            set_bump(backend, stimulus, 0, 1, -1.0)


class WholeNetworkEquivalenceTests(unittest.TestCase):
    """The N=200 model from ``config/``: legacy builder composition vs vectorised."""

    @classmethod
    def setUpClass(cls):
        cls.artifacts = load_single_ring_artifacts()

    def _legacy_rows(self):
        backend = RecordingNest()
        a = self.artifacts
        neuron = a.neuron_parameters
        labels = {}
        rings = {}
        for name in ("r1", "r2"):
            network = build_ring_network(backend, a.population_size, neuron["ring"], reset_kernel=False, configure_backend=False)
            label_nodes(labels, name, network.neurons)
            label_nodes(labels, name + "rec", network.spike_recorders)
            rings[name] = network
        readouts = {}
        for name, ring in (("f1", rings["r1"]), ("f2", rings["r2"])):
            readout = build_fourier_readout(
                backend, ring.neurons, a.fourier_weights, a.num_fourier_k,
                a.readout_weight_scale, a.output_dc_baseline,
            )
            label_nodes(labels, name, readout.all_neurons)
            label_nodes(labels, name + "rec", readout.all_recorders)
            label_nodes(labels, name + "dc", readout.dc_generator)
            readouts[name] = readout
        circuit = build_decision_circuit(backend, neuron["homeostasis"], a.homeostasis)
        for label, node in circuit.neurons.items():
            labels[node.ids[0]] = ("dec", label)
        for label, node in circuit.spike_recorders.items():
            labels[node.ids[0]] = ("drec", label.replace("_spike", ""))
        n_features = len(a.feature_names)
        full = np.zeros((2 * n_features, 2))
        full[:n_features, 0] = a.warm_weights
        full[n_features:, 1] = a.cold_weights
        connect_ring_feature_vectors(
            backend, circuit, readouts["f1"].neurons, readouts["f2"].neurons, a.feature_names,
            full[:, 0], full[:, 1], a.homeostasis.warm_cold_weight_scale,
        )
        gain = build_gain_network(backend, rings["r1"].neurons, circuit, neuron["gain"], a.gain)
        connect_gain_feedback(backend, gain, rings["r1"].neurons, a.gain.gain_to_ring_weight)
        label_nodes(labels, "left", gain.left_neurons)
        label_nodes(labels, "right", gain.right_neurons)
        label_nodes(labels, "leftrec", gain.left_spike_recorders)
        label_nodes(labels, "rightrec", gain.right_spike_recorders)
        return relabel_rows(connection_rows(backend), labels), len(backend.connect_calls)

    def _vectorised_rows(self):
        backend = RecordingNest()
        network = build_single_ring_network(backend, artifacts=self.artifacts, seed=13579)
        labels = {}
        for name in ("r1", "r2", "f1", "f2"):
            pop = getattr(network, name)
            label_nodes(labels, name, pop.neurons)
            label_nodes(labels, name + "rec", pop.recorders)
        label_nodes(labels, "f1dc", network.f1.dc_generator)
        label_nodes(labels, "f2dc", network.f2.dc_generator)
        for index, label in enumerate(network.decision.labels):
            labels[network.decision.neurons.ids[index]] = ("dec", label)
            labels[network.decision.recorders.ids[index]] = ("drec", label)
        label_nodes(labels, "left", network.gain.left.neurons)
        label_nodes(labels, "right", network.gain.right.neurons)
        label_nodes(labels, "leftrec", network.gain.left.recorders)
        label_nodes(labels, "rightrec", network.gain.right.recorders)
        rows = connection_rows(backend)
        # The generators are the one addition: one per ring neuron, rate 0, weight 4500.
        generator_rows = [row for row in rows if row[0] not in labels]
        self.assertEqual(len(generator_rows), 2 * network.population_size)
        self.assertTrue(all(w == 4500.0 for _, _, w in generator_rows))
        for gen in network.stimulus.values():
            self.assertEqual(gen.rates.tolist(), [0.0] * network.population_size)
        return relabel_rows([row for row in rows if row[0] in labels], labels), len(backend.connect_calls), network

    def test_vectorised_network_has_the_legacy_synapse_set_plus_generators(self):
        legacy, legacy_calls = self._legacy_rows()
        fast, fast_calls, network = self._vectorised_rows()
        self.assertEqual(len(legacy), len(fast))
        self.assertEqual(fast, legacy)
        self.assertEqual(network.feedback_synapses_per_side, network.population_size - 10)
        self.assertGreater(legacy_calls, 90000)  # 98 552 per-synapse calls for N=200
        self.assertEqual(fast_calls, 31)
        self.assertEqual(network.describe()["population_size"], 200)

    def test_kernel_seed_threads_and_reset(self):
        backend = RecordingNest()
        build_single_ring_network(backend, artifacts=self.artifacts, seed=7, local_num_threads=2)
        self.assertEqual(
            backend.kernel_calls,
            [("ResetKernel", None), ("set_verbosity", "M_ERROR"), ("SetKernelStatus", {"local_num_threads": 2, "rng_seed": 7})],
        )
        with self.assertRaises(TypeError):
            build_single_ring_network(backend, artifacts=self.artifacts, weights_dir="x")

    def test_artifact_validation_reports_shape_problems(self):
        with tempfile.TemporaryDirectory() as directory:
            weights_dir = Path(directory)
            np.save(weights_dir / "N_200_fourier_weights.npy", np.zeros((200, 3)))
            with self.assertRaisesRegex(BuildError, "Fourier weights .* shape \\(200, 3\\), expected \\(200, 40\\)"):
                load_single_ring_artifacts(weights_dir=str(weights_dir))
        self.assertEqual(self.artifacts.warm_weights.shape, (40,))
        self.assertEqual(self.artifacts.cold_weights.shape, (40,))
        self.assertEqual(sorted(self.artifacts.paths), sorted([
            "ring_params", "neuron_params", "homeostasis_params", "gain_params",
            "fourier_weights", "homeostasis_weights", "homeostasis_metadata",
        ]))
        self.assertEqual(self.artifacts.neuron_parameters["gain"], load_neuron_parameters(str(MODEL_PARAMS / "neuron_params.json"), "gain"))
        self.assertEqual(self.artifacts.gain, load_gain_spec(str(MODEL_PARAMS / "gain_modulation_params.json")))
        self.assertEqual(self.artifacts.homeostasis, load_homeostasis_spec(str(MODEL_PARAMS / "homeostasis_params.json")))


if __name__ == "__main__":
    unittest.main()
