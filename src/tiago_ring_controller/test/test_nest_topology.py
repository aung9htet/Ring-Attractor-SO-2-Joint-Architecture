"""Fake-NEST manifests for ring, readout, comparator, and gain topology."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support_fake_nest import FakeNodes, RecordingNest  # noqa: E402
from tiago_ring_controller.config import (  # noqa: E402
    load_gain_spec,
    load_homeostasis_spec,
)
from tiago_ring_controller.nest.comparator import (  # noqa: E402
    DecisionCircuit,
    build_decision_circuit,
    connect_ring_features,
)
from tiago_ring_controller.nest.gain import (  # noqa: E402
    build_gain_network,
    connect_gain_feedback,
)
from tiago_ring_controller.nest.kernel import (  # noqa: E402
    configure_kernel,
    recorder_events,
    spike_counts,
)
from tiago_ring_controller.nest.multi_ring import (  # noqa: E402
    SignedProductLayer,
    build_output_rings,
    build_signed_product_layer,
)
from tiago_ring_controller.nest.readout import build_fourier_readout  # noqa: E402
from tiago_ring_controller.nest.ring import (  # noqa: E402
    RingNetwork,
    build_builder_ring,
    build_legacy_ring,
    build_ring_network,
    inject_stimulus,
)


MODEL_PARAMS = SRC / "config/model_params"


class RingTopologyManifestTests(unittest.TestCase):
    def test_legacy_ring_create_connect_order_is_self_inclusive_n_squared(self):
        backend = RecordingNest()
        neuron_parameters = {"V_th": 0.8, "I_e": 450000.0}
        network = build_legacy_ring(backend, 4, neuron_parameters)

        self.assertEqual(
            backend.kernel_calls,
            [("ResetKernel", None), ("set_verbosity", "M_ERROR")],
        )
        self.assertEqual(
            [(call["model"], call["n"], call["params"]) for call in backend.create_calls],
            [
                ("iaf_psc_alpha", None, neuron_parameters),
                ("spike_recorder", None, None),
            ] * 4,
        )
        self.assertEqual([nodes.ids for nodes in network.neurons], [(1,), (3,), (5,), (7,)])
        self.assertEqual([nodes.ids for nodes in network.spike_recorders], [(2,), (4,), (6,), (8,)])
        self.assertEqual(len(backend.connect_calls), 4 + 4 * 4)
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]) for call in backend.connect_calls[:4]],
            [
                ((1,), (2,), None),
                ((3,), (4,), None),
                ((5,), (6,), None),
                ((7,), (8,), None),
            ],
        )
        recurrent = backend.connect_calls[4:]
        self.assertEqual(
            [(call["source"], call["target"]) for call in recurrent[:4]],
            [((1,), (1,)), ((1,), (3,)), ((1,), (5,)), ((1,), (7,))],
        )
        self.assertEqual(
            [(call["source"], call["target"]) for call in recurrent[4:8]],
            [((3,), (3,)), ((3,), (5,)), ((3,), (7,)), ((3,), (1,))],
        )
        self.assertEqual([call["syn_spec"]["weight"] for call in recurrent[:4]], [3.0, -0.132, -0.0, -0.132])
        self.assertTrue(all(call["conn_spec"] is None for call in recurrent))
        self.assertEqual(network.variant, "legacy")

    def test_builder_ring_retains_its_odd_geometry_and_population_scaled_weights(self):
        backend = RecordingNest()
        network = build_builder_ring(backend, 5, {})
        np.testing.assert_array_equal(network.distances, [0.0, 25.0, 50.0, 50.0, 0.0])
        recurrent = backend.connect_calls[5:]
        self.assertEqual(len(recurrent), 25)
        self.assertEqual(
            [call["syn_spec"]["weight"] for call in recurrent[:5]],
            [3.0, 2.839, 2.387, 2.387, 3.0],
        )
        self.assertEqual(network.variant, "builder")

    def test_unknown_variant_still_performs_current_kernel_and_population_side_effects(self):
        backend = RecordingNest()
        with self.assertRaisesRegex(ValueError, "Unknown ring variant"):
            build_ring_network(backend, 2, {}, variant="not-a-ring")
        self.assertEqual(len(backend.create_calls), 4)
        self.assertEqual(len(backend.connect_calls), 2)
        self.assertEqual(backend.kernel_calls[:2], [("ResetKernel", None), ("set_verbosity", "M_ERROR")])

    def test_stimulus_is_inclusive_serial_simulation_then_rate_zero(self):
        backend = RecordingNest()
        network = RingNetwork(
            neurons=[FakeNodes((100 + index,)) for index in range(12)],
            spike_recorders=[],
            distances=np.empty(0),
            population_size=12,
            variant="legacy",
        )
        stimulus = inject_stimulus(backend, network, center_index=0, half_width=5)
        self.assertEqual(stimulus.ids, (1,))
        self.assertEqual(
            backend.create_calls,
            [{"model": "poisson_generator", "n": None, "params": {"rate": 200.0}, "ids": (1,)}],
        )
        self.assertEqual(
            [call["target"] for call in backend.connect_calls],
            [(107,), (108,), (109,), (110,), (111,), (100,), (101,), (102,), (103,), (104,), (105,)],
        )
        self.assertTrue(all(call["syn_spec"] == {"weight": 4500.0} for call in backend.connect_calls))
        self.assertEqual(backend.simulate_calls, [50.0])
        self.assertEqual(backend.status_calls, [((1,), {"rate": 0.0})])


class FourierReadoutManifestTests(unittest.TestCase):
    def test_readout_population_recorders_dc_and_feature_connection_order(self):
        backend = RecordingNest()
        ring_neurons = [FakeNodes((101,)), FakeNodes((102,)), FakeNodes((103,))]
        weights = np.arange(12, dtype=float).reshape(3, 4) / 10.0
        readout = build_fourier_readout(
            backend,
            ring_neurons,
            weights,
            num_fourier_k=2,
            output_weight_scale=200.0,
            output_dc_baseline=200.0,
        )
        self.assertEqual(
            [(call["model"], call["n"], call["params"], call["ids"]) for call in backend.create_calls],
            [
                ("iaf_psc_alpha", 4, None, (1, 2, 3, 4)),
                ("spike_recorder", 4, None, (5, 6, 7, 8)),
                ("dc_generator", None, {"amplitude": 200.0}, (9,)),
            ],
        )
        self.assertEqual(tuple(readout.neurons), ("sin_pos_k1", "sin_neg_k1", "sin_pos_k2", "sin_neg_k2"))
        self.assertEqual(tuple(readout.spike_recorders), ("sin_pos_k1_recs", "sin_neg_k1_recs", "sin_pos_k2_recs", "sin_neg_k2_recs"))
        self.assertEqual(backend.connect_calls[0]["source"], (9,))
        self.assertEqual(backend.connect_calls[0]["target"], (1, 2, 3, 4))
        self.assertEqual(
            [(call["source"], call["target"]) for call in backend.connect_calls[1:5]],
            [((1,), (5,)), ((2,), (6,)), ((3,), (7,)), ((4,), (8,))],
        )
        feature_calls = backend.connect_calls[5:]
        self.assertEqual(len(feature_calls), 12)
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]["weight"]) for call in feature_calls[:3]],
            [((101,), (1,), 0.0), ((102,), (1,), 80.0), ((103,), (1,), 160.0)],
        )
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]["weight"]) for call in feature_calls[3:6]],
            [((101,), (2,), 20.0), ((102,), (2,), 100.0), ((103,), (2,), 180.0)],
        )
        np.testing.assert_array_equal(readout.weights, weights)

    def test_shape_validation_is_currently_optional_and_creation_free_on_failure(self):
        backend = RecordingNest()
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            build_fourier_readout(backend, [FakeNodes((1,))] * 3, np.zeros((3, 2)), 2, 1.0, 0.0)
        self.assertEqual(backend.create_calls, [])
        unvalidated = build_fourier_readout(
            backend,
            [FakeNodes((101,))] * 3,
            np.zeros((3, 2)),
            1,
            1.0,
            0.0,
            validate_shape=False,
        )
        self.assertEqual(len(unvalidated.neurons), 2)


class ComparatorAndGainManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.homeostasis_spec = load_homeostasis_spec(
            str(MODEL_PARAMS / "homeostasis_params.json")
        )
        cls.gain_spec = load_gain_spec(str(MODEL_PARAMS / "gain_modulation_params.json"))

    def test_decision_circuit_has_exact_six_connections_then_four_recorders(self):
        backend = RecordingNest()
        parameters = {"V_th": 0.8}
        circuit = build_decision_circuit(backend, parameters, self.homeostasis_spec)
        self.assertEqual(
            [(call["model"], call["params"]) for call in backend.create_calls],
            [("iaf_psc_alpha", parameters)] * 4 + [("spike_recorder", None)] * 4,
        )
        self.assertEqual(tuple(circuit.neurons), ("warm", "cold", "left", "right"))
        self.assertEqual(tuple(circuit.spike_recorders), ("warm_spike", "cold_spike", "left_spike", "right_spike"))
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]["weight"]) for call in backend.connect_calls[:6]],
            [
                ((2,), (4,), 20000.0),
                ((2,), (3,), -20000.0),
                ((1,), (3,), 20000.0),
                ((1,), (4,), -20000.0),
                ((3,), (4,), -20000.0),
                ((4,), (3,), -20000.0),
            ],
        )
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]) for call in backend.connect_calls[6:]],
            [
                ((1,), (5,), None),
                ((2,), (6,), None),
                ((3,), (7,), None),
                ((4,), (8,), None),
            ],
        )

    def test_ring_features_connect_only_ring1_to_warm_and_ring2_to_cold(self):
        backend = RecordingNest()
        circuit = DecisionCircuit(
            neurons={"warm": FakeNodes((1,)), "cold": FakeNodes((2,)), "left": FakeNodes((3,)), "right": FakeNodes((4,))},
            spike_recorders={},
        )
        ring_1 = {"sin_pos_k1": FakeNodes((101,)), "sin_neg_k1": FakeNodes((102,))}
        ring_2 = {"sin_pos_k1": FakeNodes((201,)), "sin_neg_k1": FakeNodes((202,))}
        weights = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)
        connect_ring_features(
            backend,
            circuit,
            ring_1,
            ring_2,
            ["sin_pos_k1", "sin_neg_k1"],
            weights,
            5.0,
        )
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]["weight"]) for call in backend.connect_calls],
            [
                ((101,), (1,), 5.0),
                ((201,), (2,), 150.0),
                ((102,), (1,), 10.0),
                ((202,), (2,), 200.0),
            ],
        )

    def test_gain_network_and_feedback_preserve_direction_and_boundary_exclusion(self):
        backend = RecordingNest()
        ring = [FakeNodes((100 + index,)) for index in range(12)]
        circuit = DecisionCircuit(
            neurons={"warm": FakeNodes((498,)), "cold": FakeNodes((499,)), "left": FakeNodes((500,)), "right": FakeNodes((501,))},
            spike_recorders={},
        )
        gain = build_gain_network(backend, ring, circuit, {"V_th": 1.35}, self.gain_spec)
        self.assertEqual([nodes.ids for nodes in gain.left_neurons], [(index,) for index in range(1, 24, 2)])
        self.assertEqual([nodes.ids for nodes in gain.right_neurons], [(index,) for index in range(25, 48, 2)])
        # 24 neuron-to-recorder calls, then six decision/ring inputs per index.
        self.assertEqual(len(backend.connect_calls), 24 + 6 * 12)
        first_inputs = backend.connect_calls[24:30]
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]["weight"]) for call in first_inputs],
            [
                ((500,), (1,), 10000.0),
                ((501,), (25,), 10000.0),
                ((500,), (25,), -30000.0),
                ((501,), (1,), -30000.0),
                ((100,), (1,), 450000.0),
                ((100,), (25,), 450000.0),
            ],
        )

        connect_gain_feedback(backend, gain, ring, self.gain_spec.gain_to_ring_weight)
        feedback = backend.connect_calls[-4:]
        self.assertEqual(
            [(call["source"], call["target"], call["syn_spec"]["weight"]) for call in feedback],
            [
                ((11,), (106,), -0.6),
                ((35,), (104,), -0.6),
                ((13,), (107,), -0.6),
                ((37,), (105,), -0.6),
            ],
        )
        feedback_sources = {call["source"] for call in feedback}
        self.assertNotIn((1,), feedback_sources)
        self.assertNotIn((23,), feedback_sources)

    def test_population_size_ten_has_no_gain_feedback_connections(self):
        backend = RecordingNest()
        ring = [FakeNodes((100 + index,)) for index in range(10)]
        circuit = DecisionCircuit(
            neurons={"warm": FakeNodes((498,)), "cold": FakeNodes((499,)), "left": FakeNodes((500,)), "right": FakeNodes((501,))},
            spike_recorders={},
        )
        gain = build_gain_network(backend, ring, circuit, {}, self.gain_spec)
        before = len(backend.connect_calls)
        connect_gain_feedback(backend, gain, ring, -0.6)
        self.assertEqual(len(backend.connect_calls), before)


class MultiRingTopologyTests(unittest.TestCase):
    def test_signed_product_create_connect_flatten_and_callback_order(self):
        backend = RecordingNest()
        callbacks = []
        layer = build_signed_product_layer(
            backend,
            [FakeNodes((100 + index,)) for index in range(4)],
            [FakeNodes((200 + index,)) for index in range(4)],
            population_size=4,
            feature_grid_size=2,
            dc_baseline=7.0,
            input_weight=11.0,
            on_population_built=lambda name, nodes, recorders: callbacks.append(
                (name, nodes.ids, recorders.ids)
            ),
        )

        expected_order = (
            "cos1_pos", "cos1_neg", "sin1_pos", "sin1_neg",
            "cos2_pos", "cos2_neg", "sin2_pos", "sin2_neg",
            "cos1cos2_pos", "cos1cos2_neg",
            "cos1sin2_pos", "cos1sin2_neg",
            "sin1cos2_pos", "sin1cos2_neg",
            "sin1sin2_pos", "sin1sin2_neg",
        )
        self.assertEqual(layer.feature_order, expected_order)
        self.assertEqual([item[0] for item in callbacks], list(expected_order))
        np.testing.assert_array_equal(layer.mapped_ring_indices, [0, 2])
        self.assertEqual(layer.n_cells, 4)
        self.assertEqual(len(layer.flat_nodes), 64)
        self.assertEqual(len(backend.create_calls), 48)
        self.assertEqual(len(backend.connect_calls), 96)
        self.assertEqual(
            [(call["model"], call["n"], call["params"]) for call in backend.create_calls[:6]],
            [
                ("iaf_psc_alpha", 4, None),
                ("spike_recorder", 4, None),
                ("dc_generator", None, {"amplitude": 7.0}),
                ("iaf_psc_alpha", 4, None),
                ("spike_recorder", 4, None),
                ("dc_generator", None, {"amplitude": 7.0}),
            ],
        )
        self.assertEqual(
            [
                (call["source"], call["target"], call["syn_spec"])
                for call in backend.connect_calls[:10]
            ],
            [
                ((9,), (1, 2, 3, 4), None),
                ((1,), (5,), None),
                ((2,), (6,), None),
                ((3,), (7,), None),
                ((4,), (8,), None),
                ((18,), (10, 11, 12, 13), None),
                ((10,), (14,), None),
                ((11,), (15,), None),
                ((12,), (16,), None),
                ((13,), (17,), None),
            ],
        )
        self.assertEqual(
            [
                (call["source"], call["target"], call["syn_spec"]["weight"])
                for call in backend.connect_calls[10:14]
            ],
            [
                ((100,), (1,), 11.0),
                ((100,), (2,), 11.0),
                ((102,), (12,), 11.0),
                ((102,), (13,), 11.0),
            ],
        )

    def test_output_all_to_all_uses_target_by_source_transpose_and_scale(self):
        backend = RecordingNest()
        layer = SignedProductLayer(
            feature_grid_size=1,
            n_cells=1,
            mapped_ring_indices=np.array([0]),
            feature_order=("fixture",),
            populations={},
            flat_nodes=FakeNodes((101, 102, 103)),
        )
        weights = {
            name: np.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float
            )
            for name in ("lift", "pitch", "yaw")
        }
        outputs = build_output_rings(
            backend,
            layer,
            weights,
            output_ring_size=2,
            output_dc_baseline=200.0,
            output_weight_scale=10.0,
            matrix_is_target_by_source=True,
        )
        self.assertEqual(tuple(outputs.populations), ("lift", "pitch", "yaw"))
        all_to_all = [
            call
            for call in backend.connect_calls
            if call["conn_spec"] == {"rule": "all_to_all"}
        ]
        self.assertEqual(len(all_to_all), 3)
        expected_matrix = (10.0 * weights["lift"].T).tolist()
        for call in all_to_all:
            self.assertEqual(call["source"], (101, 102, 103))
            self.assertEqual(len(call["target"]), 2)
            self.assertEqual(call["syn_spec"], {"weight": expected_matrix})

        # The compatibility switch retains the source-by-target branch too.
        backend = RecordingNest()
        build_output_rings(
            backend,
            layer,
            weights,
            output_ring_size=2,
            output_dc_baseline=200.0,
            output_weight_scale=10.0,
            matrix_is_target_by_source=False,
        )
        source_by_target = [
            call
            for call in backend.connect_calls
            if call["conn_spec"] == {"rule": "all_to_all"}
        ]
        self.assertEqual(
            source_by_target[0]["syn_spec"]["weight"],
            (10.0 * weights["lift"]).tolist(),
        )

    def test_output_orientation_getter_runs_at_each_legacy_wiring_stage(self):
        backend = RecordingNest()
        layer = SignedProductLayer(
            feature_grid_size=1,
            n_cells=1,
            mapped_ring_indices=np.array([0]),
            feature_order=("fixture",),
            populations={},
            flat_nodes=FakeNodes((101, 102, 103)),
        )
        weights = {
            name: np.arange(6, dtype=float).reshape(3, 2)
            for name in ("lift", "pitch", "yaw")
        }
        decisions = iter((True, False, True))
        getter_call_positions = []

        def orientation_getter():
            getter_call_positions.append(len(backend.connect_calls))
            return next(decisions)

        build_output_rings(
            backend,
            layer,
            weights,
            output_ring_size=2,
            output_dc_baseline=200.0,
            output_weight_scale=10.0,
            matrix_orientation_getter=orientation_getter,
        )
        self.assertEqual(getter_call_positions, [3, 7, 11])
        all_to_all = [
            call for call in backend.connect_calls
            if call["conn_spec"] == {"rule": "all_to_all"}
        ]
        self.assertEqual(all_to_all[0]["syn_spec"]["weight"], (10.0 * weights["lift"].T).tolist())
        self.assertEqual(all_to_all[1]["syn_spec"]["weight"], (10.0 * weights["pitch"]).tolist())
        self.assertEqual(all_to_all[2]["syn_spec"]["weight"], (10.0 * weights["yaw"].T).tolist())


class KernelUtilityTests(unittest.TestCase):
    def test_kernel_configuration_only_sets_explicit_thread_and_seed_values(self):
        backend = RecordingNest()
        configure_kernel(backend, local_num_threads=1, rng_seed=12345)
        self.assertEqual(
            backend.kernel_calls,
            [
                ("ResetKernel", None),
                ("set_verbosity", "M_ERROR"),
                ("SetKernelStatus", {"local_num_threads": 1, "rng_seed": 12345}),
            ],
        )
        backend = RecordingNest()
        configure_kernel(backend, reset_kernel=False, verbosity=None)
        self.assertEqual(backend.kernel_calls, [])

    def test_thread_value_is_passed_by_identity_and_backend_validation_propagates(self):
        class ThreadValue:
            def __int__(self):
                raise AssertionError("local_num_threads must not be coerced")

        class IdentityRecordingNest(RecordingNest):
            def SetKernelStatus(self, kernel_status):
                self.raw_kernel_status = kernel_status

        value = ThreadValue()
        backend = IdentityRecordingNest()
        configure_kernel(backend, local_num_threads=value, rng_seed=np.int64(7))
        status = backend.raw_kernel_status
        self.assertIs(status["local_num_threads"], value)
        self.assertIs(type(status["rng_seed"]), int)

        class RejectingNest(RecordingNest):
            def SetKernelStatus(self, kernel_status):
                self.rejected_status = kernel_status
                raise TypeError("backend rejected local_num_threads")

        rejecting_backend = RejectingNest()
        with self.assertRaisesRegex(TypeError, "backend rejected local_num_threads"):
            configure_kernel(rejecting_backend, local_num_threads=value)
        self.assertIs(rejecting_backend.rejected_status["local_num_threads"], value)

    def test_recorder_get_style_and_cumulative_count_dtype(self):
        backend = RecordingNest()
        recorders = [
            FakeNodes((1,), {"times": [1.0, 2.0]}),
            FakeNodes((2,), {"times": []}),
            FakeNodes((3,), {"times": [4.0]}),
        ]
        self.assertEqual(recorder_events(backend, recorders[0]), {"times": [1.0, 2.0]})
        counts = spike_counts(backend, recorders)
        np.testing.assert_array_equal(counts, [2.0, 0.0, 1.0])
        self.assertEqual(str(counts.dtype), "float64")


if __name__ == "__main__":
    unittest.main()
