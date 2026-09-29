"""Characterize signed-product ordering without constructing a real NEST graph."""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LEGACY = ROOT / "legacy"
FEATURE_ORDER = [
    "cos1_pos",
    "cos1_neg",
    "sin1_pos",
    "sin1_neg",
    "cos2_pos",
    "cos2_neg",
    "sin2_pos",
    "sin2_neg",
    "cos1cos2_pos",
    "cos1cos2_neg",
    "cos1sin2_pos",
    "cos1sin2_neg",
    "sin1cos2_pos",
    "sin1cos2_neg",
    "sin1sin2_pos",
    "sin1sin2_neg",
]


class _FakeNodes:
    def __init__(self, ids):
        self.ids = list(ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _FakeNodes(self.ids[index])
        return _FakeNodes([self.ids[index]])

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def get(self, key):
        if key != "global_id":
            raise KeyError(key)
        return self.ids[0] if len(self.ids) == 1 else list(self.ids)


class _FakeNest(types.ModuleType):
    NodeCollection = _FakeNodes

    def __init__(self):
        super().__init__("nest")
        self.next_id = 1
        self.connect_calls = 0

    def Create(self, _model, n=1, params=None):
        del params
        count = 1 if n is None else int(n)
        result = _FakeNodes(range(self.next_id, self.next_id + count))
        self.next_id += count
        return result

    def Connect(self, _source, _target, syn_spec=None):
        del syn_spec
        self.connect_calls += 1


def _load_multi_ring_module():
    fake_nest = _FakeNest()
    previous_nest = sys.modules.get("nest")
    previous_ring = sys.modules.get("ring_attractor")
    sys.modules["nest"] = fake_nest
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(LEGACY))
    try:
        ring_spec = importlib.util.spec_from_file_location(
            "ring_attractor", LEGACY / "ring_attractor.py"
        )
        ring_module = importlib.util.module_from_spec(ring_spec)
        sys.modules["ring_attractor"] = ring_module
        ring_spec.loader.exec_module(ring_module)

        spec = importlib.util.spec_from_file_location(
            "baseline_compositional_fourier_decoder",
            LEGACY / "compositional_fourier_decoder.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, fake_nest
    finally:
        sys.path.remove(str(LEGACY))
        sys.path.remove(str(SRC))
        if previous_nest is None:
            sys.modules.pop("nest", None)
        else:
            sys.modules["nest"] = previous_nest
        if previous_ring is None:
            sys.modules.pop("ring_attractor", None)
        else:
            sys.modules["ring_attractor"] = previous_ring


MODULE, FAKE_NEST = _load_multi_ring_module()


def _load_multi_ring_inference_module():
    fake_nest = _FakeNest()
    previous_nest = sys.modules.get("nest")
    previous_ring = sys.modules.get("ring_attractor")
    sys.modules["nest"] = fake_nest
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(LEGACY))
    try:
        ring_spec = importlib.util.spec_from_file_location(
            "ring_attractor", LEGACY / "ring_attractor.py"
        )
        ring_module = importlib.util.module_from_spec(ring_spec)
        sys.modules["ring_attractor"] = ring_module
        ring_spec.loader.exec_module(ring_module)

        spec = importlib.util.spec_from_file_location(
            "baseline_multi_ring_component",
            LEGACY / "multi_ring_component.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(LEGACY))
        sys.path.remove(str(SRC))
        if previous_nest is None:
            sys.modules.pop("nest", None)
        else:
            sys.modules["nest"] = previous_nest
        if previous_ring is None:
            sys.modules.pop("ring_attractor", None)
        else:
            sys.modules["ring_attractor"] = previous_ring


INFERENCE_MODULE = _load_multi_ring_inference_module()


class SignedFeatureContractTests(unittest.TestCase):
    def _trainer(self, side=2):
        trainer = object.__new__(MODULE.CompositionalFourierDecoderTrainer)
        trainer.feature_grid_size = side
        trainer._signed_feature_order = trainer._signed_feature_order_list()
        return trainer

    def test_feature_order_and_dimension(self):
        trainer = self._trainer(side=32)
        self.assertEqual(trainer._signed_term_names(), [
            "cos1", "sin1", "cos2", "sin2",
            "cos1cos2", "cos1sin2", "sin1cos2", "sin1sin2",
        ])
        self.assertEqual(trainer._signed_feature_order_list(), FEATURE_ORDER)
        self.assertEqual(trainer._signed_feature_dim(), 16384)

    def test_vector_flattening_is_feature_major_then_row_major(self):
        trainer = self._trainer(side=2)
        counts = {
            name: np.arange(4, dtype=float).reshape(2, 2) + 10.0 * index
            for index, name in enumerate(FEATURE_ORDER)
        }
        actual = trainer._signed_counts_to_vector(counts)
        expected = np.concatenate([counts[name].ravel(order="C") for name in FEATURE_ORDER])
        np.testing.assert_array_equal(actual, expected)
        maps = trainer._signed_vector_to_maps(actual)
        self.assertEqual(list(maps), FEATURE_ORDER)
        for name in FEATURE_ORDER:
            np.testing.assert_array_equal(maps[name], counts[name])

    def test_term_summaries_dispatch_through_overridden_private_layout_methods(self):
        class CustomTrainer(MODULE.CompositionalFourierDecoderTrainer):
            def _signed_vector_to_maps(self, vector):
                self.seen_vector = vector
                return {
                    "custom_pos": np.array([[1.0, -2.0]]),
                    "custom_neg": np.array([[4.0, -8.0]]),
                }

            def _signed_term_names(self):
                return ["custom"]

        trainer = object.__new__(CustomTrainer)
        vector = np.array([123.0])
        self.assertEqual(trainer._signed_vector_term_sums(vector), {"custom": 3.0})
        self.assertIs(trainer.seen_vector, vector)
        self.assertEqual(
            trainer._signed_vector_term_activity(vector), {"custom": 15.0}
        )
        self.assertIs(trainer.seen_vector, vector)

    def test_fake_nest_build_captures_mapping_and_flat_population_order(self):
        setup = object.__new__(MODULE.MultiRingAttractorSetup)
        setup.population_size = 100
        setup.feature_grid_size = 32
        setup.signed_product_dc_baseline = 180.0
        setup.signed_product_input_weight = 120.0
        setup.rings = [
            types.SimpleNamespace(ring_neurons=[_FakeNodes([100000 + i]) for i in range(100)]),
            types.SimpleNamespace(ring_neurons=[_FakeNodes([200000 + i]) for i in range(100)]),
        ]
        layer = setup.build_signed_product_grid_layer(np.empty(0))
        self.assertEqual(layer["feature_order"], FEATURE_ORDER)
        self.assertEqual(layer["n_cells"], 1024)
        np.testing.assert_array_equal(
            layer["mapped_ring_idxs"],
            [0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 31, 34, 38, 41, 44, 47,
             50, 53, 56, 59, 62, 66, 69, 72, 75, 78, 81, 84, 88, 91, 94, 97],
        )
        self.assertEqual(len(layer["flat_nodes"]), 16384)
        self.assertGreater(FAKE_NEST.connect_calls, 16384)

    def test_saved_artifact_feature_order_matches_runtime(self):
        path = ROOT / "src/config/ring_decoding_weights/N_100_J_2_multi_ring_sawtooth_weights.npz"
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data["signed_product_feature_order"].tolist(), FEATURE_ORDER)
            self.assertEqual(int(data["signed_product_population_size"]), 32)
            self.assertEqual(bool(data["output_weight_matrix_is_target_by_source"]), True)


class OrientationConventionTests(unittest.TestCase):
    def _trainer(self):
        trainer = object.__new__(MODULE.CompositionalFourierDecoderTrainer)
        trainer.joint_axes = ["x", "y"]
        trainer.population_size = 100
        trainer.output_ring_size = 100
        return trainer

    def test_forward_kinematics_is_x_then_y(self):
        trainer = self._trainer()
        q = np.array([math.pi / 2, math.pi / 2])
        expected = trainer._rot("x", q[0]) @ trainer._rot("y", q[1])
        np.testing.assert_allclose(trainer._forward_kinematics(q), expected, rtol=0.0, atol=1e-15)

    def test_lift_pitch_yaw_cardinal_cases(self):
        trainer = self._trainer()
        lift, pitch, yaw = trainer._lift_pitch_yaw_from_angles(np.array([math.pi / 2, 0.0]))
        self.assertAlmostEqual(lift, math.pi / 2, places=14)
        self.assertAlmostEqual(pitch, 0.0, places=14)
        self.assertAlmostEqual(yaw, 0.0, places=14)
        lift, pitch, yaw = trainer._lift_pitch_yaw_from_angles(np.array([0.0, math.pi / 2]))
        self.assertAlmostEqual(lift, 0.0, places=14)
        self.assertAlmostEqual(pitch, math.pi / 2, places=14)
        self.assertAlmostEqual(yaw, 0.0, places=14)

    def test_angle_to_center_uses_wrapping_and_python_rounding(self):
        trainer = self._trainer()
        self.assertEqual(trainer._angle_to_center_idx(0.0), 0)
        self.assertEqual(trainer._angle_to_center_idx(2 * math.pi), 0)
        self.assertEqual(trainer._angle_to_center_idx(-math.pi / 2), 75)
        self.assertEqual(trainer._angle_to_center_idx(math.pi), 50)

    def test_trainer_scalar_return_types_are_legacy_numpy_scalars(self):
        trainer = self._trainer()
        self.assertIsInstance(trainer._angle_to_ring_index(math.pi, 100), np.float64)
        profile = np.zeros(100)
        profile[25] = 1.0
        decoded = trainer._decode_angle_sawtooth_from_profile(profile)
        self.assertIsInstance(decoded, np.float64)
        self.assertAlmostEqual(float(decoded), math.pi / 2, places=14)

    def test_baseline_and_serial_injection_windows_remain_300_vs_400_ms(self):
        trainer = object.__new__(MODULE.CompositionalFourierDecoderTrainer)
        trainer.baseline_burnin_ms = 300.0
        trainer.sim_settle_ms = 300.0
        calls = []
        counts = iter(
            [np.array([0.0]), np.array([3.0]), np.array([3.0]), np.array([7.0])]
        )
        original_simulate = getattr(MODULE.nest, "Simulate", None)
        MODULE.nest.Simulate = lambda duration: calls.append(("simulate", duration))
        try:
            def serial_injection():
                calls.append(("inject", "q1"))
                MODULE.nest.Simulate(50.0)
                calls.append(("inject", "q2"))
                MODULE.nest.Simulate(50.0)

            delta = trainer._measure_net_stimulus_delta(
                model=None,
                q=np.array([0.0, 0.0]),
                get_counts_fn=lambda: next(counts),
                pre_stim_action=serial_injection,
            )
        finally:
            if original_simulate is None:
                delattr(MODULE.nest, "Simulate")
            else:
                MODULE.nest.Simulate = original_simulate

        np.testing.assert_array_equal(delta, [1.0])
        self.assertEqual(
            calls,
            [
                ("simulate", 300.0),
                ("simulate", 300.0),
                ("inject", "q1"),
                ("simulate", 50.0),
                ("inject", "q2"),
                ("simulate", 50.0),
                ("simulate", 300.0),
            ],
        )


class InferenceFacadeTypeTests(unittest.TestCase):
    def _decoder(self):
        decoder = object.__new__(INFERENCE_MODULE.MultiRingDecode)
        decoder.population_size = 100
        decoder.output_ring_size = 100
        decoder.num_joints = 2
        return decoder

    def test_scalar_coordinate_and_decode_methods_return_python_float(self):
        decoder = self._decoder()
        index = decoder.angle_to_ring_index(math.pi)
        angle = decoder.ring_index_to_angle(50)
        profile = np.zeros(100)
        profile[25] = 1.0
        decoded = decoder._decode_angle_sawtooth_from_profile(profile)
        self.assertIs(type(index), float)
        self.assertIs(type(angle), float)
        self.assertIs(type(decoded), float)
        self.assertEqual(index, 50.0)
        self.assertEqual(angle, math.pi)
        self.assertAlmostEqual(decoded, math.pi / 2, places=14)

    def test_array_coordinate_methods_preserve_ndarray_shape(self):
        decoder = self._decoder()
        indices = decoder.angle_to_ring_index(np.array([0.0, math.pi]))
        angles = decoder.ring_index_to_angle(np.array([0.0, 50.0]))
        self.assertIsInstance(indices, np.ndarray)
        self.assertIsInstance(angles, np.ndarray)
        np.testing.assert_array_equal(indices, [0.0, 50.0])
        np.testing.assert_array_equal(angles, [0.0, math.pi])

    def test_decode_and_signed_layer_dispatch_through_preferred_angles_override(self):
        class TrackingDecoder(INFERENCE_MODULE.MultiRingDecode):
            def _preferred_angles(self, n):
                self.preferred_angle_calls.append(n)
                return np.linspace(math.pi / 3.0, 2.0 * math.pi + math.pi / 3.0, n, endpoint=False)

        decoder = object.__new__(TrackingDecoder)
        decoder.preferred_angle_calls = []
        profile = np.array([0.0, 2.0, 0.0])
        decoded = decoder._decode_angle_sawtooth_from_profile(profile)
        self.assertEqual(decoder.preferred_angle_calls, [3])
        self.assertAlmostEqual(decoded, math.pi, places=14)

        decoder.population_size = 4
        decoder.feature_grid_size = 2
        decoder.signed_product_dc_baseline = 0.0
        decoder.signed_product_input_weight = 1.0
        decoder.joint_rings = {
            "q1": {"ring": types.SimpleNamespace(ring_neurons=[_FakeNodes([100 + i]) for i in range(4)])},
            "q2": {"ring": types.SimpleNamespace(ring_neurons=[_FakeNodes([200 + i]) for i in range(4)])},
        }
        decoder.signed_product_populations = {}
        decoder.populations = {}
        decoder._build_signed_product_layer()
        self.assertEqual(decoder.preferred_angle_calls, [3, 4])


if __name__ == "__main__":
    unittest.main()
