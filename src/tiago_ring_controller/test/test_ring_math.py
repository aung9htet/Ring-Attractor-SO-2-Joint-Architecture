"""Pure characterization of both existing ring implementations."""

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LEGACY_DIR = ROOT / "legacy"


class _FakeNodeCollection:
    pass


def _load_module(name, path):
    """Import source with a non-simulating NEST stub."""
    fake_nest = types.ModuleType("nest")
    fake_nest.NodeCollection = _FakeNodeCollection
    previous_nest = sys.modules.get("nest")
    sys.modules["nest"] = fake_nest
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(LEGACY_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(LEGACY_DIR))
        sys.path.remove(str(SRC))
        if previous_nest is None:
            sys.modules.pop("nest", None)
        else:
            sys.modules["nest"] = previous_nest


LEGACY = _load_module("baseline_legacy_ring", LEGACY_DIR / "ring_attractor.py")
BUILDER = _load_module("baseline_builder_ring", LEGACY_DIR / "builders/ring_attractor.py")
COMPONENT = _load_module("baseline_ring_component", LEGACY_DIR / "ring_component.py")


def _load_trainer_module():
    fake_nest = types.ModuleType("nest")
    previous_nest = sys.modules.get("nest")
    previous_ring = sys.modules.get("ring_attractor")
    sys.modules["nest"] = fake_nest
    sys.modules["ring_attractor"] = LEGACY
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(LEGACY_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "facade_train_ring_model", LEGACY_DIR / "train_ring_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(LEGACY_DIR))
        sys.path.remove(str(SRC))
        if previous_nest is None:
            sys.modules.pop("nest", None)
        else:
            sys.modules["nest"] = previous_nest
        if previous_ring is None:
            sys.modules.pop("ring_attractor", None)
        else:
            sys.modules["ring_attractor"] = previous_ring


TRAINER = _load_trainer_module()


def _legacy(population_size):
    ring = object.__new__(LEGACY.Ring_Attractor)
    ring.population_size = population_size
    ring.max_distance = 50
    ring.sd_1 = 10
    ring.sd_2 = 5
    return ring


def _builder(population_size):
    ring = object.__new__(BUILDER.RingAttractor)
    ring.population_size = population_size
    ring.max_distance = 50
    ring.excitation_std_dev = 10
    ring.inhibition_std_dev = 5
    return ring


class RingDistanceAndWeightTests(unittest.TestCase):
    def test_even_distance_profiles_match_at_current_sizes(self):
        expected_100 = np.concatenate([np.arange(0.0, 51.0), np.arange(49.0, 0.0, -1.0)])
        expected_200 = np.concatenate(
            [np.arange(0.0, 50.5, 0.5), np.arange(49.5, 0.0, -0.5)]
        )
        for implementation in (_legacy, _builder):
            with self.subTest(implementation=implementation.__name__, size=100):
                method = implementation(100)
                actual = (
                    method.neuron_distance()
                    if hasattr(method, "neuron_distance")
                    else method._compute_ring_distances()
                )
                np.testing.assert_array_equal(actual, expected_100)
            with self.subTest(implementation=implementation.__name__, size=200):
                method = implementation(200)
                actual = (
                    method.neuron_distance()
                    if hasattr(method, "neuron_distance")
                    else method._compute_ring_distances()
                )
                np.testing.assert_array_equal(actual, expected_200)

    def test_odd_distance_profiles_are_intentionally_different(self):
        np.testing.assert_array_equal(
            _legacy(5).neuron_distance(), np.array([0.0, 25.0, 50.0, 50.0, 25.0])
        )
        np.testing.assert_array_equal(
            _builder(5)._compute_ring_distances(), np.array([0.0, 25.0, 50.0, 50.0, 0.0])
        )

    def test_weight_profiles_match_only_at_population_size_100(self):
        distances = [0.0, 5.0, 10.0, 25.0, 50.0]
        expected_legacy = np.array([3.0, 0.992, -1.008, -0.132, -0.0])
        actual_legacy = np.array([_legacy(100).get_weight(d) for d in distances])
        actual_builder_100 = np.array([_builder(100)._compute_weight(d) for d in distances])
        actual_builder_200 = np.array([_builder(200)._compute_weight(d) for d in distances])
        np.testing.assert_array_equal(actual_legacy, expected_legacy)
        np.testing.assert_array_equal(actual_builder_100, expected_legacy)
        np.testing.assert_array_equal(
            actual_builder_200, np.array([3.0, -1.008, -0.404, -0.0, -0.0])
        )
        self.assertFalse(np.array_equal(actual_builder_200, actual_legacy))


class RingCoordinateTests(unittest.TestCase):
    def test_builder_phase_and_index_conventions(self):
        ring = _builder(100)
        np.testing.assert_allclose(
            ring.preferred_angles(4), [0.0, math.pi / 2, math.pi, 3 * math.pi / 2],
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(ring.index_to_phase(25), math.pi / 2)
        self.assertEqual(ring.angle_to_neuron_index(-math.pi / 2), 75)
        self.assertEqual(ring.angle_to_neuron_index(2 * math.pi), 0)
        self.assertEqual(ring.angle_to_neuron_index(math.pi), 50)
        self.assertEqual(ring.angle_to_ring_index(-math.pi / 2), 75.0)
        self.assertEqual(ring.ring_index_to_angle(-25), 3 * math.pi / 2)
        np.testing.assert_array_equal(ring.generate_ring_positions(3), [25, 50, 75])

    def test_legacy_component_phase_helpers(self):
        component = COMPONENT.RingAttractorComponent
        self.assertEqual(component.center_index_to_phase(25, 100), math.pi / 2)
        np.testing.assert_array_equal(component.generate_center_indices(100, 3), [25, 50, 75])

    def test_population_vector_zero_and_cardinal_behavior(self):
        ring = _builder(100)
        self.assertEqual(ring.decode_angle_from_spikes(np.zeros(100)), 0.0)
        counts = np.zeros(100)
        counts[25] = 3.0
        self.assertAlmostEqual(ring.decode_angle_from_spikes(counts), math.pi / 2, places=14)
        counts[:] = 0.0
        counts[75] = 3.0
        self.assertAlmostEqual(ring.decode_angle_from_spikes(counts), -math.pi / 2, places=14)


class FourierArtifactConventionTests(unittest.TestCase):
    def test_saved_weights_are_positive_negative_sine_pairs(self):
        for population_size, harmonics in ((100, 5), (200, 20)):
            path = ROOT / (
                "src/config/ring_decoding_weights/N_%d_fourier_weights.npy" % population_size
            )
            actual = np.load(path, allow_pickle=False)
            theta = 2.0 * math.pi * np.arange(population_size) / population_size
            expected_columns = []
            for harmonic in range(1, harmonics + 1):
                sine = np.sin(harmonic * theta)
                expected_columns.extend((np.maximum(sine, 0.0), np.maximum(-sine, 0.0)))
            expected = np.column_stack(expected_columns)
            with self.subTest(population_size=population_size):
                self.assertEqual(actual.shape, (population_size, 2 * harmonics))
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=4e-16)
                for column in range(0, 2 * harmonics, 2):
                    self.assertFalse(np.any((actual[:, column] > 0) & (actual[:, column + 1] > 0)))

    def test_root_trainer_summary_dispatches_through_preferred_angles_override(self):
        class CustomTrainer(TRAINER.DecoderTrainer):
            def _preferred_angles(self, population_size):
                self.preferred_angles_argument = population_size
                return np.zeros(population_size)

        trainer = object.__new__(CustomTrainer)
        trainer.population_size = 4
        trainer.num_fourier_k = 1
        trainer.num_positions = 2
        trainer.sim_settle_ms = 10.0
        trainer.stimulus_half_width = 1
        trainer.repeats_per_position = 1
        trainer.results = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer.params_file = str(root / "ring_params.json")
            Path(trainer.params_file).write_text(
                json.dumps(
                    {
                        "readout_weight_scale": 200.0,
                        "output_dc_baseline": 200.0,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "outputs"
            output_dir.mkdir()
            previous_cwd = os.getcwd()
            try:
                os.chdir(directory)
                trainer._save_summary(str(output_dir))
            finally:
                os.chdir(previous_cwd)

            weights = np.load(
                root / "config/ring_decoding_weights/N_4_fourier_weights.npy",
                allow_pickle=False,
            )
            np.testing.assert_array_equal(weights, np.zeros((4, 2)))
            self.assertEqual(trainer.preferred_angles_argument, 4)


if __name__ == "__main__":
    unittest.main()
