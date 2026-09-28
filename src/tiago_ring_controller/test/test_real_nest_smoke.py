"""Pinned, small real-NEST characterization of both public ring facades."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import nest
except ImportError:  # pragma: no cover - exercised on hosts without NEST
    nest = None


PINNED_NEST_VERSION = "HEAD@41892a5"
EXPECTED_COUNTS = np.array(
    [0, 2, 3, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=float,
)
EXPECTED_TIMES = [
    [],
    [29.400000000000002, 53.9],
    [24.8, 46.6, 66.8],
    [30.2, 50.800000000000004, 68.3],
    [34.1, 59.4],
    [23.0, 41.4, 58.6],
] + [[] for _ in range(14)]

EXPECTED_FULL_HOMEOSTASIS = {
    "warm_spike": 85,
    "cold_spike": 161,
    "left_spike": 19,
    "right_spike": 122,
}
EXPECTED_FULL_LEFT_GAIN = [
    (55, 11), (56, 12), (57, 7), (58, 9), (59, 8), (60, 13),
    (61, 9), (62, 7), (63, 2), (64, 6), (65, 6),
]
EXPECTED_FULL_RIGHT_GAIN = [
    (47, 2), (48, 3), (49, 4), (50, 3), (51, 3), (52, 3), (53, 4),
    (54, 4), (55, 27), (56, 28), (57, 21), (58, 25), (59, 23),
    (60, 30), (61, 25), (62, 20), (63, 8), (64, 16), (65, 19),
    (66, 4), (67, 4), (68, 3), (69, 3), (70, 3), (71, 4), (72, 3),
    (73, 1),
]
EXPECTED_FULL_STATE_RING = [
    (45, 1), (46, 2), (47, 5), (48, 5), (49, 5), (50, 5), (51, 5),
    (52, 6), (53, 6), (54, 6), (55, 11), (56, 11), (57, 10),
    (58, 11), (59, 10), (60, 11), (61, 11), (62, 10), (63, 9),
    (64, 10), (65, 10), (66, 6), (67, 6), (68, 6), (69, 5), (70, 5),
    (71, 5), (72, 5), (73, 5), (74, 2), (75, 1),
]
EXPECTED_FULL_TARGET_RING = [
    (123, 1), (124, 1), (125, 1), (126, 2), (127, 5), (128, 5),
    (129, 5), (130, 5), (131, 5), (132, 6), (133, 6), (134, 6),
    (135, 10), (136, 9), (137, 11), (138, 11), (139, 13), (140, 11),
    (141, 12), (142, 10), (143, 11), (144, 11), (145, 10), (146, 6),
    (147, 6), (148, 6), (149, 5), (150, 5), (151, 5), (152, 5),
    (153, 5), (154, 2), (155, 1), (156, 1), (157, 1),
]


@unittest.skipIf(nest is None, "NEST is not installed")
class RealNestRingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest(
                "exact spike characterization requires NEST %s, found %s"
                % (PINNED_NEST_VERSION, nest.__version__)
            )

        # Other characterization modules deliberately import these facades with
        # a tiny fake ``nest`` module.  Rebind their backend here so discovery
        # order cannot leak the fake into this pinned real-NEST smoke test.
        import builders.ring_attractor as builder_module
        import ring_attractor as legacy_module

        builder_module.nest = nest
        legacy_module.nest = nest

        cls.variants = (
            ("legacy", legacy_module.Ring_Attractor, "ring_neurons", "ring_spike_recorders", "_get_spike_counts"),
            ("builder", builder_module.RingAttractor, "neurons", "spike_recorders", "get_spike_counts"),
        )
        cls.params_file = str(SRC / "config/model_params/neuron_params.json")

    @staticmethod
    def _as_node_collection(groups):
        return nest.NodeCollection(
            [int(nest.GetStatus(group, "global_id")[0]) for group in groups]
        )

    def test_seeded_topology_delay_and_spike_events(self):
        for name, constructor, neuron_attr, recorder_attr, count_method in self.variants:
            with self.subTest(variant=name):
                nest.ResetKernel()
                nest.SetKernelStatus({"local_num_threads": 1, "rng_seed": 24680})
                model = constructor(
                    population_size=20,
                    reset_kernel=False,
                    params_file=self.params_file,
                )

                nodes = self._as_node_collection(getattr(model, neuron_attr))
                recurrent = nest.GetConnections(source=nodes, target=nodes)
                self.assertEqual(len(recurrent), 20 * 20)
                self.assertEqual(
                    set(float(value) for value in nest.GetStatus(recurrent, "delay")),
                    {1.0},
                )

                model.inject_stimulus(center_index=3, half_width=2)
                nest.Simulate(20.0)
                np.testing.assert_array_equal(getattr(model, count_method)(), EXPECTED_COUNTS)
                times = [
                    nest.GetStatus(recorder, "events")[0].get("times", []).tolist()
                    for recorder in getattr(model, recorder_attr)
                ]
                self.assertEqual(times, EXPECTED_TIMES)

    def test_seeded_full_comparator_gain_and_feedback_dynamics(self):
        import gain_modulation
        import homeostasis
        import ring_component
        import single_ring

        # Fake-NEST characterization imports can precede this test under
        # discovery; pin every facade's module global back to real NEST.
        for module in (gain_modulation, homeostasis, ring_component, single_ring):
            module.nest = nest

        model = single_ring.SingleRingModel(
            ring_params_file=str(SRC / "config/model_params/ring_params.json"),
            weights_dir=str(SRC / "config/ring_decoding_weights"),
            seed=13579,
            local_num_threads=1,
        )
        model.setup(60, 140, stimulus_half_width=5)
        nest.Simulate(300.0)

        homeostasis_counts = {
            key: len(recorder.get("events").get("times", []))
            for key, recorder in model.homeostasis.homeostasis_recorders.items()
        }
        self.assertEqual(homeostasis_counts, EXPECTED_FULL_HOMEOSTASIS)

        def sparse(values):
            array = np.asarray(values, dtype=int)
            indices = np.flatnonzero(array)
            return list(zip(indices.tolist(), array[indices].tolist()))

        left = [
            len(recorder.get("events").get("times", []))
            for recorder in model.gain_modulation.left_gain_spike_recorders
        ]
        right = [
            len(recorder.get("events").get("times", []))
            for recorder in model.gain_modulation.right_gain_spike_recorders
        ]
        self.assertEqual(sparse(left), EXPECTED_FULL_LEFT_GAIN)
        self.assertEqual(sparse(right), EXPECTED_FULL_RIGHT_GAIN)
        self.assertEqual(
            sparse(model.r1._get_ring_spike_counts()), EXPECTED_FULL_STATE_RING
        )
        self.assertEqual(
            sparse(model.r2._get_ring_spike_counts()), EXPECTED_FULL_TARGET_RING
        )


if __name__ == "__main__":
    unittest.main()
