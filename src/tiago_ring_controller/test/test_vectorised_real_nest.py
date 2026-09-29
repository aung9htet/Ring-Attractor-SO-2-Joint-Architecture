"""Real-NEST gates for the vectorised model (plan phase 2, items 1-3 and 5-6).

* equivalence with the frozen ``SingleRingModel`` under the same seed and the
  same bump protocol, with rate-profile tolerances (exact spike parity is not
  expected: the build-time generators change the per-thread RNG streams and the
  connection order changes floating-point summation);
* determinism: same seed, same counts;
* build time;
* pinned counts for the vectorised model (the phase-2 golden);
* the engine with the generator port accepts a mid-trial bump that moves r1.

See ``docs/blocks/equivalence.md`` for the measured numbers behind the tolerances.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LEGACY = ROOT / "legacy"
for path in (SRC, LEGACY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import nest
except ImportError:  # pragma: no cover - hosts without NEST
    nest = None

from tiago_ring_controller.config import source_config_path  # noqa: E402
from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE  # noqa: E402
from tiago_ring_controller.cosim import CosimConfig, GeneratorStimulusPort  # noqa: E402
from tiago_ring_controller.cosim.fakes import ring_readout  # noqa: E402
from tiago_ring_controller.cosim.runner import make_nest_engine  # noqa: E402
from tiago_ring_controller.cosim.tf import bump_datapack  # noqa: E402
from tiago_ring_controller.nest.single_ring import build_single_ring_network  # noqa: E402


PINNED_NEST_VERSION = "HEAD@41892a5"
SEED = 13579
STATE_INDEX, GOAL_INDEX, HALF_WIDTH = 60, 140, 5
SETTLE_MS = 300.0

# Phase-2 golden for the vectorised model (seed 13579, protocol of ``vectorised_run``),
# recorded in the container on 2026-09-29; see docs/blocks/equivalence.md.
GOLDEN_PATH = ROOT / "test/golden/vectorised_single_ring_seed13579.json"


def centroid(counts):
    total, _, value = ring_readout(np.asarray(counts, dtype=float))
    return total, value


def bump_width(counts, fraction=0.5):
    """Neurons above the baseline by at least ``fraction`` of the bump height."""

    counts = np.asarray(counts, dtype=float)
    baseline = float(np.median(counts))
    height = counts.max() - baseline
    if height <= 0:
        return 0
    return int(np.sum(counts >= baseline + fraction * height))


@unittest.skipIf(nest is None, "NEST is not installed")
class VectorisedModelRealNestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest(
                "requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__)
            )
        cls.ring_params = source_config_path("model_params", "ring_params.json")
        cls.weights_dir = source_config_path("ring_decoding_weights")

    # -- protocols --------------------------------------------------------
    def legacy_run(self):
        import gain_modulation
        import homeostasis
        import ring_component
        import single_ring

        for module in (gain_modulation, homeostasis, ring_component, single_ring):
            module.nest = nest
        model = single_ring.SingleRingModel(
            ring_params_file=self.ring_params, weights_dir=self.weights_dir,
            seed=SEED, local_num_threads=1,
        )
        model.r1._inject_bump(STATE_INDEX, HALF_WIDTH)   # 50 ms hidden each
        model.r2._inject_bump(GOAL_INDEX, HALF_WIDTH)
        nest.Simulate(SETTLE_MS)
        return self._summary(
            model.r1._get_ring_spike_counts(), model.r2._get_ring_spike_counts(),
            [len(r.get("events")["times"]) for r in model.gain_modulation.left_gain_spike_recorders],
            [len(r.get("events")["times"]) for r in model.gain_modulation.right_gain_spike_recorders],
            {k: len(r.get("events")["times"]) for k, r in model.homeostasis.homeostasis_recorders.items()},
        )

    def vectorised_run(self, seed=SEED):
        network = build_single_ring_network(
            nest, seed=seed, local_num_threads=1,
            ring_params_file=self.ring_params, weights_dir=self.weights_dir,
        )
        # Same protocol as the legacy facade: r1 bump for 50 ms, then r2 for 50 ms.
        network.set_bump("r1", STATE_INDEX, HALF_WIDTH)
        nest.Simulate(50.0)
        network.clear_bumps()
        network.set_bump("r2", GOAL_INDEX, HALF_WIDTH)
        nest.Simulate(50.0)
        network.clear_bumps()
        nest.Simulate(SETTLE_MS)
        counts = lambda pop: [int(n) for n in pop.recorders.get("n_events")]  # noqa: E731
        summary = self._summary(
            counts(network.r1), counts(network.r2), counts(network.gain.left), counts(network.gain.right),
            {label + "_spike": int(network.decision.recorder_for(label).get("n_events")) for label in network.decision.labels},
        )
        summary["build_seconds"] = network.build_seconds
        return summary

    @staticmethod
    def _summary(r1, r2, left, right, decision):
        r1_total, r1_centroid = centroid(r1)
        r2_total, r2_centroid = centroid(r2)
        return {
            "r1": [int(v) for v in r1], "r2": [int(v) for v in r2],
            "left": [int(v) for v in left], "right": [int(v) for v in right],
            "decision": {k: int(v) for k, v in decision.items()},
            "r1_total": r1_total, "r1_centroid": r1_centroid, "r1_width": bump_width(r1),
            "r2_total": r2_total, "r2_centroid": r2_centroid, "r2_width": bump_width(r2),
            "left_total": int(sum(left)), "right_total": int(sum(right)),
        }

    # -- gates -------------------------------------------------------------
    def test_rate_profiles_match_the_legacy_model_within_tolerance(self):
        legacy = self.legacy_run()
        fast = self.vectorised_run()
        report = {}
        for ring, index in (("r1", STATE_INDEX), ("r2", GOAL_INDEX)):
            with self.subTest(ring=ring):
                self.assertLessEqual(abs(legacy[ring + "_centroid"] - index), 3.0)
                self.assertLessEqual(abs(fast[ring + "_centroid"] - index), 3.0)
                self.assertLessEqual(abs(fast[ring + "_centroid"] - legacy[ring + "_centroid"]), 3.0)
                self.assertLessEqual(
                    abs(fast[ring + "_total"] - legacy[ring + "_total"]), 0.15 * legacy[ring + "_total"]
                )
                # Profile distance: sum |fast - legacy| over the ring relative to the legacy total.
                # A bump at the wrong place or of the wrong width scores well above 1.
                l1 = float(np.sum(np.abs(np.asarray(fast[ring]) - np.asarray(legacy[ring])))) / legacy[ring + "_total"]
                report[ring + "_l1"] = round(l1, 3)
                self.assertLessEqual(l1, 0.25)
        print("EQUIVALENCE", {**report, "legacy_r1_width": legacy["r1_width"], "fast_r1_width": fast["r1_width"],
                              "legacy_r2_width": legacy["r2_width"], "fast_r2_width": fast["r2_width"]})
        # Opponent gain: same winning side, comparable magnitude.
        self.assertEqual(
            np.sign(fast["right_total"] - fast["left_total"]),
            np.sign(legacy["right_total"] - legacy["left_total"]),
        )
        for side in ("left_total", "right_total"):
            with self.subTest(side=side):
                self.assertLessEqual(abs(fast[side] - legacy[side]), 0.35 * max(legacy[side], 20))
        for key in ("warm_spike", "cold_spike"):
            with self.subTest(key=key):
                self.assertLessEqual(
                    abs(fast["decision"][key] - legacy["decision"][key]), 0.35 * max(legacy["decision"][key], 10)
                )

    def test_build_is_fast_and_deterministic(self):
        first = self.vectorised_run()
        second = self.vectorised_run()
        self.assertLess(first["build_seconds"], 3.0)
        for key in ("r1", "r2", "left", "right", "decision"):
            self.assertEqual(first[key], second[key])
        other = self.vectorised_run(seed=SEED + 1)
        self.assertNotEqual(first["r1"], other["r1"])

    def test_pinned_counts_are_the_phase_two_golden(self):
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(golden["nest_version"], PINNED_NEST_VERSION)
        fast = self.vectorised_run()
        for key in ("r1", "r2", "left", "right", "decision", "r1_total", "r2_total", "left_total", "right_total"):
            with self.subTest(key=key):
                self.assertEqual(fast[key], golden[key])
        for key in ("r1_centroid", "r2_centroid"):
            self.assertAlmostEqual(fast[key], golden[key], places=3)

    def test_engine_with_generator_port_accepts_a_mid_trial_state_bump(self):
        config = CosimConfig.from_profile(
            COLLECTOR_PROFILE, 5, joint_min=-1.0, joint_max=1.0, rng_seed=SEED,
            nest_model="vectorised", ring_params_file=self.ring_params, weights_dir=self.weights_dir,
        )
        engine = make_nest_engine(config, backend=nest)
        self.assertIsInstance(engine.stimulus_port, GeneratorStimulusPort)
        engine.initialize()
        engine.reset()
        engine.set_datapacks({
            "goal_bump": bump_datapack("goal_bump", 0.0, GOAL_INDEX, HALF_WIDTH, "goal"),
            "state_bump": bump_datapack("state_bump", 0.0, STATE_INDEX, HALF_WIDTH, "proprioception"),
        })
        for _ in range(6):
            engine.advance(50.0)
        before = engine.get_datapacks()["ring_counts"]
        self.assertEqual(before["hidden_ms"], 0.0)
        self.assertEqual(before["readout_mode"], "node_collection")
        self.assertLessEqual(abs(before["r1_centroid"] - STATE_INDEX), 6.0)
        # Mid-trial: a new proprioceptive bump far from the current one.
        window = slice(120 - HALF_WIDTH, 120 + HALF_WIDTH + 1)
        quiet = sum(float(np.sum(np.asarray(engine.get_datapacks()["ring_counts"]["r1_delta"])[window])) for _ in [0])
        engine.set_datapacks({
            "state_bump": bump_datapack("state_bump", 300.0, 120, HALF_WIDTH, "proprioception", duration_ms=300.0)
        })
        driven = 0.0
        for _ in range(6):
            engine.advance(50.0)
            driven += float(np.sum(np.asarray(engine.get_datapacks()["ring_counts"]["r1_delta"])[window]))
        after = engine.get_datapacks()["ring_counts"]
        self.assertEqual(after["nest_step"], 12)
        self.assertEqual(after["hidden_ms"], 0.0)
        self.assertEqual(engine.stimulus_port.applied[-1]["nest_step"], 6)
        self.assertEqual(engine.stimulus_port.active, {})           # 300 ms elapsed: window off again
        self.assertGreaterEqual(engine.recalibrations, 2)
        # The generators drove the window (the attractor's response is a phase-5 question).
        self.assertLessEqual(quiet, 3.0)
        self.assertGreater(driven, 40.0)
        self.report = {"centroid_before": before["r1_centroid"], "centroid_after": after["r1_centroid"], "window_spikes": driven}
        print("MIDTRIAL", self.report)
        raster = engine.raster()
        self.assertGreater(len(raster["r1_times"]), 0)
        engine.finish_trial()
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
