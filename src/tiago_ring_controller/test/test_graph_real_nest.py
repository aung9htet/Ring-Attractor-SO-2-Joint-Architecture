"""Phase 3 gate (real NEST): the hand-built single-joint graph reproduces the phase-2 golden.

Also runs ``GraphNestEngine`` with the graph: once-mode encoders at trial
start, then a continuous proprioceptive encoder mid-trial.
"""

import json
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
except ImportError:  # pragma: no cover - hosts without NEST
    nest = None

from tiago_ring_controller.blocks import BuildContext, Decoder, Encoder, FourierReadout, Gain, Goal, Homeostasis, Joint, Ring  # noqa: E402
from tiago_ring_controller.cosim.datapack import DataPack  # noqa: E402
from tiago_ring_controller.cosim.graph_engine import GraphNestEngine  # noqa: E402
from tiago_ring_controller.graph import Graph  # noqa: E402


PINNED_NEST_VERSION = "HEAD@41892a5"
GOLDEN_PATH = ROOT / "test/golden/vectorised_single_ring_seed13579.json"
SEED = 13579
STATE_INDEX, GOAL_INDEX, HALF_WIDTH = 60, 140, 5


def single_joint_graph(state_mode="once", **sim):
    g = Graph("two ring single joint", rng_seed=SEED, local_num_threads=1, **sim)
    r1, r2 = g.add(Ring("r1")), g.add(Ring("r2"))
    f1, f2 = g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
    cmp, gain = g.add(Homeostasis("cmp")), g.add(Gain("gain"))
    enc_state = g.add(Encoder("enc_state", mode=state_mode, joint_min=-1.0, joint_max=1.0))
    enc_goal = g.add(Encoder("enc_goal", joint_min=-1.0, joint_max=1.0))
    dec, j6, goal = g.add(Decoder("dec")), g.add(Joint("j6", index=5)), g.add(Goal("goal", angle_rad=0.6))
    g.connect(r1.spikes, f1.ring)
    g.connect(r2.spikes, f2.ring)
    g.connect(f1.features, cmp.state_features)
    g.connect(f2.features, cmp.target_features)
    g.connect(cmp.left, gain.left_in)
    g.connect(cmp.right, gain.right_in)
    g.connect(r1.spikes, gain.ring)
    g.connect(gain.feedback, r1.stim)
    g.connect(enc_state.stim, r1.stim)
    g.connect(enc_goal.stim, r2.stim)
    g.connect(j6.angle, enc_state.angle)
    g.connect(goal.angle, enc_goal.angle)
    g.connect(gain.left_counts, dec.left_counts)
    g.connect(gain.right_counts, dec.right_counts)
    g.connect(dec.velocity, j6.velocity)
    return g


@unittest.skipIf(nest is None, "NEST is not installed")
class GraphRealNestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest("requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__))
        cls.golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    def test_hand_built_graph_reproduces_the_phase_two_golden(self):
        graph = single_joint_graph()
        built = graph.build(BuildContext(nest, rng_seed=SEED, local_num_threads=1))
        self.assertLess(built.build_seconds, 3.0)
        enc_state, enc_goal = graph.blocks["enc_state"], graph.blocks["enc_goal"]
        # The golden's protocol: r1 bump 50 ms, r2 bump 50 ms, then 300 ms.
        enc_state.drive_index(STATE_INDEX, 0.0, 50.0)
        nest.Simulate(50.0)
        enc_state.expire(50.0)
        enc_goal.drive_index(GOAL_INDEX, 50.0, 50.0)
        nest.Simulate(50.0)
        enc_goal.expire(100.0)
        nest.Simulate(300.0)
        counts = lambda pop: [int(n) for n in pop.recorders.get("n_events")]  # noqa: E731
        self.assertEqual(counts(graph.blocks["r1"].population), self.golden["r1"])
        self.assertEqual(counts(graph.blocks["r2"].population), self.golden["r2"])
        self.assertEqual(counts(graph.blocks["gain"].populations.left), self.golden["left"])
        self.assertEqual(counts(graph.blocks["gain"].populations.right), self.golden["right"])
        decision = graph.blocks["cmp"].population
        self.assertEqual(
            {label + "_spike": int(decision.recorder_for(label).get("n_events")) for label in decision.labels},
            self.golden["decision"],
        )

    def test_graph_engine_runs_and_accepts_continuous_proprioception(self):
        graph = single_joint_graph(state_mode="continuous")
        engine = GraphNestEngine(nest, graph)
        engine.initialize()
        engine.reset()
        enc_state = graph.blocks["enc_state"]

        def angle_for(index):
            # invert the encoder's own (collector) mapping by search
            for angle in np.linspace(-1.0, 1.0, 4001):
                if enc_state.ring_index(float(angle)) == index:
                    return float(angle)
            raise AssertionError("no angle maps to index %d" % index)

        angle_state, angle_goal = angle_for(STATE_INDEX), angle_for(GOAL_INDEX)
        self.assertEqual(enc_state.ring_index(angle_state), STATE_INDEX)
        engine.set_datapacks({
            "enc_goal.angle": DataPack("enc_goal.angle", 0.0, {"angle": angle_goal}),
            "enc_state.angle": DataPack("enc_state.angle", 0.0, {"angle": angle_state}),
        })
        for _ in range(4):
            engine.advance(50.0)
        packs = engine.get_datapacks()
        self.assertEqual(sorted(packs), ["cmp", "gain", "r1", "r2"])
        self.assertLessEqual(abs(packs["r1"]["centroid"] - STATE_INDEX), 6.0)
        self.assertLessEqual(abs(packs["r2"]["centroid"] - GOAL_INDEX), 6.0)
        self.assertGreater(packs["gain"]["right_counts"] + packs["gain"]["left_counts"], 0)
        self.assertEqual(packs["r1"]["hidden_ms"] if "hidden_ms" in packs["r1"] else 0.0, 0.0)
        # continuous: a new measured angle every tick moves the state bump
        angle_new = angle_for(120)
        window = slice(115, 126)
        driven = 0.0
        for _ in range(6):
            engine.set_datapacks({"enc_state.angle": DataPack("enc_state.angle", engine.t_ms, {"angle": angle_new})})
            engine.advance(50.0)
            driven += float(np.sum(np.asarray(engine.get_datapacks()["r1"]["counts"])[window]))
        self.assertGreater(driven, 40.0)
        self.assertGreaterEqual(engine.recalibrations, 6)
        self.assertEqual(engine.applied[-1]["index"], 120)
        raster = engine.raster()
        self.assertGreater(len(raster["r1_times"]), 0)
        self.assertIn("r2_senders", raster)
        engine.finish_trial()
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
