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
from tiago_ring_controller.config import source_config_path  # noqa: E402
from tiago_ring_controller.cosim import CosimConfig  # noqa: E402
from tiago_ring_controller.cosim.datapack import DataPack  # noqa: E402
from tiago_ring_controller.cosim.fakes import FakeRobotEngine  # noqa: E402
from tiago_ring_controller.cosim.fakes import ring_readout  # noqa: E402
from tiago_ring_controller.cosim.graph_engine import GraphNestEngine  # noqa: E402
from tiago_ring_controller.cosim.runner import build_loop, make_nest_engine, run_trial  # noqa: E402
from tiago_ring_controller.graph import (  # noqa: E402
    EXAMPLES_DIR,
    Graph,
    multi_joint_two_ring,
    run_graph,
    two_joint_forward_kinematics,
    two_ring_single_joint,
)
from tiago_ring_controller.math.circular import decode_sawtooth_profile  # noqa: E402

LEGACY = ROOT / "legacy"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))


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
            "goal": DataPack("goal", 0.0, {"angle": angle_goal}),
            "j6": DataPack("j6", 0.0, {"angle": angle_state}),
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
            engine.set_datapacks({"j6": DataPack("j6", engine.t_ms, {"angle": angle_new})})
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


def _tick_rows(record):
    """Per main tick: (left, right, r1 delta, velocities, joint position) for tick-for-tick comparison."""

    rows = []
    for tick in record.main_ticks:
        # the legacy loop carries ring_counts as an engine input, the compiled graph emits it
        counts = tick.inputs.get("ring_counts") or tick.outputs.get("ring_counts")
        cmd = tick.outputs.get("arm_velocity_cmd")
        joint = tick.inputs.get("joint_state")
        rows.append((
            None if counts is None else (counts["left"], counts["right"], tuple(counts["r1_delta"])),
            None if cmd is None else tuple(round(v, 12) for v in cmd["velocities"]),
            None if joint is None else tuple(round(p, 12) for p in joint["positions"]),
        ))
    return rows


@unittest.skipIf(nest is None, "NEST is not installed")
class GraphFileParityTests(unittest.TestCase):
    """Phase 4 gate: file run == template run == vectorised cosim runner, tick for tick."""

    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest("requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__))

    def test_example_file_equals_template_and_vectorised_runner(self):
        goal = 0.6
        from_file = run_graph(Path(EXAMPLES_DIR) / "two_ring_single_joint.graph.json", engines="nest", goals=[goal])[0]
        from_python = run_graph(two_ring_single_joint(), engines="nest", goals=[goal])[0]
        file_rows, python_rows = _tick_rows(from_file["record"]), _tick_rows(from_python["record"])
        self.assertEqual(file_rows, python_rows)
        self.assertEqual(from_file["record"].stop_reason, from_python["record"].stop_reason)
        self.assertEqual(from_file["legacy"]["scalars"]["q_final"], from_python["legacy"]["scalars"]["q_final"])

        # The legacy transceivers on the vectorised model, i.e. run_cosim_trial.py --model vectorised.
        graph = two_ring_single_joint()
        joint = graph.blocks["j5"]
        dec = graph.blocks["dec"].params
        config = CosimConfig.from_profile(
            graph.blocks["enc_state"].profile, 5, joint_min=joint.limits()[0], joint_max=joint.limits()[1],
            rng_seed=SEED, nest_model="vectorised",
            decoder_gain_positive=dec["gain_positive"], decoder_gain_negative=dec["gain_negative"],
            decoder_tau_s=dec["tau_s"], decoder_delay_steps=dec["delay_steps"],
        )
        loop = build_loop(config, [make_nest_engine(config, backend=nest), FakeRobotEngine("robot")])
        loop.initialize()
        try:
            legacy = run_trial(loop, goal, config)
        finally:
            loop.shutdown()
        legacy_rows = _tick_rows(legacy)
        self.assertEqual(len(file_rows), len(legacy_rows))
        self.assertEqual(file_rows, legacy_rows)
        self.assertEqual(from_file["record"].stop_reason, legacy.stop_reason)
        print("PARITY ticks=%d stop=%s q_final=%.4f" % (len(file_rows), legacy.stop_reason, from_file["legacy"]["scalars"]["q_final"]))
        self.assertEqual(from_file["record"].meta["graph"]["name"], "two ring single joint 5")
        self.assertEqual(from_file["record"].meta["nest_hidden_ms"], 0.0)


@unittest.skipIf(nest is None, "NEST is not installed")
class ForwardKinematicsParityTests(unittest.TestCase):
    """Phase 4 gate: the forward-kinematics template against legacy ``MultiRingDecode``."""

    ANGLES = (0.9, -1.3)

    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest("requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__))

    def _legacy(self):
        import os

        import multi_ring_component
        import ring_attractor

        for module in (multi_ring_component, ring_attractor):
            module.nest = nest
        cwd = os.getcwd()
        os.chdir(str(LEGACY))
        try:
            nest.ResetKernel()
            nest.SetKernelStatus({"local_num_threads": 1, "rng_seed": SEED})
            decoder = multi_ring_component.MultiRingDecode(reset_kernel=False).build()
            nest.Simulate(300.0)
            before = {name: decoder.get_output_counts(name) for name in ("lift", "pitch", "yaw")}
            decoder.inject_joint_bump(0, angle_rad=self.ANGLES[0])   # 50 ms each, inside inject_stimulus
            decoder.inject_joint_bump(1, angle_rad=self.ANGLES[1])
            nest.Simulate(300.0)
            profiles = {name: decoder.get_output_counts(name) - before[name] for name in before}
            rings = {"q1": decoder.get_joint_ring_counts(0), "q2": decoder.get_joint_ring_counts(1)}
            indices = (decoder._angle_to_center_idx(self.ANGLES[0]), decoder._angle_to_center_idx(self.ANGLES[1]))
        finally:
            os.chdir(cwd)
        return profiles, rings, indices

    def _graph(self):
        graph = two_joint_forward_kinematics(angles=self.ANGLES)
        built = graph.build(BuildContext(nest, rng_seed=SEED, local_num_threads=1))
        nest.Simulate(300.0)
        counts = lambda block: np.asarray(block.population.recorders.get("n_events"), dtype=float)  # noqa: E731
        outputs = {name: graph.blocks[name] for name in ("lift", "pitch", "yaw")}
        before = {name: counts(block) for name, block in outputs.items()}
        enc1, enc2 = graph.blocks["enc_q1"], graph.blocks["enc_q2"]
        enc1.drive_index(enc1.ring_index(self.ANGLES[0]), 0.0, 50.0)
        nest.Simulate(50.0)
        enc1.expire(50.0)
        enc2.drive_index(enc2.ring_index(self.ANGLES[1]), 50.0, 50.0)
        nest.Simulate(50.0)
        enc2.expire(100.0)
        nest.Simulate(300.0)
        profiles = {name: counts(block) - before[name] for name, block in outputs.items()}
        rings = {"q1": counts(graph.blocks["q1"]), "q2": counts(graph.blocks["q2"])}
        return profiles, rings, (enc1.last_index, enc2.last_index), built

    def test_output_ring_profiles_and_decoded_angles_match_the_legacy_stack(self):
        legacy_profiles, legacy_rings, legacy_indices = self._legacy()
        profiles, rings, indices, built = self._graph()
        self.assertEqual(indices, legacy_indices)
        # Build time is reported, not asserted: it is ~5 s idle and load dependent.
        report = {"build_seconds": round(built.build_seconds, 3)}
        for name in ("q1", "q2"):
            _, _, legacy_centroid = ring_readout(legacy_rings[name])
            _, _, centroid = ring_readout(rings[name])
            report[name] = (round(legacy_centroid, 2), round(centroid, 2))
            self.assertLessEqual(abs(legacy_centroid - centroid), 3.0)
        for name in ("lift", "pitch", "yaw"):
            legacy_total = float(np.sum(legacy_profiles[name]))
            total = float(np.sum(profiles[name]))
            l1 = float(np.sum(np.abs(profiles[name] - legacy_profiles[name]))) / max(legacy_total, 1.0)
            legacy_angle = decode_sawtooth_profile(legacy_profiles[name])
            angle = decode_sawtooth_profile(profiles[name])
            error = abs(np.angle(np.exp(1j * (angle - legacy_angle))))
            report[name] = {"legacy_total": legacy_total, "total": total, "l1": round(l1, 3),
                            "legacy_angle": round(float(legacy_angle), 3), "angle": round(float(angle), 3), "angle_error": round(float(error), 3)}
            self.assertGreater(legacy_total, 0.0)
            self.assertLessEqual(abs(total - legacy_total), 0.25 * legacy_total)
            self.assertLessEqual(l1, 0.5)
            self.assertLessEqual(error, 0.35)
        print("FK_PARITY", report)


@unittest.skipIf(nest is None, "NEST is not installed")
class MultiJointRealNestTests(unittest.TestCase):
    """Phase 5 gate on the fakes with real NEST: two joints from two decoders in one trajectory."""

    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest("requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__))

    def test_two_joints_move_toward_their_goals(self):
        # Goals far enough for the comparator to produce drive (docs/blocks/feedback.md: below
        # about 18 ring indices of goal distance the decision pair stays silent).
        goals = (0.6, -1.2)
        graph = multi_joint_two_ring(joints=(5, 6), goals=goals)
        result = run_graph(graph, engines="nest", goals=[goals[0]])[0]
        record = result["record"]
        final = record.final_state["joint_state"]["positions"]
        print("MULTIJOINT steps=%d stop=%s j5=%.4f j6=%.4f" % (record.n_steps, record.stop_reason, final[5], final[6]))
        for joint, goal in zip((5, 6), goals):
            with self.subTest(joint=joint):
                self.assertEqual(np.sign(final[joint]), np.sign(goal))
                self.assertLess(abs(final[joint] - goal), 0.5 * abs(goal))   # at least half way
        start = record.main_ticks[0].inputs["joint_state"]["positions"]
        self.assertTrue(all(abs(final[k] - start[k]) < 1e-6 for k in range(5)))   # uncommanded joints hold
        cmd = record.main_ticks[-1].outputs["arm_velocity_cmd"]
        self.assertEqual([c["joint_index"] for c in cmd["commands"]], [5, 6])
        self.assertIn(record.stop_reason, ("max_steps", "drive_settled"))


if __name__ == "__main__":
    unittest.main()
