"""Graph files, compiler and runner on the fakes (phase 4)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.blocks import Decoder, Encoder, Goal, Joint, Ring  # noqa: E402
from tiago_ring_controller.cosim.fake_backend import FakeNestBackend  # noqa: E402
from tiago_ring_controller.graph import (  # noqa: E402
    EXAMPLES_DIR,
    TEMPLATES,
    Graph,
    GraphError,
    compile_graph,
    dumps,
    find_primary,
    graph_cosim_config,
    loads,
    run_graph,
    three_ring_single_joint,
    two_joint_forward_kinematics,
    two_ring_single_joint,
)
from tiago_ring_controller.graph.compile import (  # noqa: E402
    DecoderTF,
    GoalBlockTF,
    JointCommandTF,
    JointSensorTF,
    LegacyViewTF,
    ProfileDecoderTF,
)


class SchemaRoundTripTests(unittest.TestCase):
    def test_templates_round_trip_byte_identically_and_match_the_example_files(self):
        for name, template in TEMPLATES.items():
            with self.subTest(template=name):
                graph = template()
                text = dumps(graph)
                again = loads(text)
                self.assertEqual(dumps(again), text)
                self.assertEqual(again.describe(), graph.describe())
                example = (Path(EXAMPLES_DIR) / (name + ".graph.json")).read_text(encoding="utf-8")
                self.assertEqual(example, text, "%s.graph.json is stale: run write_examples()" % name)
                document = json.loads(text)
                self.assertEqual(document["schema"], "ring-blocks/1")
                self.assertEqual([b["id"] for b in document["blocks"]], list(graph.declared))
                for block in document["blocks"]:
                    self.assertEqual(list(block), ["id", "type", "params"] + (["ui"] if "ui" in block else []))

    def test_ui_is_kept_and_composites_stay_one_entry(self):
        graph = three_ring_single_joint()
        graph.declared["jt"].ui = {"x": 10, "y": 20}
        document = json.loads(dumps(graph))
        entry = next(b for b in document["blocks"] if b["id"] == "jt")
        self.assertEqual((entry["type"], entry["ui"]), ("JointTriple", {"x": 10, "y": 20}))
        self.assertNotIn("jt__B", [b["id"] for b in document["blocks"]])
        self.assertIn({"from": "goal.angle", "to": "jt.goal_angle"}, document["edges"])
        again = loads(json.dumps(document))
        self.assertEqual(again.declared["jt"].ui, {"x": 10, "y": 20})
        self.assertEqual(len(again.blocks), len(graph.blocks))

    def test_file_problems_are_reported_together(self):
        document = {
            "schema": "ring-blocks/0",
            "bogus": 1,
            "simulation": {"dt_ms": 50, "nope": 1},
            "blocks": [
                {"id": "r1", "type": "Ring", "params": {"population_size": 2}},
                {"id": "x", "type": "Nope"},
                {"id": "j", "type": "Joint", "params": {}, "extra": 1},
                {"id": "j", "type": "Joint", "params": {}},
                "not a block",
            ],
            "edges": [{"from": "r1.spikes", "to": "zz.ring"}, {"from": "j.nope", "to": "r1.stim"}, {"from": "j.angle"}],
            "robot": {"engine": "fake"},
        }
        with self.assertRaises(GraphError) as raised:
            loads(json.dumps(document))
        text = str(raised.exception)
        for needle in (
            "schema must be 'ring-blocks/1'", "unknown top-level keys ['bogus']", "unknown simulation keys ['nope']",
            "Ring 'r1': population_size: 2 is below the minimum 3", "unknown block type 'Nope'",
            "block 'j': unknown keys ['extra']", "duplicate block id 'j'", "blocks[4]: needs 'id' and 'type'",
            "unknown block 'r1' in port reference 'r1.spikes'", "unknown block 'zz' in port reference 'zz.ring'",
            "Joint has no port 'nope'", "edges[2]: needs 'from' and 'to'",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)
        with self.assertRaisesRegex(GraphError, "not valid JSON"):
            loads("{")
        with self.assertRaisesRegex(GraphError, "must be a JSON object"):
            loads("[]")

    def test_save_and_load_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "g.graph.json")
            graph = two_ring_single_joint(joint=3, goal_rad=0.2)
            graph.save(path)
            again = Graph.load(path)
            self.assertEqual(again.name, "two ring single joint 3")
            self.assertEqual(again.blocks["goal"].params["angle_rad"], 0.2)
            self.assertEqual(again.blocks["j3"].index, 3)


class CompilerTests(unittest.TestCase):
    def test_single_joint_compiles_to_the_expected_transceivers(self):
        graph = two_ring_single_joint()
        compiled = compile_graph(graph, engines="fake")
        names = [type(tf).__name__ for tf in compiled.tfs]
        self.assertEqual(names, ["LegacyViewTF", "JointSensorTF", "GoalBlockTF", "DecoderTF", "JointCommandTF"])
        self.assertEqual([e.name for e in compiled.loop.engines], ["nest", "robot"])
        self.assertEqual(compiled.nest_engine.inputs, {"j5", "goal"})
        self.assertEqual(compiled.loop.lead_engine, "nest")
        primary = compiled.primary
        self.assertTrue(primary.complete)
        self.assertEqual((primary.state_ring.id, primary.goal_ring.id, primary.gain.id, primary.decoder.id, primary.joint.id),
                         ("r1", "r2", "gain", "dec", "j5"))
        self.assertEqual((primary.state_encoder.id, primary.goal_encoder.id), ("enc_state", "enc_goal"))
        config = compiled.config
        self.assertEqual((config.joint_index, config.profile, config.nest_model, config.nest_lead_steps), (5, "collector", "vectorised", 4))
        self.assertAlmostEqual(config.joint_min, -1.4172695167359413)
        self.assertEqual(config.decoder_gain_positive, graph.blocks["dec"].params["gain_positive"])
        decoder_tf = next(tf for tf in compiled.tfs if isinstance(tf, DecoderTF))
        self.assertEqual(decoder_tf.sources, {"left_counts": ("gain", "left_counts"), "right_counts": ("gain", "right_counts")})
        self.assertEqual(decoder_tf.inputs, {"gain"})
        command = next(tf for tf in compiled.tfs if isinstance(tf, JointCommandTF))
        self.assertEqual((command.inputs, command.outputs), ({"dec"}, {"arm_velocity_cmd"}))
        with self.assertRaisesRegex(GraphError, "engines must be"):
            compile_graph(graph, engines="nope")

    def test_forward_kinematics_compiles_without_the_legacy_view(self):
        compiled = compile_graph(two_joint_forward_kinematics(), engines="fake")
        names = [type(tf).__name__ for tf in compiled.tfs]
        self.assertEqual(names[:2], ["GoalBlockTF", "GoalBlockTF"])
        self.assertEqual(names.count("ProfileDecoderTF"), 3)
        self.assertNotIn("LegacyViewTF", names)
        self.assertFalse(compiled.primary.complete)
        self.assertIsNone(compiled.loop.lead_engine)
        self.assertEqual(compiled.nest_engine.inputs, {"angle_q1", "angle_q2"})

    def test_find_primary_on_a_graph_without_the_motif(self):
        g = Graph("signals only")
        goal, joint, dec = g.add(Goal("goal")), g.add(Joint("j")), g.add(Decoder("dec"))
        g.connect(goal.angle, dec.left_counts)
        g.connect(joint.angle, dec.right_counts)
        g.connect(dec.velocity, g.add(Joint("j2", index=2)).velocity)
        primary = find_primary(g)
        self.assertEqual((primary.decoder.id, primary.joint.id, primary.gain), ("dec", "j2", None))
        self.assertFalse(primary.complete)
        config = graph_cosim_config(g)
        self.assertEqual(config.joint_index, 2)
        compiled = compile_graph(g, engines="fake")
        self.assertIsNone(compiled.nest_engine)
        self.assertEqual([e.name for e in compiled.loop.engines], ["robot"])


class RunnerTests(unittest.TestCase):
    def test_all_templates_run_on_the_fakes_and_record_the_graph(self):
        for name, template in TEMPLATES.items():
            with self.subTest(template=name):
                results = run_graph(template(), engines="fake", goals=[0.4])
                record = results[0]["record"]
                self.assertEqual(record.meta["graph"]["name"], template().name)
                self.assertIn(record.stop_reason, ("drive_settled", "max_steps"))
                self.assertGreater(record.n_steps, 0)
                self.assertEqual(record.meta["nest_build"]["order"][0], record.meta["graph"]["blocks"][0]["id"])

    def test_single_joint_run_writes_the_collector_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            results = run_graph(Path(EXAMPLES_DIR) / "two_ring_single_joint.graph.json", engines="fake", goals=[0.3, 0.5], out_dir=directory)
            self.assertEqual(len(results), 2)
            legacy = results[0]["legacy"]
            self.assertEqual(legacy["scalars"]["q_goal"], 0.3)
            self.assertEqual(results[1]["legacy"]["scalars"]["q_goal"], 0.5)
            self.assertEqual(sorted(os.listdir(directory)), ["session_meta.json", "trials", "trials_summary.csv"])
            self.assertIn("trial_0002_cosim.json", os.listdir(os.path.join(directory, "trials")))
            record = results[0]["record"]
            self.assertIn("ring_counts", record.main_ticks[0].outputs)
            self.assertIn("arm_velocity_cmd", record.main_ticks[0].outputs)
            self.assertEqual(record.main_ticks[0].outputs["arm_velocity_cmd"]["joint_index"], 5)
            self.assertIsNotNone(record.meta["goal_ring_index"])
            self.assertIsNotNone(record.meta["initial_ring_index"])
            self.assertEqual(record.meta["nest_hidden_ms"], 0.0)

    def test_forward_kinematics_run_writes_json_records(self):
        with tempfile.TemporaryDirectory() as directory:
            results = run_graph(two_joint_forward_kinematics(angles=(0.3, -0.2)), engines="fake", goals=[0.0], out_dir=directory)
            self.assertNotIn("legacy", results[0])
            self.assertEqual(os.listdir(directory), ["trial_0001_graph.json"])
            record = results[0]["record"]
            self.assertEqual(record.stop_reason, "max_steps")
            last = record.main_ticks[-1].outputs
            self.assertEqual(sorted(k for k in last if k.startswith("decode_")), ["decode_lift", "decode_pitch", "decode_yaw"])
            self.assertTrue(np.isnan(last["decode_lift"]["angle"]))   # fake NEST: no spikes


if __name__ == "__main__":
    unittest.main()
