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
from tiago_ring_controller.control.trajectory import build_multi_joint_trajectory, build_receding_trajectory  # noqa: E402
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
    multi_joint_two_ring,
    resolve_joint_limits,
    run_graph,
    three_ring_single_joint,
    two_joint_forward_kinematics,
    two_ring_single_joint,
)
from tiago_ring_controller.graph.compile import (  # noqa: E402
    ArmCommandTF,
    DecoderTF,
    GoalBlockTF,
    JointCommandTF,
    JointSensorTF,
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
        self.assertEqual(names, ["JointSensorTF", "GoalBlockTF", "DecoderTF", "JointCommandTF", "ArmCommandTF"])
        self.assertEqual([e.name for e in compiled.loop.engines], ["nest", "robot"])
        self.assertEqual(compiled.nest_engine.inputs, {"j5", "goal"})
        self.assertIn("ring_counts", compiled.nest_engine.outputs)
        self.assertEqual(compiled.nest_engine.legacy_view, {"state_ring": "r1", "gain": "gain", "goal_ring": "r2"})
        goal_tf = next(tf for tf in compiled.tfs if isinstance(tf, GoalBlockTF))
        self.assertEqual(goal_tf.outputs, {"goal", "goal_bump"})
        sensor_tf = next(tf for tf in compiled.tfs if isinstance(tf, JointSensorTF))
        self.assertEqual(sensor_tf.outputs, {"j5", "state_bump"})
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
        self.assertEqual((command.inputs, command.outputs), ({"dec", "ring_counts"}, {"j5.command"}))
        merge = next(tf for tf in compiled.tfs if isinstance(tf, ArmCommandTF))
        self.assertEqual((merge.inputs, merge.outputs, merge.primary_id), ({"j5.command"}, {"arm_velocity_cmd"}, "j5.command"))
        with self.assertRaisesRegex(GraphError, "engines must be"):
            compile_graph(graph, engines="nope")

    def test_forward_kinematics_compiles_without_the_legacy_view(self):
        compiled = compile_graph(two_joint_forward_kinematics(), engines="fake")
        names = [type(tf).__name__ for tf in compiled.tfs]
        self.assertEqual(names[:2], ["GoalBlockTF", "GoalBlockTF"])
        self.assertEqual(names.count("ProfileDecoderTF"), 3)
        self.assertIsNone(compiled.nest_engine.legacy_view)
        self.assertNotIn("ring_counts", compiled.nest_engine.outputs)
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
            tick = record.main_ticks[0]
            self.assertIn("ring_counts", tick.inputs)            # engine datapack, like the legacy loop
            self.assertIn("arm_velocity_cmd", tick.outputs)
            self.assertEqual(tick.outputs["goal_bump"]["goal_rad"], 0.3)
            first = record.ticks[0]                              # the first lead tick carries the initial state bump
            self.assertIn("center_index", first.outputs["state_bump"])
            self.assertNotIn("state_bump", tick.outputs)
            self.assertEqual(sorted(k for k in results[0]["legacy"]["raster"]), sorted(
                ["r1_times", "r1_senders", "r2_times", "r2_senders", "left_times", "left_senders", "right_times", "right_senders"]))
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


URDF = """<robot name="tiago">
  <joint name="arm_6_joint" type="revolute"><limit lower="-2.0" upper="2.0"/></joint>
  <joint name="arm_7_joint" type="revolute"><limit lower="-1.39" upper="1.39"/></joint>
</robot>"""   # arm joint index 6 is arm_7_joint


class MultiJointTests(unittest.TestCase):
    def test_multi_joint_trajectory_equals_single_joint_and_integrates_each_joint(self):
        base = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.assertEqual(build_receding_trajectory(base, 5, [0.1, 0.2, 0.3], 0.05),
                         build_multi_joint_trajectory(base, {5: [0.1, 0.2, 0.3]}, 0.05))
        result = build_multi_joint_trajectory(base, {5: [0.1, 0.2], 6: [-0.1]}, 0.05)
        self.assertEqual(len(result.points), 2)
        self.assertAlmostEqual(result.points[0].positions[5], 0.605)
        self.assertAlmostEqual(result.points[0].positions[6], 0.695)
        self.assertAlmostEqual(result.points[1].positions[5], 0.615)
        self.assertAlmostEqual(result.points[1].positions[6], 0.695)   # padded with zero velocity
        self.assertEqual(result.points[1].velocities[6], 0.0)
        self.assertAlmostEqual(result.next_commanded_positions[6], 0.695)
        self.assertIsNone(build_multi_joint_trajectory(base, {5: []}, 0.05))

    def test_multi_joint_template_compiles_and_commands_both_joints_on_the_fakes(self):
        graph = multi_joint_two_ring(joints=(5, 6), goals=(0.6, -0.4))
        self.assertEqual(len(graph.blocks), 22)
        compiled = compile_graph(graph, engines="fake")
        names = [type(tf).__name__ for tf in compiled.tfs]
        self.assertEqual(names.count("JointCommandTF"), 2)
        self.assertEqual(names[-1], "ArmCommandTF")
        self.assertEqual((compiled.primary.joint.id, compiled.primary.decoder.id), ("j5", "dec_j5"))
        self.assertEqual(compiled.config.joint_index, 5)
        results = run_graph(graph, engines="fake", goals=[0.6])
        record = results[0]["record"]
        cmd = record.main_ticks[0].outputs["arm_velocity_cmd"]
        self.assertEqual([c["joint_index"] for c in cmd["commands"]], [5, 6])
        self.assertEqual(cmd["joint_index"], 5)
        self.assertEqual(len(cmd["velocities"]), 4)
        self.assertIn("legacy", results[0])
        self.assertEqual(results[0]["legacy"]["scalars"]["q_goal"], 0.6)

    def test_fake_robot_integrates_two_joints_and_settles_both(self):
        from tiago_ring_controller.cosim.datapack import DataPack
        from tiago_ring_controller.cosim.fakes import FakeRobotEngine

        robot = FakeRobotEngine("robot", home=[0.0] * 7)
        robot.initialize()
        robot.reset()
        for _ in range(20):
            robot.set_datapacks({"arm_velocity_cmd": DataPack("arm_velocity_cmd", robot.t_ms, {
                "joint_index": 5, "velocities": [0.2] * 4, "dt_s": 0.05,
                "commands": [{"joint_index": 5, "velocities": [0.2] * 4}, {"joint_index": 6, "velocities": [-0.1] * 4}],
            })})
            robot.advance(50.0)
        positions = robot.get_datapacks()["joint_state"]["positions"]
        self.assertGreater(positions[5], 0.15)
        self.assertLess(positions[6], -0.07)
        self.assertEqual(robot.published[-1]["joint_indices"], [5, 6])
        self.assertEqual(sum(abs(p) > 1e-9 for p in positions), 2)
        robot.finish_trial()
        self.assertLess(abs(robot.velocities[5]), 0.01)
        self.assertLess(abs(robot.velocities[6]), 0.01)

    def test_gazebo_engine_publishes_both_joints_in_one_trajectory(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("engine_tests", ROOT / "test/test_cosim_engines.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from tiago_ring_controller.cosim.datapack import DataPack
        from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine

        transport = module.FakeTransport()
        engine = GazeboRosEngine(transport, stepper_name="clock_wait")
        engine.initialize()
        engine.reset()
        engine.set_datapacks({"arm_velocity_cmd": DataPack("arm_velocity_cmd", 0.0, {
            "joint_index": 5, "velocities": [0.2, 0.2], "dt_s": 0.05,
            "commands": [{"joint_index": 5, "velocities": [0.2, 0.2]}, {"joint_index": 6, "velocities": [-0.4]}],
        })})
        engine.advance(50.0)
        _, points, _ = transport.published[-1]
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].positions[5] - transport.positions[5], 0.0, places=2)  # transport moved to point 0
        self.assertAlmostEqual(points[0].velocities[5], 0.2)
        self.assertAlmostEqual(points[0].velocities[6], -0.4)
        self.assertAlmostEqual(points[1].velocities[6], 0.0)
        self.assertEqual(engine.published[-1]["joint_indices"], [5, 6])
        engine.finish_trial()
        engine.shutdown()

    def test_urdf_limits_follow_into_the_encoders(self):
        graph = two_ring_single_joint(joint=6)
        graph.blocks["j6"].params["limits_source"] = "urdf"
        old = graph.blocks["j6"].limits()
        self.assertEqual(graph.blocks["enc_state"].params["joint_min"], old[0])

        class Transport:
            def get_param(self, name):
                return URDF if name == "/robot_description" else None

        changed = resolve_joint_limits(graph, Transport())
        self.assertEqual(changed["j6"]["to"], (-1.39, 1.39))
        self.assertEqual(sorted(changed["j6"]["followers"]), ["enc_goal", "enc_state"])
        self.assertEqual(graph.blocks["enc_goal"].params["joint_max"], 1.39)
        self.assertEqual(graph.blocks["j6"].limits(), (-1.39, 1.39))
        graph.blocks["j6"].params["index"] = 3
        with self.assertRaisesRegex(GraphError, "no URDF limits for arm joint 3"):
            resolve_joint_limits(graph, Transport())
        untouched = two_ring_single_joint(joint=6)
        self.assertEqual(resolve_joint_limits(untouched, Transport()), {})


if __name__ == "__main__":
    unittest.main()
