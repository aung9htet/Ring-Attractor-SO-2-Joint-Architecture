"""The velocity command path on the fakes: limits, velocity cap, measured velocity, record alignment.

See docs/archive/velocity-plan.md (part 1.3) for the defects these tests pin.
"""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tiago_ring_controller.blocks import Decoder, Goal, Joint  # noqa: E402
from tiago_ring_controller.control.trajectory import (  # noqa: E402
    CommandState,
    build_multi_joint_trajectory,
    build_receding_trajectory,
)
from tiago_ring_controller.cosim import DataPack, FakeRobotEngine, TickRecord  # noqa: E402
from tiago_ring_controller.cosim.fakes import command_limits, measured_velocity  # noqa: E402
from tiago_ring_controller.cosim.tf import TickContext  # noqa: E402
from tiago_ring_controller.graph import Graph, compile_graph, run_graph, two_ring_single_joint  # noqa: E402
from tiago_ring_controller.graph.compile import JointCommandTF  # noqa: E402


class TrajectoryLimitTests(unittest.TestCase):
    BASE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def test_positions_are_held_at_a_limit_and_the_velocity_is_the_one_travelled(self):
        result = build_multi_joint_trajectory(self.BASE, {5: [1.0] * 4}, 0.05, limits={5: (-0.7, 0.68)})
        np.testing.assert_allclose([p.positions[5] for p in result.points], [0.65, 0.68, 0.68, 0.68])
        np.testing.assert_allclose([p.velocities[5] for p in result.points], [1.0, 0.6, 0.0, 0.0])
        self.assertAlmostEqual(result.next_commanded_positions[5], 0.65)
        self.assertEqual(result.clamped_joints, (5,))
        lower = build_multi_joint_trajectory(self.BASE, {5: [-40.0]}, 0.05, limits={5: (-0.7, 0.68)})
        self.assertEqual(lower.points[0].positions[5], -0.7)
        self.assertEqual(lower.next_commanded_positions[5], -0.7)

    def test_without_a_limit_in_reach_nothing_changes(self):
        free = build_multi_joint_trajectory(self.BASE, {5: [0.1, 0.2, 0.3]}, 0.05)
        limited = build_multi_joint_trajectory(self.BASE, {5: [0.1, 0.2, 0.3]}, 0.05, limits={5: (-1.0, 1.0), 2: (0.0, 0.1)})
        self.assertEqual(free, limited)
        self.assertEqual(limited.clamped_joints, ())
        self.assertEqual(free, build_receding_trajectory(self.BASE, 5, [0.1, 0.2, 0.3], 0.05))

    def test_a_joint_outside_its_range_is_not_pulled_back_only_kept_from_going_further(self):
        outward = build_multi_joint_trajectory(self.BASE, {6: [1.0, 1.0]}, 0.05, limits={6: (-0.5, 0.65)})
        self.assertEqual([p.positions[6] for p in outward.points], [0.7, 0.7])
        self.assertEqual(outward.clamped_joints, (6,))
        inward = build_multi_joint_trajectory(self.BASE, {6: [-1.0, -1.0]}, 0.05, limits={6: (-0.5, 0.65)})
        np.testing.assert_allclose([p.positions[6] for p in inward.points], [0.65, 0.6])
        self.assertEqual(inward.clamped_joints, ())
        resting = build_multi_joint_trajectory(self.BASE, {6: [0.0]}, 0.05, limits={6: (-0.5, 0.65)})
        self.assertEqual((resting.points[0].positions[6], resting.clamped_joints), (0.7, ()))

    def test_command_state_consumes_one_clamped_step(self):
        state = CommandState()
        state.prime(self.BASE)
        for _ in range(5):
            state.build_multi({5: [1.0] * 4}, 0.05, limits={5: (-0.7, 0.68)})
        self.assertAlmostEqual(state.commanded_positions[5], 0.68)
        self.assertEqual(state.commanded_velocities[5], 0.0)


class FakeRobotLimitTests(unittest.TestCase):
    def test_fake_robot_enforces_the_limits_of_the_command(self):
        robot = FakeRobotEngine("robot")
        robot.initialize()
        robot.reset()
        command = {"joint_index": 5, "velocities": [-0.5] * 4, "dt_s": 0.05,
                   "commands": [{"joint_index": 5, "velocities": [-0.5] * 4, "limits": [-0.2, 0.2]}]}
        self.assertEqual(command_limits(command), {5: (-0.2, 0.2)})
        self.assertEqual(command_limits({"joint_index": 5, "velocities": [0.1]}), {})
        for _ in range(30):
            robot.set_datapacks({"arm_velocity_cmd": DataPack("arm_velocity_cmd", robot.t_ms, command)})
            robot.advance(50.0)
            state = robot.get_datapacks()["joint_state"]
            self.assertGreaterEqual(state["commanded_positions"][5], -0.2)
            self.assertGreaterEqual(state["positions"][5], -0.2)
        self.assertEqual(state["clamped_joints"], [5])
        self.assertAlmostEqual(state["positions"][5], -0.2, places=3)
        robot.advance(50.0)
        self.assertEqual(robot.get_datapacks()["joint_state"]["clamped_joints"], [])

    def test_measured_velocity_prefers_the_finite_difference_estimate(self):
        self.assertEqual(measured_velocity({"velocities": [1.0, 2.0], "velocities_fd": [3.0, 4.0]}, 1), 4.0)
        self.assertEqual(measured_velocity({"velocities": [1.0, 2.0]}, 1), 2.0)
        self.assertEqual(measured_velocity({"positions": [0.0]}, 0), 0.0)
        joint = Joint("j", index=1)
        self.assertEqual(joint.read({"positions": [0.5, 0.25], "velocities": [-1.45, -1.45], "velocities_fd": [0.0, 0.1]}),
                         {"angle": 0.25, "velocity_measured": 0.1})
        self.assertEqual(joint.read({"positions": [0.5, 0.25]})["velocity_measured"], 0.0)


class GraphCommandLimitTests(unittest.TestCase):
    """A signal-only graph whose decoder asks for far too much: the joint stops at its limit."""

    def graph(self):
        g = Graph("limit test", dt_ms=50.0, nest_lead_steps=0, max_steps=40)
        goal = g.add(Goal("goal", angle_rad=-50.0))
        sense = g.add(Joint("sense", index=0))
        dec = g.add(Decoder("dec", gain_positive=0.5, gain_negative=-0.5, tau_s=0.05, horizon=4))
        moved = g.add(Joint("moved", index=2, joint_min=-0.3, joint_max=0.2, max_velocity=0.4))
        g.connect(goal.angle, dec.left_counts)       # signed input = sense.angle - goal.angle = +50
        g.connect(sense.angle, dec.right_counts)
        g.connect(dec.velocity, moved.velocity)
        return g

    def test_velocity_is_capped_and_the_commanded_position_stops_at_the_joint_limit(self):
        record = run_graph(self.graph(), engines="fake", goals=[-50.0])[0]["record"]
        self.assertEqual(record.stop_reason, "max_steps")
        clipped, held = [], []
        for tick in record.main_ticks:
            cmd = tick.outputs["arm_velocity_cmd"]
            self.assertLessEqual(max(abs(v) for v in cmd["velocities"]), 0.4)
            self.assertEqual(cmd["commands"][0]["limits"], [-0.3, 0.2])
            clipped.append(cmd["velocity_clamped"])
            state = tick.inputs["joint_state"]
            self.assertLessEqual(state["commanded_positions"][2], 0.2)
            self.assertLessEqual(state["positions"][2], 0.2)
            held.append(state["clamped_joints"])
        self.assertTrue(all(clipped[1:]))            # the decoder asked for up to 25 rad/s
        self.assertEqual(held[-1], [2])
        self.assertEqual(held[1], [])                # far from the limit at first
        final = record.final_state["joint_state"]["positions"]
        self.assertAlmostEqual(final[2], 0.2, places=3)
        self.assertEqual(final[0], 0.0)              # the uncommanded joint held

    def test_an_ordinary_command_passes_unchanged(self):
        graph = self.graph()
        graph.blocks["dec"].params.update(gain_positive=0.002, gain_negative=-0.002)   # 0.1 rad/s
        graph.simulation.max_steps = 10
        record = run_graph(graph, engines="fake", goals=[-50.0])[0]["record"]
        last = record.main_ticks[-1]
        self.assertFalse(last.outputs["arm_velocity_cmd"]["velocity_clamped"])
        self.assertEqual(last.inputs["joint_state"]["clamped_joints"], [])
        self.assertAlmostEqual(last.outputs["arm_velocity_cmd"]["velocities"][-1], 0.1, places=3)


class RecordAlignmentTests(unittest.TestCase):
    def test_ring_fields_of_the_consumed_sample_come_from_its_own_nest_step(self):
        compiled = compile_graph(two_ring_single_joint(), engines="fake")
        tf = next(tf for tf in compiled.tfs if isinstance(tf, JointCommandTF))
        tf.reset()
        ctx = TickContext(t_ms=0.0, tick=0)

        def ring(step):
            return DataPack("ring_counts", 0.0, {"nest_step": step, "r1_spike_count": 20.0 + step,
                                                 "r1_bump_index": 100 + step, "r1_centroid": 100.5 + step})

        def decoded(step):
            return DataPack("dec", 0.0, {"velocity": [0.1, 0.2, 0.3, 0.4], "settled": False, "consecutive_settled": 0,
                                         "consumed": {"nest_step": step, "signed_spike": 1.0}})

        self.assertEqual(tf({"ring_counts": ring(1)}, ctx), {})           # lead: no horizon yet
        for newest in (2, 3):                                             # lead: the horizon still starts at step 1
            tf({"ring_counts": ring(newest), "dec": decoded(1)}, ctx)
        for consumed_step in (1, 2, 3):                                   # main ticks: NEST is three steps ahead
            pack = tf({"ring_counts": ring(consumed_step + 3), "dec": decoded(consumed_step)}, ctx)["j5.command"]
            consumed = pack["consumed"]
            self.assertEqual(consumed["nest_step"], consumed_step)
            self.assertEqual((consumed["r1_spike_count"], consumed["r1_bump_index"], consumed["r1_centroid"]),
                             (20.0 + consumed_step, 100 + consumed_step, 100.5 + consumed_step))
        self.assertEqual(sorted(tf._ring_views), [3, 4, 5, 6])            # older views are dropped
        self.assertEqual(pack["limits"], list(compiled.primary.joint.limits()))
        tf.reset()
        self.assertEqual(tf._ring_views, {})

    def test_feedback_sweep_analysis_reads_the_engine_datapack(self):
        archive = str(Path(__file__).resolve().parents[1] / "scripts" / "archive")
        if archive not in sys.path:
            sys.path.insert(0, archive)
        import sweep_feedback

        graph = two_ring_single_joint()
        run_graph(graph, engines="fake", goals=[0.4])                     # builds the encoder (ring size)
        encoder = graph.blocks["enc_state"]
        encoder._joint_index = 5
        ticks = [
            TickRecord(
                tick=k, phase="main", t_ms=50.0 * k,
                inputs={"ring_counts": DataPack("ring_counts", 0.0, {"r1_centroid": 100.0 + 0.5 * k}),
                        "joint_state": DataPack("joint_state", 0.0, {"positions": [0.0] * 5 + [0.005 * k, 0.0]})},
                outputs={}, advanced=("nest", "robot"),
            )
            for k in range(30)
        ]

        class Record:
            main_ticks = ticks

        metrics = sweep_feedback.analyse(Record, encoder)
        self.assertEqual(metrics["samples"], 30)
        self.assertTrue(np.isfinite(metrics["tracking_error"]))
        self.assertIsNotNone(metrics["lag_ticks"])


if __name__ == "__main__":
    unittest.main()
