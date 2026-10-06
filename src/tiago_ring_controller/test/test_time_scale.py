"""``simulation.time_scale``: NEST steps per robot tick (docs/archive/velocity/speed.md), on the fakes."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tiago_ring_controller.graph import Graph, compile_graph, load_transport_calibration, run_graph_trial, two_ring_single_joint  # noqa: E402
from tiago_ring_controller.config import source_config_path  # noqa: E402
from tiago_ring_controller.graph.calibration import calibration_report  # noqa: E402
from tiago_ring_controller.graph.graph import Simulation  # noqa: E402
from tiago_ring_controller.graph.templates import ARM_MAX_VELOCITY, BOUNDED  # noqa: E402


class SimulationTests(unittest.TestCase):
    def test_validated_and_written_only_when_set(self):
        self.assertNotIn("time_scale", Simulation().to_dict())            # files from before it existed do not change
        self.assertEqual(Simulation(time_scale=3).to_dict()["time_scale"], 3)
        self.assertEqual(Simulation(time_scale=3).validate(), [])
        for bad in (0, -1, 1.5, True):
            self.assertTrue(Simulation(time_scale=bad).validate(), bad)

    def test_graph_file_round_trip(self):
        graph = two_ring_single_joint(joint=5, bounds="bounded", time_scale=2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "g.graph.json")
            graph.save(path)
            self.assertEqual(Graph.load(path).simulation.time_scale, 2)


class TemplateTests(unittest.TestCase):
    def test_bounded_default_is_fast_and_the_legacy_graph_is_untouched(self):
        legacy = two_ring_single_joint(joint=5)
        self.assertEqual(legacy.simulation.time_scale, 1)
        self.assertEqual(legacy.blocks["j5"].params["max_velocity"], 1.0)
        bounded = two_ring_single_joint(joint=5, bounds="bounded")
        self.assertEqual(bounded.simulation.time_scale, BOUNDED["time_scale"])
        self.assertEqual(bounded.blocks["j5"].params["max_velocity"], ARM_MAX_VELOCITY[5])
        self.assertEqual(two_ring_single_joint(joint=5, bounds="bounded", bounded={"time_scale": 2}).simulation.time_scale, 2)

    def test_decoder_dead_zone_and_settle_threshold_scale_and_the_gain_does_not(self):
        artifact = load_transport_calibration(source_config_path("calibration", BOUNDED["transport"]))
        one = two_ring_single_joint(joint=5, bounds="bounded", time_scale=1).blocks["dec"].params
        three = two_ring_single_joint(joint=5, bounds="bounded", time_scale=3).blocks["dec"].params
        self.assertEqual(one["dead_zone"], artifact["dead_zone"])
        self.assertEqual(three["dead_zone"], 3 * artifact["dead_zone"])
        self.assertEqual(three["drive_threshold"], 3 * one["drive_threshold"])
        self.assertEqual((three["gain_positive"], three["tau_s"], three["n_settle"]), (one["gain_positive"], one["tau_s"], one["n_settle"]))

    def test_calibration_report_accepts_the_scaled_dead_zone(self):
        for k in (1, 3):
            row = calibration_report(two_ring_single_joint(joint=5, bounds="bounded", time_scale=k))[0]
            self.assertTrue(row["uses_transport_calibration"] and row["network_matches"] and row["dt_matches"], k)
            self.assertIsNone(row["warning"])


class EngineTests(unittest.TestCase):
    def kernel_time(self, k, steps=6):
        graph = two_ring_single_joint(joint=5, bounds="bounded", calibration="legacy", time_scale=k)
        graph.simulation.max_steps = steps
        graph.blocks["dec"].params["n_settle"] = 10 ** 6
        compiled = compile_graph(graph, engines="fake")
        compiled.loop.initialize()
        try:
            record = run_graph_trial(compiled, 0.4)
            return compiled.nest_engine.kernel_time_ms, record
        finally:
            compiled.loop.shutdown()

    def test_k_nest_steps_per_tick_and_one_datapack_per_tick(self):
        one, record_one = self.kernel_time(1)
        three, record_three = self.kernel_time(3)
        self.assertEqual(three, 3 * one)
        self.assertEqual(record_three.n_steps, record_one.n_steps)
        steps = [t.inputs["r1"].data["nest_step"] for t in record_three.main_ticks if t.inputs.get("r1") is not None]
        self.assertEqual(steps, sorted(set(steps)))                        # one ring datapack per robot tick

    def test_run_graph_refuses_a_time_scale_for_graph_files(self):
        import run_graph

        with self.assertRaises(SystemExit):
            run_graph.load(run_graph.parse_args(["--template", "multi_joint_two_ring", "--time-scale", "2"]))
        graph = run_graph.load(run_graph.parse_args(["--template", "two_ring_single_joint", "--bounds", "bounded", "--time-scale", "2"]))
        self.assertEqual(graph.simulation.time_scale, 2)


if __name__ == "__main__":
    unittest.main()
