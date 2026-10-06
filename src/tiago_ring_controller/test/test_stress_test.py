"""scripts/stress_test.py without ROS or NEST: the random plan, sharded and resumed runs, the report (docs/stress-test.md)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stress_test  # noqa: E402
from tiago_ring_controller.graph.templates import joint_limits  # noqa: E402


class PlanTests(unittest.TestCase):
    def test_same_seed_same_plan_all_joints_inside_the_limits(self):
        plan = stress_test.make_plan(200, 7)
        self.assertEqual(plan, stress_test.make_plan(200, 7))
        self.assertNotEqual(plan["trials"], stress_test.make_plan(200, 8)["trials"])
        self.assertEqual([t["id"] for t in plan["trials"]], list(range(200)))
        self.assertEqual({t["joint"] for t in plan["trials"]}, set(range(7)))
        self.assertEqual((plan["screened"], plan["rejected"]), (False, 0))
        for trial in plan["trials"]:
            for joint, angle in enumerate(trial["pose"]):
                lo, hi = joint_limits(joint)
                self.assertTrue(lo <= angle <= hi)
            lo, hi = joint_limits(trial["joint"])
            self.assertTrue(lo <= trial["goal_rad"] <= hi)
            self.assertAlmostEqual(trial["pose"][trial["joint"]], trial["start_rad"])
            self.assertAlmostEqual(trial["goal_rad"], lo + trial["goal_fraction"] * (hi - lo))
        # nothing is hand-picked: short moves and moves across most of the range both occur
        distances = [abs(t["goal_fraction"] - t["start_fraction"]) for t in plan["trials"]]
        self.assertLess(min(distances), 0.02)
        self.assertGreater(max(distances), 0.8)

    def test_rejected_draws_are_drawn_again(self):
        plan = stress_test.make_plan(40, 3, accept=lambda trial: trial["joint"] != 2)
        self.assertEqual(len(plan["trials"]), 40)
        self.assertTrue(plan["screened"] and plan["rejected"] > 0)
        self.assertNotIn(2, {t["joint"] for t in plan["trials"]})
        with self.assertRaisesRegex(RuntimeError, "only 0 of 5 trials accepted"):
            stress_test.make_plan(5, 3, accept=lambda trial: False, max_draws=50)

    def test_moveit_accept_checks_the_way_to_the_pose_and_the_move(self):
        calls = []

        class Screen:
            def pose_path_is_valid(self, start, end):
                calls.append((list(start), list(end)))
                return len(calls) != 2                               # the move itself collides

        trial = stress_test.make_plan(1, 1)["trials"][0]
        reset = [0.0] * 7
        self.assertFalse(stress_test.moveit_accept(Screen(), reset)(trial))
        goal_pose = list(trial["pose"])
        goal_pose[trial["joint"]] = trial["goal_rad"]
        self.assertEqual(calls, [(reset, trial["pose"]), (trial["pose"], goal_pose)])


class RunTests(unittest.TestCase):
    def test_shards_do_not_overlap_and_a_run_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = str(Path(directory) / "plan.json")
            self.assertEqual(stress_test.main(["plan", "--trials", "5", "--seed", "2", "--out", plan_path]), 0)
            out = str(Path(directory) / "nest_v4")
            common = ["run", "--plan", plan_path, "--system", "F", "--engines", "fake", "--out", out]
            self.assertEqual(stress_test.main(common + ["--shard", "1/2"]), 0)
            first = json.loads((Path(out) / "results_shard1_of_2.json").read_text())
            self.assertEqual([r["trial"] for r in first["trials"]], [0, 2, 4])
            self.assertEqual(stress_test.main(common + ["--shard", "2/2"]), 0)
            rows, failed = stress_test.load_results(out)
            self.assertEqual((sorted(rows), failed), ([0, 1, 2, 3, 4], []))
            plan = json.loads(Path(plan_path).read_text())
            for trial in plan["trials"]:
                row = rows[trial["id"]]
                self.assertEqual((row["joint"], row["system"]), (trial["joint"], "F"))
                self.assertAlmostEqual(row["start_rad"], trial["start_rad"], places=6)      # the arm started in the planned pose
                self.assertNotIn("trace", row)
            stamp = (Path(out) / "results_shard1_of_2.json").stat().st_mtime_ns
            self.assertEqual(stress_test.main(common + ["--shard", "1/2"]), 0)               # nothing left to do
            self.assertEqual(len(json.loads((Path(out) / "results_shard1_of_2.json").read_text())["trials"]), 3)
            self.assertGreaterEqual((Path(out) / "results_shard1_of_2.json").stat().st_mtime_ns, stamp)

    def test_pid_needs_gazebo(self):
        with tempfile.TemporaryDirectory() as directory:
            plan_path = str(Path(directory) / "plan.json")
            stress_test.main(["plan", "--trials", "1", "--out", plan_path])
            with self.assertRaisesRegex(SystemExit, "only runs against Gazebo"):
                stress_test.main(["run", "--plan", plan_path, "--system", "PID", "--engines", "nest", "--out", directory])


def row(trial, system, error, **extra):
    lo, hi = joint_limits(trial["joint"])
    final = trial["goal_rad"] - error
    result = {"trial": trial["id"], "system": system, "joint": trial["joint"], "goal_rad": trial["goal_rad"], "start_rad": trial["start_rad"],
              "final_rad": final, "goal_error_rad": error, "time_to_90_s": 1.5, "settling_time_s": 2.0, "reached_tolerance": abs(error) < 0.05,
              "overshoot_rad": 0.01, "peak_speed": 0.5, "no_drive": False, "wrong_direction": False, "clamp_ticks": 0, "n_steps": 100,
              "rho": stress_test.radians_per_index(trial["joint"]), "start_fraction": trial["start_fraction"],
              "goal_fraction": trial["goal_fraction"], "planned_start_rad": trial["start_rad"], "pose": trial["pose"], "base_shift_m": 0.0}
    result.update(extra)
    return result


class ReportTests(unittest.TestCase):
    def test_bins(self):
        self.assertEqual([stress_test.distance_bin(d) for d in (0.0, 0.049, 0.05, 0.2, 0.49, 0.5, 1.0)], [0, 0, 1, 2, 3, 4, 4])
        rows = [{"joint": 0, "distance": 0.02, "error": 0.01, "rho": 0.02, "start_fraction": 0.1, "goal_fraction": 0.12},
                {"joint": 0, "distance": 0.02, "error": -0.03, "rho": 0.02, "start_fraction": 0.1, "goal_fraction": 0.12},
                {"joint": 6, "distance": 0.9, "error": 0.2, "rho": 0.04, "start_fraction": 0.05, "goal_fraction": 0.95}]
        grid, count = stress_test.joint_distance_grid(rows)
        self.assertAlmostEqual(grid[0, 0], 0.02)
        self.assertAlmostEqual(grid[6, 4], 0.2)
        self.assertEqual((int(count.sum()), int(np.isfinite(grid).sum())), (3, 2))
        grid, count = stress_test.start_goal_grid(rows, 5)
        self.assertAlmostEqual(grid[0, 0], 1.0)                      # median of 0.5 and 1.5 ring indices
        self.assertAlmostEqual(grid[0, 4], 5.0)
        self.assertEqual(int(count[0, 0]), 2)

    def test_tables_from_result_folders(self):
        plan = stress_test.make_plan(30, 4)
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "nest_plan.json").write_text(json.dumps(plan))
            for label, system, error in (("v3", "B", 0.1), ("v4", "F", 0.01)):
                (data / ("nest_" + label)).mkdir()
                rows = [row(t, system, error if t["id"] % 2 else -error, lost_bump=False) for t in plan["trials"]]
                rows[3]["goal_error_rad"] = 0.3 if system == "F" else rows[3]["goal_error_rad"]
                (data / ("nest_" + label) / "results.json").write_text(json.dumps({"trials": rows, "failed": []}))
            out = data / "report"
            self.assertEqual(stress_test.main(["report", "--data", str(data), "--out", str(out), "--no-figures"]), 0)
            text = (out / "tables.md").read_text()
        self.assertIn("No results under `gazebo_*`.", text)
        self.assertIn("Plan: 30 trials, seed 4, not screened", text)
        self.assertIn("| v3 (two rings) | 30 | 0.100 / 0.100 / 0.100 |", text)
        self.assertIn("| v4 (three rings) | 30 | 0.010 / 0.010 / 0.300 |", text)
        worst = text.split("### The ten largest v4 errors")[1].splitlines()[4]
        self.assertTrue(worst.startswith("| 3 | arm_%d |" % (plan["trials"][3]["joint"] + 1)))


if __name__ == "__main__":
    unittest.main()
