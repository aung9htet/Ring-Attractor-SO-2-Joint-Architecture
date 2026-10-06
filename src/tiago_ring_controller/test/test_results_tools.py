"""The comparison tools without ROS or NEST: base restore and reset retries of the Gazebo engine,
the report on fixture data, the video grid, the blind-band measure (docs/results.md)."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts", ROOT / "test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fit_comparator  # noqa: E402
import results_report  # noqa: E402
from test_cosim_engines import FakeTransport  # noqa: E402
from tiago_ring_controller.cosim.engine import EngineError  # noqa: E402
from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine  # noqa: E402
from tiago_ring_controller.graph import calibration_report, three_ring_single_joint, two_ring_single_joint  # noqa: E402


class BaseTransport(FakeTransport):
    """A fake simulation whose robot base can be read and set; ``states`` are the play_motion results in order."""

    def __init__(self, states=("SUCCEEDED",), **kwargs):
        super().__init__(**kwargs)
        self.states = list(states)
        self.base = (-0.2, 0.01, 0.1)

    def play_motion(self, motion_name, timeout_s):
        super().play_motion(motion_name, timeout_s)
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def model_pose(self, name):
        return self.base

    def set_model_pose(self, name, x, y, yaw):
        self.calls.append(("set_model_pose", name, x, y, yaw))
        self.base = (x, y, yaw)
        return True


def engine_for(transport, **kwargs):
    engine = GazeboRosEngine(transport, max_step_size_s=0.001, **kwargs)
    engine.initialize()
    return engine


class GazeboResetTests(unittest.TestCase):
    def test_reset_puts_the_base_back_and_logs_where_it_was(self):
        transport = BaseTransport()
        engine = engine_for(transport)
        engine.reset()
        self.assertIn(("set_model_pose", "tiago", 0.0, 0.0, 0.0), transport.calls)
        self.assertEqual(engine.base_log["found"], (-0.2, 0.01, 0.1))
        self.assertEqual(engine.base_log["start"], (0.0, 0.0, 0.0))
        transport.base = (0.03, 0.04, 0.0)                      # the arm pushed the robot during the trial
        engine.finish_trial()
        shift = engine.base_shift()
        self.assertAlmostEqual(shift["trial_m"], 0.05)
        self.assertAlmostEqual(shift["found_m"], math.hypot(0.2, 0.01))

    def test_base_is_left_alone_when_asked_and_unknown_poses_give_nan(self):
        transport = BaseTransport()
        engine = engine_for(transport, base_pose=None)
        engine.reset()
        self.assertFalse(any(isinstance(c, tuple) and c[0] == "set_model_pose" for c in transport.calls))
        plain = engine_for(FakeTransport())                    # a transport that cannot tell
        plain.reset()
        plain.finish_trial()
        self.assertTrue(all(math.isnan(v) for v in plain.base_shift().values()))

    def test_failed_reset_motion_is_retried_from_the_last_reset_pose(self):
        transport = BaseTransport(states=("SUCCEEDED",))
        engine = engine_for(transport)
        engine.reset()
        home = list(transport.positions)
        transport.positions = [p + 0.5 for p in home]           # a trial left the arm somewhere else
        transport.states = ["ABORTED", "SUCCEEDED"]
        engine.reset()
        self.assertEqual(engine.reset_failures, 1)
        moved_back = [points[0].positions for _, points, _ in transport.published]
        self.assertIn(tuple(home), [tuple(p) for p in moved_back])
        transport.states = ["ABORTED"]
        with self.assertRaisesRegex(EngineError, "ended in state ABORTED"):
            engine.reset()


def validation_row(system, joint, goal, final, **extra):
    q = list(np.linspace(0.0, final, 60)) + [final] * 60
    row = {"system": system, "joint": joint, "goal_rad": goal, "start_rad": 0.0, "final_rad": final, "goal_error_rad": goal - final,
           "time_to_90_s": 2.0, "settling_time_s": 3.0, "reached_tolerance": abs(goal - final) < 0.05, "overshoot_rad": 0.01,
           "peak_speed": 0.5, "no_drive": False, "wrong_direction": False, "clamp_ticks": 0, "wall_s": 20.0, "tick_wall_ms": 60.0,
           "n_steps": 120, "rho": 0.02, "trace": {"dt_s": 0.05, "q": q, "v_cmd": [0.0] * 119, "ring": [0.0] * 119}}
    row.update(extra)
    return row


class ResultsReportTests(unittest.TestCase):
    def write(self, directory):
        compare = Path(directory) / "data" / "compare"
        (compare / "f3").mkdir(parents=True)
        goals = [0.5, -0.5]
        legacy = [{"joint": 5, "q_goal": g, "q_start": 0.0, "q_final": g * 0.7, "goal_error_rad": g * 0.3, "time_to_90_s": 6.0,
                   "settling_time_s": float("nan"), "reached_tolerance": False, "overshoot_rad": 0.0, "peak_speed": 0.2, "no_drive": False,
                   "wrong_direction": False, "n_steps": 200, "wall_s": 40.0, "tick_wall_ms": 55.0,
                   "trace": {"dt_s": 0.05, "q": [0.0, g * 0.7, g * 0.7], "v_cmd": []}} for g in goals]
        (compare / "legacy_trials_j5.json").write_text(json.dumps({"trials": legacy}))
        (compare / "b3.json").write_text(json.dumps({"trials": [validation_row("B", 5, g, g * 0.8) for g in goals], "failed": []}))
        (compare / "f3" / "validation.json").write_text(json.dumps({
            "trials": [validation_row("F", 5, g, g * 0.98, lost_bump=False, body_gap_index=0.4, goal_gap_index=0.5,
                                      goal_bump_error_index=0.3, base_shift_m=0.02) for g in goals], "failed": [{"joint": 5}]}))
        return Path(directory) / "data"

    def test_tables_from_files_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.write(directory)
            self.assertEqual(results_report.main(["--data", str(data), "--out", directory, "--no-figures"]), 0)
            text = (Path(directory) / "tables.md").read_text()
        self.assertIn("| L (original scripts) | 2 | 0.150 / 0.150 / 0.150 |", text)
        self.assertIn("| B3 (v3, bounded ring, 3x) | 2 | 0.100 / 0.100 / 0.100 |", text)
        v4 = next(line for line in text.splitlines() if line.startswith("| F3 (v4, 3x)"))
        self.assertIn("0.010 / 0.010 / 0.010", v4)
        self.assertEqual([c.strip() for c in v4.split("|")][11:15], ["0", "0", "0", "2"])        # no drive, wrong way, clamped, base pushed
        self.assertIn("Missing inputs: `pid_trials_j*.json`", text)
        self.assertIn("| v3, bounded ring, 3x | 2 | -0.090 | 100% | 0% | 0.500 | no difference shown |", text)
        self.assertIn("Every system ran the same 2 goals.", text)
        self.assertIn("| goal error, all joints, median / p90 | <= 0.045 / 0.12 rad", text)

    def test_stills_and_repeat_spread(self):
        row = results_report.normalise(validation_row("F", 5, 0.5, 0.49), "F3")
        self.assertAlmostEqual(row["hold"], 0.0)
        rows = [results_report.normalise(validation_row("F", 5, 0.5, f), "F3") for f in (0.49, 0.50, 0.52)]
        self.assertEqual(results_report.repeat_spread(rows)[2], 1)
        self.assertAlmostEqual(results_report.repeat_spread(rows)[1], 0.03)


class MeasureTests(unittest.TestCase):
    def test_blind_band_is_the_run_of_driveless_offsets_around_zero(self):
        offsets = list(range(-6, 7))
        drive = [-9, -8, -7, 0.2, 0, 0, 0, 0, -0.5, 3, 6, 8, 9]         # nothing from -3 to +2 (+2 points the wrong way)
        self.assertEqual(fit_comparator.blind_band(offsets, drive), (5, -0.5))
        sharp = [o * 2.0 for o in offsets]
        self.assertEqual(fit_comparator.blind_band(offsets, sharp), (0, 0.0))

    def test_v4_decoders_are_not_checked_against_the_transport_calibration(self):
        self.assertEqual(calibration_report(three_ring_single_joint()), [])
        self.assertEqual(len(calibration_report(two_ring_single_joint())), 1)

    def test_video_grid(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is not installed")
        import record_video

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, frames in enumerate((5, 9)):
                path = str(Path(directory) / ("v%d.mp4" % index))
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, record_video.SIZE)
                for _ in range(frames):
                    writer.write(np.full((record_video.SIZE[1], record_video.SIZE[0], 3), 40 * (index + 1), dtype=np.uint8))
                writer.release()
                paths.append(path)
            out = str(Path(directory) / "grid.mp4")
            self.assertEqual(record_video.main(["grid", "--columns", "2", "--out", out] + paths), 0)
            capture = cv2.VideoCapture(out)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 9)
            self.assertEqual((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))), (1280, 360))


if __name__ == "__main__":
    unittest.main()
