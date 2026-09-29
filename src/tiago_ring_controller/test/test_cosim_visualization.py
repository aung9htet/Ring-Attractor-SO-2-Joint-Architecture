"""Observer hook and the off-screen ring monitor."""

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.config import load_joint_calibration  # noqa: E402
from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE  # noqa: E402
from tiago_ring_controller.cosim import CosimConfig, LoopObserver  # noqa: E402
from tiago_ring_controller.cosim.runner import build_loop, make_fake_engines, run_trial  # noqa: E402

try:
    import matplotlib  # noqa: F401
except ImportError:  # pragma: no cover
    matplotlib = None


CALIBRATION_PATH = SRC / "config/calibration/velocity_calibration.json"


def make_config(**overrides):
    calibration = load_joint_calibration(str(CALIBRATION_PATH), 5)
    return CosimConfig.from_profile(COLLECTOR_PROFILE, 5, calibration, max_steps=12, **overrides)


class CountingObserver(LoopObserver):
    def __init__(self):
        self.resets = 0
        self.ticks = []
        self.ends = []

    def on_reset(self, loop):
        self.resets += 1

    def on_tick(self, record):
        self.ticks.append((record.phase, record.tick))

    def on_trial_end(self, record):
        self.ends.append(record.stop_reason)


class ObserverHookTests(unittest.TestCase):
    def test_observers_see_every_tick_and_do_not_change_the_record(self):
        config = make_config()
        observer = CountingObserver()
        observed = run_trial(build_loop(config, make_fake_engines(config), observers=[observer]), 0.5, config)
        plain = run_trial(build_loop(config, make_fake_engines(config)), 0.5, config)
        self.assertEqual(observer.resets, 1)
        self.assertEqual(len(observer.ticks), len(observed.ticks))
        self.assertEqual(observer.ticks[:4], [("lead", -4), ("lead", -3), ("lead", -2), ("lead", -1)])
        self.assertEqual(observer.ends, [observed.stop_reason])
        self.assertEqual(observed.to_dict(include_timing=False), plain.to_dict(include_timing=False))


@unittest.skipIf(matplotlib is None, "matplotlib is not installed")
class RingMonitorTests(unittest.TestCase):
    def test_offscreen_monitor_renders_frames_for_each_trial(self):
        from tiago_ring_controller.cosim.visualization import RingMonitor

        config = make_config()
        with tempfile.TemporaryDirectory() as tmp:
            monitor = RingMonitor(100, 5, dt_ms=config.dt_ms, history=20, render_every=4,
                                  show=False, frame_dir=tmp)
            self.assertFalse(monitor.show)
            loop = build_loop(config, make_fake_engines(config), observers=[monitor])
            first = run_trial(loop, 0.5, config)
            second = run_trial(loop, -0.4, config)
            expected = sum(len(r.ticks) // 4 + 1 for r in (first, second))
            self.assertEqual(monitor.frames_written, expected)
            self.assertEqual(monitor.trial_index, 2)
            self.assertEqual(monitor.goal_rad, -0.4)
            # The first lead sub-step has no NEST sample yet.
            self.assertEqual(len(monitor.steps), len(second.ticks) - 1)
            self.assertEqual(len(monitor.position), second.n_steps)
            self.assertEqual(monitor.stop_reason, second.stop_reason)
            files = sorted(os.listdir(tmp))
            self.assertEqual(len(files), expected)
            self.assertTrue(files[0].startswith("trial_001_frame_"))
            frame = monitor.frame()
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 4)
            self.assertGreater(frame.shape[0], 100)
            monitor.close()


if __name__ == "__main__":
    unittest.main()
