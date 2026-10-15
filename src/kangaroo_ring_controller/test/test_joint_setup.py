"""Behaviour tests for kangaroo_ring_controller/joint_setup.py (DECISIONS.md D35). ROS-free and NEST-free:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_joint_setup.py
"""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import RESULTS  # noqa: E402
from kangaroo_ring_controller.joint_setup import JointSetup, load, newest_setup, save, uncalibrated  # noqa: E402
from tiago_ring_controller.control.controller import DecoderParameters  # noqa: E402
from tiago_ring_controller.control.profiles import CALIBRATION_PROFILE  # noqa: E402

D34_LEFT_J3 = RESULTS / 'calibration/left_j3_seed2026_20261014_133218/calibration.json'


class Load(unittest.TestCase):
    def test_a_recorded_calibration_loads_as_recorded(self):
        # Literal values from the D34 run (docs/experiments/2026-10-14_kangaroo_settle_guard_recalibration.md).
        s = load(D34_LEFT_J3)
        self.assertEqual((s.side, s.joint, s.profile.name), ('left', 3, 'calibration'))
        self.assertEqual(s.limits, (-1.166365957762566, 2.4754874757242735))
        self.assertEqual(s.start, (0.239, 1.321, 1.57, 0.8))
        self.assertEqual((s.decoder.gain_positive, s.decoder.gain_negative, s.decoder.tau_s, s.decoder.delay_steps),
                         (0.003985985067710629, -0.006697309694590344, 0.700012642913498, 4))
        self.assertEqual(s.calibration, str(D34_LEFT_J3))

    def test_save_writes_the_recorded_file_again(self):
        # Q4: the schema is unchanged, so saving a recorded setup with its run fields reproduces the file exactly.
        recorded = json.loads(D34_LEFT_J3.read_text())
        run = {k: recorded[k] for k in ('seed', 'n_trials', 'fit_every', 'beta', 'limits_file', 'balanced', 'trials')}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'calibration.json'
            save(load(D34_LEFT_J3), path, **run)
            self.assertEqual(json.loads(path.read_text()), recorded)
            self.assertEqual(load(path), replace(load(D34_LEFT_J3), calibration=str(path)))

    def test_a_calibration_from_before_start_poses_starts_from_pal_home(self):
        s = load(RESULTS / 'calibration/right_j1_seed2026_20261013_100119/calibration.json')  # no start_pose (pre D25)
        self.assertEqual(s.start, (-0.239, 1.321, -1.57, 0.8))  # PAL 4-DoF home, right (KB 4c)


def write_cal(root, name, start_pose=None):
    """Minimal calibration.json as calibrate.py writes it."""
    d = Path(root) / name
    d.mkdir(parents=True)
    side, joint = name.split('_')[0], int(name.split('_')[1][1])
    path = d / 'calibration.json'
    save(JointSetup(side, joint, CALIBRATION_PROFILE, DecoderParameters(0.004, -0.004, 0.3, 0), (-1.0, 1.0),
                    start_pose or ()), path)
    return str(path)


class Newest(unittest.TestCase):
    def test_the_newest_calibration_by_time_is_chosen(self):
        with tempfile.TemporaryDirectory() as root:
            write_cal(root, 'right_j1_seed2026_20261013_100119')
            newest = write_cal(root, 'right_j1_seed100_20261013_164230', start_pose=(-0.239, 0.161, -1.57, 0.8))
            s = newest_setup('right', 1, root)
            self.assertEqual(s.calibration, newest)  # by time, though 'seed100' sorts before 'seed2026'
            self.assertEqual((s.start, s.limits, s.decoder.gain_positive),
                             ((-0.239, 0.161, -1.57, 0.8), (-1.0, 1.0), 0.004))

    def test_without_a_calibration_the_joint_is_uncalibrated(self):
        with tempfile.TemporaryDirectory() as root:
            s = newest_setup('left', 2, root)
            self.assertEqual(s, uncalibrated('left', 2))
            self.assertEqual((s.calibration, s.decoder, s.profile.name), (None, None, 'collector'))
            self.assertEqual(s.limits, (-2.618, 2.618))  # PAL 4-DoF URDF, arm_left_3_joint
            self.assertEqual(s.start, (0.239, 1.321, 1.57, 0.8))  # PAL home, left


if __name__ == '__main__':
    unittest.main()
