"""Behaviour tests for kangaroo_ring_controller/refit.py (DECISIONS.md D39, F15). ROS-free and NEST-free:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_refit.py
"""
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import RESULTS  # noqa: E402
from kangaroo_ring_controller.joint_setup import JointSetup, newest_setup, save  # noqa: E402
from kangaroo_ring_controller.refit import refit, save_refit  # noqa: E402
from tiago_ring_controller.control.controller import DecoderParameters  # noqa: E402
from tiago_ring_controller.control.profiles import CALIBRATION_PROFILE  # noqa: E402
from tiago_ring_controller.math.control import decoder_features  # noqa: E402

DT = 0.05  # s, control tick
H = [0, 5, 10, 20, 20, 10, 5, 0, 0, 0, 0, 0]  # one drive shape, one sign throughout


def predicted_dq(setup, spikes):
    """Δq the setup's decoder gives a drive [rad]: k_pos·S_pos + k_neg·S_neg (TIAGo's fitted model)."""
    f = decoder_features(spikes, DT, setup.decoder.tau_s, setup.decoder.delay_steps)
    return setup.decoder.gain_positive * f['S_pos'] + setup.decoder.gain_negative * f['S_neg']


class Refit(unittest.TestCase):
    def test_a_batch_fit_recovers_gains_that_explain_every_trial(self):
        # Linearity oracle: for a one-signed shape h, S(s·h) = s·S(h) at any tau and delay. Trials want 0.01·s rad
        # for +s·h and -0.03·s rad for -s·h, so a correct fit predicts +0.30 for +30·h and -0.90 for -30·h.
        trials = [dict(spike_history=[sign * s * x for x in H], dt=DT, dq_desired=dq * s, dq_actual=dq * s)
                  for s in range(1, 26) for sign, dq in ((1, 0.01), (-1, -0.03))]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'calibration.json'
            save(JointSetup('left', 0, CALIBRATION_PROFILE, DecoderParameters(1e-4, -1e-4, 0.3, 0), (-1.0, 1.0),
                            (0.239, 1.321, 1.57, 0.8)), path, seed=7, trials=trials)
            setup = refit(path, last=50)
        self.assertAlmostEqual(predicted_dq(setup, [30 * x for x in H]), 0.30, delta=0.003)
        self.assertAlmostEqual(predicted_dq(setup, [-30 * x for x in H]), -0.90, delta=0.009)

    def test_a_saved_refit_is_the_joints_newest_calibration_and_names_its_source(self):
        source = RESULTS / 'calibration/left_j3_seed2026_20261014_133218/calibration.json'  # D34
        with tempfile.TemporaryDirectory() as root:
            path = save_refit(source, root=root)
            self.assertEqual(newest_setup('left', 3, Path(root) / 'calibration'),
                             replace(refit(source), calibration=str(path)))
            record = json.loads(path.read_text())
            self.assertEqual((record['refit_of'], record['refit_trials'], record['seed']), (str(source), 50, 2026))

    def test_refits_of_one_joint_and_seed_made_in_the_same_second_are_all_saved(self):
        # Review finding: the run dir was named only by side, joint, seed and time, so the second refit collided.
        sources = [RESULTS / f'calibration/left_j3_seed2026_{stamp}/calibration.json'
                   for stamp in ('20261014_001609', '20261014_133218')]  # D28 and D34, both seed 2026
        with tempfile.TemporaryDirectory() as root:
            paths = [save_refit(s, root=root) for s in sources]
            self.assertEqual(len({p.parent for p in paths}), 2)

    def test_what_cannot_be_refit_is_refused_by_name(self):
        source = RESULTS / 'calibration/left_j3_seed2026_20261014_133218/calibration.json'
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'no trials'):  # a refit has no trials of its own
                refit(save_refit(source, root=root))
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            refit(source, last=1)

    def test_the_source_is_recorded_as_an_absolute_path(self):
        # D40 found refits named by a path relative to where refit_calibration.py ran (only valid from there).
        source = RESULTS / 'calibration/left_j3_seed2026_20261014_133218/calibration.json'
        with tempfile.TemporaryDirectory() as root:
            path = save_refit(os.path.relpath(source), root=root)
            self.assertEqual(json.loads(path.read_text())['refit_of'], str(source.resolve()))


if __name__ == '__main__':
    unittest.main()
