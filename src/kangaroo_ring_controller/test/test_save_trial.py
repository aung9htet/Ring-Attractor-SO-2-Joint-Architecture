"""Behaviour test for kangaroo_ring_controller.save_trial (DECISIONS.md D38). NEST-free and ROS-free:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_save_trial.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import RESULTS, save_trial  # noqa: E402
from kangaroo_ring_controller.joint_setup import load  # noqa: E402

D34_LEFT_J3 = RESULTS / 'calibration/left_j3_seed2026_20261014_133218/calibration.json'


class SaveTrial(unittest.TestCase):
    def test_a_trial_is_saved_with_its_joint_and_calibration_in_a_new_run_dir(self):
        with tempfile.TemporaryDirectory() as root:
            path, trial = save_trial('ring_reach', load(D34_LEFT_J3), {'seed': 100, 'q_final': 1.755}, root,
                                     from_home=True)
            self.assertEqual(path.name, 'trial.json')
            self.assertEqual(path.parent.parent, Path(root) / 'ring_reach')
            self.assertRegex(path.parent.name, r'^left_j3_seed100_\d{8}_\d{6}$')
            self.assertEqual(json.loads(path.read_text()), trial)
            self.assertEqual(trial, {'seed': 100, 'q_final': 1.755, 'side': 'left', 'joint': 3,
                                     'calibration': str(D34_LEFT_J3), 'from_home': True})


if __name__ == '__main__':
    unittest.main()
