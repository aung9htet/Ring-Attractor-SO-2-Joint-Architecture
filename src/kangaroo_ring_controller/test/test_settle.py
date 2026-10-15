"""Behaviour tests for kangaroo_ring_controller/settle.py: settle counting starts at the first threshold crossing
(docs/kangaroo/DECISIONS.md F14, Q14 (b)). ROS-free and NEST-free:
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_settle.py
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import RESULTS  # noqa: E402
from kangaroo_ring_controller.settle import GuardedDriveCore  # noqa: E402
from tiago_ring_controller.control.controller import DecoderParameters, DriveControlCore  # noqa: E402

DT = 0.05  # s, control tick


def settle_tick(core, spikes):
    """1-based tick at which `core` first reports settled, or None."""
    return next((i + 1 for i, s in enumerate(spikes) if core.step(s).is_settled), None)


class SettleGuard(unittest.TestCase):
    def test_no_drive_never_settles(self):
        spikes = [0.0] * 30
        self.assertEqual(settle_tick(DriveControlCore(DT), spikes), 10)  # TIAGo: n_settle ticks under 5
        self.assertIsNone(settle_tick(GuardedDriveCore(DT), spikes))

    def test_once_armed_settles_as_tiago(self):
        # No filter (alpha 0), no delay: |d| = 20 >= 5 on ticks 1-5, 0 from tick 6, so 10 quiet ticks end at tick 15.
        spikes = [20.0] * 5 + [0.0] * 20
        for core in (DriveControlCore(DT, alpha=0.0), GuardedDriveCore(DT, alpha=0.0)):
            self.assertEqual(settle_tick(core, spikes), 15, type(core).__name__)

    def test_a_slow_delayed_decoder_is_not_stopped_before_its_drive_arrives(self):
        # D28 reach left j3 up: TIAGo's rule stopped it at tick 10 with the drive already at 42 (F14).
        trial = json.loads((RESULTS / 'ring_reach/left_j3_seed100_20261014_002045/trial.json').read_text())
        spikes = trial['signed_spike']
        decoder = DecoderParameters(trial['decoder_gain_positive'], trial['decoder_gain_negative'],
                                    trial['decoder_tau'], trial['decoder_delay_steps'])
        scale = trial['signed_drive'][1] / spikes[1]
        self.assertEqual(settle_tick(DriveControlCore(DT, decoder, spike_scale=scale), spikes), 10)
        self.assertIsNone(settle_tick(GuardedDriveCore(DT, decoder, spike_scale=scale), spikes))


if __name__ == '__main__':
    unittest.main()
