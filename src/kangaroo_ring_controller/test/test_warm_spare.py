"""Behaviour tests for kangaroo_ring_controller/warm_spare.py (DECISIONS.md D37). NEST-free and ROS-free: the ring
builder is a stand-in that records its calls, as the spare's rules do not depend on what a ring is.
    python3 -m unittest discover -s src/kangaroo_ring_controller/test -v -p test_warm_spare.py
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller.warm_spare import WarmSpare  # noqa: E402

LEFT_J0_SEED5 = ('left', 0, 5)  # a selection: side, joint, seed


class Builder:
    """Stand-in for RingDrive(newest_setup(side, joint), seed): a ring that carries its setup and seed."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, selected):
        self.calls.append(selected)
        if self.fail:
            raise RuntimeError('NEST failed')
        side, joint, seed = selected
        return SimpleNamespace(setup=(side, joint), seed=seed)


class Take(unittest.TestCase):
    def test_the_prepared_ring_is_handed_out_once_for_its_setup_and_seed(self):
        spare = WarmSpare(Builder())
        spare.prepare(LEFT_J0_SEED5)
        ring = spare.take(('left', 0), 5)
        self.assertEqual((ring.setup, ring.seed), (('left', 0), 5))
        self.assertIsNone(spare.take(('left', 0), 5))  # used up

    def test_a_ring_for_another_setup_or_seed_is_not_handed_out(self):
        for setup, seed in ((('left', 0), 6), (('left', 1), 5)):  # e.g. a newer calibration changed the setup
            spare = WarmSpare(Builder())
            spare.prepare(LEFT_J0_SEED5)
            self.assertIsNone(spare.take(setup, seed))


class Due(unittest.TestCase):
    def test_a_build_is_due_only_when_the_selection_changes(self):
        spare = WarmSpare(Builder())
        self.assertTrue(spare.due(LEFT_J0_SEED5))
        spare.prepare(LEFT_J0_SEED5)
        self.assertFalse(spare.due(LEFT_J0_SEED5))
        self.assertTrue(spare.due(('left', 0, 6)))  # another seed
        self.assertTrue(spare.due(('right', 0, 5)))  # another arm

    def test_after_a_reach_takes_it_another_is_due(self):
        spare = WarmSpare(Builder())
        spare.prepare(LEFT_J0_SEED5)
        spare.take(('left', 0), 5)
        self.assertTrue(spare.due(LEFT_J0_SEED5))

    def test_a_failed_build_is_not_retried_until_the_selection_changes(self):
        build = Builder(fail=True)
        spare = WarmSpare(build)
        with self.assertRaisesRegex(RuntimeError, 'NEST failed'):  # the caller reports it
            spare.prepare(LEFT_J0_SEED5)
        self.assertFalse(spare.due(LEFT_J0_SEED5))
        self.assertTrue(spare.due(('left', 1, 5)))
        self.assertIsNone(spare.take(('left', 0), 5))
        self.assertEqual(build.calls, [LEFT_J0_SEED5])


if __name__ == '__main__':
    unittest.main()
