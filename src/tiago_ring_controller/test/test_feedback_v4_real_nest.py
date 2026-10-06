"""Real-NEST gate of architecture v4 (docs/v4-tuning.md).

The goal ring's projection must leave no blind band wider than two indices
around the goal, and the three-ring graph must reach its goals within the
settling tolerance with the motor gain 30% low, right and 30% high, after a
push and after the joint was held, without losing a bump.
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

try:
    import nest
except ImportError:  # pragma: no cover - host without NEST
    nest = None

from tiago_ring_controller.evaluation.motion import SETTLE_TOLERANCE  # noqa: E402
from tiago_ring_controller.graph.templates import joint_limits  # noqa: E402

PINNED_NEST_VERSION = "HEAD@41892a5"
JOINT = 5
#: (start, goal) as fractions of the joint range
MOVES = [(0.5, 0.9), (0.5, 0.1), (0.2, 0.6), (0.8, 0.3), (0.25, 0.95), (0.75, 0.05)]


@unittest.skipIf(nest is None, "NEST is not installed")
class FeedbackRealNestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest("requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__))

    def trials(self, moves=MOVES, **kwargs):
        from validate_velocity import run_trial

        lo, hi = joint_limits(JOINT)
        return [run_trial(JOINT, "F", lo + s * (hi - lo), lo + g * (hi - lo), "nest", None, 13579, **kwargs) for s, g in moves]

    def check(self, label, rows):
        errors = np.abs([r["goal_error_rad"] for r in rows])
        print("V4 %s: goal error median %.3f max %.3f rad, predicted - state %.1f indices, lost %d" % (
            label, np.median(errors), errors.max(), np.median(np.abs([r["body_gap_index"] for r in rows])), sum(r["lost_bump"] for r in rows)))
        self.assertFalse(any(r["lost_bump"] or r["wrong_direction"] or r["no_drive"] for r in rows))
        self.assertLess(errors.max(), SETTLE_TOLERANCE)

    def test_goals_are_reached_whatever_the_motor_gain(self):
        for scale in (0.7, 1.0, 1.3):
            self.check("motor gain x%.1f" % scale, self.trials(motor_gain_scale=scale))

    def test_a_push_and_a_held_joint_are_corrected(self):
        self.check("push 0.2 rad", self.trials(MOVES[:4], push_rad=0.2))
        self.check("held 2 s", self.trials(MOVES[:4], hold_s=2.0))

    def test_reference_projection_leaves_no_blind_band(self):
        import argparse
        import tempfile

        import fit_comparator

        with tempfile.TemporaryDirectory() as out:
            args = argparse.Namespace(bounds="bounded", set=None, reach=6, sharp=True, engines="nest", out=out)
            fit_comparator.fine_map(args)
            import json

            rows = json.load(open(Path(out) / "fine_map.json"))["rows"]
        self.assertLessEqual(max(r["blind_width"] for r in rows), 2)


if __name__ == "__main__":
    unittest.main()
