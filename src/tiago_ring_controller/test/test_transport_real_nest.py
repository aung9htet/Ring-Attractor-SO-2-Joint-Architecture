"""Real-NEST gate of the transport calibration (docs/archive/velocity/calibration.md).

Twelve trials on two seeds must reproduce the checked-in artifact within the
documented tolerance, and with the calibrated decoder the fake robot must end
where the state bump ended.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import nest
except ImportError:  # pragma: no cover - host without NEST
    nest = None

from tiago_ring_controller.graph import (  # noqa: E402
    load_transport_calibration,
    run_transport_trials,
    two_ring_single_joint,
)
from tiago_ring_controller.training.transport import fit_transport  # noqa: E402

PINNED_NEST_VERSION = "HEAD@41892a5"
PAIRS = [(60, 110), (110, 60), (35, 135), (135, 35), (85, 150), (135, 60)]


@unittest.skipIf(nest is None, "NEST is not installed")
class TransportRealNestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest("requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__))
        cls.trials, cls.rows = run_transport_trials(
            lambda goal, seed: two_ring_single_joint(goal_rad=goal, calibration="transport", **({} if seed is None else {"seed": seed})),
            PAIRS, [13579, 7], engines="nest")

    def test_small_calibration_reproduces_the_artifact(self):
        fit = fit_transport(self.trials)
        artifact = load_transport_calibration()
        print("TRANSPORT model=%s a=%.6f theta=%g cv_median=%.3f cv_p90=%.3f counts=%s" % (
            fit.model, fit.index_per_spike, fit.dead_zone, fit.cv["dead_zone"].median_abs_rel, fit.cv["dead_zone"].p90_abs_rel, fit.counts))
        self.assertEqual(fit.model, "dead_zone")
        self.assertEqual(fit.counts["ok"], len(self.trials))
        self.assertTrue(10.0 <= fit.dead_zone <= 30.0)
        self.assertTrue(0.004 <= fit.index_per_spike <= 0.0065)
        self.assertLess(abs(fit.index_per_spike - artifact["index_per_spike"]), 0.15 * artifact["index_per_spike"])
        self.assertLess(fit.cv["dead_zone"].median_abs_rel, 0.10)
        self.assertLess(fit.cv["dead_zone"].p90_abs_rel, fit.cv["linear"].p90_abs_rel)

    def test_calibrated_joint_follows_the_bump(self):
        rho = (1.4171452891568528 + 1.4172695167359413) / 190.0
        ratios = [row["joint_displacement"] / (rho * row["displacement"]) for row in self.rows]
        print("TRACKING joint/bump min=%.3f max=%.3f" % (min(ratios), max(ratios)))
        for row, ratio in zip(self.rows, ratios):
            with self.subTest(seed=row["seed"], start=row["start_index"], goal=row["goal_index"]):
                self.assertAlmostEqual(ratio, 1.0, delta=0.2)
        self.assertAlmostEqual(sum(ratios) / len(ratios), 1.0, delta=0.06)


if __name__ == "__main__":
    unittest.main()
