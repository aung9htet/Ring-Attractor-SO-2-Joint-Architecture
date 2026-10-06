"""Real-NEST gate of the bounded ring (docs/archive/velocity/bounded-ring.md).

Twelve trials inside the arc must reproduce the bounded transport artifact and
the joint must follow the bump; goals at both joint limits must end inside the
limits with the bump inside the wall, and the ring's joint estimate
(``RingState``) must agree with the joint the fake robot ends at.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import nest
except ImportError:  # pragma: no cover - host without NEST
    nest = None

from tiago_ring_controller.config import source_config_path  # noqa: E402
from tiago_ring_controller.graph import (  # noqa: E402
    compile_graph,
    load_transport_calibration,
    run_graph_trial,
    run_transport_trials,
    two_ring_single_joint,
)
from tiago_ring_controller.graph.calibration import transport_motif  # noqa: E402
from tiago_ring_controller.graph.templates import BOUNDED, bounded_settings  # noqa: E402
from tiago_ring_controller.training.transport import fit_transport  # noqa: E402

PINNED_NEST_VERSION = "HEAD@41892a5"
PAIRS = [(70, 120), (120, 70), (45, 145), (145, 45), (95, 155), (145, 70)]
JOINT = 5


def bounded_graph(goal, seed=13579, time_scale=None):
    return two_ring_single_joint(joint=JOINT, goal_rad=goal, bounds="bounded", time_scale=time_scale,
                                 **({} if seed is None else {"seed": seed}))


@unittest.skipIf(nest is None, "NEST is not installed")
class BoundedRingRealNestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest("requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__))
        # the transport is a property of the network per 50 ms step: one NEST step per tick
        cls.trials, cls.rows = run_transport_trials(lambda goal, seed: bounded_graph(goal, seed, time_scale=1), PAIRS, [13579, 7], engines="nest")

    def test_small_calibration_reproduces_the_bounded_artifact(self):
        fit = fit_transport(self.trials)
        artifact = load_transport_calibration(source_config_path("calibration", BOUNDED["transport"]))
        print("BOUNDED TRANSPORT model=%s a=%.6f theta=%g cv_median=%.3f counts=%s" % (
            fit.model, fit.index_per_spike, fit.dead_zone, fit.cv["dead_zone"].median_abs_rel, fit.counts))
        self.assertEqual(fit.model, "dead_zone")
        self.assertEqual(fit.counts["ok"], len(self.trials))
        self.assertLess(abs(fit.index_per_spike - artifact["index_per_spike"]), 0.15 * artifact["index_per_spike"])
        self.assertLess(fit.cv["dead_zone"].median_abs_rel, 0.10)

    def test_calibrated_joint_follows_the_bump(self):
        motif = transport_motif(bounded_graph(0.0))
        rho = motif["state_encoder"].radians_per_index(int(motif["state_ring"].params["population_size"]))
        ratios = [row["joint_displacement"] / (rho * row["displacement"]) for row in self.rows]
        print("BOUNDED TRACKING joint/bump min=%.3f max=%.3f" % (min(ratios), max(ratios)))
        for row, ratio in zip(self.rows, ratios):
            with self.subTest(seed=row["seed"], start=row["start_index"], goal=row["goal_index"]):
                self.assertAlmostEqual(ratio, 1.0, delta=0.2)
        self.assertAlmostEqual(sum(ratios) / len(ratios), 1.0, delta=0.06)

    def test_time_scale_runs_the_same_network_path_faster(self):
        # docs/archive/velocity/speed.md: at time_scale 3 the ring's counts per tick are the sums of three
        # steps of the time_scale 1 run, spike for spike.  The trials end at different network
        # times (the settle rule counts robot ticks), so the end points may differ by a few indices.
        for start, goal in ((0.0, 1.0), (0.6, -0.8)):
            runs = {}
            for k in (1, 3):
                compiled = compile_graph(bounded_graph(goal, time_scale=k), engines="nest")
                compiled.robot_engine.home[JOINT] = start
                compiled.loop.initialize()
                try:
                    record = run_graph_trial(compiled, goal)
                finally:
                    compiled.loop.shutdown()
                packs = {}
                for inputs in [t.inputs for t in record.ticks] + [record.end_state]:
                    if inputs.get("r1") is not None:
                        packs[int(inputs["r1"]["nest_step"])] = np.array(inputs["r1"]["counts"])
                runs[k] = (packs, record)
            one, three = runs[1][0], runs[3][0]
            ticks = [s for s in sorted(three) if all(n in one for n in (3 * s - 2, 3 * s - 1, 3 * s))]
            same = [np.array_equal(three[s], one[3 * s - 2] + one[3 * s - 1] + one[3 * s]) for s in ticks]
            finals = [runs[k][1].final_state["joint_state"]["positions"][JOINT] for k in (1, 3)]
            print("TIME SCALE %+.1f -> %+.1f: %d of %d ticks identical, joint %+.3f / %+.3f, steps %d / %d"
                  % (start, goal, sum(same), len(same), finals[0], finals[1], runs[1][1].n_steps, runs[3][1].n_steps))
            with self.subTest(start=start, goal=goal):
                self.assertGreater(len(same), 30)
                self.assertTrue(all(same))
                self.assertLess(runs[3][1].n_steps, 0.6 * runs[1][1].n_steps)
                self.assertAlmostEqual(finals[1], finals[0], delta=0.15)

    def test_goals_at_the_limits_stay_inside_and_the_ring_reads_the_joint(self):
        graph = bounded_graph(0.0)
        joint = graph.blocks["j%d" % JOINT]
        lo, hi = joint.params["joint_min"], joint.params["joint_max"]
        wall = bounded_settings(200)["wall"]
        for start, goal in ((0.0, hi), (0.0, lo), (0.5 * hi, hi), (0.5 * lo, lo)):
            with self.subTest(start=start, goal=goal):
                compiled = compile_graph(bounded_graph(goal), engines="nest")
                compiled.robot_engine.home[JOINT] = start
                compiled.loop.initialize()
                try:
                    record = run_graph_trial(compiled, goal)
                finally:
                    compiled.loop.shutdown()
                positions = [t.inputs["joint_state"]["positions"][JOINT] for t in record.main_ticks]
                final = record.final_state["joint_state"]["positions"][JOINT]
                centroids = [t.inputs["r1"].data["centroid_circular"] for t in record.main_ticks if t.inputs.get("r1") is not None]
                centroids = [c for c in centroids if np.isfinite(c)]
                estimates = [t.outputs["state"].data["angle"] for t in record.main_ticks[-5:]]
                estimates = [e for e in estimates if np.isfinite(e)]
                print("LIMIT start %+.3f goal %+.3f: final %+.3f, joint range seen [%+.3f, %+.3f], bump [%.1f, %.1f], ring %+.3f"
                      % (start, goal, final, min(positions), max(positions), min(centroids), max(centroids), np.mean(estimates)))
                self.assertTrue(lo - 1e-6 <= min(positions) and max(positions) <= hi + 1e-6)
                self.assertTrue(wall[0] <= min(centroids) and max(centroids) <= wall[1])
                self.assertLess(abs(goal - final), 0.2)
                # at a limit the joint and the bump part by up to ~6 indices (2026-10-03: 0.07-0.12 rad);
                # inside the arc they agree to ~0.02 rad (docs/archive/velocity/bounded-ring.md)
                self.assertLess(abs(np.mean(estimates) - final), 0.15)


if __name__ == "__main__":
    unittest.main()
