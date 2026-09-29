"""Real-NEST parity: the cosim NEST engine vs the legacy per-tick loop.

Both drive the *same* unchanged ``SingleRingModel`` with the same seed and
the same two 50 ms stimulus injections in the collector's order (goal, then
state).  The legacy side is the collector's ``run_ring_simulation`` arithmetic
(cumulative event counts before/after ``Simulate(50)``); the cosim side is
``NestEngine`` with NodeCollection ``n_events`` deltas, once per step mode.
"""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
LEGACY = ROOT / "legacy"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

try:
    import nest
except ImportError:  # pragma: no cover - exercised on hosts without NEST
    nest = None

from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE  # noqa: E402
from tiago_ring_controller.cosim import CosimConfig  # noqa: E402
from tiago_ring_controller.cosim.runner import make_nest_engine  # noqa: E402
from tiago_ring_controller.cosim.tf import bump_datapack  # noqa: E402


PINNED_NEST_VERSION = "HEAD@41892a5"
SEED = 13579
GOAL_INDEX = 140
STATE_INDEX = 60
HALF_WIDTH = 5
STEPS = 6
DT_MS = 50.0


@unittest.skipIf(nest is None, "NEST is not installed")
class RealNestCosimParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if nest.__version__ != PINNED_NEST_VERSION:
            raise unittest.SkipTest(
                "parity check requires NEST %s, found %s" % (PINNED_NEST_VERSION, nest.__version__)
            )
        import gain_modulation
        import homeostasis
        import ring_component
        import single_ring

        for module in (gain_modulation, homeostasis, ring_component, single_ring):
            module.nest = nest
        cls.single_ring = single_ring
        cls.ring_params = str(SRC / "config/model_params/ring_params.json")
        cls.weights_dir = str(SRC / "config/ring_decoding_weights")

    def _legacy_counts(self):
        model = self.single_ring.SingleRingModel(
            ring_params_file=self.ring_params, weights_dir=self.weights_dir,
            seed=SEED, local_num_threads=1,
        )
        model.r2._inject_bump(GOAL_INDEX, HALF_WIDTH)
        model.r1._inject_bump(STATE_INDEX, HALF_WIDTH)

        def total(recorders):
            return sum(len(s) for s in model._collect_spikes(recorders))

        def r1_counts():
            return np.array(
                [len(s) for s in model._collect_spikes(model.r1.ring_attractor.ring_spike_recorders)],
                dtype=float,
            )

        rows = []
        for _ in range(STEPS):
            left_before = total(model.gain_modulation.left_gain_spike_recorders)
            right_before = total(model.gain_modulation.right_gain_spike_recorders)
            r1_before = r1_counts()
            nest.Simulate(DT_MS)
            left = int(total(model.gain_modulation.left_gain_spike_recorders) - left_before)
            right = int(total(model.gain_modulation.right_gain_spike_recorders) - right_before)
            delta = np.maximum(r1_counts() - r1_before, 0.0)
            rows.append((left, right, delta.tolist()))
        return rows

    def _cosim_counts(self, step_mode):
        config = CosimConfig.from_profile(
            COLLECTOR_PROFILE, 5, joint_min=-1.0, joint_max=1.0, rng_seed=SEED,
            nest_step_mode=step_mode, ring_params_file=self.ring_params, weights_dir=self.weights_dir,
        )
        engine = make_nest_engine(config, backend=nest)
        engine.initialize()
        engine.reset()
        engine.set_datapacks({
            "goal_bump": bump_datapack("goal_bump", 0.0, GOAL_INDEX, HALF_WIDTH, "goal"),
            "state_bump": bump_datapack("state_bump", 0.0, STATE_INDEX, HALF_WIDTH, "proprioception"),
        })
        rows = []
        for _ in range(STEPS):
            engine.advance(DT_MS)
            counts = engine.get_datapacks()["ring_counts"]
            rows.append((counts["left"], counts["right"], counts["r1_delta"]))
        readout_mode = counts["readout_mode"]
        hidden_ms = counts["hidden_ms"]
        raster = engine.raster()
        engine.finish_trial()
        engine.shutdown()
        return rows, readout_mode, hidden_ms, raster

    def test_simulate_mode_reproduces_the_legacy_loop_exactly(self):
        legacy = self._legacy_counts()
        cosim, readout_mode, hidden_ms, raster = self._cosim_counts("simulate")
        self.assertEqual(readout_mode, "node_collection")
        self.assertEqual(hidden_ms, 100.0)
        self.assertEqual(cosim, legacy)
        self.assertGreater(sum(l + r for l, r, _ in legacy), 0)
        self.assertGreater(len(raster["r1_times"]), 0)

    def test_prepare_run_cleanup_mode_reproduces_the_legacy_loop_exactly(self):
        legacy = self._legacy_counts()
        cosim, readout_mode, hidden_ms, _ = self._cosim_counts("run")
        self.assertEqual(readout_mode, "node_collection")
        self.assertEqual(cosim, legacy)


if __name__ == "__main__":
    unittest.main()
