"""The bounded ring on the fakes: arc mapping, ring readout of the joint, wall, limit fade, comparator target.

docs/archive/velocity/bounded-ring.md; the real-NEST gates are in test_bounded_ring_real_nest.py.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support_fake_nest import RecordingNest  # noqa: E402
from tiago_ring_controller.blocks import Decoder, Gain, Ring, RingState  # noqa: E402
from tiago_ring_controller.blocks.base import BuildContext  # noqa: E402
from tiago_ring_controller.control.mapping import ArcMapping, RingMapper  # noqa: E402
from tiago_ring_controller.cosim import DataPack, FakeRobotEngine  # noqa: E402
from tiago_ring_controller.cosim.fakes import circular_centroid, ring_readout  # noqa: E402
from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine  # noqa: E402
from tiago_ring_controller.graph import compile_graph, run_graph, two_ring_single_joint  # noqa: E402
from tiago_ring_controller.graph.templates import BOUNDED, bounded_settings  # noqa: E402
from tiago_ring_controller.nest.populations import wall_profile  # noqa: E402
from tiago_ring_controller.training.comparator import (  # noqa: E402
    bounded_ramp,
    decision_summary,
    fit_comparator,
    monotonicity,
    ramp,
    sawtooth_ramp,
)


def bump(center, size=200, width=4.0):
    distance = (np.arange(size) - center + size / 2) % size - size / 2
    return np.round(10 * np.exp(-distance ** 2 / (2 * width ** 2)))


class ArcMappingTests(unittest.TestCase):
    def test_bounded_arc_is_centred_opposite_the_seam_and_round_trips(self):
        arc = ArcMapping.bounded(-1.4, 1.4, 200, 0.7)
        self.assertEqual(arc.arc, (30.0, 170.0))
        self.assertAlmostEqual(arc.radians_per_index, 2.8 / 140)
        for index in range(30, 171):
            self.assertEqual(arc.index(arc.angle(index)), index)
        self.assertEqual((arc.index(-5.0), arc.index(5.0)), (30, 170))     # outside the range lands on the arc ends
        self.assertTrue(arc.contains(30) and not arc.contains(29) and arc.contains(25, margin=5))
        with self.assertRaises(ValueError):
            ArcMapping.bounded(1.0, -1.0, 200, 0.7)
        with self.assertRaises(ValueError):
            ArcMapping.bounded(-1.0, 1.0, 200, 1.0)

    def test_taper_fades_one_end_only(self):
        arc = ArcMapping.bounded(-1.0, 1.0, 200, 0.7)
        upper, lower = arc.taper(10, "upper"), arc.taper(10, "lower")
        self.assertEqual((upper[100], upper[160], upper[165], upper[170], upper[180]), (1.0, 1.0, 0.5, 0.0, 0.0))
        self.assertEqual((upper[20], lower[180]), (1.0, 1.0))                 # moving back into the arc is never faded
        self.assertEqual((lower[30], lower[35], lower[40], lower[20]), (0.0, 0.5, 1.0, 0.0))

    def test_ring_mapper_wraps_every_mapping(self):
        legacy = RingMapper("collector", -1.0, 1.0, 200, 5)
        self.assertEqual(legacy.arc, (5.0, 195.0))
        self.assertAlmostEqual(legacy.radians_per_index, 2.0 / 190)
        self.assertEqual(legacy.index(legacy.angle(77)), 77)
        circular = RingMapper("circular", 0.0, 1.0, 200)
        self.assertAlmostEqual(circular.radians_per_index, 2 * math.pi / 200)
        self.assertEqual(circular.index(circular.angle(50)), 50)
        bounded = RingMapper("bounded", -1.0, 1.0, 200, 5, 0.7)
        self.assertEqual((bounded.arc, bounded.effective_half_width), ((30.0, 170.0), 5))
        self.assertEqual(RingMapper("analysis", -1.0, 1.0, 200, 5).effective_half_width, 10)
        with self.assertRaises(ValueError):
            RingMapper("nope", -1.0, 1.0, 200)


class RingStateTests(unittest.TestCase):
    def test_reads_the_joint_angle_back_including_a_bump_on_the_seam(self):
        state = RingState("s", mapping="bounded", joint_min=-1.0, joint_max=1.0, arc_fraction=0.7, window_ticks=1)
        out = state.read(bump(100))
        self.assertAlmostEqual(out["angle"], 0.0, places=6)
        self.assertGreater(out["strength"], 0.9)
        self.assertTrue(out["in_arc"])
        arc = ArcMapping.bounded(-1.0, 1.0, 200, 0.7)
        self.assertAlmostEqual(state.read(bump(55))["angle"], arc.angle(55), places=6)
        seam = state.read(bump(199))                     # straddles the seam, beyond the upper arc end
        self.assertAlmostEqual(seam["index"], 199.0, places=6)
        self.assertFalse(seam["in_arc"])
        self.assertEqual(seam["angle"], 1.0)                # clamped to the joint limit
        low = state.read(bump(2))                        # just past the seam on the other side: below the arc
        self.assertAlmostEqual(low["index"], 2.0, places=6)
        self.assertEqual(low["angle"], -1.0)
        self.assertTrue(math.isnan(state.read(np.zeros(200))["angle"]))
        # the linear centroid of the legacy readout is 100 indices off for the same bump
        self.assertAlmostEqual(circular_centroid(bump(199)), 199.0, places=6)
        self.assertGreater(abs(ring_readout(bump(199))[2] - 199.0), 50.0)

    def test_window_sums_steps_once_each(self):
        state = RingState("s", mapping="collector", joint_min=-1.0, joint_max=1.0, window_ticks=3)
        state.step(bump(60), nest_step=1)
        state.step(bump(60), nest_step=1)                # the same NEST step again is ignored
        state.step(bump(64), nest_step=2)
        self.assertEqual(len(state.window), 2)
        out = state.step(bump(68), nest_step=3)
        mapper = RingMapper("collector", -1.0, 1.0, 200, 5)
        self.assertAlmostEqual(out["index"], 64.0, places=3)
        self.assertAlmostEqual(out["angle"], mapper.angle(64.0), places=6)
        state.reset()
        self.assertEqual(len(state.window), 0)


class WallAndFadeTests(unittest.TestCase):
    def test_wall_profile(self):
        profile = wall_profile(200, 15, 185, 0.9)
        self.assertEqual((profile[15], profile[100], profile[185]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(profile[14], 0.9)
        self.assertAlmostEqual(profile[0], 0.9)
        tapered = wall_profile(200, 15, 185, 0.8, taper=10)
        self.assertAlmostEqual(tapered[10], 0.9)
        self.assertAlmostEqual(tapered[2], 0.8)

    def test_ring_build_scales_the_bias_current_outside_the_wall(self):
        backend = RecordingNest()
        ring = Ring("r", population_size=20, wall=[5, 14], outside_current_scale=0.9)
        ring.build(BuildContext(backend), {})
        calls = [value for _, value in backend.status_calls if isinstance(value, list) and value and "I_e" in value[0]]
        self.assertEqual(len(calls), 1)
        currents = [entry["I_e"] for entry in calls[0]]
        self.assertAlmostEqual(currents[0], 0.9 * currents[10])
        self.assertEqual(currents[5], currents[14])
        self.assertIsNone(Ring("plain").current_profile())
        with self.assertRaises(Exception):
            Ring("bad", wall=[50, 10], outside_current_scale=0.9).current_profile()

    def test_gain_drive_fades_toward_each_limit(self):
        left, right = Gain("g", limit_arc=[30, 170], limit_taper=10.0).drive_profiles(200)
        self.assertEqual((right[150], right[165], right[170], right[30]), (1.0, 0.5, 0.0, 1.0))
        self.assertEqual((left[30], left[35], left[50], left[170]), (0.0, 0.5, 1.0, 1.0))
        self.assertEqual(Gain("g").drive_profiles(200), (None, None))
        left, right = Gain("g", limit_arc=[30, 170], limit_taper=0.0).drive_profiles(200)   # hard cut, the template's choice
        self.assertEqual((right[169], right[170], right[20], left[31], left[30], left[180]), (1.0, 0.0, 1.0, 1.0, 0.0, 1.0))

    def test_ramp_current_lifts_warm_and_cold_only(self):
        backend = RecordingNest()
        graph = two_ring_single_joint(joint=5, bounds="bounded", calibration="legacy", bounded={"ramp_current": 8.0e5})
        graph.build(BuildContext(backend))
        cmp = graph.blocks["cmp"]
        lifted = [(node, value["I_e"]) for node, value in backend.status_calls if isinstance(value, dict) and "I_e" in value]
        bias = BuildContext(backend).neuron_parameters("homeostasis")["I_e"]
        self.assertEqual([value for _, value in lifted], [bias + 8.0e5] * 2)
        self.assertEqual([str(node) for node, _ in lifted], [str(tuple(cmp.population.node(label))) for label in ("warm", "cold")])
        backend = RecordingNest()
        two_ring_single_joint(joint=5, bounds="bounded", calibration="legacy", bounded={"ramp_current": 0.0}).build(BuildContext(backend))
        self.assertFalse([v for _, v in backend.status_calls if isinstance(v, dict) and "I_e" in v])

    def test_off_bounded_parameters_leave_the_fingerprint_alone(self):
        from tiago_ring_controller.graph.calibration import network_fingerprint

        plain = two_ring_single_joint(joint=5)
        explicit = two_ring_single_joint(joint=5)
        explicit.blocks["cmp"].params["ramp_current"] = 0.0
        explicit.blocks["gain"].params["limit_taper"] = 3.0          # governed by limit_arc, which is off
        self.assertEqual(network_fingerprint(plain), network_fingerprint(explicit))
        explicit.blocks["cmp"].params["ramp_current"] = 1.0
        self.assertNotEqual(network_fingerprint(plain), network_fingerprint(explicit))


class ComparatorTargetTests(unittest.TestCase):
    def test_bounded_ramp_is_continuous_odd_about_the_seam_and_rises_across_the_arc(self):
        theta = np.linspace(0.0, 2 * np.pi, 2001)[:-1]
        values = bounded_ramp(theta, 0.8)
        self.assertLess(np.max(np.abs(np.diff(values))), 0.01)                    # continuous
        np.testing.assert_allclose(bounded_ramp(-theta, 0.8), 1.0 - values, atol=1e-12)
        arc = (theta >= np.pi * 0.2) & (theta <= np.pi * 1.8)
        self.assertTrue(np.all(np.diff(values[arc]) > 0))
        self.assertAlmostEqual(float(bounded_ramp(np.pi, 0.8)), 0.5)
        legacy = sawtooth_ramp(theta)
        self.assertGreater(np.max(np.abs(np.diff(np.append(legacy, legacy[0])))), 0.9)   # the legacy target jumps at the seam
        self.assertLess(np.max(np.abs(np.diff(np.append(values, values[0])))), 0.01)

    def test_fit_on_sine_features_recovers_a_monotone_ramp(self):
        positions = np.arange(1, 200, 2)
        theta = 2 * np.pi * positions / 200
        harmonics = np.arange(1, 21)
        signed = np.sin(np.outer(theta, harmonics)) * 40.0
        counts = np.zeros((len(positions), 40))
        counts[:, 0::2] = np.maximum(signed, 0) + 20.0
        counts[:, 1::2] = np.maximum(-signed, 0) + 20.0
        fit = fit_comparator(counts, positions, 200, lambda t: bounded_ramp(t, 0.8), "bounded", window_ms=350.0)
        self.assertLess(fit.test_rms, 0.01)
        values = ramp(fit, counts, 350.0)
        self.assertEqual(monotonicity(values, positions, 30, 170)["wrong_way"], 0)
        self.assertEqual(fit.artifact().shape, (80, 2))

    def test_decision_summary(self):
        rows = [{"state": s, "goal": g, "drive": float(np.sign(g - s) * (abs(g - s) > 8) * 50)}
                for s in range(30, 171, 10) for g in range(30, 171, 10)]
        rows.append({"state": 40, "goal": 120, "drive": -5.0})
        rows = [r for r in rows if not (r["state"] == 40 and r["goal"] == 120 and r["drive"] > 0)]
        summary = decision_summary(rows, 30, 170)
        self.assertEqual(summary["wrong_sign"], 1)
        self.assertEqual(summary["dead_zone_max"], 0.0)


class SettleAndStartTests(unittest.TestCase):
    def test_with_a_dead_zone_the_settle_check_reads_the_raw_count(self):
        decoder = Decoder("d", gain_positive=0.001, gain_negative=-0.001, tau_s=0.05, dead_zone=20.0,
                          drive_threshold=5.0, n_settle=3, horizon=1)
        decoder.configure(50.0)
        for _ in range(10):
            out = decoder.step(0.0, 15.0)            # 15 spikes/tick: inside the dead zone, the bump still creeps
        self.assertEqual(out["consumed"]["decoded_velocity"], 0.0)
        self.assertFalse(out["settled"])
        for _ in range(10):
            out = decoder.step(0.0, 0.0)
        self.assertTrue(out["settled"])
        plain = Decoder("p", tau_s=0.05, n_settle=3, horizon=1)
        plain.configure(50.0)
        sample = plain.consume(0.0, 15.0)
        self.assertEqual(sample["settle_drive"], sample["delayed_drive"])

    def test_engines_start_from_a_chosen_pose(self):
        robot = FakeRobotEngine("robot")
        robot.initialize()
        robot.set_start_pose({4: -1.2})
        robot.reset()
        self.assertEqual(robot.get_datapacks()["joint_state"]["positions"][4], -1.2)

        import importlib.util

        spec = importlib.util.spec_from_file_location("engine_tests", ROOT / "test/test_cosim_engines.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        transport = module.FakeTransport()
        engine = GazeboRosEngine(transport, settle_timeout_s=0.5)
        engine.initialize()
        engine.set_start_pose({5: 0.9})
        engine.reset()
        _, points, _ = transport.published[-1]
        self.assertEqual(points[-1].positions[5], 0.9)
        self.assertEqual(points[-1].positions[4], transport.positions[4])         # the other joints hold
        self.assertGreaterEqual(points[-1].time_from_start_s, 2.0)
        self.assertEqual(engine.command_state.commanded_positions[5], 0.9)       # the fake moved there


class BoundedTemplateTests(unittest.TestCase):
    def test_settings_place_the_wall_and_the_limits_around_the_arc(self):
        settings = bounded_settings(200, dict(arc_fraction=0.7, wall_margin=15))
        self.assertEqual((settings["limit_arc"], settings["wall"]), ([30, 170], [15, 185]))
        self.assertEqual(bounded_settings(200)["arc_fraction"], BOUNDED["arc_fraction"])

    def test_bounded_graph_builds_consistently_and_runs_on_the_fakes(self):
        graph = two_ring_single_joint(joint=5, bounds="bounded", calibration="legacy")
        b = bounded_settings(200)
        enc, state = graph.blocks["enc_state"], graph.blocks["state"]
        self.assertEqual((enc.params["mapping"], enc.params["arc_fraction"]), ("bounded", b["arc_fraction"]))
        self.assertEqual((state.params["mapping"], state.params["joint_min"]), ("bounded", enc.params["joint_min"]))
        self.assertEqual(graph.blocks["r1"].params["wall"], b["wall"])
        self.assertEqual(graph.blocks["gain"].params["limit_arc"], b["limit_arc"])
        self.assertTrue(graph.blocks["cmp"].params["weights_artifact"].startswith("homeostasis/"))
        record = run_graph(graph, engines="fake", goals=[0.4])[0]["record"]
        self.assertIn("state", record.main_ticks[-1].outputs)
        self.assertIn("centroid_circular", record.main_ticks[-1].inputs["r1"].data)
        legacy = two_ring_single_joint(joint=5)
        self.assertNotIn("state", legacy.blocks)
        self.assertIn("state", two_ring_single_joint(joint=5, state_readout=True).blocks)
        with self.assertRaises(ValueError):
            two_ring_single_joint(bounds="nope")


if __name__ == "__main__":
    unittest.main()
