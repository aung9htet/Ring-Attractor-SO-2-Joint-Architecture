"""Transport calibration: the fit on synthetic trials, geometry, decoder dead zone, artifact wiring.

The real-NEST gate is in test_transport_real_nest.py.  Method: docs/archive/velocity/calibration.md.
"""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.blocks import Decoder  # noqa: E402
from tiago_ring_controller.control.profiles import LEGACY_CONTROL_PROFILES  # noqa: E402
from tiago_ring_controller.graph import (  # noqa: E402
    calibration_report,
    compile_graph,
    load_transport_calibration,
    network_fingerprint,
    resolve_joint_limits,
    run_transport_trials,
    transport_decoder_params,
    transport_motif,
    two_ring_single_joint,
)
from tiago_ring_controller.graph.calibration import ACTUATOR_TOLERANCE, index_pairs  # noqa: E402
from tiago_ring_controller.graph.templates import decoder_params  # noqa: E402
from tiago_ring_controller.math.control import dead_zone  # noqa: E402
from tiago_ring_controller.training.transport import (  # noqa: E402
    TransportOptions,
    TransportTrial,
    classify_trial,
    decoder_gains,
    fit_transport,
    trial_displacement,
    trials_from_arrays,
    trials_to_arrays,
    unwrapped_centroid,
)

A_TRUE, THETA_TRUE = 0.0052, 18.0


def synthetic_trial(start, distance, rate, seed, a=A_TRUE, theta=THETA_TRUE, ring_size=200, noise=0.3, stuck_after=None):
    """A bump that moves ``a * dz(rate)`` per tick until it has covered ``distance`` indices."""

    rng = np.random.default_rng(seed)
    direction = np.sign(distance)
    left, right, centroid = [0.0, 25.0, 60.0], [0.0, 25.0, 60.0], [start, start, start]   # injection ticks
    position = float(start)
    while abs(position - start) < abs(distance) and len(left) < 600:
        count = max(rate + rng.normal(0.0, 4.0), 0.0)
        moving = stuck_after is None or len(left) < stuck_after
        if moving:
            position += direction * a * max(count - theta, 0.0)
        left.append(count if direction < 0 else 0.0)
        right.append(count if direction > 0 else 0.0)
        centroid.append(position)
    for _ in range(12):                                                                    # settled
        left.append(0.0)
        right.append(0.0)
        centroid.append(position)
    trace = np.array(centroid) + rng.normal(0.0, noise, len(centroid))
    return TransportTrial(np.array(left), np.array(right), trace, ring_size, start_index=int(start),
                          goal_index=int(start + distance), group=seed % 4, label="synthetic")


def synthetic_set():
    trials = []
    seed = 0
    for start, distance in ((100, 50), (100, -50), (60, 70), (150, -80), (80, 30), (120, -30)):
        for rate in (35.0, 60.0, 95.0):
            seed += 1
            trials.append(synthetic_trial(start, distance, rate, seed))
    return trials


class TransportFitTests(unittest.TestCase):
    def test_fit_recovers_the_gain_and_the_dead_zone_and_prefers_it_over_the_linear_model(self):
        fit = fit_transport(synthetic_set())
        self.assertEqual(fit.model, "dead_zone")
        self.assertLessEqual(abs(fit.dead_zone - THETA_TRUE), 2.0)
        self.assertAlmostEqual(fit.index_per_spike, A_TRUE, delta=0.0004)
        self.assertEqual(fit.counts["ok"], 18)
        self.assertLess(fit.cv["dead_zone"].p90_abs_rel, 0.08)
        self.assertGreater(fit.cv["linear"].p90_abs_rel, 2 * fit.cv["dead_zone"].p90_abs_rel)
        self.assertEqual(fit.candidates["linear"][1], 0.0)
        self.assertEqual(sorted(fit.as_dict()), ["candidates", "counts", "cv", "dead_zone", "folds", "index_per_spike", "model"])

    def test_a_network_without_a_dead_zone_keeps_the_linear_model(self):
        trials = []
        for seed, (start, distance, rate) in enumerate(((100, 50, 40.0), (100, -50, 80.0), (60, 70, 60.0), (150, -80, 50.0),
                                                        (90, 40, 90.0), (120, -40, 35.0), (70, 60, 70.0), (140, -60, 45.0))):
            trials.append(synthetic_trial(start, distance, rate, seed, a=0.004, theta=0.0))
        fit = fit_transport(trials)
        self.assertEqual((fit.model, fit.dead_zone), ("linear", 0.0))
        self.assertAlmostEqual(fit.index_per_spike, 0.004, delta=0.0003)

    def test_trials_the_decoder_cannot_represent_are_counted_and_left_out(self):
        options = TransportOptions()
        still = synthetic_trial(100, 2, 60.0, 1)
        self.assertEqual(classify_trial(still, options), "no_motion")
        edge = synthetic_trial(150, 45, 60.0, 2)
        self.assertEqual(classify_trial(edge, options), "edge")
        silent = TransportTrial(np.zeros(40), np.zeros(40), np.full(40, np.nan), 200)
        self.assertEqual(classify_trial(silent, options), "silent")
        stuck = synthetic_trial(100, 60, 90.0, 3, stuck_after=60)
        self.assertEqual(classify_trial(stuck, options), "ok")                    # not without a fitted gain
        self.assertEqual(classify_trial(stuck, options, A_TRUE, THETA_TRUE), "stuck")
        good = synthetic_set()
        fit = fit_transport(good + [still, edge, silent, stuck])
        self.assertEqual((fit.counts["ok"], fit.counts["no_motion"], fit.counts["edge"], fit.counts["silent"], fit.counts["stuck"]),
                         (18, 1, 1, 1, 1))
        self.assertAlmostEqual(fit.index_per_spike, fit_transport(good).index_per_spike)
        with self.assertRaisesRegex(ValueError, "only 0 usable trials"):
            fit_transport([still, silent])

    def test_centroid_is_circular_and_unwrapped_across_the_seam(self):
        size = 200
        counts = np.zeros((30, size))
        centers = 190.0 + np.arange(30)                       # crosses 199 -> 0
        for tick, center in enumerate(centers):
            for offset in range(-4, 5):
                counts[tick, int(round(center + offset)) % size] = 5 - abs(offset)
        counts[7] = 0.0                                       # a silent tick
        trace = unwrapped_centroid(counts)
        self.assertTrue(np.isnan(trace[7]))
        good = np.isfinite(trace)
        np.testing.assert_allclose(trace[good], centers[good], atol=0.05)
        self.assertGreater(trace[-1], 215.0)                  # kept counting past the seam
        with self.assertRaises(ValueError):
            unwrapped_centroid(np.zeros(10))

    def test_trials_round_trip_through_arrays(self):
        trials = synthetic_set()[:3]
        again = trials_from_arrays(trials_to_arrays(trials))
        for a, b in zip(trials, again):
            np.testing.assert_array_equal(a.left, b.left)
            np.testing.assert_array_equal(a.centroid, b.centroid)
            self.assertEqual((a.start_index, a.goal_index, a.ring_size, str(a.group)), (b.start_index, b.goal_index, b.ring_size, b.group))
            self.assertEqual(trial_displacement(a), trial_displacement(b))


class GeometryTests(unittest.TestCase):
    LIMITS = (-1.4172695167359413, 1.4171452891568528)

    def test_radians_per_index_is_the_slope_of_the_mapping(self):
        for name, profile in LEGACY_CONTROL_PROFILES.items():
            with self.subTest(profile=name):
                rho = profile.radians_per_index(self.LIMITS[0], self.LIMITS[1], 200, 5)
                positions = np.linspace(self.LIMITS[0] + 0.05, self.LIMITS[1] - 0.05, 4001)
                indices = [profile.joint_to_ring_index(q, self.LIMITS[0], self.LIMITS[1], 200, 5) for q in positions]
                slope = np.polyfit(positions, indices, 1)[0]
                self.assertAlmostEqual(rho * slope, 1.0, delta=0.002)

    def test_index_to_joint_inverts_the_mapping(self):
        for name, profile in LEGACY_CONTROL_PROFILES.items():
            first, span = profile.index_range(200, 5)
            for index in range(int(first) + 1, int(first + span) - 1, 7):
                with self.subTest(profile=name, index=index):
                    q = profile.ring_index_to_joint(index, self.LIMITS[0], self.LIMITS[1], 200, 5)
                    self.assertEqual(profile.joint_to_ring_index(q, self.LIMITS[0], self.LIMITS[1], 200, 5), index)

    def test_encoder_exposes_the_geometry_before_and_after_the_build(self):
        graph = two_ring_single_joint()
        encoder = graph.blocks["enc_state"]
        rho = encoder.radians_per_index(200)
        self.assertAlmostEqual(rho, (self.LIMITS[1] - self.LIMITS[0]) / 190.0)
        self.assertEqual(encoder.ring_index(encoder.index_to_angle(77, 200)) if encoder.ring_size else 77, 77)
        with self.assertRaises(ValueError):
            encoder.radians_per_index()                       # ring size unknown before the build
        compile_graph(graph, engines="fake").loop.reset()
        self.assertEqual(encoder.radians_per_index(), rho)
        self.assertEqual(encoder.ring_index(encoder.index_to_angle(77)), 77)

    def test_decoder_gain_is_a_rho_over_dt(self):
        positive, negative = decoder_gains(0.0053, 0.0149, 0.05)
        self.assertAlmostEqual(positive, 0.0053 * 0.0149 / 0.05)
        self.assertEqual(negative, -positive)
        self.assertAlmostEqual(decoder_gains(0.0053, 0.0149, 0.05, actuator_gain=0.9)[0], positive / 0.9)
        with self.assertRaises(ValueError):
            decoder_gains(0.0053, 0.0149, 0.0)


class DecoderDeadZoneTests(unittest.TestCase):
    def test_dead_zone_function(self):
        self.assertEqual(dead_zone(7.0, 0.0), 7.0)
        np.testing.assert_array_equal(dead_zone(np.array([-30.0, -10.0, 0.0, 10.0, 30.0]), 20.0), [-10.0, 0.0, 0.0, 0.0, 10.0])
        self.assertEqual(dead_zone(-25.0, 20.0), -5.0)

    def test_zero_dead_zone_is_todays_decoder_and_a_dead_zone_removes_the_offset(self):
        counts = [(0, 30), (0, 60), (5, 90), (40, 0), (10, 12), (0, 0)]
        plain = Decoder("a", gain_positive=0.001, gain_negative=-0.001, tau_s=0.1)
        zero = Decoder("b", gain_positive=0.001, gain_negative=-0.001, tau_s=0.1, dead_zone=0.0)
        dead = Decoder("c", gain_positive=0.001, gain_negative=-0.001, tau_s=1e-6, dead_zone=20.0)
        for block in (plain, zero, dead):
            block.configure(50.0)
        for left, right in counts:
            a, b, c = plain.consume(left, right), zero.consume(left, right), dead.consume(left, right)
            self.assertEqual(a, b)
            self.assertEqual(c["signed_spike"], float(right - left))             # the record keeps the raw count
            self.assertAlmostEqual(c["filtered_drive"], dead_zone(float(right - left), 20.0), places=6)
        self.assertAlmostEqual(dead.buffer[-3]["decoded_velocity"], -0.001 * 20.0, places=6)   # (40, 0): -40 -> -20


class ArtifactWiringTests(unittest.TestCase):
    def test_checked_in_artifact_matches_the_example_network(self):
        calibration = load_transport_calibration()
        self.assertEqual(calibration["schema"], "transport-calibration/1")
        self.assertEqual(calibration["network_fingerprint"], network_fingerprint(two_ring_single_joint()))
        self.assertEqual(calibration["dt_ms"], 50.0)
        self.assertGreater(calibration["index_per_spike"], 0.0)
        self.assertEqual(calibration["fit"]["model"], calibration["model"])

    def test_fingerprint_follows_the_network_not_the_joint(self):
        base = network_fingerprint(two_ring_single_joint(joint=5))
        self.assertEqual(base, network_fingerprint(two_ring_single_joint(joint=2, goal_rad=-1.0, seed=7)))
        changed = two_ring_single_joint()
        changed.blocks["gain"].params["gain_to_ring_weight"] = -0.7
        self.assertNotEqual(base, network_fingerprint(changed))
        slower = two_ring_single_joint()
        slower.simulation.dt_ms = 25.0
        self.assertNotEqual(base, network_fingerprint(slower))
        motif = transport_motif(two_ring_single_joint())
        self.assertEqual({role: block.id for role, block in motif.items()}, {
            "decoder": "dec", "joint": "j5", "gain": "gain", "state_ring": "r1", "comparator": "cmp",
            "state_readout": "f1", "goal_readout": "f2", "goal_ring": "r2", "state_encoder": "enc_state", "goal_encoder": "enc_goal"})

    def test_template_gains_scale_with_the_joint_range(self):
        calibration = load_transport_calibration()
        five, four = decoder_params(5, calibration="transport"), decoder_params(4, calibration="transport")
        rho5 = (1.4171452891568528 + 1.4172695167359413) / 190.0
        self.assertAlmostEqual(five["gain_positive"], calibration["index_per_spike"] * rho5 / 0.05)
        self.assertEqual(five["gain_negative"], -five["gain_positive"])
        self.assertEqual((five["dead_zone"], five["tau_s"], five["delay_steps"]),
                         (calibration["dead_zone"], calibration["decoder_tau_s"], calibration["decoder_delay_steps"]))
        self.assertAlmostEqual(four["gain_positive"] / five["gain_positive"], (2.0944328056118007 + 2.0945012771678044) / (190.0 * rho5))
        legacy = decoder_params(5, calibration="legacy")
        self.assertEqual((legacy["gain_positive"], "dead_zone" in legacy), (0.0011, False))
        with self.assertRaises(ValueError):
            decoder_params(5, calibration="nope")

    def test_actuator_gain_is_applied_only_outside_the_tolerance(self):
        calibration = dict(load_transport_calibration())
        plain = transport_decoder_params(calibration, 0.015, joint=5)["gain_positive"]
        calibration["actuator"] = {"5": {"displacement_gain": 1.0 - ACTUATOR_TOLERANCE / 2}, "4": {"displacement_gain": 0.9}}
        self.assertEqual(transport_decoder_params(calibration, 0.015, joint=5)["gain_positive"], plain)
        self.assertAlmostEqual(transport_decoder_params(calibration, 0.015, joint=4)["gain_positive"], plain / 0.9)
        self.assertEqual(transport_decoder_params(calibration, 0.015, joint=1)["gain_positive"], plain)

    def test_report_says_whether_a_graph_uses_the_calibration_and_flags_a_stale_one(self):
        graph = two_ring_single_joint(calibration="transport")
        row = calibration_report(graph)[0]
        self.assertEqual((row["decoder"], row["joint"], row["uses_transport_calibration"], row["network_matches"], row["warning"]),
                         ("dec", 5, True, True, None))
        self.assertFalse(calibration_report(two_ring_single_joint(calibration="legacy"))[0]["uses_transport_calibration"])
        graph.blocks["gain"].params["ring_to_gain_weight"] = 400000.0
        stale = calibration_report(graph)[0]
        self.assertFalse(stale["network_matches"])
        self.assertIn("rerun scripts/calibrate_transport.py", stale["warning"])
        self.assertEqual(compile_graph(graph, engines="fake").meta["calibration"][0]["warning"], stale["warning"])

    def test_urdf_limits_rescale_the_decoder_gains_with_the_mapping(self):
        graph = two_ring_single_joint(joint=6, calibration="transport")
        graph.blocks["j6"].params["limits_source"] = "urdf"
        before = graph.blocks["dec"].params["gain_positive"]
        old = graph.blocks["j6"].limits()

        class Transport:
            def get_param(self, name):
                return '<robot><joint name="arm_7_joint"><limit lower="-1.39" upper="1.39"/></joint></robot>'

        changed = resolve_joint_limits(graph, Transport())
        factor = 2.78 / (old[1] - old[0])
        self.assertAlmostEqual(changed["j6"]["decoder"]["gain_factor"], factor)
        self.assertAlmostEqual(graph.blocks["dec"].params["gain_positive"], before * factor)
        self.assertTrue(calibration_report(graph)[0]["uses_transport_calibration"])      # still a * rho / dt

    def test_transport_trials_run_on_the_fakes(self):
        self.assertEqual(index_pairs([35, 60], [35, 60, 85], 22), [(35, 60), (35, 85), (60, 35), (60, 85)])
        seen = []
        trials, rows = run_transport_trials(
            lambda goal, seed: two_ring_single_joint(goal_rad=goal, **({} if seed is None else {"seed": seed})),
            [(60, 110)], [1, 2], engines="fake", on_trial=seen.append)
        self.assertEqual(len(trials), 2)
        self.assertEqual([(r["seed"], r["start_index"], r["goal_index"], r["status"]) for r in rows],
                         [(1, 60, 110, "silent"), (2, 60, 110, "silent")])        # the fake NEST does not spike
        self.assertEqual((trials[0].start_index, trials[0].goal_index, trials[0].ring_size), (60, 110, 200))
        self.assertEqual(seen, rows)


if __name__ == "__main__":
    unittest.main()
