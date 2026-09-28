"""Characterize analytic/ridge fitting and pure evaluation/output contracts."""

import math
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.contracts import DecoderSpec  # noqa: E402
from tiago_ring_controller.evaluation.metrics import (  # noqa: E402
    aligned_position_error,
    analysis_region_statistics,
    centroid_change,
    circular_angle_error,
    circular_index_error,
    circular_index_nearest,
    circular_signed_difference,
    collector_batch_statistics,
    decode_circular_centroid,
    goal_error,
    movement_direction,
    overshoot,
    relative_goal_error_after_time,
    scalar_readout_metrics,
    settling_time,
)
from tiago_ring_controller.evaluation.plots import (  # noqa: E402
    break_circular_wraps_for_plot,
    evenly_spaced_indices,
    final_absolute_error,
    gain_imbalance,
    goal_from_timeseries,
    mean_std_box,
    normalize_profile_for_display,
    paired_by_trial_index,
    representative_trial_index,
)
from tiago_ring_controller.evaluation.serialization import (  # noqa: E402
    PID_BATCH_STAT_FIELDS,
    PID_TIMESERIES_FIELDS,
    PID_TRIAL_SCALAR_FIELDS,
    RING_BATCH_STAT_FIELDS,
    RING_RASTER_FIELDS,
    RING_TIMESERIES_FIELDS,
    RING_TRIAL_SCALAR_FIELDS,
    array_payload,
    array_payload_schema,
    assert_exact_fields,
    json_safe,
    legacy_json_text,
    ordered_fields,
    slug,
)
from tiago_ring_controller.features import embed_homeostasis_weights  # noqa: E402
from tiago_ring_controller.math.circular import (  # noqa: E402
    circular_signed_difference as ring_circular_signed_difference,
)
from tiago_ring_controller.training.analytic import (  # noqa: E402
    build_fourier_weights,
    build_homeostasis_weight_matrix,
    build_scalar_ramp_weights,
)
from tiago_ring_controller.training.calibration import (  # noqa: E402
    fit_decoder_from_trials,
    fit_signed_direction_gains,
    update_decoder_parameters,
)
from tiago_ring_controller.training.ridge import (  # noqa: E402
    fit_normalized_ridge,
    fit_ridge,
)


class AnalyticArtifactTests(unittest.TestCase):
    def test_fourier_construction_matches_both_checked_in_artifacts(self):
        for population_size, harmonics in ((100, 5), (200, 20)):
            expected = np.load(
                SRC / (
                    "config/ring_decoding_weights/N_%d_fourier_weights.npy"
                    % population_size
                ),
                allow_pickle=False,
            )
            actual = build_fourier_weights(population_size, harmonics)
            with self.subTest(population_size=population_size):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(str(actual.dtype), "float64")
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=4e-16)

    def test_homeostasis_push_pull_embedding_layout_is_exact(self):
        signed = np.array([1.5, -2.0])
        expected = np.array(
            [
                [1.5, 0.0],
                [-1.5, 0.0],
                [-2.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.5],
                [0.0, -1.5],
                [0.0, -2.0],
                [0.0, 2.0],
            ]
        )
        np.testing.assert_array_equal(embed_homeostasis_weights(signed), expected)
        np.testing.assert_array_equal(build_homeostasis_weight_matrix(signed), expected)

    def test_scalar_ramp_matches_checked_in_artifact(self):
        with np.load(
            SRC / "config/ring_decoding_weights/N_100_J_2_multi_ring_sawtooth_scalar_weights.npz",
            allow_pickle=False,
        ) as archive:
            expected = archive["scalar_ramp_weights"]
        np.testing.assert_array_equal(build_scalar_ramp_weights(100, 100.0), expected)


class RidgeSolverTests(unittest.TestCase):
    def test_primal_branch_matches_the_script_equation(self):
        rng = np.random.RandomState(11)
        features = rng.normal(size=(9, 3))
        targets = rng.normal(size=(9, 2))
        ridge_lambda = 1.0e-4
        expected = np.linalg.solve(
            features.T @ features + ridge_lambda * np.eye(3),
            features.T @ targets,
        )
        actual = fit_ridge(features, targets, ridge_lambda)
        np.testing.assert_array_equal(actual, expected)

    def test_dual_branch_matches_equation_and_primal_dual_identity(self):
        rng = np.random.RandomState(22)
        features = rng.normal(size=(4, 7))
        targets = rng.normal(size=(4, 3))
        ridge_lambda = 0.03
        dual = features.T @ np.linalg.solve(
            features @ features.T + ridge_lambda * np.eye(4), targets
        )
        primal = np.linalg.solve(
            features.T @ features + ridge_lambda * np.eye(7),
            features.T @ targets,
        )
        np.testing.assert_array_equal(fit_ridge(features, targets, ridge_lambda), dual)
        np.testing.assert_allclose(dual, primal, rtol=1e-10, atol=1e-12)

    def test_normalized_ridge_matches_independent_legacy_reference(self):
        features = np.array(
            [[1.0, 5.0, -1.0], [2.0, 5.0, 0.5], [4.0, 5.0, 2.0], [8.0, 5.0, 4.0]]
        )
        targets = np.array([-0.2, 0.0, 0.4, 0.9])
        ridge_lambda = 1e-6
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0.0] = 1.0
        normalized = (features - mean) / std
        augmented = np.concatenate([normalized, np.ones((4, 1))], axis=1)
        theta = np.linalg.solve(
            augmented.T @ augmented + ridge_lambda * np.eye(4),
            augmented.T @ targets,
        )
        weights = theta[:-1] / std
        bias = float(theta[-1] - (mean / std) @ theta[:-1])
        predictions = features @ weights + bias

        fit = fit_normalized_ridge(features, targets, ridge_lambda)
        np.testing.assert_array_equal(fit.feature_mean, mean)
        np.testing.assert_array_equal(fit.feature_std, std)
        np.testing.assert_array_equal(fit.theta_normalized, theta)
        np.testing.assert_array_equal(fit.weights, weights)
        self.assertEqual(fit.bias, bias)
        np.testing.assert_array_equal(fit.predictions, predictions)
        self.assertEqual(fit.rmse, float(np.sqrt(np.mean((targets - predictions) ** 2))))


class CalibrationFitTests(unittest.TestCase):
    def test_direction_gain_regularization_clipping_and_too_few_samples(self):
        self.assertIsNone(fit_signed_direction_gains([[1.0, 0.0]], [1.0]))
        features = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        targets = np.array([100.0, 200.0, -100.0, -200.0])
        self.assertEqual(fit_signed_direction_gains(features, targets), (0.01, -0.01))

    def test_decoder_grid_search_keeps_arrays_schema_and_skips_zero_activity(self):
        trials = [
            {"spike_history": np.array([0.0, 0.0]), "dt": 0.1, "dq_desired": 99.0, "dq_actual": 99.0},
            {"spike_history": np.array([4.0, 0.0]), "dt": 0.1, "dq_desired": 0.2, "dq_actual": 0.18},
            {"spike_history": np.array([-5.0, 0.0]), "dt": 0.1, "dq_desired": -0.25, "dq_actual": -0.2},
        ]
        fit = fit_decoder_from_trials(trials, tau_candidates=(0.3,), delay_candidates=(0,))
        self.assertIsNotNone(fit)
        self.assertEqual((fit.tau, fit.delay_steps, fit.n_valid), (0.3, 0, 2))
        self.assertEqual(fit.X.shape, (2, 2))
        np.testing.assert_array_equal(fit.y_desired, [0.2, -0.25])
        np.testing.assert_array_equal(fit.y_actual, [0.18, -0.2])
        self.assertEqual(tuple(fit.as_legacy_dict()), (
            "tau", "delay_steps", "k_pos", "k_neg", "desired_mse",
            "actual_mse", "desired_mae", "actual_mae", "n_valid",
            "X", "y_desired", "y_actual", "y_pred",
        ))

    def test_decoder_ema_updates_tau_and_replaces_integer_delay(self):
        trials = [
            {"spike_history": np.array([4.0, 0.0]), "dt": 0.1, "dq_desired": 0.2, "dq_actual": 0.18},
            {"spike_history": np.array([-5.0, 0.0]), "dt": 0.1, "dq_desired": -0.25, "dq_actual": -0.2},
        ]
        fit = fit_decoder_from_trials(trials, tau_candidates=(0.5,), delay_candidates=(1,))
        current = DecoderSpec(0.001, -0.002, 0.3, 4, "fixture")
        updated = update_decoder_parameters(current, fit, beta=0.2)
        self.assertEqual(updated.gain_positive, 0.8 * current.gain_positive + 0.2 * fit.k_pos)
        self.assertEqual(updated.gain_negative, 0.8 * current.gain_negative + 0.2 * fit.k_neg)
        self.assertEqual(updated.tau, 0.8 * 0.3 + 0.2 * 0.5)
        self.assertEqual(updated.delay_steps, 1)
        self.assertEqual(updated.source, "calibration_fit")
        self.assertIs(update_decoder_parameters(current, None), current)


class EvaluationMetricTests(unittest.TestCase):
    def test_circular_centroid_zero_activity_cardinal_and_strength(self):
        rates = np.zeros((4, 3))
        rates[1, 1] = 3.0
        rates[3, 2] = 2.0
        centroid, strength = decode_circular_centroid(rates)
        self.assertTrue(np.isnan(centroid[0]))
        np.testing.assert_allclose(centroid[1:], [1.0, 3.0], rtol=0.0, atol=2e-16)
        np.testing.assert_array_equal(strength, [0.0, 1.0, 1.0])

    def test_half_open_circular_difference_and_wrap_metrics(self):
        self.assertEqual(circular_signed_difference(50, 0, 100), -50.0)
        self.assertEqual(circular_signed_difference(0, 99, 100), 1.0)
        np.testing.assert_array_equal(circular_index_nearest([99, 1], [0, 99], 100), [-1.0, 101.0])
        np.testing.assert_array_equal(circular_index_error([99, 1], [0, 99], 100), [-1.0, 2.0])
        np.testing.assert_allclose(circular_angle_error([-math.pi + 0.1], [math.pi - 0.1]), [0.2], rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(centroid_change([99.0, 1.0, np.nan, 3.0], 100), [np.nan, 2.0, np.nan, np.nan])
        signed, absolute = goal_error([99.0, 1.0, np.nan], 0.0, 100)
        np.testing.assert_array_equal(signed, [1.0, -1.0, np.nan])
        np.testing.assert_array_equal(absolute, [1.0, 1.0, np.nan])

    def test_ring_circular_difference_preserves_numpy_scalar_and_rejects_lists(self):
        result = ring_circular_signed_difference(
            np.float64(0.0), np.float64(99.0), 100
        )
        self.assertIsInstance(result, np.float64)
        self.assertEqual(result, np.float64(1.0))
        with self.assertRaises(TypeError):
            ring_circular_signed_difference([0.0], [99.0], 100)

    def test_relative_goal_error_uses_strict_time_mask_then_fallback(self):
        self.assertEqual(
            relative_goal_error_after_time([0, 30000, 30001], [20, 10, 5], 100, 30000),
            0.05,
        )
        self.assertAlmostEqual(
            relative_goal_error_after_time([0, 10], [20, 10], 100, 30000),
            0.15,
        )
        self.assertTrue(np.isnan(relative_goal_error_after_time([0], [np.nan], 100, 30000)))

    def test_trial_direction_alignment_settling_and_overshoot_edge_semantics(self):
        self.assertEqual(movement_direction(1e-9), 0.0)
        self.assertEqual(movement_direction(1.0001e-9), 1.0)
        self.assertAlmostEqual(aligned_position_error(1.0, 0.8, 0.5), 0.2)
        self.assertAlmostEqual(aligned_position_error(0.0, 0.2, -0.5), 0.2)
        times = np.arange(7) * 0.1
        errors = [0.2, 0.05, 0.049, 0.04, 0.03, 0.02, 0.01]
        # 0.05 does not satisfy the strict '< tolerance' criterion.
        self.assertEqual(settling_time(errors, times, 0.05, 5), 0.2)
        self.assertAlmostEqual(overshoot([0.0, 0.8, 1.2, 1.1], 1.0), 0.2)

    def test_collector_statistics_preserve_thresholds_and_stop_reason_schema(self):
        trials = []
        for error, stop, predicted in ((0.01, "drive_settled", 0.4), (0.06, "max_steps", 0.1)):
            trials.append({
                "scalars": {
                    "abs_position_error_rad": error,
                    "aligned_error_rad": error,
                    "dq_desired": 0.5,
                    "dq_actual": 0.45,
                    "decoder_predicted_dq_rad": predicted,
                    "stop_reason": stop,
                }
            })
        stats = collector_batch_statistics(trials)
        self.assertEqual(stats["n_trials"], 2)
        self.assertAlmostEqual(stats["mean_abs_error_rad"], 0.035)
        self.assertEqual(stats["success_rate_le_0p02_rad"], 0.5)
        self.assertEqual(stats["success_rate_le_0p05_rad"], 0.5)
        self.assertEqual(stats["success_rate_le_0p10_rad"], 1.0)
        self.assertEqual(stats["max_steps_count"], 1)
        self.assertEqual(stats["drive_settled_count"], 1)

    def test_empty_analysis_region_retains_none_metrics_and_zero_counts(self):
        stats = analysis_region_statistics([])
        self.assertEqual(stats["n_trials"], 0)
        self.assertIsNone(stats["mean_abs_error_rad"])
        self.assertIsNone(stats["success_rate_abs_error_le_0p10_rad"])
        self.assertEqual(stats["max_steps_count"], 0)
        self.assertEqual(stats["drive_settled_count"], 0)

    def test_scalar_metric_preserves_noncircular_index_mae_and_constant_signal_nan(self):
        metrics = scalar_readout_metrics(
            saw_indices=[99.0, 1.0, np.nan],
            true_indices=[0.0, 2.0, 50.0],
            scalar_counts=[1.0, 2.0, 3.0],
            output_ring_size=100,
            sawtooth_errors_deg=[2.0, 4.0],
        )
        self.assertEqual(metrics["sawtooth_index_mae"], 50.0)
        self.assertEqual(metrics["sawtooth_mae_deg"], 3.0)
        self.assertAlmostEqual(metrics["signal_coverage_pct"], 200.0 / 3.0)
        constant = scalar_readout_metrics([0.0, 1.0], [0.0, 1.0], [4.0, 4.0], 100)
        self.assertTrue(np.isnan(constant["scalar_vs_sawtooth_correlation"]))
        self.assertTrue(all(np.isnan(value) for value in constant["scalar_norm"]))


class EvaluationSerializationAndPlotDataTests(unittest.TestCase):
    def test_field_orders_and_schema_family_suffixes_are_exact(self):
        self.assertEqual(RING_TRIAL_SCALAR_FIELDS[:3], ("joint_index", "batch_idx", "iteration_idx"))
        self.assertEqual(RING_TRIAL_SCALAR_FIELDS[-4:], ("decoder_gain_positive", "decoder_gain_negative", "decoder_tau", "decoder_delay_steps"))
        self.assertEqual(PID_TRIAL_SCALAR_FIELDS[-4:], ("kp", "ki", "kd", "output_limit"))
        self.assertEqual(RING_BATCH_STAT_FIELDS[-1], "drive_settled_count")
        self.assertEqual(PID_BATCH_STAT_FIELDS[-1], "error_settled_count")
        self.assertEqual(RING_TIMESERIES_FIELDS[3], "signed_spike")
        self.assertEqual(PID_TIMESERIES_FIELDS, ("time", "joint_position", "joint_velocity", "position_error", "p_term", "i_term", "d_term", "decoded_velocity"))
        self.assertEqual(RING_RASTER_FIELDS[:2], ("r1_times", "r1_senders"))

    def test_ordered_payload_schema_and_exact_field_rejection(self):
        values = OrderedDict((("a", [1, 2]), ("b", np.array(3.0))))
        ordered = ordered_fields(values, ("b", "a"))
        self.assertEqual(tuple(ordered), ("b", "a"))
        payload = array_payload(values, ("a", "b"))
        self.assertEqual(tuple(payload), ("a", "b"))
        self.assertEqual(
            array_payload_schema(payload),
            (
                {"key": "a", "shape": [2], "dtype": "int64"},
                {"key": "b", "shape": [], "dtype": "float64"},
            ),
        )
        assert_exact_fields(values, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "Payload fields differ"):
            assert_exact_fields(values, ("b", "a"))

    def test_json_safe_slug_and_permissive_nan_text(self):
        value = {
            np.int64(2): np.array([np.float64(1.5)]),
            "flag": np.bool_(True),
            "nested": (np.int32(3),),
        }
        self.assertEqual(json_safe(value), {"2": [1.5], "flag": True, "nested": [3]})
        self.assertIn("NaN", legacy_json_text({"value": float("nan")}))
        self.assertEqual(slug("joint 3/x:y"), "joint_3_x_y")

    def test_plot_data_helpers_preserve_pairing_wrap_and_display_only_normalization(self):
        timeseries = {
            "joint_position": np.array([0.0, 0.5]),
            "position_error": np.array([1.0, -0.2]),
            "right_gain_spikes": np.array([5, 2]),
            "left_gain_spikes": np.array([1, 3]),
        }
        np.testing.assert_array_equal(goal_from_timeseries(timeseries), [1.0, 0.3])
        self.assertEqual(final_absolute_error(timeseries), 0.2)
        np.testing.assert_array_equal(gain_imbalance(timeseries), [4, -1])
        self.assertEqual(representative_trial_index([4.0, 1.0, 3.0, 2.0]), 2)
        self.assertIsNone(representative_trial_index([]))
        ring, pid = paired_by_trial_index([1, 2, 3], [9, 8])
        np.testing.assert_array_equal(ring, [1, 2])
        np.testing.assert_array_equal(pid, [9, 8])
        wrapped = break_circular_wraps_for_plot([98.0, 2.0, 3.0], 100)
        np.testing.assert_array_equal(wrapped, [98.0, np.nan, 3.0])
        normalized = normalize_profile_for_display([2.0, 4.0, 6.0])
        np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0], rtol=0.0, atol=5e-10)
        self.assertEqual(mean_std_box([1.0, 3.0]), (2.0, 1.0, 3.0))
        self.assertEqual(evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices(3, 5), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
