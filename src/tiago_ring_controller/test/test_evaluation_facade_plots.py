"""Offline facade/figure contracts for extracted evaluation helpers."""

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Plot-only tests do not construct a NEST network.  The default system Python
# used by ``unittest discover`` may not have the pinned NEST path installed, so
# provide an import-only stand-in and remove it immediately after the legacy
# modules have loaded.  This does not mask NEST for the real-NEST smoke suite.
_remove_nest_stub = False
try:  # pragma: no cover - depends on the invoking environment
    import nest as _nest_probe  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised in source-only CI
    sys.modules["nest"] = types.ModuleType("nest")
    _remove_nest_stub = True

import multi_ring_sawtooth_scalar_decoder as scalar_module  # noqa: E402
import single_ring as single_ring_module  # noqa: E402

if _remove_nest_stub:
    del sys.modules["nest"]
import single_joint_data_visualizer as visualizer  # noqa: E402
from tiago_ring_controller.evaluation.metrics import (  # noqa: E402
    scalar_readout_metrics,
)
from tiago_ring_controller.evaluation.serialization import json_safe  # noqa: E402


class EvaluationFacadeDelegationTests(unittest.TestCase):
    def test_json_safe_recursive_callback_is_used_for_legacy_override_dispatch(self):
        calls = []

        def convert_child(value):
            calls.append(value)
            return json_safe(value, recursive_converter=convert_child)

        value = {"items": (np.int64(2), np.array([3.0]))}
        self.assertEqual(
            json_safe(value, recursive_converter=convert_child),
            {"items": [2, [3.0]]},
        )
        self.assertEqual(len(calls), 3)
        self.assertIs(calls[0], value["items"])

    def test_scalar_metric_facade_keeps_private_feature_hooks_and_key_order(self):
        inference = object.__new__(scalar_module.MultiRingScalarReadoutInference)
        inference.output_ring_size = 100
        inference.num_joints = 2

        hook_calls = {"kinematics": 0, "index": 0, "signal": 0, "decode": 0}

        def kinematics(q):
            hook_calls["kinematics"] += 1
            return (
                float(q[0] + q[1]),
                float(q[1]),
                float(q[0] + 2.0 * q[1]),
            )

        def to_index(angle, size):
            hook_calls["index"] += 1
            return float(angle * 10.0 + size / 2.0)

        def has_signal(profile):
            hook_calls["signal"] += 1
            return bool(np.any(profile))

        def decode(profile):
            hook_calls["decode"] += 1
            return float(profile[0])

        inference._lift_pitch_yaw_from_angles = kinematics
        inference._angle_to_ring_index = to_index
        inference._profile_has_signal = has_signal
        inference._decode_sawtooth_index = decode

        vary_angles = np.array([0.0, 0.25, 0.5])
        profiles = np.array([[10.0, 1.0], [20.0, 1.0], [30.0, 1.0]])
        vis = {
            "q1_fixed": {
                "vary_angles": vary_angles,
                "fixed_jt": 0,
                "fixed_val": 0.0,
                "vary_jt": 1,
                "lift_counts": profiles,
                "pitch_counts": profiles,
                "yaw_counts": profiles,
                "lift_scalar": np.array([1.0, 2.0, 4.0]),
                "pitch_scalar": np.array([2.0, 4.0, 8.0]),
                "yaw_scalar": np.array([3.0, 6.0, 12.0]),
            }
        }

        actual = inference._compute_scalar_metrics(vis)
        self.assertEqual(
            hook_calls,
            {"kinematics": 9, "index": 9, "signal": 9, "decode": 9},
        )

        pitch_true = [50.0, 52.5, 55.0]
        expected = scalar_readout_metrics(
            [10.0, 20.0, 30.0],
            pitch_true,
            [2.0, 4.0, 8.0],
            100,
            vary_angles=vary_angles,
        )
        # The facade alone computes the circular-angle errors through its legacy
        # private hooks; all downstream scalar reductions match the pure helper.
        for key in (
            "sawtooth_index_mae",
            "scalar_vs_sawtooth_correlation",
            "scalar_vs_true_correlation",
            "scalar_rmse_normalized",
            "signal_coverage_pct",
            "vary_angles",
            "true_norm",
            "saw_norm",
            "scalar_norm",
            "scalar_counts_raw",
            "scalar_count_min",
            "scalar_count_max",
        ):
            if isinstance(expected[key], list):
                np.testing.assert_allclose(
                    actual["q1_fixed"]["pitch"][key],
                    expected[key],
                    rtol=0.0,
                    atol=0.0,
                )
            else:
                self.assertEqual(actual["q1_fixed"]["pitch"][key], expected[key])
        self.assertEqual(
            tuple(actual["q1_fixed"]["pitch"]),
            (
                "sawtooth_mae_deg",
                "sawtooth_index_mae",
                "scalar_vs_sawtooth_correlation",
                "scalar_vs_true_correlation",
                "scalar_rmse_normalized",
                "signal_coverage_pct",
                "vary_angles",
                "true_norm",
                "saw_norm",
                "scalar_norm",
                "scalar_counts_raw",
                "scalar_count_min",
                "scalar_count_max",
            ),
        )

    def test_single_ring_metric_facades_preserve_private_difference_dispatch(self):
        model = object.__new__(single_ring_module.SingleRingModel)
        model.population_size = 100
        calls = []

        def difference(target, current):
            calls.append((target, current))
            delta = target - current
            return ((delta + 50.0) % 100.0) - 50.0

        model._circular_signed_difference = difference
        centroid = np.array([99.0, 1.0, np.nan, 3.0])
        np.testing.assert_array_equal(
            model._compute_centroid_change(centroid),
            [np.nan, 2.0, np.nan, np.nan],
        )
        signed, absolute = model._compute_goal_error(centroid, 0.0)
        np.testing.assert_array_equal(signed, [1.0, -1.0, np.nan, -3.0])
        np.testing.assert_array_equal(absolute, [1.0, 1.0, np.nan, 3.0])
        self.assertEqual(len(calls), 2)

        np.testing.assert_array_equal(
            model._break_wraps_for_plot(np.array([98.0, 2.0, 3.0])),
            [98.0, np.nan, 3.0],
        )
        model.get_goal_distance_trace = lambda **_kwargs: (
            np.array([0.0, 30000.0, 30001.0]),
            np.array([20.0, 10.0, 5.0]),
        )
        self.assertEqual(
            model.get_relative_goal_error_after_time(7, avg_after_t=30000.0),
            0.05,
        )


class FigureStructureTests(unittest.TestCase):
    @staticmethod
    def _write_trial(path, include_gain):
        payload = {
            "time": np.array([0.0, 0.1, 0.2]),
            "joint_position": np.array([0.0, 0.4, 0.8]),
            "position_error": np.array([1.0, 0.6, 0.2]),
            "decoded_velocity": np.array([0.5, 0.3, 0.1]),
        }
        if include_gain:
            payload.update(
                {
                    "left_gain_spikes": np.array([1.0, 2.0, 3.0]),
                    "right_gain_spikes": np.array([4.0, 2.0, 1.0]),
                }
            )
        np.savez_compressed(str(path), **payload)

    def test_visualizer_figures_keep_axes_labels_sizes_and_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ring_trials = root / "data" / "joint_3" / "trials"
            pid_trials = root / "data" / "joint_3" / "pid" / "trials"
            output = root / "figures"
            ring_trials.mkdir(parents=True)
            pid_trials.mkdir(parents=True)
            output.mkdir()
            self._write_trial(ring_trials / "trial_0001_timeseries.npz", True)
            self._write_trial(pid_trials / "trial_0001_timeseries.npz", False)

            captured = {}
            original_savefig = visualizer._savefig

            def capture_savefig(fig, path, fs, fw):
                captured[Path(path).name] = {
                    "size": tuple(fig.get_size_inches()),
                    "axes": len(fig.axes),
                    "xlabels": tuple(ax.get_xlabel() for ax in fig.axes),
                    "ylabels": tuple(ax.get_ylabel() for ax in fig.axes),
                    "titles": tuple(ax.get_title() for ax in fig.axes),
                }
                return original_savefig(fig, path, fs, fw)

            with mock.patch.object(visualizer, "BASE", str(root / "data")), mock.patch.object(
                visualizer, "_savefig", side_effect=capture_savefig
            ), contextlib.redirect_stdout(io.StringIO()):
                visualizer.plot_fig01(3, str(output), 10, "normal")
                visualizer.plot_fig02(3, str(output), 10, "normal")
                visualizer.plot_fig03(3, str(output), 10, "normal")
                visualizer.plot_fig05(3, str(output), 10, "normal")
                visualizer.plot_app10(3, str(output), 10, "normal")

            expected = {
                "fig_01_example_joint_trajectory.png": ((14.0, 5.0), 2),
                "fig_02_error_over_time.png": ((10.0, 5.0), 1),
                "fig_03_velocity_command_over_time.png": ((10.0, 5.0), 1),
                "fig_05_paired_final_error_scatter.png": ((6.0, 6.0), 1),
                "appendix_gain_imbalance_vs_velocity.png": ((7.0, 6.0), 1),
            }
            self.assertEqual(set(captured), set(expected))
            for filename, (size, axes) in expected.items():
                with self.subTest(filename=filename):
                    self.assertEqual(captured[filename]["size"], size)
                    self.assertEqual(captured[filename]["axes"], axes)
                    image = plt.imread(str(output / filename))
                    self.assertGreater(image.shape[0], 0)
                    self.assertGreater(image.shape[1], 0)

            self.assertEqual(
                captured["fig_01_example_joint_trajectory.png"]["xlabels"],
                ("Time (s)", "Time (s)"),
            )
            self.assertEqual(
                captured["fig_05_paired_final_error_scatter.png"]["ylabels"],
                ("Ring final error (rad)",),
            )

    def test_scalar_plot_family_keeps_grid_shape_and_canvas_dimensions(self):
        inference = object.__new__(scalar_module.MultiRingScalarReadoutInference)
        inference.output_ring_size = 100
        angles = np.array([-1.0, 0.0, 1.0])
        metric = {
            "sawtooth_mae_deg": 1.0,
            "sawtooth_index_mae": 2.0,
            "scalar_vs_sawtooth_correlation": 0.9,
            "scalar_vs_true_correlation": 0.8,
            "scalar_rmse_normalized": 0.1,
            "signal_coverage_pct": 100.0,
            "vary_angles": angles.tolist(),
            "true_norm": [0.1, 0.5, 0.9],
            "saw_norm": [0.2, 0.5, 0.8],
            "scalar_norm": [0.15, 0.55, 0.85],
            "scalar_counts_raw": [1.0, 2.0, 3.0],
            "scalar_count_min": 1.0,
            "scalar_count_max": 3.0,
        }
        metrics = {
            slice_key: {
                joint_name: dict(metric)
                for joint_name in ("lift", "pitch", "yaw")
            }
            for slice_key in ("q1_fixed", "q2_fixed")
        }
        vis = {
            slice_key: {
                "vary_angles": angles,
                "lift_scalar": np.array([1.0, 2.0, 3.0]),
                "pitch_scalar": np.array([2.0, 3.0, 4.0]),
                "yaw_scalar": np.array([3.0, 4.0, 5.0]),
            }
            for slice_key in ("q1_fixed", "q2_fixed")
        }

        with tempfile.TemporaryDirectory() as directory:
            captured = []
            original_close = scalar_module.plt.close

            def capture_close(fig):
                captured.append((tuple(fig.get_size_inches()), len(fig.axes)))
                return original_close(fig)

            with mock.patch.object(
                scalar_module.plt, "close", side_effect=capture_close
            ):
                inference._plot_scalar_readout_vs_sawtooth(metrics, directory)
                inference._plot_scalar_readout_vs_true(metrics, directory)
                inference._plot_sawtooth_vs_true_index(metrics, directory)
                inference._plot_scalar_readout_traces(vis, directory)
                inference._plot_scalar_error_summary(metrics, directory)

            self.assertEqual(
                captured,
                [
                    ((14.0, 8.0), 6),
                    ((14.0, 8.0), 6),
                    ((14.0, 8.0), 6),
                    ((10.0, 8.0), 2),
                    ((14.0, 8.0), 6),
                ],
            )
            filenames = (
                "scalar_readout_vs_sawtooth_index.png",
                "scalar_readout_vs_true_index.png",
                "sawtooth_vs_true_index.png",
                "scalar_readout_traces.png",
                "scalar_error_summary.png",
            )
            for filename in filenames:
                image = plt.imread(str(Path(directory) / filename))
                self.assertGreater(image.shape[0], 0)
                self.assertGreater(image.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
