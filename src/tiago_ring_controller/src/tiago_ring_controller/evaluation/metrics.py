"""Side-effect-free metrics extracted from the current analysis workflows."""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def decode_circular_centroid(rate_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decode per-column centroid and resultant strength as ``SingleRingModel``."""

    n_neurons = rate_matrix.shape[0]
    angles = 2.0 * np.pi * np.arange(n_neurons) / n_neurons
    centroid_idx = np.full(rate_matrix.shape[1], np.nan)
    strength = np.zeros(rate_matrix.shape[1])

    for time_idx in range(rate_matrix.shape[1]):
        activity = rate_matrix[:, time_idx]
        if np.sum(activity) <= 0:
            continue
        resultant = np.sum(activity * np.exp(1j * angles))
        theta = np.angle(resultant) % (2.0 * np.pi)
        centroid_idx[time_idx] = theta / (2.0 * np.pi) * n_neurons
        strength[time_idx] = np.abs(resultant) / np.sum(activity)
    return centroid_idx, strength


def circular_signed_difference(target_idx: Any, current_idx: Any, size: float) -> Any:
    """Return the signed half-open circular difference ``[-N/2, N/2)``."""

    delta = target_idx - current_idx
    return ((delta + size / 2.0) % size) - size / 2.0


def circular_index_nearest(decoded: Any, true: Any, size: float) -> np.ndarray:
    """Map decoded indices onto the copy nearest each true index."""

    decoded_array = np.asarray(decoded, dtype=float)
    true_array = np.asarray(true, dtype=float)
    n = float(size)
    return true_array + ((decoded_array - true_array + n / 2.0) % n - n / 2.0)


def circular_index_error(decoded: Any, true: Any, size: float) -> np.ndarray:
    """Return the multi-ring analysis's signed circular index error."""

    decoded_array = np.asarray(decoded, dtype=float)
    true_array = np.asarray(true, dtype=float)
    n = float(size)
    return ((decoded_array - true_array + n / 2.0) % n) - n / 2.0


def circular_angle_error(decoded_angle: Any, true_angle: Any) -> np.ndarray:
    """Return absolute wrapped angular error in radians."""

    return np.abs(np.angle(np.exp(1j * (np.asarray(decoded_angle) - true_angle))))


def centroid_change(
    centroid_idx: Sequence[float],
    size: float,
    difference_function: Optional[Callable[[Any, Any], Any]] = None,
    preserve_input_type: bool = False,
) -> np.ndarray:
    """Return per-step circular change with a leading NaN."""

    centroid = (
        centroid_idx
        if preserve_input_type
        else np.asarray(centroid_idx, dtype=float)
    )
    delta = np.full_like(centroid, np.nan, dtype=float)
    previous = centroid[:-1]
    current = centroid[1:]
    valid = ~(np.isnan(previous) | np.isnan(current))
    if difference_function is None:
        difference = circular_signed_difference(current, previous, size)
    else:
        difference = difference_function(current, previous)
    delta[1:] = np.where(
        valid,
        difference,
        np.nan,
    )
    return delta


def goal_error(
    centroid_idx: Sequence[float],
    goal_idx: float,
    size: float,
    difference_function: Optional[Callable[[Any, Any], Any]] = None,
    preserve_input_type: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return signed and absolute goal error for a centroid trace."""

    centroid = (
        centroid_idx
        if preserve_input_type
        else np.asarray(centroid_idx, dtype=float)
    )
    if difference_function is None:
        difference = circular_signed_difference(goal_idx, centroid, size)
    else:
        difference = difference_function(goal_idx, centroid)
    signed = np.where(
        np.isnan(centroid),
        np.nan,
        difference,
    )
    return signed, np.abs(signed)


def relative_goal_error_after_time(
    bin_centers_ms: Sequence[float],
    goal_distance: Sequence[float],
    population_size: int,
    avg_after_ms: float = 30000.0,
    preserve_input_type: bool = False,
) -> float:
    """Mirror the post-threshold mean and all-finite fallback in ``single_ring``."""

    if preserve_input_type:
        centers = bin_centers_ms
        relative_error = goal_distance / float(population_size)
    else:
        centers = np.asarray(bin_centers_ms, dtype=float)
        relative_error = np.asarray(goal_distance, dtype=float) / float(
            population_size
        )
    mask = centers > avg_after_ms
    if np.any(mask):
        post_values = relative_error[mask]
        post_finite = post_values[np.isfinite(post_values)]
        post_mean = float(np.mean(post_finite)) if post_finite.size > 0 else np.nan
    else:
        post_mean = np.nan

    if np.isfinite(post_mean):
        return post_mean
    all_finite = relative_error[np.isfinite(relative_error)]
    all_mean = float(np.mean(all_finite)) if all_finite.size > 0 else np.nan
    if np.isfinite(all_mean):
        return all_mean
    return np.nan


def movement_direction(desired_displacement: float) -> float:
    """Return the trial direction using the existing 1e-9 dead band."""

    if abs(desired_displacement) > 1e-9:
        return float(np.sign(desired_displacement))
    return 0.0


def aligned_position_error(
    goal_position: float,
    final_position: float,
    desired_displacement: float,
) -> float:
    """Positive means undershoot and negative means overshoot."""

    return (goal_position - final_position) * movement_direction(desired_displacement)


def settling_time(
    position_error: Sequence[float],
    time_s: Sequence[float],
    tolerance: float = 0.05,
    consecutive_steps: int = 5,
) -> float:
    """Return the first strict-tolerance run, else the last sample time."""

    error = np.abs(np.asarray(position_error))
    times = np.asarray(time_s)
    for index in range(len(error) - consecutive_steps + 1):
        if np.all(error[index:index + consecutive_steps] < tolerance):
            return float(times[index])
    return float(times[-1])


def overshoot(
    joint_position: Sequence[float],
    goal_position: float,
    preserve_input_type: bool = False,
) -> float:
    """Return maximum excursion beyond the goal using visualizer semantics."""

    positions = (
        joint_position if preserve_input_type else np.asarray(joint_position)
    )
    distance = positions - goal_position
    start_side = np.sign(positions[0] - goal_position)
    past_goal = distance * (-start_side)
    return float(np.max(np.clip(past_goal, 0, None)))


def collector_batch_statistics(
    trials: Sequence[Mapping[str, Any]],
    prediction_key: str = "decoder_predicted_dq_rad",
    settled_stop_reason: str = "drive_settled",
) -> Dict[str, Any]:
    """Compute the current Ring/PID collector batch statistics schema.

    Pass ``prediction_key='predicted_dq_rad'`` and
    ``settled_stop_reason='error_settled'`` for the PID schema.
    """

    scalars = [trial["scalars"] for trial in trials]
    abs_errors = np.array([row["abs_position_error_rad"] for row in scalars])
    aligned = np.array([row["aligned_error_rad"] for row in scalars])
    desired = np.array([row["dq_desired"] for row in scalars])
    actual = np.array([row["dq_actual"] for row in scalars])
    predicted = np.array([row[prediction_key] for row in scalars])

    settled_field = "%s_count" % settled_stop_reason
    return {
        "n_trials": len(trials),
        "mean_abs_error_rad": float(np.mean(abs_errors)),
        "std_abs_error_rad": float(np.std(abs_errors)),
        "median_abs_error_rad": float(np.median(abs_errors)),
        "max_abs_error_rad": float(np.max(abs_errors)),
        "min_abs_error_rad": float(np.min(abs_errors)),
        "mean_aligned_error_rad": float(np.mean(aligned)),
        "std_aligned_error_rad": float(np.std(aligned)),
        "mean_dq_desired_rad": float(np.mean(desired)),
        "mean_dq_actual_rad": float(np.mean(actual)),
        "mean_dq_pred_rad": float(np.mean(predicted)),
        "mae_pred_vs_desired_rad": float(np.mean(np.abs(predicted - desired))),
        "mae_pred_vs_actual_rad": float(np.mean(np.abs(predicted - actual))),
        "success_rate_le_0p02_rad": float(np.mean(abs_errors <= 0.02)),
        "success_rate_le_0p05_rad": float(np.mean(abs_errors <= 0.05)),
        "success_rate_le_0p10_rad": float(np.mean(abs_errors <= 0.10)),
        "max_steps_count": int(
            sum(1 for row in scalars if row["stop_reason"] == "max_steps")
        ),
        settled_field: int(
            sum(1 for row in scalars if row["stop_reason"] == settled_stop_reason)
        ),
    }


def analysis_region_statistics(
    region_trials: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute ``TiagoDecoderAnalysis._compute_region_statistics`` exactly."""

    abs_errors = np.array(
        [trial["abs_position_error_rad"] for trial in region_trials], dtype=float
    )
    aligned = np.array(
        [trial["aligned_error_rad"] for trial in region_trials], dtype=float
    )
    desired = np.array([trial["dq_desired"] for trial in region_trials], dtype=float)
    actual = np.array([trial["dq_actual"] for trial in region_trials], dtype=float)
    predicted = np.array(
        [trial["decoder_predicted_dq_rad"] for trial in region_trials], dtype=float
    )
    present = len(abs_errors) > 0
    return {
        "n_trials": int(len(region_trials)),
        "mean_abs_error_rad": float(np.mean(abs_errors)) if present else None,
        "std_abs_error_rad": float(np.std(abs_errors)) if present else None,
        "median_abs_error_rad": float(np.median(abs_errors)) if present else None,
        "max_abs_error_rad": float(np.max(abs_errors)) if present else None,
        "min_abs_error_rad": float(np.min(abs_errors)) if present else None,
        "mean_aligned_error_rad": float(np.mean(aligned)) if present else None,
        "std_aligned_error_rad": float(np.std(aligned)) if present else None,
        "mean_dq_desired_rad": float(np.mean(desired)) if present else None,
        "mean_dq_actual_rad": float(np.mean(actual)) if present else None,
        "mean_dq_pred_rad": float(np.mean(predicted)) if present else None,
        "mae_pred_vs_desired_rad": (
            float(np.mean(np.abs(predicted - desired))) if present else None
        ),
        "mae_pred_vs_actual_rad": (
            float(np.mean(np.abs(predicted - actual))) if present else None
        ),
        "success_rate_abs_error_le_0p02_rad": (
            float(np.mean(abs_errors <= 0.02)) if present else None
        ),
        "success_rate_abs_error_le_0p05_rad": (
            float(np.mean(abs_errors <= 0.05)) if present else None
        ),
        "success_rate_abs_error_le_0p10_rad": (
            float(np.mean(abs_errors <= 0.10)) if present else None
        ),
        "max_steps_count": int(
            sum(1 for trial in region_trials if trial["stop_reason"] == "max_steps")
        ),
        "drive_settled_count": int(
            sum(
                1
                for trial in region_trials
                if trial["stop_reason"] == "drive_settled"
            )
        ),
    }


def scalar_readout_metrics(
    saw_indices: Sequence[float],
    true_indices: Sequence[float],
    scalar_counts: Sequence[float],
    output_ring_size: int,
    sawtooth_errors_deg: Sequence[float] = (),
    vary_angles: Optional[Sequence[float]] = None,
    constant_signal_callback: Optional[Callable[[float, float], None]] = None,
    legacy_vary_angles_tolist: bool = False,
) -> Dict[str, Any]:
    """Preserve the scalar decoder's normalization and noncircular index MAE."""

    saw = np.asarray(saw_indices, dtype=float)
    true = np.asarray(true_indices, dtype=float)
    scalar = np.asarray(scalar_counts, dtype=float)
    saw_norm = saw / output_ring_size
    true_norm = true / output_ring_size

    scalar_min = float(np.nanmin(scalar))
    scalar_max = float(np.nanmax(scalar))
    scalar_span = scalar_max - scalar_min
    scalar_is_constant = scalar_span <= 1e-6
    if scalar_is_constant:
        if constant_signal_callback is not None:
            constant_signal_callback(scalar_min, scalar_max)
        scalar_norm = np.full_like(scalar, np.nan, dtype=float)
    else:
        scalar_norm = (scalar - scalar_min) / (scalar_span + 1e-9)

    valid = ~np.isnan(saw)
    n_valid = int(np.sum(valid))
    coverage = 100.0 * n_valid / len(saw)
    if n_valid >= 2:
        index_mae = float(np.nanmean(np.abs(saw - true)))
    else:
        index_mae = float("nan")

    if n_valid >= 2 and not scalar_is_constant:
        scalar_vs_saw = float(np.corrcoef(saw_norm[valid], scalar_norm[valid])[0, 1])
        scalar_vs_true = float(
            np.corrcoef(true_norm[valid], scalar_norm[valid])[0, 1]
        )
        scalar_rmse = float(
            np.sqrt(np.mean((scalar_norm[valid] - true_norm[valid]) ** 2))
        )
    else:
        scalar_vs_saw = float("nan")
        scalar_vs_true = float("nan")
        scalar_rmse = float("nan")

    errors = list(sawtooth_errors_deg)
    result = {
        "sawtooth_mae_deg": float(np.mean(errors)) if errors else float("nan"),
        "sawtooth_index_mae": index_mae,
        "scalar_vs_sawtooth_correlation": scalar_vs_saw,
        "scalar_vs_true_correlation": scalar_vs_true,
        "scalar_rmse_normalized": scalar_rmse,
        "signal_coverage_pct": coverage,
    }
    if vary_angles is not None:
        result["vary_angles"] = (
            vary_angles.tolist()
            if legacy_vary_angles_tolist
            else np.asarray(vary_angles).tolist()
        )
    result.update(
        {
            "true_norm": true_norm.tolist(),
            "saw_norm": saw_norm.tolist(),
            "scalar_norm": scalar_norm.tolist(),
            "scalar_counts_raw": scalar.tolist(),
            "scalar_count_min": scalar_min,
            "scalar_count_max": scalar_max,
        }
    )
    return result


__all__ = [
    "aligned_position_error",
    "analysis_region_statistics",
    "centroid_change",
    "circular_angle_error",
    "circular_index_error",
    "circular_index_nearest",
    "circular_signed_difference",
    "collector_batch_statistics",
    "decode_circular_centroid",
    "goal_error",
    "movement_direction",
    "overshoot",
    "relative_goal_error_after_time",
    "scalar_readout_metrics",
    "settling_time",
]
