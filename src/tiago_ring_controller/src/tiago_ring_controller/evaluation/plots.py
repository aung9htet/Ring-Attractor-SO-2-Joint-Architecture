"""Pure data preparation used by the existing matplotlib workflows."""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


RING_COLOR = "#C0392B"
PID_COLOR = "#2980B9"
SETTLE_TOLERANCE = 0.05
SETTLE_STEPS = 5


def goal_from_timeseries(
    timeseries: Mapping[str, Any],
    preserve_input_type: bool = False,
) -> np.ndarray:
    """Reconstruct q_goal with the visualizer's pointwise expression."""

    if preserve_input_type:
        return timeseries["joint_position"] + timeseries["position_error"]
    return np.asarray(timeseries["joint_position"]) + np.asarray(
        timeseries["position_error"]
    )


def final_absolute_error(timeseries: Mapping[str, Any]) -> float:
    """Return the collector visualizer's final absolute error."""

    return abs(float(np.asarray(timeseries["position_error"])[-1]))


def representative_trial_index(final_errors: Sequence[float]) -> Optional[int]:
    """Return the zero-based trial nearest the median, using first-tie argmin."""

    errors = np.asarray(final_errors, dtype=float)
    if len(errors) == 0:
        return None
    return int(np.argmin(np.abs(errors - np.median(errors))))


def gain_imbalance(
    timeseries: Mapping[str, Any],
    preserve_input_type: bool = False,
) -> np.ndarray:
    """Return the plotted ``right - left`` gain spike series."""

    if preserve_input_type:
        return timeseries["right_gain_spikes"] - timeseries["left_gain_spikes"]
    return np.asarray(timeseries["right_gain_spikes"]) - np.asarray(
        timeseries["left_gain_spikes"]
    )


def mean_std_box(data: Sequence[float]) -> Tuple[float, float, float]:
    """Return mean and mean±population-std used by the custom bar glyph."""

    values = np.asarray(data)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, mean - std, mean + std


def paired_by_trial_index(
    ring_values: Sequence[float],
    pid_values: Sequence[float],
    preserve_input_type: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preserve the visualizer's index matching (which is not true pairing)."""

    if preserve_input_type:
        ring = ring_values
        pid = pid_values
    else:
        ring = np.asarray(ring_values)
        pid = np.asarray(pid_values)
    count = min(len(ring), len(pid))
    return ring[:count], pid[:count]


def break_circular_wraps_for_plot(
    centroid_idx: Sequence[float],
    population_size: int,
    preserve_input_type: bool = False,
) -> np.ndarray:
    """Insert a NaN at each displayed jump larger than half the ring."""

    values = (
        centroid_idx.copy()
        if preserve_input_type
        else np.asarray(centroid_idx).copy()
    )
    for index in range(1, len(values)):
        if np.isnan(values[index]) or np.isnan(values[index - 1]):
            continue
        if abs(values[index] - values[index - 1]) > population_size / 2:
            values[index] = np.nan
    return values


def normalize_profile_for_display(profile: Sequence[float]) -> np.ndarray:
    """Min-max normalize a profile with the multi-ring display convention."""

    values = np.asarray(profile, dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return (values - minimum) / (maximum - minimum + 1e-9)


def evenly_spaced_indices(length: int, count: int) -> Sequence[int]:
    """Select indices exactly as ``DemoGraphRecorder`` does."""

    if length <= 0:
        return []
    if count >= length:
        return list(range(length))

    raw = np.linspace(0, length - 1, count)
    chosen = []
    used = set()
    for value in raw:
        index = int(round(value))
        index = max(0, min(length - 1, index))
        if index not in used:
            chosen.append(index)
            used.add(index)

    if len(chosen) < count:
        for index in range(length):
            if index not in used:
                chosen.append(index)
                used.add(index)
            if len(chosen) == count:
                break

    chosen = sorted(chosen)
    if 0 not in chosen:
        chosen[0] = 0
    if length - 1 not in chosen:
        chosen[-1] = length - 1
    return chosen


def smooth_circular_counts(counts: Sequence[float]) -> np.ndarray:
    """Apply the demo's display-only three-tap circular smoothing."""

    values = np.asarray(counts, dtype=float)
    if values.size < 3:
        return values
    return (
        0.15 * np.roll(values, 1)
        + 0.70 * values
        + 0.15 * np.roll(values, -1)
    )


def circular_centroid_from_counts(counts: Sequence[float]) -> float:
    """Return the demo's nonnegative circular activity centroid."""

    values = np.maximum(np.asarray(counts, dtype=float), 0.0)
    size = len(values)
    if size == 0 or np.sum(values) <= 1.0e-9:
        return 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    resultant = np.sum(values * np.exp(1j * theta)) / np.sum(values)
    if np.abs(resultant) <= 1.0e-9:
        return 0.0
    return float(np.angle(resultant) % (2.0 * np.pi))


def theta_to_ring_index(theta: float, population_size: int) -> float:
    return float(
        (theta % (2.0 * np.pi)) * population_size / (2.0 * np.pi)
    )


__all__ = [
    "PID_COLOR",
    "RING_COLOR",
    "SETTLE_STEPS",
    "SETTLE_TOLERANCE",
    "break_circular_wraps_for_plot",
    "circular_centroid_from_counts",
    "evenly_spaced_indices",
    "final_absolute_error",
    "gain_imbalance",
    "goal_from_timeseries",
    "mean_std_box",
    "normalize_profile_for_display",
    "paired_by_trial_index",
    "representative_trial_index",
    "smooth_circular_counts",
    "theta_to_ring_index",
]
